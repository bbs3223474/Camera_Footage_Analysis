import os
import cv2
import subprocess
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import onnxruntime as ort

# --- 2. 核心参数配置 ---
SOURCE_DIR = r"H:\Videos"  # 你的视频源路径
SAVE_DIR = r"E:\process\clips"      # 结果保存路径
MODEL_PATH = "yolov8n.onnx"          # 必须使用你导出的 ONNX 文件
NUM_PROCESSES = 6                    # RX 9070 建议 4 个进程并行
TARGET_CLASSES = [0, 1, 2, 3]        # 人、自行车、汽车、摩托车
CONF_LEVEL = 0.35
IMG_SIZE = 1024                      # 高分辨率识别小目标
STRIDE = 10                           # 隔帧检测
BUFFER_SEC = 2                       # 动作前后冗余时间

def get_video_info_via_ffmpeg(path):
    """
    方案一：使用 ffprobe 强制探测被损坏的视频时长和 FPS
    """
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2: return 25.0, 0.0
        
        # 处理 FPS (例如 "25/1")
        fps_str = lines[0]
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 25.0
        else:
            fps = float(fps_str)
            
        duration = float(lines[1])
        return fps, duration
    except:
        return 25.0, 0.0

def process_single_video(video_name):
    video_path = os.path.join(SOURCE_DIR, video_name)
    print(f"[*] [开始处理] {video_name}")
    save_name_base = os.path.splitext(video_name)[0]
    
    # 1. 强行获取视频参数
    print(f"[*] [1/5] 正在通过 FFprobe 获取时长: {video_name}")
    fps, duration = get_video_info_via_ffmpeg(video_path)
    print(f"[*] [2/5] 获取成功: {fps} FPS, 时长 {duration}s")
    if duration == 0:
        return f"[-] 失败: {video_name} (无法读取时长，文件可能严重损坏)"
    
    try:
        # 1. 直接创建底层的 DirectML 会话 (跳过 YOLO 的自检)
        providers = [('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        
        # 获取模型输入节点的名称
        input_name = session.get_inputs()[0].name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] [错误] OpenCV 无法打开文件: {video_name}")
            return f"[-] 失败: {video_name}"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        active_seconds = set()
        frame_idx = 0
        
        while True:
            print(f"[*] [3/5] 正在读取第一帧进行初始化...")
            ret, frame = cap.read()
            if not ret: break
            
            # 隔帧检测
            if frame_idx % STRIDE == 0:
                # 图像预处理 (缩放、归一化)
                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                img = img.transpose((2, 0, 1)) # HWC to CHW
                img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                
                # 执行推理 - 这将直接运行在你的 RX 9070 上
                outputs = session.run(None, {input_name: img})
                
                # 解析结果 (简单过滤：只要有对应类别的框，置信度达标就记录)
                # YOLOv8 ONNX 输出形状通常是 [1, 84, 21504]
                preds = np.squeeze(outputs[0])
                # 提取最高得分和对应的类别
                scores = np.max(preds[4:, :], axis=0)
                class_ids = np.argmax(preds[4:, :], axis=0)
                
                # 筛选符合条件的目标
                mask = (scores > CONF_LEVEL) & np.isin(class_ids, TARGET_CLASSES)
                if np.any(mask):
                    active_seconds.add(int(frame_idx / fps))
            
            frame_idx += 1
        cap.release()

        if not active_seconds:
            return f"[-] 未发现目标: {video_name}"

        # 2. 合并片段并剪辑 (FFmpeg 部分保持不变)
        sorted_secs = sorted(list(active_seconds))
        segments = []
        if sorted_secs:
            start, end = sorted_secs[0], sorted_secs[0]
            for s in sorted_secs[1:]:
                if s <= end + BUFFER_SEC + 1:
                    end = s
                else:
                    segments.append((max(0, start-BUFFER_SEC), min(duration, end+BUFFER_SEC)))
                    start, end = s, s
            segments.append((max(0, start-BUFFER_SEC), min(duration, end+BUFFER_SEC)))

        for i, (ts, te) in enumerate(segments):
            out = os.path.join(SAVE_DIR, f"{save_name_base}_c{i}.mp4")
            cmd = [
                'ffmpeg', '-y', 
                '-fflags', '+genpts',  # 关键：修复损坏的时间戳索引
                '-ss', str(round(ts, 2)), 
                '-t', str(round(dur, 2)), 
                '-i', video_path, 
                '-c', 'copy', 
                '-tag:v', 'hvc1',      # 关键：修复 Win11 缩略图/属性显示
                '-an', out_file
            ]
            subprocess.run(cmd, capture_output=True)

        return f"[+] 成功: {video_name} ({len(segments)}段)"

    except Exception as e:
        return f"[!] 失败: {video_name} -> {str(e)}"

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.mp4', '.mkv', '.avi'))]
    print(f"AMD RX 9070 底层加速引擎已启动... 并发进程: {NUM_PROCESSES}")
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        list(executor.map(process_single_video, files))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
