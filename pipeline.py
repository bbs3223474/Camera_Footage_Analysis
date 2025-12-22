import os
import subprocess
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import onnxruntime as ort

# --- 核心参数配置 ---
SOURCE_DIR = r"H:\Videos"  # 你的视频源路径
SAVE_DIR = r"E:\process\clips"      # 结果保存路径
MODEL_PATH = "yolov8n.onnx"
NUM_PROCESSES = 6        # 并发进程数，5700X3D 建议 4-6 个进程，与下方STRIDE配合修改，STRIDE越大，运算压力越小，并发数可以越多
STRIDE = 10              # 跳帧数量，每隔多少帧分析一次，可适当增加以提高解析速度
IMG_SIZE = 1024          # 统一推理分辨率，中高端GPU推荐至少1024分辨率以提高分析精度
CONF_LEVEL = 0.35        # 最低可信度，增大数字以提高分析精度，降低数字以覆盖更全面的结果
BUFFER_SEC = 2           # 缓冲秒数，检测到动作后，额外截取之前或之后多少秒的视频
TARGET_CLASSES = [0, 1, 2, 3] # 人、自行车、汽车、摩托车

def get_video_info(path):
    """使用 ffprobe 获取帧率和时长，即使索引损坏也能尝试估算"""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split('\n')
        fps = eval(lines[0]) if '/' in lines[0] else float(lines[0])
        duration = float(lines[1]) if len(lines) > 1 else 2700.0
        return fps, duration
    except:
        return 25.0, 2700.0 # 默认 25帧，45分钟兜底

def process_single_video(video_name):
    video_path = os.path.join(SOURCE_DIR, video_name)
    save_name_base = os.path.splitext(video_name)[0]
    
    fps, duration = get_video_info(video_path)
    print(f"[*] 启动任务: {video_name} ({fps} FPS)")

    # 1. 启动 FFmpeg 管道模式
    # -vf fps={fps/STRIDE}: 让 FFmpeg 帮我们在解码层跳帧，极大减轻 Python 压力
    # -s {IMG_SIZE}x{IMG_SIZE}: 强制缩放到推理尺寸
    # -f image2pipe: 输出原始图像流
    ffmpeg_cmd = [
        'ffmpeg', 
        '-loglevel', 'error',
        '-hwaccel', 'd3d11va',        # 1. 开启 D3D11 硬件加速接口
        '-hwaccel_device', '0',        # 2. 指定第一块显卡
        '-i', video_path,
        '-vf', f'fps={fps/STRIDE},scale={IMG_SIZE}:{IMG_SIZE}', # 硬件层缩放
        '-f', 'image2pipe', 
        '-pix_fmt', 'bgr24', 
        '-vcodec', 'rawvideo', 
        '-'
    ]
    
    # 2. 初始化 DirectML
    opts = ort.SessionOptions()
    session = ort.InferenceSession(
        MODEL_PATH, 
        sess_options=opts, 
        providers=[('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name

    active_seconds = set()
    pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)
    
    # 每帧的数据大小 (1024*1024*3 字节)
    frame_size = IMG_SIZE * IMG_SIZE * 3
    count = 0

    try:
        while True:
            # 直接从管道读取原始字节
            raw_frame = pipe.stdout.read(frame_size)
            if not raw_frame: break
            
            # 快速转换为张量
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((3, IMG_SIZE, IMG_SIZE))
            img = frame.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # RX 9070 推理
            outputs = session.run(None, {input_name: img})
            preds = np.squeeze(outputs[0])
            
            # 简单解析逻辑 (YOLOv8 输出 [84, 21504])
            scores = np.max(preds[4:, :], axis=0)
            class_ids = np.argmax(preds[4:, :], axis=0)
            
            if np.any((scores > CONF_LEVEL) & np.isin(class_ids, TARGET_CLASSES)):
                current_sec = (count * STRIDE) / fps
                active_seconds.add(int(current_sec))
            
            count += 1
            if count % 100 == 0:
                print(f"  > {video_name}: 已处理 {int((count*STRIDE)/fps)} 秒...")

        pipe.stdout.close()
        pipe.wait()

        if not active_seconds:
            return f"[-] 未发现目标: {video_name}"

        # 3. 合并片段并剪辑
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

        # 4. 修复式剪辑导出
        for i, (ts, te) in enumerate(segments):
            out_file = os.path.join(SAVE_DIR, f"{save_name_base}_part{i}.mp4")
            # +genpts 和 -tag:v hvc1 解决 Win11 不识别问题
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error', '-fflags', '+genpts',
                '-ss', str(round(ts, 2)), '-t', str(round(te-ts, 2)),
                '-i', video_path, '-c', 'copy', '-tag:v', 'hvc1', '-an', out_file
            ])
        
        return f"[+] 成功: {video_name} (导出 {len(segments)} 段)"

    except Exception as e:
        if pipe: pipe.terminate()
        return f"[!] 报错: {video_name} -> {str(e)}"

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.mp4', '.mkv', '.avi'))]
    
    print(f"==========================================")
    print(f"🚀 GPU 管道流引擎已就绪")
    print(f"模式: FFmpeg Pipe + DirectML 推理")
    print(f"==========================================\n")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        futures = {executor.submit(process_single_video, f): f for f in files}
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
