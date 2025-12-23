import os
import cv2
import subprocess
import multiprocessing
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. 核心参数配置 ---
SOURCE_DIR = r"H:\Videos" # 视频源目录
SAVE_DIR = r"E:\process\motion" # 导出剪辑后视频的目录
NUM_PROCESSES = 6        # 并发任务数，数量越高理论速度越快，但CPU、GPU和硬盘读写消耗越高
STRIDE = 10               # 跳帧：每 10 帧检测一次。数值越大处理越快，数值越小越灵敏
BUFFER_SEC = 3           # 剪辑前后保留的缓冲时间（秒）
MIN_MOTION_AREA = 500    # 运动阈值：变化像素点超过此值视为有物体动（夜视建议 500-800）
STOP_AFTER_SILENT = 15   # 画面无变化超过 15 秒则自动断开剪辑

def get_video_info(path):
    """获取视频的基本参数"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=avg_frame_rate,duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', path]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().split('\n')
        fps = eval(res[0]) if '/' in res[0] else float(res[0])
        duration = float(res[1]) if len(res) > 1 else 2700.0
        return fps, duration
    except Exception as e:
        return 25.0, 0.0

def process_motion_video(video_name):
    video_path = os.path.join(SOURCE_DIR, video_name)
    save_name_base = os.path.splitext(video_name)[0]
    
    # --- 实时进度打印 ---
    print(f"🕒 [正在分析] >>> {video_name}")
    sys.stdout.flush()

    fps, duration = get_video_info(video_path)
    if duration == 0:
        return f"❌ [跳过] 文件损坏或无法读取: {video_name}"

    # 尝试开启 d3d11va 硬件加速解码，降低 CPU 压力
    ffmpeg_cmd = [
        'ffmpeg', '-loglevel', 'error', '-hwaccel', 'd3d11va', '-reinit_filter', '0',
        '-i', video_path,
        '-vf', f'fps={fps},scale=640:360',  # 缩小分辨率检测，速度提升数倍
        '-f', 'image2pipe', '-pix_fmt', 'gray', '-vcodec', 'rawvideo', '-'
    ]
    
    pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    prev_frame = None
    active_seconds = set()
    frame_idx = 0
    
    try:
        # 1. 扫描阶段
        while True:
            raw_frame = pipe.stdout.read(640 * 360) # 读取灰度图数据
            if not raw_frame: break
            
            if frame_idx % STRIDE == 0:
                curr_frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((360, 640))
                if prev_frame is not None:
                    # 帧差法计算
                    frame_diff = cv2.absdiff(prev_frame, curr_frame)
                    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                    motion_score = np.count_nonzero(thresh) # 统计变化像素点数量
                    
                    if motion_score > MIN_MOTION_AREA:
                        active_seconds.add(int(frame_idx / fps))
                prev_frame = curr_frame
            frame_idx += 1

        pipe.stdout.close()
        stderr_output = pipe.stderr.read().decode()
        pipe.wait()

        if stderr_output:
            print(f"⚠️ [内核警告] {video_name}: {stderr_output[:100]}...")

        if not active_seconds:
            return f"⚪ [静止跳过] {video_name}"

        # 2. 逻辑分段 (满足 15 秒无动作自动断开)
        sorted_secs = sorted(list(active_seconds))
        segments = []
        if sorted_secs:
            start, end = sorted_secs[0], sorted_secs[0]
            for s in sorted_secs[1:]:
                if s <= end + STOP_AFTER_SILENT:
                    end = s
                else:
                    segments.append((max(0, start - BUFFER_SEC), min(duration, end + BUFFER_SEC)))
                    start, end = s, s
            segments.append((max(0, start - BUFFER_SEC), min(duration, end + BUFFER_SEC)))

        # 3. 物理剪辑导出
        clip_count = 0
        for i, (ts, te) in enumerate(segments):
            dur = te - ts
            if dur < 1.0: continue # 忽略小于 1 秒的瞬时闪烁
            
            out_file = os.path.join(SAVE_DIR, f"{save_name_base}_part{i}.mp4")
            res = subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error', '-fflags', '+genpts',
                '-ss', str(round(ts, 2)), '-t', str(round(dur, 2)),
                '-i', video_path, '-c', 'copy', '-tag:v', 'hvc1', '-an', out_file
            ], capture_output=True)
            
            if res.returncode == 0:
                clip_count += 1
            else:
                print(f"❌ [剪辑出错] {video_name} 段{i}: {res.stderr.decode()[:100]}")

        return f"✅ [处理完成] {video_name} -> 检出 {clip_count} 个动态片段"

    except Exception as e:
        if pipe: pipe.terminate()
        return f"🔥 [运行时崩溃] {video_name}: {str(e)}"

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    # 支持多种视频格式
    video_exts = ('.mp4', '.mkv', '.avi', '.mov', '.flv')
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(video_exts)]
    
    print(f"==========================================")
    print(f"🚀 监控视频自动化分析引擎 v2.0")
    print(f"待处理总数: {len(files)}")
    print(f"核心策略: 帧间位移检测 + 静止15s自动断开")
    print(f"==========================================\n")

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        # 提交所有任务
        futures = {executor.submit(process_motion_video, f): f for f in files}
        
        # 实时获取结果
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
                sys.stdout.flush() # 强制刷新控制台，实时看到结果
            except Exception as e:
                print(f"💥 线程执行异常: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
