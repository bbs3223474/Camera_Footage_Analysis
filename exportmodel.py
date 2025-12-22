from ultralytics import YOLO

# 1. 加载官方预训练的权重文件（会自动下载）
# 如果你追求极致速度，用 yolov8n.pt；如果追求更高精度，用 yolov8s.pt
model = YOLO("yolov8n.pt") 

# 2. 导出为 ONNX 格式
# imgsz: 设置为 1024 可以提升对小目标的识别率，非常适合监控场景
# dynamic: 设置为 True 可以处理不同尺寸的视频
# opset: 建议设为 12 以获得最佳兼容性
model.export(format="onnx", imgsz=1024, dynamic=True, opset=12)
