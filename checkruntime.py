import onnxruntime as ort
print("ONNX Runtime 导入成功！版本:", ort.__version__)
print("可用 Provider:", ort.get_available_providers())
