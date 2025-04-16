from ultralytics import YOLO

model = YOLO('yolov8s.yaml').load('yolov8s.pt')

path = model.export(format="onnx")  # export the model to ONNX format
