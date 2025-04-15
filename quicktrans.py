import cv2
from PIL import Image

from ultralytics import YOLO

# model = YOLO("yolov8m.pt")
model = YOLO('yolov8s-p2.yaml').load('yolov8s.pt')

path = model.export(format="onnx")  # export the model to ONNX format
