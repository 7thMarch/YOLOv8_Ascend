import cv2
from PIL import Image

from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
model = YOLO('yolov8s-p2.yaml').load('yolov8s.pt')

# # from PIL
# im1 = Image.open("images/bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# # from ndarray
# im2 = cv2.imread("images/bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])



# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO()  # load a pretrained model (recommended for training)

# # Use the model
# model.train(data="coco8.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# image_path = '/data/zhangpengjie/zhangpengjie/Workspace/Experiments/YOLO/images/img00234.jpg'
# results = model(image_path)  # predict on an image

# path = model.export(format="onnx")  # export the model to ONNX format
