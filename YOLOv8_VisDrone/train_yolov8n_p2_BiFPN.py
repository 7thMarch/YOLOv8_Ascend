import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

if __name__ == '__main__':
    modelcfg_path = 'E:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\model\\yolov8n-p2-BiFPN.yaml'
    checkpoint_path = 'E:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\checkpoints\\yolov8n.pt'
    datayaml_path = 'E:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\datasets\\VisDrone.yaml'

    # Load a model
    model = YOLO(modelcfg_path).load(checkpoint_path)  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=datayaml_path, imgsz=640, batch=16, workers=8, cache=False, epochs=100, device='cuda')  # train the model

    # Test
    metrics = model.val(data=datayaml_path, imgsz=640, split='test')  # evaluate model performance on the validation set

    # results = model("ultralytics\\assets\\bus.jpg")  # predict on an image
    # path = model.export(format="onnx", opset=13)  # export the model to ONNX format
