from ultralytics import YOLO

cfg_path = 'G:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\model\\yolov8s-p2-BiFPN.yaml'
ckp_path = 'G:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\checkpoints\\yolov8s.pt'

# Load a pretrained YOLO11n model
model = YOLO(cfg_path)
model.load(ckp_path)

yaml_path = 'G:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\datasets\\coco8.yaml'
# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=yaml_path,  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

metrics = model.val()
