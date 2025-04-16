from PIL import Image

from ultralytics import YOLO

ckp_path = 'G:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\checkpoints\\yolov8s.pt'
model = YOLO(ckp_path)

img_path = 'G:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\data\\coco8\\images\\val\\000000000036.jpg'
im1 = Image.open(img_path)

results = model.predict(source=im1)  # save plotted images
for result in results:
    print(result.boxes)  # Print detection boxes
    result.show()  # Display the annotated image
    result.save(filename="result.jpg")  # Save annotated image
