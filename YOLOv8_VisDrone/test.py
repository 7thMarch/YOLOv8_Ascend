import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
random.seed(0)

import glob
import cv2
from pathlib import Path
from ultralytics import YOLO

classToNames = {
  0: 'pedestrian',
  1: 'people',
  2: 'bicycle',
  3: 'car',
  4: 'van',
  5: 'truck',
  6: 'tricycle',
  7: 'awning-tricycle',
  8: 'bus',
  9: 'motor'
}

class_colors = {name: [random.randint(0, 255) for _ in range(3)] for name in classToNames.values()}

def save_detection_result_image(img_path, boxes, names, confs, save_path):
    """
    在图像上绘制检测框、类别和置信度，并保存结果图像。

    参数:
    - img_path: 输入图像路径
    - result: 模型单张图像的预测结果 (results[0])
    - save_dir: 保存结果图像的目录
    - model_names: 类别名称映射 (如 model.names)
    """

    img = cv2.imread(img_path)
    if img is None:
        print(f"[警告] 无法读取图像: {img_path}")
        return

    for box, name, conf in zip(boxes, names, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{name} {conf:.2f}"
        color = class_colors.get(name, (0, 255, 0))

        # 绘制检测框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # # 添加标签背景和文字
        # (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        # cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 保存结果图像
    cv2.imwrite(save_path, img)
    # print(f"[信息] 保存结果图像: {save_path}")


if __name__ == '__main__':
    # 参数
    datayaml_path = 'E:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\datasets\\VisDrone.yaml'
    checkpoint_path = r'E:\MyReposity\YOLOv8_Ascend\runs\detect\train_n\weights\best.pt'
    save_path = r'E:\MyReposity\YOLOv8_Ascend\YOLOv8_VisDrone\outcomes\n'

    # Load Images
    dataroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\images'
    image_paths = glob.glob(dataroot_path+'\\*')

    # Load a model
    model = YOLO(checkpoint_path) # load a pretrained model (recommended for training)

    # Save Outcomes
    save_path = Path(save_path)
    for img_path in image_paths:
        img_path = Path(img_path)
        
        # Predict with the model
        results = model(img_path)  # predict on an image

        # Access the results
        for result in results:
            xywh = result.boxes.xywh  # center-x, center-y, width, height
            xywhn = result.boxes.xywhn  # normalized
            xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
            xyxyn = result.boxes.xyxyn  # normalized
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
            confs = result.boxes.conf  # confidence score of each box

            save_detection_result_image(img_path, xyxy, names, confs, save_path / img_path.relative_to(dataroot_path))
