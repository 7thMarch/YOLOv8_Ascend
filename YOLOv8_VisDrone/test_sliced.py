import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
random.seed(0)

import glob
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

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
    save_path = r'E:\MyReposity\YOLOv8_Ascend\YOLOv8_VisDrone\outcomes\n_sliced'

    # Load Images
    dataroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\images'
    image_paths = glob.glob(dataroot_path+'\\*')

    # Load a model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        # model_path="sar/sar/weights/last.pt",
        # 模型权重文件
        model_path=checkpoint_path,
        confidence_threshold=0.3,
        device="cuda:0",  # or "cpu"
    )

    # Save Outcomes
    save_path = Path(save_path)
    for img in image_paths:
        image_path = Path(img)
        image = cv2.imread(image_path)

        height, width, depth = image.shape

        # 使用 Sahi 进行切片推理
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        detections = result.object_prediction_list
        
        # 处理 Sahi 的检测结果           
        # detection_data = []
        xyxy = []
        names = []
        confs = []
        class_names = {}
        for detection in detections:
            bbox = detection.bbox
            score = detection.score.value
            category_id = detection.category.id
            category_name = detection.category.name           

            # 保存分类名称
            if category_id not in class_names:
                class_names[category_id] = category_name

            xyxy.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            names.append(category_name)
            confs.append(score)

        
        save_detection_result_image(img, xyxy, names, confs, save_path / image_path.relative_to(dataroot_path))
        