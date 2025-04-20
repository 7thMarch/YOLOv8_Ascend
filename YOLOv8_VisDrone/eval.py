import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
random.seed(0)

import glob
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def box_iou(box1, box2):
    """
    计算两个框的 IoU（交并比）
    输入：
    - box1: [x1, y1, x2, y2]
    - box2: [x1, y1, x2, y2]
    
    输出：
    - IoU 值
    """
    # 计算面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算交集
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # 计算并集
    union_area = area1 + area2 - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def evaluate_with_confidence(xyxy_real, classes_real, xyxy_pred, classes_pred, confs, iou_thresh=0.5):
    """
    在单图或多图下评估预测结果，按置信度排序评估 P/R/F1/AP
    """
    predictions = list(zip(xyxy_pred, classes_pred, confs))
    predictions.sort(key=lambda x: x[2], reverse=True)  # 按置信度降序排列

    matched_gt = set()
    tp = []
    fp = []

    for pred_box, pred_cls, conf in predictions:
        match_found = False
        for i, (gt_box, gt_cls) in enumerate(zip(xyxy_real, classes_real)):
            if i in matched_gt:
                continue
            if pred_cls != gt_cls:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou >= iou_thresh:
                match_found = True
                matched_gt.add(i)
                break
        if match_found:
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.array(tp)
    fp = np.array(fp)

    # 累加 TP/FP，得到 Precision 和 Recall 曲线
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    total_gt = len(xyxy_real)

    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (total_gt + 1e-6)
    
    # F1 曲线
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # 如果 precision 为空（没有任何匹配）
    if len(precision) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 近似 mAP@0.5：对 PR 曲线积分
    ap = np.trapz(precision, recall)

    # 返回最终最大 F1、最后一个 P/R、AP
    return precision[-1], recall[-1], f1.max(), ap

def read_yolo_txt_to_xyxy(txt_path, img_width, img_height):
    """
    读取 YOLO 标签文件（.txt），并转为 xyxy 格式（像素坐标）

    参数:
    - txt_path: 标签文件路径
    - img_width, img_height: 图像宽高（像素）

    返回:
    - numpy 数组 (N, 5)，格式: [class_id, x1, y1, x2, y2]
    """
    boxes = []
    classes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            x1 = (xc - w / 2) * img_width
            y1 = (yc - h / 2) * img_height
            x2 = (xc + w / 2) * img_width
            y2 = (yc + h / 2) * img_height
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return classes, np.array(boxes)

if __name__ == '__main__':
    # 参数
    datayaml_path = 'E:\\MyReposity\\YOLOv8_Ascend\\YOLO算法改进\\core\\configs\\datasets\\VisDrone.yaml'
    checkpoint_path = r'E:\MyReposity\YOLOv8_Ascend\runs\detect\train_n\weights\best.pt'
    save_path = r'E:\MyReposity\YOLOv8_Ascend\YOLOv8_VisDrone\outcomes\n'

    # Load Images
    dataroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\images'
    labelroot_path = r'D:\datasets\VisDrone\VisDrone2019-DET-test-dev\labels'
    image_paths = glob.glob(dataroot_path+'\\*')

    # Load a model
    model = YOLO(checkpoint_path) # load a pretrained model (recommended for training)

    # Save Outcomes
    all_precision = []
    all_recall = []
    all_f1 = []
    all_map = []
    save_path = Path(save_path)
    for img_path in image_paths:
        img_path = Path(img_path)
        label_path = (labelroot_path / img_path.relative_to(dataroot_path)).with_suffix(".txt")

        img = cv2.imread(img_path)
        h,w,_ = img.shape
        classes_real, xyxy_real = read_yolo_txt_to_xyxy(label_path, w, h)
        
        # Predict with the model
        result = model(img_path)[0]  # predict on an image

        # Access the results
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box

        xyxy_pred = xyxy
        classes_pred = [cls.item() for cls in result.boxes.cls.int()]
        P, R, F1, mAP = evaluate_with_confidence(xyxy_real, classes_real, xyxy_pred, classes_pred, confs)

        print(f"Precision: {P:.3f}, Recall: {R:.3f}, F1: {F1:.3f}, mAP@0.5: {mAP:.3f}")
        all_precision.append(P)
        all_recall.append(R)
        all_f1.append(F1)
        all_map.append(mAP)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_f1 = np.mean(all_f1)
    avg_map = np.mean(all_map)

    print(f"Average Precision: {avg_precision:.3f}, Average Recall: {avg_recall:.3f}, Average F1: {avg_f1:.3f}, Average mAP: {avg_map:.3f}")

