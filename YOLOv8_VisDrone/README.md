# YOLOv8 + VisDrone

以YOLOv8为基础模型，在VisDrone数据集上完成目标检测。

任务难点在于VisDrone数据集有许多远距离的小目标，为了提升对这些小目标的检测精度，需要做如下改进：

1. 改进YOLOv8模型，引入P2小目标检测头、BiFPN特征金字塔网络。
2. 通过滑窗切片，将高分辨率图像切成640*480的小分辨率图像，使小目标在图像中的相对占比增大。

目标检测中对小目标的定义有两大类，分别是基于相对尺度定义（即待检测目标在整体图像中所占的相对面积较小）和基于绝对尺度定义（即从目标绝对像素大小这一角度考虑来对小目标进行定义）。目前最为通用的定义来自于目标检测领域的通用数据集——MS COCO数据集，将小目标定义为分辨率小于32像素×32像素的目标。

## 模型训练

## 模型验证/测试

调用`val`函数，参数可以参考[官方文档](https://docs.ultralytics.com/zh/modes/val/#arguments-for-yolo-model-validation)。

# 参考文献

- [YOLOv5 小目标检测、无人机视角小目标检测](https://blog.csdn.net/u012505617/article/details/121753656)
- [基于YOLOv8训练无人机视角Visdrone2019数据集](https://blog.csdn.net/weixin_45679938/article/details/142439297)
- [VisDrone 数据集](https://docs.ultralytics.com/zh/datasets/detect/visdrone/#dataset-yaml)
