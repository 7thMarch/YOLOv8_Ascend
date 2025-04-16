# YOLO算法改进

在本地系统上，基于Pytorch开发基于YOLO的小目标检测算法。

YOLO算法的部署主要依赖[ultralytics](https://github.com/ultralytics/ultralytics)库（实际开发时将ultralytics库直接安装到python环境中）。

## 模型结构修改

为了适应对小目标检测的需求，引入P2小目标检测层、BiFPN特征金字塔融合。

下图是YOLOv8网络结构的极简示意图，对应于[yolov8s.yaml](./core/configs/model/yolov8s.yaml)。

<img src=./Images/YOLOv8.png width='500px'>

### P2小目标检测层

正常的YOLOv8对象检测模型输出层是P3、P4、P5三个输出层，为了提升对小目标的检测能力，额外引入P2层（P2层做的卷积次数少，特征图的尺寸（分辨率）较大，更加利于小目标识别）。Backbone部分的结果没有改变，但是Neck跟Head部分模型结构做了调整。

因为没有引入新的结构，因此**只需要在描述模型结构的`yaml`文件中引入新的设计即可**。

下图是YOLOv8-p2网络结构的极简示意图，对应于[yolov8-p2.yaml](./core/configs/model/yolov8-p2.yaml)。

<img src=./Images/YOLOv8-p2.png width='500px'>

### BiFPN特征金字塔

BiFPN 是一种更聪明、更轻量、对小目标更友好的多尺度特征融合结构。在不明显增加模型体积的情况下，它能有效提升检测精度，特别是在小目标和复杂场景中。

主要改动点有以下几方面：

- 可学习的特征融合权重：利用可学习的权重参数对待融合特征进行加权，替代传统的简单加法或 concat
- 增加本层的跳跃连接：将本层融合前的原始特征作为额外的融合信息，使特征融合更加丰富

由于加入了新的模块BiConcat3和BiConcat2，**需要在`ultralytics/nn/task.py`中注册新的组件**。

```
# task.py
from ultralytics.nn.biFPN import BiConcat2, BiConcat3

# 添加bifpn_concat结构
elif m in [Concat, BiFPN_Concat2, BiFPN_Concat3]:
    c2 = sum(ch[x] for x in f)
```

```
# biFPN.py
import math
import numpy as np
import torch
import torch.nn as nn

class BiConcat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiConcat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiConcat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiConcat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
 
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)
```

下图是YOLOv8-p2-BiFPN1网络结构的极简示意图，对应于[yolov8-p2-BiFPN1.yaml](./core/configs/model/yolov8-p2-BiFPN.yaml)。

<img src=./Images/YOLOv8-p2-BiFPN1.png width='500px'>

## 模型训练与测试

### 数据集

利用COCO数据集进行模型的初步训练、测试，轻量化的要求下，可以使用COCO8、COCO128。

如果希望导入自己的小目标数据集，需要根据YOLO的格式保存，并将数据集信息保存至`yaml`文件中。

数据集组织如下：

```
- root_path
    - images
        - train
            - xxxxxxxx.png
            - ......
        - val
            - xxxxxxxx.png
            - ......
    - labels
        - train
            - xxxxxxxx.txt
            - ......
        - val
            - xxxxxxxx.png
            - ......
```

其中labels中应该包含若干行，每一行存储`id x y w h`，其中x,y,w,h是根据图像大小归一化的、范围0-1的浮点数，例如：

```
25 0.475759 0.414523 0.951518 0.672422
0 0.671279 0.617945 0.645759 0.726859
```

`yaml`文件的存储格式参考[coco.yaml](./core/configs/datasets/coco.yaml)。

### 模型结构

利用`yaml`文件组织新的模型结构，直接调用`ultralytics`中的`YOLO`模块构建模型：

```
from ultralytics import YOLO

model = YOLO('yolov8s-p2-BiFPN.yaml')
```

### 模型训练与测试

参考官方文档，直接调用接口进行模型的训练、测试。
