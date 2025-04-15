# MindX SDK

## 1 导入资源

```
from mindx.sdk import Tensor  # mxVision 中的 Tensor 数据结构
from mindx.sdk import base  # mxVision 推理接口
```

## 2 初始化资源

```
base.mx_init()  # 初始化 mxVision 资源
DEVICE_ID = 0  # 设备id
model_path = 'model/yolov5s_bs1.om'  # 模型路径
model = base.model(modelPath=model_path, deviceId=DEVICE_ID)  # 初始化 base.model 类
```

## 3 数据前处理

```
# img -> npy (b, c, h ,w)
img = Tensor(img)  # 将numpy转为转为Tensor类
```

## 4 模型推理

```
output = model.infer([img])[0]
```


## 5 数据后处理

```
output.to_host()  # 将 Tensor 数据转移到内存
output = np.array(output)  # 将数据转为 numpy array 类型
```