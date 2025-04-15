# AscendCL

## 1 导入资源

```
import acl  # AscnedCL推理所需库文件
import constants as const  # 其中包含acl相关状态值，用于判断推理时程序状态是否正常
```

## 2 初始化资源
```
ret = acl.init()
ret = acl.rt.set_device(device_id)  # 指定运算的Device
context, ret = acl.rt.create_context(device_id)  # 显式创建一个Context 
```


## 3 加载模型

```
# 加载模型
model_id, ret = acl.mdl.load_from_file(model_path)  # 加载离线模型文件, 返回标识模型的ID
model_desc = acl.mdl.create_desc()  # 初始化模型描述信息, 包括模型输入个数、输入维度、输出个数、输出维度等信息
ret = acl.mdl.get_desc(model_desc, model_id)  # 根据加载成功的模型的ID, 获取该模型的描述信息
```

## 4 数据加载

### 4.1 输入数据集

```
input_list = [img, ]  # 0. 初始化输入数据列表

input_dataset = acl.mdl.create_dataset()  # 1. 创建输入数据
input_num = acl.mdl.get_num_inputs(model_desc)  # 得到模型输入个数
for i in range(input_num):
    input_data = input_list[i]  # 2. 得到每个输入数据

    # 3. 得到每个输入数据流的指针(input_ptr)和所占字节数(size)
    size = input_data.size * input_data.itemsize  # 得到所占字节数
    bytes_data=input_data.tobytes()  # 将每个输入数据转换为字节流
    input_ptr=acl.util.bytes_to_ptr(bytes_data)  # 得到输入数据指针

    dataset_buffer = acl.create_data_buffer(input_ptr, size)  # 4. 为每个输入创建 buffer
    _, ret = acl.mdl.add_dataset_buffer(input_dataset, dataset_buffer)  # 5. 将每个 buffer 添加到输入数据中
print("Create model input dataset success")
```

### 4.2 输出数据集
```
# 准备输出数据集
output_dataset = acl.mdl.create_dataset()  # 1. 创建输出数据
output_size = acl.mdl.get_num_outputs(model_desc)  # 得到模型输出个数
for i in range(output_size):
    size = acl.mdl.get_output_size_by_index(model_desc, i)  # 得到每个输出所占内存大小

    # 2. 为输出分配内存
    buf, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_NORMAL_ONLY)
    # 3. 为每个输出创建 buffer
    dataset_buffer = acl.create_data_buffer(buf, size)
    # 4. 将每个 buffer 添加到输出数据中
    _, ret = acl.mdl.add_dataset_buffer(output_dataset, dataset_buffer)  

    if ret:  # 若分配出现错误, 则释放内存
        acl.rt.free(buf)
        acl.destroy_data_buffer(dataset_buffer)
print("Create model output dataset success")
```

## 5 模型推理

```
# 模型推理, 得到的输出将写入 output_dataset 中
ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
if ret != const.ACL_SUCCESS:  # 判断推理是否出错
    print("Execute model failed for acl.mdl.execute error ", ret)
```

## 6 输出数据解析

```
# 解析 output_dataset, 得到模型输出列表
model_output = [] # 模型输出列表
for i in range(output_size):
    buf = acl.mdl.get_dataset_buffer(output_dataset, i)  # 1. 获取每个输出buffer
    data_addr = acl.get_data_buffer_addr(buf)  # 获取输出buffer的地址
    size = int(acl.get_data_buffer_size(buf))  # 获取输出buffer的字节数
    byte_data = acl.util.ptr_to_bytes(data_addr, size)  # 2. 将指针转为字节流数据
    dims = tuple(acl.mdl.get_output_dims(model_desc, i)[0]["dims"])  # 从模型信息中得到每个输出的维度信息
    output_data = np.frombuffer(byte_data, dtype=np.float32).reshape(dims)  # 3. 将 output_data 以流的形式读入转化成 ndarray 对象
    model_output.append(output_data) # 添加到模型输出列表
```

### 7 释放资源

```
# 释放输入资源, 包括数据结构和内存
num = acl.mdl.get_dataset_num_buffers(input_dataset)  # 获取输入个数
for i in range(num):
    data_buf = acl.mdl.get_dataset_buffer(input_dataset, i)  # 获取每个输入buffer
    if data_buf:
        ret = acl.destroy_data_buffer(data_buf)  # 销毁每个输入buffer (销毁 aclDataBuffer 类型)
ret = acl.mdl.destroy_dataset(input_dataset)  # 销毁输入数据 (销毁 aclmdlDataset类型的数据)

# 释放输出资源, 包括数据结构和内存
num = acl.mdl.get_dataset_num_buffers(output_dataset)  # 获取输出个数
for i in range(num):
    data_buf = acl.mdl.get_dataset_buffer(output_dataset, i)   # 获取每个输出buffer
    if data_buf:
        ret = acl.destroy_data_buffer(data_buf)  # 销毁每个输出buffer (销毁 aclDataBuffer 类型)
ret = acl.mdl.destroy_dataset(output_dataset)  # 销毁输出数据 (销毁 aclmdlDataset类型的数据)

# 卸载模型
if model_id:
    ret = acl.mdl.unload(model_id)

# 释放模型描述信息
if model_desc:
    ret = acl.mdl.destroy_desc(model_desc)

# 释放 Context
if context:
    acl.rt.destroy_context(context)

# 释放Device
acl.rt.reset_device(device_id)
acl.finalize()

print("Release acl resource success") 
```