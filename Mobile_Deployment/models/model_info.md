# 模型信息文档

**生成时间**: 2026-02-09  
**模型类型**: YOLOv3 + MobileNetV1  
**任务**: 实验鼠检测（二分类：mouse / other）

---

## 📦 文件清单

### 1. yolov3_mouse_fp32.onnx
- **格式**: ONNX (Open Neural Network Exchange)
- **大小**: 92.34 MB
- **精度**: FP32 (32位浮点)
- **Opset版本**: 11
- **来源**: PaddleDetection训练模型导出

### 2. infer_cfg.yml
- **用途**: 推理配置文件
- **包含信息**: 
  - 图像预处理参数
  - 输入尺寸
  - 归一化参数
  - NMS阈值

### 3. label_list.txt
- **用途**: 类别标签文件
- **内容**:
  ```
  mouse
  other
  ```

---

## 🔧 模型输入输出规格

### 输入张量

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `image` | [1, 3, 608, 608] | float32 | RGB图像，归一化后的像素值 |
| `im_shape` | [1, 2] | float32 | 原始图像尺寸 [height, width] |
| `scale_factor` | [1, 2] | float32 | 缩放因子 [scale_y, scale_x] |

### 输出张量

| 名称 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `multiclass_nms3_0.tmp_0` | [N, 6] | float32 | 检测结果：[class_id, score, x1, y1, x2, y2] |
| `multiclass_nms3_0.tmp_2` | [N] | int32 | 每个batch的检测框数量 |

**注意**: N是动态的，取决于检测到的目标数量

---

## 🖼️ 图像预处理流程

### 1. 读取图像
```python
import cv2
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### 2. Resize到608×608
```python
original_h, original_w = image.shape[:2]
target_size = 608

# 保持宽高比的resize
scale = target_size / max(original_h, original_w)
new_h = int(original_h * scale)
new_w = int(original_w * scale)

resized = cv2.resize(image, (new_w, new_h))

# 填充到608×608
padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
padded[:new_h, :new_w, :] = resized
```

### 3. 归一化
```python
# 转换为float32并归一化到[0, 1]
normalized = padded.astype(np.float32) / 255.0

# 应用ImageNet均值和标准差
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
normalized = (normalized - mean) / std
```

### 4. 转换为CHW格式
```python
# HWC -> CHW
image_chw = normalized.transpose(2, 0, 1)

# 添加batch维度
image_batch = np.expand_dims(image_chw, axis=0)
```

### 5. 准备其他输入
```python
im_shape = np.array([[original_h, original_w]], dtype=np.float32)
scale_factor = np.array([[scale, scale]], dtype=np.float32)
```

---

## 📊 后处理流程

### 1. 解析输出
```python
# boxes: [N, 6] - [class_id, score, x1, y1, x2, y2]
boxes = output[0]
num_boxes = output[1][0]

# 只取有效的检测框
valid_boxes = boxes[:num_boxes]
```

### 2. 过滤低置信度
```python
confidence_threshold = 0.5
filtered_boxes = valid_boxes[valid_boxes[:, 1] > confidence_threshold]
```

### 3. 映射回原始图像坐标
```python
for box in filtered_boxes:
    class_id = int(box[0])
    confidence = box[1]
    x1, y1, x2, y2 = box[2:6]
    
    # 坐标已经是原始图像尺寸，无需额外转换
    # 但需要确保在图像范围内
    x1 = max(0, min(x1, original_w))
    y1 = max(0, min(y1, original_h))
    x2 = max(0, min(x2, original_w))
    y2 = max(0, min(y2, original_h))
```

---

## 🎯 性能指标

### 训练集性能
- **mAP@0.5**: 93.63%
- **训练轮数**: 50 epochs
- **数据集**: 10,816张图像（8,653训练 + 2,163验证）

### 推理性能（预估）
- **模型大小**: 92.34 MB
- **推理延迟**: 
  - CPU (iPhone 12): ~300-500ms
  - GPU (Metal): ~100-200ms
- **内存占用**: ~200-300 MB

---

## 🔍 类别说明

| ID | 类别名 | 说明 |
|----|--------|------|
| 0 | mouse | 实验鼠 |
| 1 | other | 其他物体 |

---

## ⚠️ 注意事项

1. **Batch Size限制**: 由于使用了multiclass_nms3算子，模型只支持batch_size=1
2. **输入尺寸**: 固定为608×608，其他尺寸可能导致精度下降
3. **颜色空间**: 输入必须是RGB格式（不是BGR）
4. **坐标系统**: 输出坐标是绝对像素值，不是归一化坐标
5. **NMS**: 模型内部已包含NMS，无需额外处理

---

## 📱 移动端部署建议

### iOS (ONNX Runtime)
```swift
// 推荐配置
let options = ORTSessionOptions()
options.graphOptimizationLevel = .all
options.executionMode = .sequential

// 使用CoreML加速（如果可用）
options.appendCoreMLExecutionProvider()
```

### 优化建议
1. **量化**: 可在移动端使用ONNX Runtime的动态量化
2. **模型优化**: 使用onnxruntime-tools进行图优化
3. **缓存**: 缓存预处理后的图像以提高连续推理速度
4. **异步推理**: 在后台线程执行推理，避免阻塞UI

---

## 🔗 相关资源

- **PaddleDetection**: https://github.com/PaddlePaddle/PaddleDetection
- **ONNX Runtime**: https://onnxruntime.ai/
- **ONNX Runtime React Native**: https://github.com/microsoft/onnxruntime-react-native
- **训练文档**: `/hy-tmp/paddle_detection_mouse/Training_Pipeline.md`

---

**最后更新**: 2026-02-09

