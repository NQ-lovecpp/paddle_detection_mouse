# Linux服务器端工作总结

**完成时间**: 2026-02-09  
**工作环境**: Linux服务器 (2× Tesla T4, CUDA 11.6, PaddlePaddle 2.5.1)

---

## ✅ 已完成的工作

### 1. 项目初始化 ✅
- 创建项目目录: `/hy-tmp/paddle_detection_mouse/Mobile_Deployment/`
- 编写完整的项目文档和指南
- 建立文件组织结构

### 2. 模型导出 (Paddle Inference格式) ✅
**任务**: 将训练好的模型导出为可部署的推理格式

**执行命令**:
```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/B_baseline_1gpu/yolov3_my_dog_mouse_voc/best_model.pdparams \
    --output_dir=output/inference_model_baseline
```

**输出文件**:
- `model.pdmodel` (125KB) - 网络结构
- `model.pdiparams` (93MB) - 模型权重  
- `infer_cfg.yml` (351B) - 推理配置

**状态**: ✅ 成功

---

### 3. 离线量化 (Post-Training Quantization) ✅
**任务**: 使用PTQ方法将FP32模型压缩为INT8

**背景**: 
- 原计划使用量化感知训练(QAT)，但因GPU内存不足失败
- 改用离线量化(PTQ)，无需重新训练，速度快（<1分钟）

**执行命令**:
```bash
python tools/post_quant.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/post_quant/yolov3_darknet53_ptq.yml \
    -o weights=output/B_baseline_1gpu/yolov3_my_dog_mouse_voc/best_model.pdparams \
    --output_dir=output/ptq_baseline_int8
```

**量化配置**:
- 量化方式: INT8 (8-bit权重 + 8-bit激活)
- 量化器: HistQuantizer (直方图量化)
- 校准批次: 10 batches
- 算子融合: 已启用 (Conv+BN融合，共47层)

**输出文件**:
- `model.pdmodel` (283KB) - 量化后的网络结构
- `model.pdiparams` (93MB) - 量化后的模型权重
- `infer_cfg.yml` (351B) - 推理配置

**状态**: ✅ 成功

---

### 4. ONNX格式转换 ✅
**任务**: 将Paddle模型转换为ONNX格式，用于移动端部署

#### 4.1 FP32模型转换 ✅
**执行命令**:
```bash
paddle2onnx \
    --model_dir output/inference_model_baseline/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_fp32.onnx
```

**ONNX模型信息**:
- **文件大小**: 92.34 MB
- **Opset版本**: 11
- **输入张量**:
  - `image`: [1, 3, 608, 608] (float32) - RGB图像
  - `im_shape`: [1, 2] (float32) - 原始图像尺寸
  - `scale_factor`: [1, 2] (float32) - 缩放因子
- **输出张量**:
  - `multiclass_nms3_0.tmp_0`: [N, 6] (float32) - 检测框 [class_id, score, x1, y1, x2, y2]
  - `multiclass_nms3_0.tmp_2`: [N] (int32) - 检测框数量

**验证结果**: ✅ 模型有效，通过onnx.checker验证

**状态**: ✅ 成功

#### 4.2 INT8量化模型转换 ❌
**尝试命令**:
```bash
paddle2onnx \
    --model_dir output/ptq_baseline_int8/yolov3_darknet53_ptq \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_int8.onnx
```

**失败原因**: 
- paddle2onnx不支持量化算子（fake_quantize_dequantize等）
- 这是已知限制，量化模型转ONNX需要特殊处理

**解决方案**: 
- 使用FP32 ONNX模型
- 在移动端使用ONNX Runtime的动态量化功能
- 或使用CoreML转换工具（iOS专用）

**状态**: ❌ 不支持（预期行为）

---

### 5. 文件准备与文档编写 ✅
**任务**: 整理所有部署文件，编写详细文档

**完成的文档**:
1. **README.md** - 项目总览和完整流程
2. **model_info.md** - 模型详细信息和使用说明
3. **MAC_SETUP_GUIDE.md** - Mac环境设置和React Native开发指南
4. **LINUX_WORK_SUMMARY.md** - 本文档

**部署文件清单**:
```
Mobile_Deployment/
├── models/
│   ├── yolov3_mouse_fp32.onnx      (92.34 MB) - ONNX模型
│   ├── infer_cfg.yml                (351 B)    - 推理配置
│   ├── label_list.txt               (12 B)     - 类别标签
│   └── model_info.md                           - 模型文档
├── README.md                                    - 项目总览
├── MAC_SETUP_GUIDE.md                          - Mac开发指南
├── LINUX_WORK_SUMMARY.md                       - 本文档
└── progress_tracker.md                         - 进度追踪
```

**状态**: ✅ 完成

---

## 📊 性能指标总结

### 训练模型性能
- **架构**: YOLOv3 + MobileNetV1
- **数据集**: 10,816张图像（mouse/other二分类）
- **训练轮数**: 50 epochs
- **mAP@0.5**: 93.63%
- **模型大小**: 93 MB (FP32)

### ONNX模型规格
- **格式**: ONNX Opset 11
- **输入尺寸**: 608×608 (固定)
- **Batch Size**: 1 (固定，由于NMS算子限制)
- **文件大小**: 92.34 MB
- **预期推理速度**: 
  - iPhone 12 CPU: ~300-500ms
  - iPhone 12 GPU (Metal): ~100-200ms

---

## 🔧 技术要点

### 1. 模型导出注意事项
- 必须使用完整路径指定权重文件
- 配置文件中的`pretrain_weights`会被`-o weights=`覆盖
- 导出后的模型包含完整的推理图（含NMS）

### 2. 量化策略选择
- **QAT (量化感知训练)**: 精度最高，但需要重新训练，耗时长
- **PTQ (离线量化)**: 速度快（<1分钟），精度略低但可接受
- 本项目选择PTQ，因为GPU资源受限

### 3. ONNX转换限制
- 量化模型无法直接转ONNX（算子不支持）
- multiclass_nms3算子限制batch_size=1
- 需要在移动端实现额外的量化（如果需要）

### 4. 图像预处理要求
```python
# 关键参数
input_size = 608
mean = [0.485, 0.456, 0.406]  # ImageNet均值
std = [0.229, 0.224, 0.225]   # ImageNet标准差
color_format = 'RGB'           # 不是BGR！
```

---

## 📦 文件传输到Mac

### 需要传输的文件
```bash
# 在Linux服务器上打包
cd /hy-tmp/paddle_detection_mouse
tar -czf Mobile_Deployment.tar.gz Mobile_Deployment/

# 文件大小约 93 MB
```

### 传输方式
**方式1: scp**
```bash
# 在Mac上执行
scp user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment.tar.gz ~/Downloads/
```

**方式2: rsync**
```bash
# 在Mac上执行
rsync -avz --progress user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment/ ~/Projects/Mobile_Deployment/
```

**方式3: 云存储**
- 上传到Google Drive / Dropbox / 百度网盘
- 在Mac上下载

---

## 🎯 下一步工作 (Mac环境)

### 阶段1: 环境准备
1. 安装Xcode和Command Line Tools
2. 安装Node.js (推荐使用nvm)
3. 安装CocoaPods
4. 验证开发环境

### 阶段2: React Native项目
1. 创建TypeScript项目
2. 安装ONNX Runtime React Native
3. 安装图像处理和UI库
4. 配置iOS权限

### 阶段3: 模型集成
1. 将ONNX模型添加到iOS bundle
2. 实现ModelService（模型加载和推理）
3. 实现ImageProcessor（图像预处理）
4. 实现后处理逻辑（解析输出，绘制边界框）

### 阶段4: UI开发
1. 主屏幕
2. 检测屏幕
3. 相机/相册集成
4. 结果可视化

### 阶段5: 测试优化
1. 模拟器测试
2. 真机测试
3. 性能优化
4. 用户体验优化

---

## 📝 重要提醒

### 1. 模型使用限制
- ⚠️ **Batch Size固定为1**: 不支持批量推理
- ⚠️ **输入尺寸固定608×608**: 其他尺寸可能影响精度
- ⚠️ **颜色格式必须是RGB**: 不是OpenCV默认的BGR

### 2. 性能优化建议
- 使用CoreML加速（iOS专用）
- 启用ONNX Runtime的图优化
- 考虑降低输入分辨率（如416×416）以提升速度
- 使用Metal GPU加速

### 3. 精度验证
- 在移动端测试时，对比服务器端推理结果
- 确保预处理流程完全一致
- 检查坐标映射是否正确

---

## 🔗 相关资源

### 文档
- [PaddleDetection官方文档](https://github.com/PaddlePaddle/PaddleDetection)
- [Paddle2ONNX文档](https://github.com/PaddlePaddle/Paddle2ONNX)
- [ONNX Runtime文档](https://onnxruntime.ai/)
- [React Native文档](https://reactnative.dev/)

### 工具
- [Netron](https://netron.app/) - ONNX模型可视化
- [ONNX Runtime Perf](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/perf) - 性能分析

---

## ✅ 检查清单

在转移到Mac之前，确认以下内容：

- [x] FP32模型已成功导出为ONNX格式
- [x] ONNX模型已通过验证
- [x] 所有配置文件已准备就绪
- [x] 文档已完整编写
- [x] 文件已整理到Mobile_Deployment目录
- [ ] 文件已传输到Mac（待执行）

---

**Linux服务器端工作完成！** 🎉

现在可以将`Mobile_Deployment`目录传输到Mac，继续React Native应用开发工作。

---

**最后更新**: 2026-02-09 19:55

