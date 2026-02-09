# 实验鼠检测模型 - 移动端部署项目

> **项目目标**: 将训练好的YOLOv3模型量化、转换为ONNX格式，并开发React Native iPhone应用  
> **源模型**: B_baseline_1gpu (YOLOv3 + MobileNetV1)  
> **创建日期**: 2026-02-09  

---

## 📋 项目概览

### 工作流程
```
训练好的模型 (best_model.pdparams)
    ↓
[Linux服务器] 模型导出 → Paddle Inference格式
    ↓
[Linux服务器] 量化训练 (QAT) → INT8压缩
    ↓
[Linux服务器] 转换为ONNX格式
    ↓
[Mac环境] React Native应用开发
    ↓
[iPhone] 部署测试
```

### 技术栈
- **模型框架**: PaddlePaddle 2.5.1
- **量化工具**: PaddleSlim
- **转换工具**: Paddle2ONNX
- **移动端框架**: React Native + TypeScript
- **推理引擎**: ONNX Runtime React Native
- **目标平台**: iOS (iPhone)

---

## 🚀 阶段一：Linux服务器端工作

### 1.1 模型导出 (Paddle Inference格式)

**目的**: 将训练权重导出为可部署的推理模型

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/B_baseline_1gpu/best_model.pdparams \
    --output_dir=output/inference_model_baseline
```

**输出文件**:
- `model.pdmodel` - 网络结构
- `model.pdiparams` - 模型权重
- `infer_cfg.yml` - 推理配置

---

### 1.2 量化感知训练 (QAT)

**目的**: 将FP32模型压缩为INT8，减小体积3-4倍，提升移动端推理速度

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/quant/yolov3_mobilenet_v1_qat.yml \
    -o \
    epoch=20 \
    LearningRate.base_lr=0.0001 \
    pretrain_weights=output/B_baseline_1gpu/best_model.pdparams \
    save_dir=output/quant_baseline_int8 \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/quant_baseline_int8/vdl_log
```

**参数说明**:
- `epoch=20`: 量化微调20轮（比完整训练短）
- `base_lr=0.0001`: 小学习率微调
- `pretrain_weights`: 从baseline最佳模型开始

**预期效果**:
- 模型体积: ~94MB → ~25MB
- 精度损失: <2%
- 推理速度: 提升1.5-2倍

---

### 1.3 导出量化模型

```bash
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/quant/yolov3_mobilenet_v1_qat.yml \
    -o weights=output/quant_baseline_int8/best_model.pdparams \
    --output_dir=output/inference_model_quant
```

---

### 1.4 转换为ONNX格式

**安装依赖** (如果未安装):
```bash
pip install paddle2onnx onnx
```

**转换FP32模型**:
```bash
paddle2onnx \
    --model_dir output/inference_model_baseline/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_fp32.onnx
```

**转换INT8量化模型**:
```bash
paddle2onnx \
    --model_dir output/inference_model_quant/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_int8.onnx
```

**参数说明**:
- `--opset_version 11`: ONNX算子集版本（兼容性好）
- 可选: `--opset_version 13` (更新的版本)

---

### 1.5 验证ONNX模型

```bash
python3 -c "
import onnx
import os

# 验证FP32模型
print('='*60)
print('验证 FP32 模型')
print('='*60)
model_fp32 = onnx.load('output/yolov3_mouse_fp32.onnx')
onnx.checker.check_model(model_fp32)
print(f'✅ FP32模型有效')
print(f'Opset版本: {model_fp32.opset_import[0].version}')
print(f'输入: {[i.name for i in model_fp32.graph.input]}')
print(f'输出: {[o.name for o in model_fp32.graph.output]}')
print(f'文件大小: {os.path.getsize(\"output/yolov3_mouse_fp32.onnx\") / 1024 / 1024:.2f} MB')

print()
print('='*60)
print('验证 INT8 量化模型')
print('='*60)
model_int8 = onnx.load('output/yolov3_mouse_int8.onnx')
onnx.checker.check_model(model_int8)
print(f'✅ INT8模型有效')
print(f'Opset版本: {model_int8.opset_import[0].version}')
print(f'输入: {[i.name for i in model_int8.graph.input]}')
print(f'输出: {[o.name for o in model_int8.graph.output]}')
print(f'文件大小: {os.path.getsize(\"output/yolov3_mouse_int8.onnx\") / 1024 / 1024:.2f} MB')

print()
print('='*60)
print('压缩比对比')
print('='*60)
size_fp32 = os.path.getsize('output/yolov3_mouse_fp32.onnx') / 1024 / 1024
size_int8 = os.path.getsize('output/yolov3_mouse_int8.onnx') / 1024 / 1024
print(f'FP32: {size_fp32:.2f} MB')
print(f'INT8: {size_int8:.2f} MB')
print(f'压缩比: {size_fp32/size_int8:.2f}x')
"
```

---

### 1.6 复制ONNX模型到部署目录

```bash
# 创建模型存储目录
mkdir -p /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models

# 复制ONNX模型
cp output/yolov3_mouse_fp32.onnx /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/
cp output/yolov3_mouse_int8.onnx /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/

# 复制推理配置文件
cp output/inference_model_baseline/yolov3_my_dog_mouse_voc/infer_cfg.yml \
   /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/infer_cfg.yml

# 复制标签文件
cp dataset/mouse_other_voc/label_list.txt \
   /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/label_list.txt

echo "✅ 模型文件已复制到 Mobile_Deployment/models/"
ls -lh /hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/
```

---

## 🍎 阶段二：Mac环境工作

### 2.1 环境准备

**前置要求**:
- macOS 12.0+
- Xcode 14.0+
- Node.js 16+
- CocoaPods
- React Native CLI

**安装Node.js** (如果未安装):
```bash
# 使用Homebrew
brew install node

# 或使用nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

**安装React Native CLI**:
```bash
npm install -g react-native-cli
```

---

### 2.2 创建React Native项目

```bash
cd /path/to/your/workspace

# 创建TypeScript项目
npx react-native init MouseDetectionApp --template react-native-template-typescript

cd MouseDetectionApp
```

---

### 2.3 安装依赖

```bash
# ONNX Runtime (核心推理引擎)
npm install onnxruntime-react-native

# 文件系统操作
npm install react-native-fs

# 图片选择器
npm install react-native-image-picker

# 相册访问
npm install @react-native-camera-roll/camera-roll

# 相机功能
npm install react-native-vision-camera

# UI组件库
npm install react-native-paper
npm install react-native-vector-icons

# 类型定义
npm install --save-dev @types/react-native-vector-icons
```

**iOS依赖安装**:
```bash
cd ios
pod install
cd ..
```

---

### 2.4 项目结构

```
MouseDetectionApp/
├── src/
│   ├── models/
│   │   ├── yolov3_mouse_int8.onnx      # ONNX模型文件
│   │   └── label_list.txt               # 类别标签
│   ├── services/
│   │   ├── ModelService.ts              # 模型加载与推理
│   │   └── ImageProcessor.ts            # 图像预处理
│   ├── components/
│   │   ├── CameraView.tsx               # 相机组件
│   │   ├── ImagePicker.tsx              # 图片选择器
│   │   └── DetectionResult.tsx          # 检测结果显示
│   ├── screens/
│   │   ├── HomeScreen.tsx               # 主页
│   │   └── DetectionScreen.tsx          # 检测页面
│   └── utils/
│       ├── BoundingBox.ts               # 边界框绘制
│       └── NMS.ts                       # 非极大值抑制
├── App.tsx
└── package.json
```

---

### 2.5 核心代码实现

#### ModelService.ts - 模型推理服务

```typescript
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

export interface Detection {
  class: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

export class ModelService {
  private session: InferenceSession | null = null;
  private labels: string[] = [];
  
  async loadModel(modelPath: string): Promise<void> {
    try {
      // 加载ONNX模型
      this.session = await InferenceSession.create(modelPath);
      console.log('✅ 模型加载成功');
      
      // 加载标签
      const labelPath = modelPath.replace('.onnx', '_labels.txt');
      const labelContent = await RNFS.readFile(labelPath, 'utf8');
      this.labels = labelContent.split('\n').filter(l => l.trim());
    } catch (error) {
      console.error('❌ 模型加载失败:', error);
      throw error;
    }
  }
  
  async detect(imageData: Float32Array, width: number, height: number): Promise<Detection[]> {
    if (!this.session) {
      throw new Error('模型未加载');
    }
    
    try {
      // 创建输入张量 [1, 3, height, width]
      const inputTensor = new Tensor('float32', imageData, [1, 3, height, width]);
      
      // 执行推理
      const feeds = { image: inputTensor };
      const results = await this.session.run(feeds);
      
      // 解析输出
      const detections = this.parseOutput(results, width, height);
      
      return detections;
    } catch (error) {
      console.error('❌ 推理失败:', error);
      throw error;
    }
  }
  
  private parseOutput(results: any, imgWidth: number, imgHeight: number): Detection[] {
    // YOLOv3输出解析逻辑
    // 根据实际模型输出格式调整
    const detections: Detection[] = [];
    
    // TODO: 实现YOLOv3输出解析
    // 1. 获取边界框、置信度、类别概率
    // 2. 应用NMS
    // 3. 过滤低置信度检测
    
    return detections;
  }
}
```

#### ImageProcessor.ts - 图像预处理

```typescript
import { Image } from 'react-native';

export class ImageProcessor {
  static async preprocessImage(
    imagePath: string,
    targetWidth: number = 608,
    targetHeight: number = 608
  ): Promise<Float32Array> {
    // 1. 加载图片
    // 2. Resize到模型输入尺寸
    // 3. 归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    // 4. 转换为Float32Array
    
    // TODO: 实现图像预处理
    const imageData = new Float32Array(3 * targetWidth * targetHeight);
    return imageData;
  }
}
```

---

### 2.6 配置Info.plist权限

编辑 `ios/MouseDetectionApp/Info.plist`，添加相机和相册权限：

```xml
<key>NSCameraUsageDescription</key>
<string>需要访问相机以检测实验鼠</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>需要访问相册以选择图片</string>
<key>NSPhotoLibraryAddUsageDescription</key>
<string>需要保存检测结果到相册</string>
```

---

### 2.7 运行应用

```bash
# iOS模拟器
npx react-native run-ios

# 真机调试 (需要Apple Developer账号)
npx react-native run-ios --device "Your iPhone Name"
```

---

## 📊 性能指标

### 模型对比

| 指标 | FP32模型 | INT8量化模型 |
|------|----------|--------------|
| 文件大小 | ~94 MB | ~25 MB |
| 推理速度 | 基准 | 1.5-2x 快 |
| 精度损失 | 0% | <2% |
| 内存占用 | 高 | 低 |

### 目标性能

- **推理延迟**: <200ms (iPhone 12+)
- **帧率**: >5 FPS (实时检测)
- **精度**: mAP@0.5 >90%
- **应用大小**: <50 MB

---

## 🔧 故障排查

### 常见问题

**1. ONNX模型加载失败**
- 检查模型文件路径
- 验证ONNX模型格式
- 确认opset版本兼容性

**2. 推理速度慢**
- 使用INT8量化模型
- 降低输入分辨率
- 启用GPU加速

**3. 检测精度低**
- 调整置信度阈值
- 检查图像预处理流程
- 验证NMS参数

---

## 📝 待办事项

- [x] 创建项目文件夹
- [ ] 导出baseline模型
- [ ] 执行量化训练
- [ ] 转换为ONNX格式
- [ ] 验证ONNX模型
- [ ] 创建React Native项目
- [ ] 集成ONNX Runtime
- [ ] 实现推理引擎
- [ ] 开发UI界面
- [ ] 真机测试

---

## 📚 参考资料

- [PaddleDetection文档](https://github.com/PaddlePaddle/PaddleDetection)
- [Paddle2ONNX文档](https://github.com/PaddlePaddle/Paddle2ONNX)
- [ONNX Runtime React Native](https://github.com/microsoft/onnxruntime-react-native)
- [React Native文档](https://reactnative.dev/)

---

**最后更新**: 2026-02-09

