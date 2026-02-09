# 快速参考指南

**项目**: 实验鼠检测模型移动端部署  
**更新**: 2026-02-09

---

## 📁 文件结构

```
Mobile_Deployment/
├── models/
│   ├── yolov3_mouse_fp32.onnx    # 92.34 MB - ONNX模型文件
│   ├── infer_cfg.yml              # 351 B - 推理配置
│   ├── label_list.txt             # 12 B - 类别标签 (mouse, other)
│   └── model_info.md              # 模型详细文档
├── README.md                      # 项目总览
├── MAC_SETUP_GUIDE.md            # Mac环境设置指南
├── LINUX_WORK_SUMMARY.md         # Linux工作总结
└── QUICK_REFERENCE.md            # 本文档
```

---

## 🚀 快速开始 (Mac)

### 1. 传输文件到Mac
```bash
# 方式1: scp
scp -r user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment ~/Projects/

# 方式2: 打包后传输
# 在Linux服务器上：
cd /hy-tmp/paddle_detection_mouse
tar -czf Mobile_Deployment.tar.gz Mobile_Deployment/

# 在Mac上：
scp user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment.tar.gz ~/Downloads/
cd ~/Downloads && tar -xzf Mobile_Deployment.tar.gz
```

### 2. 安装开发环境
```bash
# 安装Node.js (使用nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.zshrc
nvm install 18
nvm use 18

# 安装CocoaPods
sudo gem install cocoapods

# 安装Watchman
brew install watchman
```

### 3. 创建React Native项目
```bash
cd ~/Projects
npx react-native init MouseDetectionApp --template react-native-template-typescript
cd MouseDetectionApp

# 安装依赖
npm install onnxruntime-react-native react-native-fs react-native-image-picker
cd ios && pod install && cd ..
```

### 4. 添加模型文件
```bash
# 复制模型到项目
mkdir -p src/models
cp ~/Projects/Mobile_Deployment/models/* src/models/

# 在Xcode中添加到bundle
# 打开 ios/MouseDetectionApp.xcworkspace
# 右键项目 -> Add Files -> 选择 src/models/ 下的文件
```

### 5. 运行应用
```bash
# 启动Metro
npm start

# 运行iOS (新终端)
npm run ios
```

---

## 📊 模型信息速查

| 项目 | 值 |
|------|-----|
| **模型架构** | YOLOv3 + MobileNetV1 |
| **任务** | 二分类目标检测 (mouse/other) |
| **输入尺寸** | 608×608×3 (RGB) |
| **输出** | 检测框 [class_id, score, x1, y1, x2, y2] |
| **文件大小** | 92.34 MB |
| **精度** | mAP@0.5 = 93.63% |
| **Batch Size** | 1 (固定) |

---

## 🔧 关键代码片段

### 模型加载
```typescript
import { InferenceSession } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

const modelPath = `${RNFS.MainBundlePath}/yolov3_mouse_fp32.onnx`;
const session = await InferenceSession.create(modelPath);
```

### 图像预处理
```typescript
// 1. Resize到608×608 (保持宽高比，padding)
// 2. 归一化: (pixel/255.0 - mean) / std
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];
// 3. HWC -> CHW
// 4. 添加batch维度
```

### 推理
```typescript
const feeds = {
  image: imageTensor,        // [1, 3, 608, 608]
  im_shape: imShapeTensor,   // [1, 2]
  scale_factor: scaleTensor, // [1, 2]
};
const results = await session.run(feeds);
```

### 解析输出
```typescript
const boxes = results['multiclass_nms3_0.tmp_0'].data; // [N, 6]
const numBoxes = results['multiclass_nms3_0.tmp_2'].data[0];

for (let i = 0; i < numBoxes; i++) {
  const classId = boxes[i * 6];
  const score = boxes[i * 6 + 1];
  const x1 = boxes[i * 6 + 2];
  const y1 = boxes[i * 6 + 3];
  const x2 = boxes[i * 6 + 4];
  const y2 = boxes[i * 6 + 5];
  
  if (score > 0.5) {
    // 绘制边界框
  }
}
```

---

## ⚠️ 常见问题

### Q1: 模型加载失败
**A**: 确保模型文件已添加到Xcode项目的bundle中，检查文件路径是否正确。

### Q2: 推理速度慢
**A**: 
- 使用CoreML加速: `options.appendCoreMLExecutionProvider()`
- 降低输入分辨率（如416×416）
- 在后台线程执行推理

### Q3: 检测结果不准确
**A**: 
- 检查图像预处理是否正确（RGB格式，归一化参数）
- 确认输入尺寸为608×608
- 验证坐标映射逻辑

### Q4: 内存占用过高
**A**: 
- 及时释放不用的图像数据
- 使用图像压缩
- 限制推理频率

---

## 📚 文档导航

- **项目总览**: `README.md`
- **模型详情**: `models/model_info.md`
- **Mac设置**: `MAC_SETUP_GUIDE.md`
- **Linux总结**: `LINUX_WORK_SUMMARY.md`
- **本文档**: `QUICK_REFERENCE.md`

---

## 🔗 有用链接

- [ONNX Runtime React Native](https://github.com/microsoft/onnxruntime-react-native)
- [React Native文档](https://reactnative.dev/)
- [React Native Vision Camera](https://github.com/mrousavy/react-native-vision-camera)
- [Netron (模型可视化)](https://netron.app/)

---

## 📞 支持

如有问题，请参考：
1. 详细文档 (`MAC_SETUP_GUIDE.md`)
2. 模型信息 (`models/model_info.md`)
3. 训练文档 (`../Training_Pipeline.md`)

---

**祝开发顺利！** 🎉

