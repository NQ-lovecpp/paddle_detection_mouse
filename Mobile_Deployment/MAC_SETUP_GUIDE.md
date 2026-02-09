# Mac环境设置指南

**目标**: 在Mac上开发React Native应用，集成ONNX模型进行实验鼠检测

---

## 📋 前置要求

### 系统要求
- macOS 12.0 或更高版本
- Xcode 14.0 或更高版本
- 至少 10GB 可用磁盘空间

### 必需软件
- [x] Xcode (从App Store安装)
- [x] Xcode Command Line Tools
- [x] Homebrew
- [x] Node.js 16+
- [x] CocoaPods
- [x] Watchman (可选，但推荐)

---

## 🚀 步骤1: 安装开发环境

### 1.1 安装Homebrew（如果未安装）
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.2 安装Node.js
```bash
# 方式1: 使用Homebrew
brew install node

# 方式2: 使用nvm（推荐）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.zshrc  # 或 ~/.bash_profile
nvm install 18
nvm use 18
nvm alias default 18
```

验证安装：
```bash
node --version  # 应显示 v18.x.x
npm --version   # 应显示 9.x.x
```

### 1.3 安装Watchman
```bash
brew install watchman
```

### 1.4 安装CocoaPods
```bash
sudo gem install cocoapods
pod --version  # 验证安装
```

### 1.5 安装Xcode Command Line Tools
```bash
xcode-select --install
```

---

## 📦 步骤2: 创建React Native项目

### 2.1 创建项目
```bash
# 进入工作目录
cd ~/Projects  # 或你喜欢的目录

# 创建TypeScript项目
npx react-native init MouseDetectionApp --template react-native-template-typescript

cd MouseDetectionApp
```

### 2.2 安装核心依赖
```bash
# ONNX Runtime（推理引擎）
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

### 2.3 安装iOS依赖
```bash
cd ios
pod install
cd ..
```

---

## 📁 步骤3: 项目结构设置

### 3.1 创建目录结构
```bash
mkdir -p src/{models,services,components,screens,utils,types}
```

### 3.2 复制模型文件
```bash
# 从Linux服务器下载的文件
# 将以下文件复制到项目中：
# - yolov3_mouse_fp32.onnx -> src/models/
# - label_list.txt -> src/models/
# - infer_cfg.yml -> src/models/
```

在React Native中，需要将模型文件添加到iOS bundle：

**方式1: 使用Xcode**
1. 打开 `ios/MouseDetectionApp.xcworkspace`
2. 右键点击项目 -> Add Files to "MouseDetectionApp"
3. 选择 `src/models/` 下的所有文件
4. 确保勾选 "Copy items if needed" 和 "Add to targets: MouseDetectionApp"

**方式2: 修改Xcode项目配置**
编辑 `ios/MouseDetectionApp.xcodeproj/project.pbxproj`，添加资源引用（较复杂，推荐方式1）

---

## 💻 步骤4: 核心代码实现

### 4.1 创建类型定义
创建 `src/types/index.ts`:
```typescript
export interface Detection {
  classId: number;
  className: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

export interface ModelConfig {
  inputSize: number;
  mean: number[];
  std: number[];
  confidenceThreshold: number;
}
```

### 4.2 创建模型服务
创建 `src/services/ModelService.ts`:
```typescript
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Detection, ModelConfig } from '../types';

export class ModelService {
  private session: InferenceSession | null = null;
  private labels: string[] = [];
  private config: ModelConfig = {
    inputSize: 608,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    confidenceThreshold: 0.5,
  };

  async initialize(): Promise<void> {
    try {
      // 加载模型
      const modelPath = `${RNFS.MainBundlePath}/yolov3_mouse_fp32.onnx`;
      console.log('Loading model from:', modelPath);
      
      this.session = await InferenceSession.create(modelPath);
      console.log('✅ Model loaded successfully');

      // 加载标签
      const labelPath = `${RNFS.MainBundlePath}/label_list.txt`;
      const labelContent = await RNFS.readFile(labelPath, 'utf8');
      this.labels = labelContent.split('\n').filter(l => l.trim());
      console.log('✅ Labels loaded:', this.labels);
    } catch (error) {
      console.error('❌ Model initialization failed:', error);
      throw error;
    }
  }

  async detect(
    imageData: Float32Array,
    originalWidth: number,
    originalHeight: number
  ): Promise<Detection[]> {
    if (!this.session) {
      throw new Error('Model not initialized');
    }

    try {
      const { inputSize } = this.config;
      
      // 计算缩放因子
      const scale = inputSize / Math.max(originalWidth, originalHeight);
      
      // 创建输入张量
      const imageTensor = new Tensor('float32', imageData, [1, 3, inputSize, inputSize]);
      const imShapeTensor = new Tensor('float32', 
        new Float32Array([originalHeight, originalWidth]), [1, 2]);
      const scaleFactorTensor = new Tensor('float32', 
        new Float32Array([scale, scale]), [1, 2]);

      // 执行推理
      const feeds = {
        image: imageTensor,
        im_shape: imShapeTensor,
        scale_factor: scaleFactorTensor,
      };

      console.log('Running inference...');
      const startTime = Date.now();
      const results = await this.session.run(feeds);
      const inferenceTime = Date.now() - startTime;
      console.log(`✅ Inference completed in ${inferenceTime}ms`);

      // 解析输出
      const detections = this.parseOutput(results);
      console.log(`Found ${detections.length} detections`);

      return detections;
    } catch (error) {
      console.error('❌ Inference failed:', error);
      throw error;
    }
  }

  private parseOutput(results: any): Detection[] {
    const detections: Detection[] = [];
    
    // 获取输出张量
    const boxes = results['multiclass_nms3_0.tmp_0'];
    const numBoxes = results['multiclass_nms3_0.tmp_2'];
    
    if (!boxes || !numBoxes) {
      console.warn('No detection outputs found');
      return detections;
    }

    const boxData = boxes.data as Float32Array;
    const count = numBoxes.data[0];

    // 解析每个检测框
    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      const classId = Math.round(boxData[offset]);
      const confidence = boxData[offset + 1];
      const x1 = boxData[offset + 2];
      const y1 = boxData[offset + 3];
      const x2 = boxData[offset + 4];
      const y2 = boxData[offset + 5];

      // 过滤低置信度
      if (confidence < this.config.confidenceThreshold) {
        continue;
      }

      detections.push({
        classId,
        className: this.labels[classId] || `class_${classId}`,
        confidence,
        bbox: { x1, y1, x2, y2 },
      });
    }

    return detections;
  }

  dispose(): void {
    // 清理资源
    this.session = null;
  }
}
```

### 4.3 创建图像预处理服务
创建 `src/services/ImageProcessor.ts`:
```typescript
import { Image } from 'react-native';

export class ImageProcessor {
  static async preprocessImage(
    imagePath: string,
    targetSize: number = 608
  ): Promise<{
    imageData: Float32Array;
    originalWidth: number;
    originalHeight: number;
  }> {
    // TODO: 实现图像预处理
    // 1. 加载图像
    // 2. Resize并padding
    // 3. 归一化
    // 4. 转换为Float32Array
    
    // 这里需要使用原生模块或第三方库来处理图像
    // 推荐使用 react-native-image-resizer 或 react-native-fast-image
    
    throw new Error('Not implemented yet');
  }
}
```

---

## 🎨 步骤5: UI开发

### 5.1 主屏幕
创建 `src/screens/HomeScreen.tsx`:
```typescript
import React from 'react';
import { View, StyleSheet, TouchableOpacity, Text } from 'react-native';
import { useNavigation } from '@react-navigation/native';

export const HomeScreen: React.FC = () => {
  const navigation = useNavigation();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>实验鼠检测</Text>
      
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Detection')}>
        <Text style={styles.buttonText}>开始检测</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 40,
    color: '#333',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
});
```

---

## 🔧 步骤6: 配置权限

### 6.1 编辑Info.plist
打开 `ios/MouseDetectionApp/Info.plist`，添加：
```xml
<key>NSCameraUsageDescription</key>
<string>需要访问相机以检测实验鼠</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>需要访问相册以选择图片</string>
<key>NSPhotoLibraryAddUsageDescription</key>
<string>需要保存检测结果到相册</string>
```

---

## 🏃 步骤7: 运行应用

### 7.1 启动Metro Bundler
```bash
npm start
```

### 7.2 运行iOS模拟器
在新终端窗口：
```bash
npm run ios
# 或指定设备
npm run ios -- --simulator="iPhone 14 Pro"
```

### 7.3 真机调试
```bash
# 连接iPhone到Mac
# 在Xcode中选择你的设备
npm run ios -- --device
```

---

## 🐛 故障排查

### 问题1: CocoaPods安装失败
```bash
cd ios
pod deintegrate
pod install --repo-update
```

### 问题2: Metro Bundler缓存问题
```bash
npm start -- --reset-cache
```

### 问题3: Xcode构建失败
1. 清理构建: Product -> Clean Build Folder (Cmd+Shift+K)
2. 删除DerivedData: `rm -rf ~/Library/Developer/Xcode/DerivedData`
3. 重新安装pods: `cd ios && pod install`

### 问题4: ONNX Runtime加载失败
- 确保模型文件已添加到Xcode项目
- 检查文件路径是否正确
- 查看Xcode控制台的详细错误信息

---

## 📚 下一步

1. ✅ 完成图像预处理实现
2. ✅ 实现相机/相册选择功能
3. ✅ 开发检测结果可视化
4. ✅ 优化推理性能
5. ✅ 添加错误处理和加载状态
6. ✅ 真机测试和性能调优

---

## 🔗 有用的资源

- [React Native官方文档](https://reactnative.dev/)
- [ONNX Runtime React Native](https://github.com/microsoft/onnxruntime-react-native)
- [React Native Vision Camera](https://github.com/mrousavy/react-native-vision-camera)
- [React Native Image Picker](https://github.com/react-native-image-picker/react-native-image-picker)

---

**最后更新**: 2026-02-09

