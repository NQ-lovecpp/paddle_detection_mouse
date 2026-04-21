# 🐭 实验鼠目标检测 — 从训练到 iOS 实时部署

> **基于 PaddleDetection 框架的二分类目标检测项目（mouse / other），覆盖数据工程、模型训练、轻量化到 iOS 移动端实时推理的全流程。**

| 项目 | 详情 |
|------|------|
| **框架** | PaddleDetection 2.6 / PaddlePaddle 2.5.1 |
| **训练环境** | 2× Tesla T4 GPU, CUDA 11.6 |
| **数据集** | 7,286 张 VOC 格式（二分类 mouse / other） |
| **最优模型** | PicoDet-S — mAP@0.5 = **94.30%**, ONNX 仅 **4.4 MB** |
| **iOS 部署** | React Native + ONNX Runtime C API + CoreML ANE, iPhone 13 Pro **14 FPS** |

---

## 📑 目录

- [项目成果](#-项目成果)
- [项目架构](#-项目架构)
- [数据工程](#-数据工程)
- [实验设计](#-实验设计)
- [模型对比](#-模型对比)
- [移动端部署](#-移动端部署)
- [仓库结构](#-仓库结构)
- [快速上手](#-快速上手)
- [文档索引](#-文档索引)
- [许可证](#-许可证)

---

## 🏆 项目成果

1. **标准化数据集** — 整合 3 个异构数据源，构建 **7,286 张**标准化二分类 VOC 数据集；通过受控实验量化数据规模对精度的影响（训练集 ×3 → mAP **+47 pp**）。

2. **高精度轻量模型** — PicoDet-S mAP@0.5 = **94.30%**（验证集 1,458 张），服务器端推理 ~78 FPS；较 YOLOv3 基线精度 +2.76%，模型体积缩小 **21×**。

3. **完整部署链路** — 打通「训练 → 导出 → ONNX 转换 → CoreML ANE 加速 → iOS 实时部署」全链路，ONNX 模型仅 **4.4 MB**；iOS 端 **14 FPS**（iPhone 13 Pro，热机稳定）。

4. **全流程文档** — 环境搭建、实验设计、数据处理、部署调优均有完整文档，降低后续复现成本。

---

## 🏗 项目架构

```
数据工程                    模型训练                    移动端部署
┌────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│ 3 个数据源  │─────▶│ PaddleDetection  │─────▶│ Paddle Inference     │
│ 标注清洗    │      │ 2× T4 数据并行   │      │      ↓               │
│ 标签统一    │      │ Linear Scaling   │      │ paddle2onnx          │
│ VOC 格式化  │      │ Rule 学习率缩放  │      │      ↓               │
│            │      │                  │      │ ONNX (4.4 MB)        │
│ 7,286 张   │      │ PicoDet-S 94.30% │      │      ↓               │
│ mouse/other│      │ YOLOv3   91.54%  │      │ CoreML ANE 加速      │
└────────────┘      └──────────────────┘      │      ↓               │
                                               │ React Native App     │
                                               │ iPhone 13 Pro 14 FPS │
                                               └──────────────────────┘
```

---

## 📊 数据工程

### 数据源整合

| 数据源 | 来源 | 图片数 | 处理方式 |
|--------|------|--------|----------|
| `dataset/dog_mouse_other_voc` | 原始三分类数据集 | ~3,536 | `dog` 标签统一为 `other` |
| `RawData/wb-img` | 网络采集 mouse 图片 | — | 标注清洗后合入 |
| `RawData/dog_mouse_other_voc` | 补充数据 | — | mouse + other 合入 |

**最终数据集**：`dataset/mouse_other_voc/` — **7,286 张**（训练集 5,828 / 验证集 1,458），VOC XML 格式，二分类（mouse / other）。

### 数据规模实验

通过训练集对比实验（1,942 张 vs 5,828 张，验证集不变），定量验证**数据规模是精度的首要驱动因素**：

| 模型 | 训练集 | mAP@0.5 | 说明 |
|------|--------|---------|------|
| YOLOv3 | 1,942 张（1/3） | ~44% | 数据不足，欠拟合 |
| YOLOv3 | 5,828 张（全量） | 91.5% | 数据 ×3 → mAP **+47 pp** |

---

## 🧪 实验设计

采用 **4×2 对比实验矩阵**，分离 GPU 数、batch size、数据量对精度的独立影响：

```
                  ┌─────────────┬─────────────┐
                  │  1/3 数据    │  全量数据    │
┌─────────────────┼─────────────┼─────────────┤
│ YOLOv3  单卡    │ Y1 baseline │ Y3          │
│ YOLOv3  双卡    │ Y2          │ Y4          │
├─────────────────┼─────────────┼─────────────┤
│ PicoDet 单卡    │ L1          │ —           │
│ PicoDet 双卡    │ L2 / L3 / L4│ —           │
└─────────────────┴─────────────┴─────────────┘
```

### 多卡训练

- 在双 Tesla T4 GPU 环境下配置**数据并行训练**，以 TCP 协议完成跨卡梯度同步
- 实测加速比 **2.2×**，验证梯度带宽在当前模型规模下不构成瓶颈
- 以 **Linear Scaling Rule**（Goyal et al. 2017）为依据：`lr ∝ total_batch_size`

---

## 📈 模型对比

| 指标 | YOLOv3-MobileNetV1 | PicoDet-S |
|------|-------------------|-----------|
| mAP@0.5 | 91.54% | **94.30%** |
| 模型大小 (ONNX) | 92.34 MB | **~4.4 MB** |
| 服务器推理 (T4) | ~41 FPS | **~78 FPS** |
| iOS 推理 | 3 FPS ⚠️ | **14 FPS** ✅ |
| CoreML 兼容 | ❌ `multiclass_nms3` 不支持 | ✅ 全部为 CoreML 兼容算子 |
| 输入尺寸 | 608×608 | 320×320 |
| 收敛速度 | ~80+ epoch | ~70 epoch |

> YOLOv3 因 PaddleDetection 自定义算子（`multiclass_nms3`）不兼容 CoreML，无法走 ANE 加速，iOS 端仅 3 FPS；最终以 PicoDet-S 替代。

---

## 📱 移动端部署

### 部署链路

```
PicoDet-S best_model.pdparams
    → tools/export_model.py  (Paddle Inference)
    → paddle2onnx            (ONNX, opset 11, 4.4 MB)
    → React Native App
        ├── ONNX Runtime C API (推理引擎)
        └── CoreML ANE        (硬件加速)
    → iPhone 13 Pro: 14 FPS (热机稳定)
```

### 性能迭代

iOS 端历经**四轮性能迭代**：1 FPS → 14 FPS 热机稳定。

### 技术栈

- **前端框架**：React Native + TypeScript
- **推理引擎**：ONNX Runtime React Native（C API）
- **硬件加速**：CoreML ANE（Apple Neural Engine）
- **模型格式**：ONNX (opset 11)

---

## 📁 仓库结构

```
paddle_detection_mouse/
├── README.md                          ← 本文件
├── Training_Pipeline.md               ← 全流程训练手册
├── Next_Steps_Guide.md                ← 后续实验与部署指导
├── Plan_Restart.md                    ← 项目复现与优化计划
│
├── PaddleDetection-release-2.6/       ← PaddleDetection 框架（含自定义配置）
│   ├── configs/
│   │   ├── picodet/runs/              ← PicoDet 实验配置 (L1–L4, M1, C1)
│   │   ├── yolov3/runs/              ← YOLOv3 实验配置 (Y1–Y6, C2–C3)
│   │   └── datasets/mouse_other_voc.yml
│   ├── dataset/mouse_other_voc/       ← 数据集目录 (gitignored)
│   └── scripts/                       ← 自动化训练脚本
│
├── Mobile_Deployment/                 ← iOS 移动端部署
│   ├── MouseDetectionApp/             ← React Native 应用源码
│   ├── MouseDetectionApp_code/        ← 应用核心代码备份
│   ├── models/                        ← ONNX 模型 & 配置
│   │   └── picodet_s_320_mouse_L1_nonms.onnx  (4.4 MB)
│   └── tools/                         ← 部署工具脚本
│
├── Scripts/                           ← 数据处理脚本
│   ├── merge_dataset.py               ← 三源数据集合并脚本
│   ├── vdl2tb.py                      ← VisualDL → TensorBoard 转换
│   └── voc_viewer/                    ← VOC 数据集可视化工具
│
├── Papers/                            ← 项目论文 (LaTeX)
│   ├── mouse_detection/               ← 英文版
│   └── mouse_detection_zh/            ← 中文版
│
└── Docs/                              ← 详细文档集
    ├── 入门paddledetection/
    ├── 服务器上的paddle搭建/
    ├── 模型导出和移动端部署/
    ├── 模型测试/
    ├── 训练实验总结与面试复述.md
    └── ...
```

---

## 🚀 快速上手

### 环境要求

- PaddlePaddle 2.5.1 + CUDA 11.6
- PaddleDetection release/2.6
- Python 3.8+
- 1–2× NVIDIA GPU（Tesla T4 或同级）

### 训练

```bash
cd PaddleDetection-release-2.6

# 单卡训练 PicoDet-S
python tools/train.py \
    -c configs/picodet/runs/L1_picodet_1gpu.yml \
    --eval --use_vdl=true

# 双卡数据并行训练 PicoDet-S
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/picodet/runs/L2_picodet_2gpu.yml \
    --eval --use_vdl=true
```

### 导出 & ONNX 转换

```bash
# 导出 Paddle Inference 格式
python tools/export_model.py \
    -c configs/picodet/runs/L1_picodet_1gpu.yml \
    -o weights=output/L1_picodet_1gpu/best_model.pdparams \
    --output_dir=output/inference_model_picodet_L1

# 转换为 ONNX
paddle2onnx \
    --model_dir output/inference_model_picodet_L1/picodet_s_320_voc_mouse \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file Mobile_Deployment/models/picodet_s_320_mouse_L1.onnx
```

### iOS 部署

```bash
cd Mobile_Deployment/MouseDetectionApp
npm install
cd ios && pod install && cd ..
npx react-native run-ios
```

> 详细步骤请参考 [Mobile_Deployment/README.md](Mobile_Deployment/README.md) 和 [Next_Steps_Guide.md](Next_Steps_Guide.md)。

---

## 📚 文档索引

| 文档 | 说明 |
|------|------|
| [Training_Pipeline.md](Training_Pipeline.md) | 全流程训练手册：环境检查、数据预处理、Anchor 聚类、训练/评估/推理/导出、模型压缩、ONNX 转换 |
| [Next_Steps_Guide.md](Next_Steps_Guide.md) | 后续实验指导：训练监控、学习率缩放策略、4×2 实验设计、模型压缩方案、iOS 部署更新 |
| [Plan_Restart.md](Plan_Restart.md) | 项目复现与优化计划：代码考古、数据集统计、原始训练参数、改进方案 |
| [Mobile_Deployment/README.md](Mobile_Deployment/README.md) | 移动端部署项目：模型导出、量化、ONNX 转换、React Native 应用开发 |
| [Docs/训练实验总结与面试复述.md](Docs/训练实验总结与面试复述.md) | 训练实验总结与面试复述要点 |

---

## 📜 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

