# 项目复现与优化计划：实验鼠检测模型（PaddleDetection 2.6）

> **生成日期**: 2026-02-07  
> **项目根目录**: `/hy-tmp/paddle_detection_mouse`  
> **PaddleDetection 目录**: `/hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6`  
> **所有训练命令均在 PaddleDetection 目录下执行**

---

## 〇、代码考古 — 项目分析摘要

### 0.1 项目背景与原始目标

本项目是一个基于 PaddleDetection release/2.6 的目标检测项目，最初在 2024 年 7-8 月实习期间开发，目标是：

- **检测三类目标**: `dog`（狗）、`mouse`（实验鼠）、`other`（其他动物）
- **模型架构**: YOLOv3 + MobileNetV1 backbone
- **数据格式**: Pascal VOC (XML 标注)
- **最终目标**: 模型训练 → 导出 → ONNX 转换 → 移动端部署

### 0.2 发现的自定义配置文件（核心文件清单）

| 文件 | 用途 | 路径 |
|------|------|------|
| **主配置** | YOLOv3 老鼠检测入口 | `configs/yolov3/yolov3_my_dog_mouse_voc.yml` |
| **数据集配置** | 定义数据路径与类别数 | `configs/datasets/mouse_other_voc.yml` ✅ 新建 |
| **旧数据集配置** | 已废弃，被上面替代 | `configs/datasets/dog_mouse_other_voc.yml` ⚠️ 废弃 |
| **运行时配置** | GPU/保存间隔/输出目录 | `configs/runtime.yml` |
| **优化器配置** | epoch/学习率/衰减策略 | `configs/yolov3/_base_/optimizer_270e.yml` |
| **网络结构配置** | YOLOv3+MobileNetV1 | `configs/yolov3/_base_/yolov3_mobilenet_v1.yml` |
| **数据读取配置** | batch_size/数据增强 | `configs/yolov3/_base_/yolov3_reader.yml` |

### 0.3 数据集统计（最新 — 合并后）

> **更新于 2026-02-07**: 已将三个数据源合并为统一的 `mouse_other_voc` 二分类数据集。
> 原始 `dog_mouse_other_voc` (3 类) 已废弃，所有 `dog` 标签已改为 `other`。

```
数据集: dataset/mouse_other_voc/    ← 合并后的新数据集
├── images/          → 10,816 张 JPG
├── annotations/     → 10,816 个 XML (VOC 格式)
├── label_list.txt   → mouse, other (2类)
├── train.txt        → 8,653 条（训练集, 80%）
└── val.txt          → 2,163 条（验证集, 20%）

数据来源:
  1. 原 dataset/dog_mouse_other_voc/  → dog 改标签为 other, 全部合入
  2. RawData/wb-img/                  → 纯 mouse, 全部合入
  3. RawData/dog_mouse_other_voc/     → mouse + other, 全部合入

类别分布（合并后总体）:
  - mouse: ~7,000+ 张 (主要类别)
  - other: ~3,800+ 张 (含原 dog 改名后的数据)
```

**旧数据集对比**:
```
旧: dataset/dog_mouse_other_voc/ → 3,536 张, 3 类 (dog/mouse/other)
新: dataset/mouse_other_voc/     → 10,816 张, 2 类 (mouse/other)   ← 数据量 3× 增长
```

### 0.4 原始训练参数

```yaml
# optimizer_270e.yml（已被作者修改）
epoch: 50                    # 原始270, 作者改为50
base_lr: 0.01                # 基础学习率
milestones: [30, 40]         # 衰减节点
gamma: 0.1                   # 衰减系数
warmup_steps: 1500           # 预热步数

# yolov3_reader.yml
batch_size: 8                # 训练批次大小
input_size: 多尺度 [320~608]   # 多尺度训练

# runtime.yml
save_dir: output/yolov3_mouse_other_voc    # ✅ 已更新
snapshot_epoch: 5            # 每5个epoch保存
```

### 0.5 已安装的关键依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| PaddlePaddle | 2.5.1 | 深度学习框架 |
| PaddleSlim | 2.6.0 | 模型压缩（蒸馏/量化/剪枝） |
| VisualDL | 2.5.3 | 训练可视化 |
| paddle2onnx | 1.2.11 | ONNX 模型转换（兼容 Paddle 2.5.1） |
| pycocotools | 2.0.11 | COCO 格式评估工具 |

### 0.6 已有训练历史（output/ 目录, ~2.8GB）

| 实验名 | 说明 | 状态 |
|--------|------|------|
| `yolov3_my_dog_mouse_voc/` | 最新一轮训练, epoch 4~44 | 有 model_final |
| `trained_models/yolov3_my_dog_mouse_voc_0废弃/` | 第一轮（废弃） | 有 model_final |
| `trained_models/yolov3_my_dog_mouse_voc_1err/` | 第二轮（有错误） | 有 model_final |
| `trained_models/yolov3_my_dog_mouse_voc_2non/` | 第三轮（无检测框） | 有 model_final |
| `yolov3_mobilenet_v1_roadsign/` | 路标检测基线实验 | 有 model_final |

### 0.7 发现的关键问题与修复状态

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| 1 | 路径分隔符为 Windows 风格 (`\`) | ✅ 已修复 | `merge_dataset.py` 生成的新 train/val.txt 使用 Linux 正斜杠 |
| 2 | 验证集文件名 `valid.txt` vs `val.txt` | ✅ 已修复 | 新配置 `mouse_other_voc.yml` 中正确设置为 `val.txt` |
| 3 | XML 中含 Windows 硬编码路径 | ✅ 已修复 | `merge_dataset.py` 已清除 `<path>` 标签，更新 `<folder>` 和 `<filename>` |
| 4 | `dog` 类标签需改为 `other` | ✅ 已修复 | 所有原 dog 标注的 `<name>` 已改为 other |
| 5 | 默认学习率适配 8 GPU | ⚠️ 需注意 | 训练时通过 `-o LearningRate.base_lr=` 覆盖 |
| 6 | Anchor 尺寸可能不适合老鼠目标 | 📋 待做 | 需运行 `tools/anchor_cluster.py` 重新聚类 |

#### 问题 5 详解：学习率线性缩放

原始配置的 `base_lr: 0.01` 是为 8 GPU 设计的。当前环境为 1~2 GPU，根据线性缩放法则：
```
单卡: 0.01 / 8 = 0.00125
双卡: 0.01 / 4 = 0.0025
```

#### 问题 6 详解：Anchor 聚类

当前 Anchor 是 COCO 数据集的默认值：
```yaml
anchors: [[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]]
```
如果实验鼠在图中是中小目标，大 Anchor（如 `[373,326]`）基本无用，需要重新聚类。

**聚类命令**（训练前执行）：
```bash
python tools/anchor_cluster.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -n 9 -s 608 -m v2 -i 1000
```

---

## 一、环境验证

### 1.1 当前环境信息（已验证 ✅）

| 项目 | 值 |
|------|-----|
| GPU | 2x Tesla T4 (15GB VRAM each) |
| GPU Compute Capability | 7.5 |
| Driver Version | 535.86.10 |
| CUDA Version (Driver) | 12.2 |
| CUDA Runtime | 11.6 |
| cuDNN | 8.4 |
| PaddlePaddle | 2.5.1 (GPU) |
| 多卡状态 | ✅ PaddlePaddle works well on 2 GPUs |

### 1.2 环境验证命令

每次新开终端/环境后，建议运行以下命令确认环境正常：

```bash
# 一行命令快速验证 GPU 和 PaddlePaddle
python3 -c "
import paddle
print('Paddle:', paddle.__version__)
print('CUDA compiled:', paddle.is_compiled_with_cuda())
print('GPU count:', paddle.device.cuda.device_count())
print('cuDNN:', paddle.device.get_cudnn_version())
paddle.utils.run_check()
"
```

---

## 二、训练前准备（前置步骤）

> **✅ 数据修复已完成**: 路径分隔符、文件名不匹配、dog→other 标签修改、数据合并等问题
> 已全部通过 `Scripts/merge_dataset.py` 一次性修复，无需手动操作。

### 2.1 验证数据完整性

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

python3 -c "
import os
missing_img, missing_ann = 0, 0
for split in ['train.txt', 'val.txt']:
    path = f'dataset/mouse_other_voc/{split}'
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        img = os.path.join('dataset/mouse_other_voc', parts[0])
        ann = os.path.join('dataset/mouse_other_voc', parts[1])
        if not os.path.exists(img):
            print(f'Missing: {img}'); missing_img += 1
        if not os.path.exists(ann):
            print(f'Missing: {ann}'); missing_ann += 1
    print(f'{split}: {len(lines)} 条')
print('✅ 数据完整' if missing_img + missing_ann == 0 else '⚠️ 存在缺失！')
"
```

### 2.2 安装训练依赖

```bash
pip install paddleslim visualdl paddle2onnx pycocotools
```

### 2.3 Anchor 聚类（推荐在训练前执行）

```bash
python tools/anchor_cluster.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -n 9 -s 608 -m v2 -i 1000
```

---

## 三、分阶段训练计划

> **约定**: 以下所有命令均在 `/hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6` 目录下执行。
> 通过 `-o` 参数覆盖配置文件中的值，**不修改原始 yml 文件**。

---

### 阶段 1：冒烟测试（Smoke Test）

#### 目的
- 验证数据读取管道（Data Pipeline）是否正常工作
- 确认 GPU 训练流程无报错
- 确认输出目录可以正确写入
- **不关心模型精度**，只关心"能跑通"

#### 关键参数调整

| 参数 | 原值 | 冒烟测试值 | 说明 |
|------|------|-----------|------|
| `epoch` | 50 | **2** | 仅跑2个epoch |
| `base_lr` | 0.01 | **0.00125** | 适配单卡 (÷8) |
| `snapshot_epoch` | 5 | **1** | 每个epoch保存 |
| `TrainReader.batch_size` | 8 | **2** | 最小batch避免OOM |
| `worker_num` | 2 | **0** | 调试时关闭多进程 |
| `log_iter` | 20 | **5** | 更频繁打印日志 |
#### 单卡冒烟测试命令

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=2 \
    LearningRate.base_lr=0.00125 \
    snapshot_epoch=1 \
    log_iter=5 \
    worker_num=0 \
    TrainReader.batch_size=2 \
    save_dir=output/smoke_test \
    --eval
```

#### 双卡冒烟测试命令

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=2 \
    LearningRate.base_lr=0.0025 \
    snapshot_epoch=1 \
    log_iter=5 \
    worker_num=0 \
    TrainReader.batch_size=4 \
    save_dir=output/smoke_test_2gpu \
    --eval
```

#### 验收标准
- [ ] 训练开始后能正常打印 loss 数值
- [ ] 2 个 epoch 结束后无报错
- [ ] `output/smoke_test/` 目录下生成了 `.pdparams` 和 `.pdopt` 文件
- [ ] `--eval` 能输出 mAP 数值（即使很低也正常）

---

### 阶段 2：基线复现（Baseline）

#### 目的
- 使用原始配置进行完整训练
- 建立 mAP 性能基线
- 记录训练曲线，作为后续优化的对照组

#### 关键参数调整

| 参数 | 原值 | 基线值 | 说明 |
|------|------|-------|------|
| `epoch` | 50 | **50** | 保持不变 |
| `base_lr` | 0.01 | **0.0025** | 适配双卡 (÷4) |
| `milestones` | [30, 40] | **[30, 40]** | 保持不变 |
| `TrainReader.batch_size` | 8 | **8** | 单卡8, T4 15GB应可承受 |
| `snapshot_epoch` | 5 | **5** | 保持不变 |
| `warmup_steps` | 1500 | **500** | 数据集较小，适当缩短预热 |
#### 单卡基线训练

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    LearningRate.base_lr=0.00125 \
    save_dir=output/baseline_1gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/baseline_1gpu/vdl_log
```

#### 双卡基线训练（加速版本）

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    LearningRate.base_lr=0.0025 \
    save_dir=output/baseline_2gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/baseline_2gpu/vdl_log
```

#### 基线评估命令

```bash
python tools/eval.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/baseline_1gpu/best_model.pdparams \
    --classwise
```

#### 基线推理可视化

```bash
python tools/infer.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/baseline_1gpu/best_model.pdparams \
    --infer_img=dataset/mouse_other_voc/images/mouse_00001.jpg \
    --output_dir=output/baseline_1gpu/infer_vis/ \
    --draw_threshold=0.3
```

#### 验收标准
- [ ] 50 个 epoch 正常完成
- [ ] mAP(0.50) > 50%（合理预期范围）
- [ ] 各类别 AP 数据已记录
- [ ] 通过 VisualDL 观察 loss 曲线正常收敛

---

### 阶段 3：扩展与优化（Scaling & Optimization）

基于阶段 2 的基线结果，提出以下 3 种优化变体：

---

#### 变体 A：更换更强 Backbone — ResNet50 + FPN

**目的**: MobileNetV1 是轻量级 backbone，精度有限。ResNet50 具有更强的特征提取能力。

**预期效果**: mAP 提升 5~15 个百分点，推理速度略有下降。

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

# 使用 YOLOv3-DarkNet53（官方配置，更强的backbone）
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_darknet53_270e_voc.yml \
    -o \
    num_classes=2 \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    TrainDataset.dataset_dir=dataset/mouse_other_voc \
    TrainDataset.anno_path=train.txt \
    TrainDataset.label_list=label_list.txt \
    EvalDataset.dataset_dir=dataset/mouse_other_voc \
    EvalDataset.anno_path=val.txt \
    EvalDataset.label_list=label_list.txt \
    TestDataset.anno_path=dataset/mouse_other_voc/label_list.txt \
    save_dir=output/variant_a_darknet53 \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/variant_a_darknet53/vdl_log
```

> **备选**: PPYOLOE-S 是更现代的检测器，精度/速度平衡更好：

```bash
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
    -o \
    num_classes=2 \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    TrainDataset='{name: VOCDataSet, dataset_dir: dataset/mouse_other_voc, anno_path: train.txt, label_list: label_list.txt, data_fields: [image, gt_bbox, gt_class, difficult]}' \
    EvalDataset='{name: VOCDataSet, dataset_dir: dataset/mouse_other_voc, anno_path: val.txt, label_list: label_list.txt, data_fields: [image, gt_bbox, gt_class, difficult]}' \
    metric=VOC \
    save_dir=output/variant_a_ppyoloe_s \
    --eval
```

---

#### 变体 B：增大 Batch Size + 线性缩放学习率

**目的**: 利用双卡 T4 的显存优势，增大 batch size 以提升训练稳定性和速度。

**线性缩放法则**: `new_lr = base_lr × (new_bs × num_gpus) / (original_bs × original_gpus)`

| 方案 | Batch Size (每卡) | GPU 数 | 等效 BS | 学习率 |
|------|-------------------|--------|---------|--------|
| 原始 | 8 | 8 | 64 | 0.01 |
| 基线 | 8 | 1 | 8 | 0.00125 |
| **变体B** | **16** | **2** | **32** | **0.005** |

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.005 \
    TrainReader.batch_size=16 \
    save_dir=output/variant_b_large_bs \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/variant_b_large_bs/vdl_log
```

> **注意**: 如果 batch_size=16 在 T4 (15GB) 上 OOM，降为 12 并相应调整 lr：
> `LearningRate.base_lr=0.00375 TrainReader.batch_size=12`

---

#### 变体 C：Anchor 重新聚类（针对老鼠小目标优化）

**目的**: 默认 Anchor 是 COCO 数据集的统计结果，可能不适合老鼠检测场景。通过对本数据集的 bounding box 进行 K-Means 聚类，生成更合适的 Anchor。

**步骤 1: 运行 Anchor 聚类工具**

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

python tools/anchor_cluster.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -n 9 \
    -s 608 \
    -m v2 \
    -i 1000
```

参数说明：
- `-n 9`: 聚类出 9 个 Anchor（YOLOv3 标配 3 个尺度 × 3 个 Anchor）
- `-s 608`: 输入尺寸 608×608
- `-m v2`: 使用 v2 版本的聚类算法
- `-i 1000`: 迭代 1000 次

**步骤 2: 将聚类结果写入训练命令**

假设聚类输出了 9 个新 Anchor（以下为示例值，需替换为实际结果）：
```
[15, 20], [25, 40], [45, 35], [40, 70], [75, 60], [70, 130], [130, 100], [180, 220], [350, 300]
```

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

# ⚠️ 下面的 anchors 值需要替换为 anchor_cluster.py 的实际输出
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    save_dir=output/variant_c_custom_anchors \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/variant_c_custom_anchors/vdl_log
```

> **注意**: `Gt2YoloTarget` 中的 anchors 也需要同步更新。如果通过 `-o` 覆盖 reader 中的 anchors 比较复杂，建议复制一份 `yolov3_reader.yml` 单独修改。

---

## 四、完整执行顺序清单

```
┌─────────────────────────────────────────────────────────┐
│  Step 0. 环境验证 + 依赖安装                              │
│    → python3 -c "import paddle; ..."                     │
│    → pip install paddleslim visualdl paddle2onnx          │
├─────────────────────────────────────────────────────────┤
│  Step 1. 数据验证 (✅ 已完成合并)                          │
│    → python3 数据完整性验证脚本                             │
│    → 10,816 对图片-标注, 2 类: mouse/other                │
├─────────────────────────────────────────────────────────┤
│  Step 2. Anchor 聚类（推荐）                               │
│    → python tools/anchor_cluster.py                       │
│    → 记录 9 个聚类结果                                     │
├─────────────────────────────────────────────────────────┤
│  Step 3. 冒烟测试（~5 分钟）                               │
│    → 单卡冒烟测试命令                                      │
│    → 确认无报错后进入下一步                                  │
├─────────────────────────────────────────────────────────┤
│  Step 4. 基线训练（~2-4 小时, 50 epoch）                   │
│    → 单卡或双卡基线训练命令                                 │
│    → 记录 mAP 结果，查看各类别 AP                          │
├─────────────────────────────────────────────────────────┤
│  Step 5. 优化实验（选择 1-2 个变体）                        │
│    → 变体 A: DarkNet53 Backbone                           │
│    → 变体 B: 大 Batch Size                                │
│    → 变体 C: 自定义 Anchor                                │
├─────────────────────────────────────────────────────────┤
│  Step 6. 模型压缩（按需）                                  │
│    → 蒸馏 / 量化 / 剪枝 / 联合策略                         │
│    → 详见 Training_Pipeline.md 第 8-11 节                  │
├─────────────────────────────────────────────────────────┤
│  Step 7. 模型导出与部署                                    │
│    → export_model.py 导出                                 │
│    → paddle2onnx 转换                                     │
│    → deploy/python/infer.py 验证                          │
└─────────────────────────────────────────────────────────┘
```

> **📖 详细全流程手册**: 请参阅 [`Training_Pipeline.md`](./Training_Pipeline.md)，
> 包含训练、评估、推理、蒸馏、量化、剪枝、ONNX 导出的完整命令和参数说明。

---

## 五、模型导出命令（供训练完成后使用）

```bash
# 1. 导出最佳模型为推理模型
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/baseline_1gpu/best_model.pdparams \
    --output_dir=output/inference_model

# 2. 转换为 ONNX
paddle2onnx \
    --model_dir output/inference_model/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_detection.onnx

# 3. Python 部署推理测试
python deploy/python/infer.py \
    --model_dir=output/inference_model/yolov3_my_dog_mouse_voc \
    --image_file=dataset/mouse_other_voc/images/mouse_00001.jpg \
    --device=GPU \
    --threshold=0.3
```

> **更多压缩+导出方案**（蒸馏、量化、剪枝后的导出命令）请参见 [`Training_Pipeline.md`](./Training_Pipeline.md) 第 7-12 节。

---

## 六、监控与调试技巧

### 6.1 VisualDL 可视化

```bash
# 启动 VisualDL 查看训练曲线
visualdl --logdir output/baseline_1gpu/vdl_log --host 0.0.0.0 --port 8040
```

### 6.2 GPU 监控

```bash
# 实时监控 GPU 使用率和显存
watch -n 1 nvidia-smi
```

### 6.3 如果训练中断，恢复训练

```bash
# 从 checkpoint 恢复（以 epoch 20 为例）
python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -r output/baseline_1gpu/20 \
    -o \
    LearningRate.base_lr=0.00125 \
    save_dir=output/baseline_1gpu \
    --eval
```

---

## 附录：快速参考表

### PaddleDetection CLI 常用参数

| 参数 | 含义 | 示例 |
|------|------|------|
| `-c` | 指定配置文件 | `-c configs/yolov3/xxx.yml` |
| `-o` | 覆盖配置参数 | `-o epoch=10 LearningRate.base_lr=0.001` |
| `--eval` | 边训边评估 | `--eval` |
| `-r` | 恢复训练 | `-r output/xxx/20` |
| `--use_vdl` | 启用 VisualDL | `--use_vdl=true` |
| `--vdl_log_dir` | VDL 日志目录 | `--vdl_log_dir=vdl_log/` |

### 学习率线性缩放快查表

| GPU 数 | Batch/卡 | 等效 BS | 学习率 |
|--------|----------|---------|--------|
| 1 | 8 | 8 | 0.00125 |
| 2 | 8 | 16 | 0.0025 |
| 2 | 16 | 32 | 0.005 |
| 4 | 8 | 32 | 0.005 |
| 8 | 8 | 64 | 0.01 (原始) |
