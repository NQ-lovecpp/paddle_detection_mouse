# 实验鼠检测模型 — 全流程训练手册

> **项目**: 基于 PaddleDetection 2.6 的二分类目标检测（mouse / other）  
> **环境**: 2× Tesla T4, PaddlePaddle 2.5.1, CUDA 11.6  
> **数据集**: `mouse_other_voc` (10,816 张图片, VOC 格式, 二分类)  
> **更新日期**: 2026-02-07  
> **所有命令均在 `/hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6` 目录下执行**

---

## 目录

0. [前置环境检查与依赖安装](#0-前置环境检查与依赖安装)
1. [数据预处理](#1-数据预处理)
2. [Anchor 聚类分析](#2-anchor-聚类分析)
3. [VisualDL 可视化配置](#3-visualdl-可视化配置)
4. [训练 — 多套配置方案](#4-训练--多套配置方案)
5. [模型评估](#5-模型评估)
6. [模型推理](#6-模型推理)
7. [模型导出](#7-模型导出)
8. [模型蒸馏（Distillation）](#8-模型蒸馏distillation)
9. [模型量化（Quantization）](#9-模型量化quantization)
10. [模型剪枝（Pruning）](#10-模型剪枝pruning)
11. [联合压缩策略](#11-联合压缩策略)
12. [ONNX 转换与部署](#12-onnx-转换与部署)

---

## 0. 前置环境检查与依赖安装

### 0.1 一键环境验证

```bash
python3 -c "
import paddle
print('='*50)
print(f'PaddlePaddle: {paddle.__version__}')
print(f'CUDA compiled: {paddle.is_compiled_with_cuda()}')
print(f'GPU count: {paddle.device.cuda.device_count()}')
print(f'cuDNN: {paddle.device.get_cudnn_version()}')
for i in range(paddle.device.cuda.device_count()):
    print(f'  GPU {i}: {paddle.device.cuda.get_device_name(i)}')
paddle.utils.run_check()
print('='*50)
"
```

### 0.2 安装必要依赖

```bash
# 核心依赖（训练+压缩+可视化+ONNX导出）
pip install paddleslim visualdl paddle2onnx pycocotools

# 验证安装
python3 -c "
import paddleslim; print(f'PaddleSlim: {paddleslim.__version__}')
import visualdl;   print(f'VisualDL: {visualdl.__version__}')
import paddle2onnx; print(f'Paddle2ONNX: {paddle2onnx.__version__}')
"
```

### 0.3 查看 GPU 拓扑

```bash
nvidia-smi topo -m
# T4 通常走 PCIe 总线通信, 输出中 PHB/PIX 表示 PCIe 互联
```

---

## 1. 数据预处理

### 1.1 当前数据集概览

```
dataset/mouse_other_voc/
├── images/           → 10,816 张 JPG
├── annotations/      → 10,816 个 XML (Pascal VOC 格式)
├── label_list.txt    → mouse, other
├── train.txt         → 8,653 条 (80%)
└── val.txt           → 2,163 条 (20%)
```

### 1.2 数据完整性验证

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
        if len(parts) != 2:
            print(f'Bad line in {split}: {line.strip()}')
            continue
        img = os.path.join('dataset/mouse_other_voc', parts[0])
        ann = os.path.join('dataset/mouse_other_voc', parts[1])
        if not os.path.exists(img):
            print(f'Missing: {img}')
            missing_img += 1
        if not os.path.exists(ann):
            print(f'Missing: {ann}')
            missing_ann += 1
    print(f'{split}: {len(lines)} 条, 缺失图片={missing_img}, 缺失标注={missing_ann}')
print('验证完成 ✅' if missing_img + missing_ann == 0 else '⚠️ 存在缺失文件！')
"
```

### 1.3 重新划分数据集（可选）

如果需要修改训练集/验证集比例，可使用以下脚本：

```bash
python3 -c "
import os, random
random.seed(42)

base = 'dataset/mouse_other_voc'
# 获取所有图片-标注配对
pairs = []
for f in sorted(os.listdir(os.path.join(base, 'images'))):
    if f.endswith('.jpg'):
        stem = f[:-4]
        xml_path = os.path.join(base, 'annotations', stem + '.xml')
        if os.path.exists(xml_path):
            pairs.append((f'./images/{f}', f'./annotations/{stem}.xml'))

random.shuffle(pairs)
split = int(len(pairs) * 0.8)
train_pairs = pairs[:split]
val_pairs = pairs[split:]

# 备份并写入
for name, data in [('train.txt', train_pairs), ('val.txt', val_pairs)]:
    path = os.path.join(base, name)
    if os.path.exists(path):
        os.rename(path, path + '.bak')
    with open(path, 'w') as f:
        for img, ann in data:
            f.write(f'{img} {ann}\n')
    print(f'{name}: {len(data)} 条')
print(f'总计: {len(pairs)} 对')
"
```

### 1.4 数据增强配置说明

PaddleDetection 的 YOLOv3 Reader 默认使用以下数据增强（无需修改源码）：

```yaml
# configs/yolov3/_base_/yolov3_reader.yml 中的关键配置
TrainReader:
  inputs_def:
    num_max_boxes: 50
  sample_transforms:
    - Decode: {}
    - Mixup: {alpha: 1.5, beta: 1.5}        # Mixup 数据增强
    - RandomDistort: {}                       # 随机颜色扰动
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}  # 随机扩展
    - RandomCrop: {}                          # 随机裁剪
    - RandomFlip: {}                          # 随机翻转
  batch_transforms:
    - BatchRandomResize:                      # 多尺度训练
        target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        random_size: True
        random_interp: True
        keep_ratio: False
    - NormalizeImage:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        is_scale: True
    - Permute: {}
    - PadGT: {}
  batch_size: 8
  shuffle: true
  drop_last: true
  worker_num: 2
```

如果需要调整数据增强策略（比如关闭 Mixup 进行 debug），可通过 `-o` 覆盖：

```bash
# 关闭 Mixup（调试用）
-o "TrainReader.sample_transforms=[{Decode: {}}, {RandomDistort: {}}, {RandomExpand: {fill_value: [123.675, 116.28, 103.53]}}, {RandomCrop: {}}, {RandomFlip: {}}]"
```

---

## 2. Anchor 聚类分析

YOLOv3 的 Anchor 对检测性能影响很大。默认 Anchor 来自 COCO 数据集，需要对本数据集重新聚类。

### 2.1 运行 Anchor 聚类

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

python tools/anchor_cluster.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -n 9 \
    -s 608 \
    -m v2 \
    -i 1000
```

输出：
```
[02/07 21:50:47] ppdet.anchor_cluster INFO: 9 anchor cluster result: [w, h]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [74, 115]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [95, 238]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [155, 174]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [158, 356]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [247, 253]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [256, 466]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [388, 343]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [388, 520]
[02/07 21:50:47] ppdet.anchor_cluster INFO: [345, 298]
```


**参数说明**：
- `-n 9`: 聚类 9 个 Anchor（YOLOv3 = 3 尺度 × 3 Anchor）
- `-s 608`: 输入分辨率 608×608
- `-m v2`: 使用改进的 K-Means 算法
- `-i 1000`: 迭代 1000 次

### 2.2 使用聚类结果

将输出的 Anchor 值替换到训练命令中（示例，**以实际输出为准**）：

```bash
-o "YOLOv3Head.anchors=[[74, 115],[95, 238],[155, 174],[158, 356],[247, 253],[256, 466],[388, 343],[388, 520],[345, 298]]"
```

---

## 3. VisualDL 可视化配置

### 3.1 训练时启用 VisualDL

在任何 `tools/train.py` 命令后添加：

```bash
--use_vdl=true \
--vdl_log_dir=output/<实验名>/vdl_log
```

PaddleDetection 会自动记录：
- **loss 变化趋势**（train loss, loc_loss, cls_loss, obj_loss）
- **mAP 变化趋势**（每次 eval 时记录）
- **学习率衰减曲线**

### 3.2 启动 VisualDL 服务

```bash
# 新开一个终端窗口
visualdl --logdir output/ --host 0.0.0.0 --port 8040

# --logdir 指向包含 vdl_log 的父目录，可以同时对比多个实验
# 浏览器访问: http://<服务器IP>:8040
```

### 3.3 对比多个实验

```bash
# 同时查看所有实验的曲线
visualdl --logdir \
    output/smoke_test/vdl_log,\
    output/baseline_2gpu/vdl_log,\
    output/variant_darknet53/vdl_log \
    --host 0.0.0.0 --port 8040
```

VisualDL 支持：
- **Scalar**: loss / mAP 曲线
- **Image**: 数据增强可视化
- **Graph**: 模型计算图
- **Histogram**: 权重分布

---

## 4. 训练 — 多套配置方案

> **通用约定**:
> - 配置文件: `configs/yolov3/yolov3_my_dog_mouse_voc.yml`
> - 所有参数通过 `-o` 覆盖，**不修改原始 yml 文件**
> - VisualDL 全程开启

---

### 方案 A: 冒烟测试（Smoke Test）

**目的**: 验证数据管道、GPU 训练流程、输出目录均正常，不关心精度。

```bash
# =====================================================
# 方案 A: 冒烟测试 (单卡, ~5 分钟)
# =====================================================
# 目的: 确认数据读取、前向/反向、模型保存均可用
# 参数: 2 epoch, bs=2, 关闭多进程加载
# 预期: 能跑完不报错即可
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    max_iters=1000 \
    LearningRate.base_lr=0.00125 \
    snapshot_epoch=1 \
    log_iter=22 \
    worker_num=8 \
    use_shared_memory=true \
    TrainReader.batch_size=26 \
    TrainReader.buf_size=120000 \
    TrainReader.mixup_epoch=-1 \
    save_dir=output/A_safe_test \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/A_safe_test/vdl_log
```

output:
```
[02/09 04:04:25] ppdet.metrics.metrics INFO: Accumulating evaluatation results...
[02/09 04:04:25] ppdet.metrics.metrics INFO: mAP(0.50, integral) = 93.63%
[02/09 04:04:25] ppdet.engine INFO: Total sample number: 2163, average FPS: 41.01975843968071
[02/09 04:04:25] ppdet.engine INFO: Best test bbox ap is 0.939.
```

visualdl：
```
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

visualdl \
    --logdir output/A_smoke_test/vdl_log,vdl_dir/scalar \
    --host 0.0.0.0 --port 8040
```


**验收标准**:
- [ ] 训练正常打印 loss
- [ ] 2 epoch 无报错
- [ ] `output/A_smoke_test/` 下有 `.pdparams` + `.pdopt`
- [ ] eval 输出 mAP 数值

---

### 方案 B: 基线训练 — 单卡 MobileNetV1（Baseline）

**目的**: 完整训练建立性能基线。

```bash
# =====================================================
# 方案 B: 基线训练 (单卡, ~2-3 小时)
# =====================================================
# 目的: 建立 mAP 基线，作为后续优化的对照
# 参数: 50 epoch, bs=8, lr=0.00125 (原始0.01÷8)
# Backbone: MobileNetV1 (轻量, 快速收敛)
# 预期 mAP(0.50): 60-75%
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    LearningRate.base_lr=0.00125 \
    save_dir=output/B_baseline_1gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/B_baseline_1gpu/vdl_log
```

```
[02/08 01:59:45] ppdet.utils.checkpoint INFO: Save checkpoint: output/A_smoke_test/yolov3_my_dog_mouse_voc
[02/08 01:59:45] ppdet.engine INFO: Eval iter: 0
[02/08 01:59:52] ppdet.engine INFO: Eval iter: 100
[02/08 01:59:59] ppdet.engine INFO: Eval iter: 200
[02/08 02:00:06] ppdet.engine INFO: Eval iter: 300
[02/08 02:00:14] ppdet.engine INFO: Eval iter: 400
[02/08 02:00:22] ppdet.engine INFO: Eval iter: 500
[02/08 02:00:30] ppdet.engine INFO: Eval iter: 600
[02/08 02:00:38] ppdet.engine INFO: Eval iter: 700
[02/08 02:00:47] ppdet.engine INFO: Eval iter: 800
[02/08 02:00:55] ppdet.engine INFO: Eval iter: 900
[02/08 02:01:03] ppdet.engine INFO: Eval iter: 1000
[02/08 02:01:09] ppdet.engine INFO: Eval iter: 1100
[02/08 02:01:15] ppdet.engine INFO: Eval iter: 1200
[02/08 02:01:21] ppdet.engine INFO: Eval iter: 1300
[02/08 02:01:27] ppdet.engine INFO: Eval iter: 1400
[02/08 02:01:33] ppdet.engine INFO: Eval iter: 1500
[02/08 02:01:39] ppdet.engine INFO: Eval iter: 1600
[02/08 02:01:45] ppdet.engine INFO: Eval iter: 1700
[02/08 02:01:52] ppdet.engine INFO: Eval iter: 1800
[02/08 02:01:59] ppdet.engine INFO: Eval iter: 1900
[02/08 02:02:06] ppdet.engine INFO: Eval iter: 2000
[02/08 02:02:13] ppdet.engine INFO: Eval iter: 2100
[02/08 02:02:16] ppdet.metrics.metrics INFO: Accumulating evaluatation results...
[02/08 02:02:16] ppdet.metrics.metrics INFO: mAP(0.50, integral) = 52.30%
[02/08 02:02:16] ppdet.engine INFO: Total sample number: 2163, average FPS: 14.319184430019613
[02/08 02:02:16] ppdet.engine INFO: Best test bbox ap is 0.526.
```

---

### 方案 C: 双卡加速基线

**目的**: 利用 2× T4 加速训练，缩短实验周期。

```bash
# =====================================================
# 方案 C: 双卡基线 (双卡, ~1-2 小时)
# =====================================================
# 目的: 双卡加速, 缩短迭代周期
# 参数: 50 epoch, bs=8/卡, 等效bs=16, lr=0.0025
# 线性缩放: 0.01 × (16/64) = 0.0025
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    LearningRate.base_lr=0.0025 \
    save_dir=output/C_baseline_2gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/C_baseline_2gpu/vdl_log
```

---

### 方案 D: 更强 Backbone — DarkNet53

**目的**: DarkNet53 比 MobileNetV1 特征提取更强，代价是推理速度下降。

```bash
# =====================================================
# 方案 D: DarkNet53 backbone (双卡, ~3-5 小时)
# =====================================================
# 目的: 用更强的 backbone 提升精度上限
# 配置: yolov3_darknet53_270e_voc.yml (官方VOC配置)
# 修改: 指向 mouse_other_voc 数据集, num_classes=2
# 预期 mAP: 比 MobileNetV1 高 5-15 个点
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

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
    save_dir=output/D_darknet53_2gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/D_darknet53_2gpu/vdl_log
```

---

### 方案 E: 大 Batch Size + 线性缩放

**目的**: 增大 batch size 提升训练稳定性，观察是否有精度增益。

```bash
# =====================================================
# 方案 E: 大 Batch Size (双卡, ~2-3 小时)
# =====================================================
# 目的: 探索更大 batch size 是否有精度增益
# 参数: bs=16/卡, 等效bs=32, lr=0.005
# 线性缩放: 0.01 × (32/64) = 0.005
# 注意: 如果 T4 (15GB) OOM, 改 bs=12, lr=0.00375
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.005 \
    TrainReader.batch_size=16 \
    save_dir=output/E_large_bs_2gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/E_large_bs_2gpu/vdl_log
```

---

### 方案 F: 自定义 Anchor + 加长训练

**目的**: 结合 Anchor 聚类结果，针对本数据集特征优化检测。

```bash
# =====================================================
# 方案 F: 自定义 Anchor (双卡, ~4-6 小时)
# =====================================================
# 目的: 用数据集聚类的 Anchor 替代 COCO 默认值
# 前置: 先运行 anchor_cluster.py 获取实际值
# 注意: 下面的 anchors 是示例值, 必须替换为实际聚类结果!
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

# ⚠️ 将下面的 anchors 替换为 anchor_cluster.py 的输出
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    epoch=100 \
    LearningRate.base_lr=0.0025 \
    save_dir=output/F_custom_anchors_2gpu \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/F_custom_anchors_2gpu/vdl_log
```

---

### 训练恢复（断点续训）

如果训练中断（如服务器重启），可从最近的 checkpoint 恢复：

```bash
# -r 指定 checkpoint 的 epoch 编号
python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -r output/B_baseline_1gpu/20 \
    -o \
    LearningRate.base_lr=0.00125 \
    save_dir=output/B_baseline_1gpu \
    --eval
```

---

## 5. 模型评估

### 5.1 单模型评估

```bash
# 评估最佳模型, --classwise 输出每个类别的 AP
python tools/eval.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o \
    weights=output/B_baseline_1gpu/best_model.pdparams \
    --classwise
```

### 5.2 对比多方案

```bash
# 批量评估所有实验
for dir in output/B_baseline_1gpu output/C_baseline_2gpu output/D_darknet53_2gpu output/E_large_bs_2gpu; do
    echo "======== Evaluating: $dir ========"
    python tools/eval.py \
        -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
        -o weights=${dir}/best_model.pdparams \
        --classwise 2>&1 | grep -E "mAP|mouse|other"
    echo ""
done
```

---

## 6. 模型推理

### 6.1 单图推理

```bash
python tools/infer.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/B_baseline_1gpu/best_model.pdparams \
    --infer_img=dataset/mouse_other_voc/images/mouse_00001.jpg \
    --output_dir=output/infer_vis/ \
    --draw_threshold=0.3
```

### 6.2 批量推理

```bash
# 对整个文件夹推理
python tools/infer.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/B_baseline_1gpu/best_model.pdparams \
    --infer_dir=dataset/mouse_other_voc/images/ \
    --output_dir=output/infer_vis_batch/ \
    --draw_threshold=0.3
```

---

## 7. 模型导出

### 7.1 导出 Paddle Inference 模型

```bash
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    -o weights=output/B_baseline_1gpu/best_model.pdparams \
    --output_dir=output/inference_model
```

导出后的目录结构：
```
output/inference_model/yolov3_my_dog_mouse_voc/
├── infer_cfg.yml       # 推理配置 (含预处理信息)
├── model.pdmodel       # 网络结构
└── model.pdiparams     # 模型权重
```

### 7.2 Python 部署推理

```bash
python deploy/python/infer.py \
    --model_dir=output/inference_model/yolov3_my_dog_mouse_voc \
    --image_file=dataset/mouse_other_voc/images/mouse_00001.jpg \
    --device=GPU \
    --threshold=0.3
```

---

## 8. 模型蒸馏（Distillation）

### 8.1 蒸馏原理

知识蒸馏让一个大模型（Teacher）的知识迁移到小模型（Student）：

```
Teacher (YOLOv3-ResNet34, 更大更准)
    │
    │  输出 soft labels / feature maps
    │
    ▼
Student (YOLOv3-MobileNetV1, 更小更快)
    │
    │  同时学习 ground truth + teacher 的输出
    │
    ▼
压缩后的模型: 精度接近 Teacher, 速度接近 Student
```

### 8.2 PaddleDetection 原生蒸馏配置

PaddleDetection 提供了 `configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml`：

```yaml
# 原生配置内容:
_BASE_: [
  '../../yolov3/yolov3_r34_270e_coco.yml',    # Teacher: YOLOv3-ResNet34
]
pretrain_weights: https://...yolov3_r34_270e_coco.pdparams  # Teacher 预训练权重

slim: Distill
distill_loss: DistillYOLOv3Loss

DistillYOLOv3Loss:
  weight: 1000    # 蒸馏损失权重
```

### 8.3 蒸馏训练命令

```bash
# =====================================================
# 蒸馏: Teacher=ResNet34, Student=MobileNetV1
# =====================================================
# 前置: 需要安装 PaddleSlim (pip install paddleslim)
# 原理: Student 同时学习 GT 标签和 Teacher 的输出
# 官方数据: mAP 从 29.4 提升到 31.0 (+1.6, COCO)
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    save_dir=output/distill_r34_to_mv1 \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/distill_r34_to_mv1/vdl_log
```

### 8.4 蒸馏模型评估

```bash
python tools/eval.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml \
    -o weights=output/distill_r34_to_mv1/best_model.pdparams \
    --classwise
```

### 8.5 蒸馏模型导出

```bash
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml \
    -o weights=output/distill_r34_to_mv1/best_model.pdparams \
    --output_dir=output/inference_model_distill
```

---

## 9. 模型量化（Quantization）

### 9.1 量化原理

将模型权重和激活从 FP32 压缩到 INT8，大幅减小模型体积和加速推理：

```
FP32 模型 (94.2 MB)
    │
    │  量化感知训练 (QAT)
    │  在训练时模拟 INT8 量化误差
    │
    ▼
INT8 模型 (25.4 MB, 3.7× 压缩)
```

### 9.2 在线量化（QAT - Quantization Aware Training）

在训练过程中加入量化模拟，让模型适应量化误差：

```bash
# =====================================================
# 在线量化 (QAT): FP32 → INT8
# =====================================================
# 前置: 需要一个训练好的 FP32 模型
# 原理: 在训练时插入量化/反量化节点
# 配置: 8-bit 权重量化 + 8-bit 激活量化
# 官方数据: 模型体积 94.2MB → 25.4MB, mAP 30.5 (COCO)
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

# 以 baseline 最佳模型为起点做量化微调
python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/quant/yolov3_mobilenet_v1_qat.yml \
    -o \
    epoch=20 \
    LearningRate.base_lr=0.0001 \
    pretrain_weights=output/B_baseline_1gpu/best_model.pdparams \
    save_dir=output/quant_mv1_int8 \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/quant_mv1_int8/vdl_log
```

### 9.3 离线量化（PTQ - Post Training Quantization）

不需要重新训练，直接对导出模型进行量化（更快但精度略低）：

```bash
# =====================================================
# 离线量化 (PTQ): 无需重训, 直接量化
# =====================================================
# 原理: 用少量数据统计权重和激活的分布, 确定量化范围
# 速度: 几分钟完成
# 精度: 比 QAT 略低, 但省时省力
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/post_quant.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/post_quant/yolov3_darknet53_ptq.yml \
    -o \
    weights=output/B_baseline_1gpu/best_model.pdparams \
    --output_dir=output/ptq_mv1_int8
```

### 9.4 量化模型导出

```bash
# QAT 模型导出
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/quant/yolov3_mobilenet_v1_qat.yml \
    -o weights=output/quant_mv1_int8/best_model.pdparams \
    --output_dir=output/inference_model_quant
```

---

## 10. 模型剪枝（Pruning）

### 10.1 剪枝原理

移除模型中不重要的卷积核/通道，减小计算量：

```
原始模型 (GFLOPs: X)
    │
    │  FPGM/L1 范数剪枝
    │  按重要性排序, 移除冗余通道
    │
    ▼
剪枝模型 (GFLOPs: 0.3X~0.7X)
    │
    │  微调恢复精度
    ▼
最终模型: 更快, 精度损失 <2%
```

### 10.2 FPGM 剪枝

```bash
# =====================================================
# FPGM 剪枝 (Geometric Median)
# =====================================================
# 原理: 剪除与几何中值最接近的通道 (冗余通道)
# 配置: 18 个卷积层, 剪枝率 10%-40%
# 前置: 需要训练好的 FP32 模型
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/prune/yolov3_prune_fpgm.yml \
    -o \
    epoch=50 \
    LearningRate.base_lr=0.00125 \
    pretrain_weights=output/B_baseline_1gpu/best_model.pdparams \
    save_dir=output/prune_fpgm \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/prune_fpgm/vdl_log
```

### 10.3 L1 范数剪枝

```bash
# L1 范数剪枝 — 另一种剪枝标准
python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/prune/yolov3_prune_l1_norm.yml \
    -o \
    epoch=50 \
    LearningRate.base_lr=0.00125 \
    pretrain_weights=output/B_baseline_1gpu/best_model.pdparams \
    save_dir=output/prune_l1 \
    --eval
```

### 10.4 剪枝模型导出

```bash
python tools/export_model.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/prune/yolov3_prune_fpgm.yml \
    -o weights=output/prune_fpgm/best_model.pdparams \
    --output_dir=output/inference_model_prune
```

---

## 11. 联合压缩策略

PaddleDetection 支持多种压缩方法组合使用，进一步提升压缩率。

### 11.1 蒸馏 + 剪枝

```bash
# =====================================================
# 联合策略: 蒸馏 + 剪枝
# =====================================================
# 官方数据: GFLOPs 减少 69.4%, 模型体积减少 67.2%
# 原理: 先用 Teacher 指导, 同时进行通道剪枝
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0,1

python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/extensions/yolov3_mobilenet_v1_coco_distill_prune.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    save_dir=output/distill_prune \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/distill_prune/vdl_log
```

### 11.2 剪枝 + 量化

```bash
# =====================================================
# 联合策略: 剪枝 + 量化
# =====================================================
# 原理: 先剪枝减少通道数, 再量化到 INT8
# 效果: 模型体积和推理延迟双重压缩
# =====================================================

cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/extensions/yolov3_mobilenetv1_prune_qat.yml \
    -o \
    epoch=50 \
    LearningRate.base_lr=0.00125 \
    pretrain_weights=output/B_baseline_1gpu/best_model.pdparams \
    save_dir=output/prune_quant \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/prune_quant/vdl_log
```

---

## 12. ONNX 转换与部署

### 12.1 Paddle → ONNX

```bash
# 从导出的 inference model 转为 ONNX
paddle2onnx \
    --model_dir output/inference_model/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_detection.onnx

# 验证 ONNX 模型
python3 -c "
import onnx
model = onnx.load('output/yolov3_mouse_detection.onnx')
onnx.checker.check_model(model)
print(f'ONNX model is valid. Opset: {model.opset_import[0].version}')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
"
```

### 12.2 量化模型 → ONNX

```bash
paddle2onnx \
    --model_dir output/inference_model_quant/yolov3_my_dog_mouse_voc \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file output/yolov3_mouse_detection_int8.onnx
```

---

## 附录 A: 全流程执行顺序总览

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 0.  环境验证 + 依赖安装                                        │
 │           pip install paddleslim visualdl paddle2onnx pycocotools   │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 1.  数据验证 (1.2 节)                                         │
 │           确认 10,816 对图片-标注完整无缺失                            │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 2.  Anchor 聚类 (2.1 节)                                     │
 │           python tools/anchor_cluster.py → 记录 9 个聚类结果          │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 3.  启动 VisualDL (3.2 节)                                   │
 │           visualdl --logdir output/ --port 8040                     │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 4.  冒烟测试 — 方案 A (~5 分钟)                                │
 │           验证数据管道 + GPU 训练可用                                  │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 5.  基线训练 — 方案 B/C (~2-3 小时)                            │
 │           建立 mAP 基线, 记录各类别 AP                                │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 6.  优化实验 — 方案 D/E/F (各 3-6 小时)                        │
 │           DarkNet53 / 大 BS / 自定义 Anchor                          │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 7.  选出最优模型 → 评估 (5.1 节)                               │
 │           对比 mAP, 选出精度最高的方案                                 │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 8.  模型压缩 (按需选择)                                        │
 │           8a. 蒸馏 → 提精度不增大小                                   │
 │           8b. 量化 → INT8 压缩 3-4×                                 │
 │           8c. 剪枝 → 减 GFLOPs                                     │
 │           8d. 联合策略 → 蒸馏+剪枝 / 剪枝+量化                        │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 9.  模型导出 (7.1 节)                                         │
 │           export_model.py → Paddle Inference 格式                   │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 10. ONNX 转换 (12.1 节)                                      │
 │           paddle2onnx → .onnx 文件                                  │
 ├─────────────────────────────────────────────────────────────────────┤
 │  Step 11. 部署测试                                                   │
 │           deploy/python/infer.py 验证端到端推理                       │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 附录 B: 实验结果记录表

| 方案 | Backbone | GPU | BS(卡) | 等效BS | LR | Epoch | mAP(0.50) | mouse AP | other AP | 耗时 |
|------|----------|-----|--------|--------|-----|-------|-----------|----------|----------|------|
| B 基线 | MobileNetV1 | 1×T4 | 8 | 8 | 0.00125 | 50 | — | — | — | — |
| C 双卡 | MobileNetV1 | 2×T4 | 8 | 16 | 0.0025 | 50 | — | — | — | — |
| D DarkNet53 | DarkNet53 | 2×T4 | 8 | 16 | 0.0025 | 80 | — | — | — | — |
| E 大BS | MobileNetV1 | 2×T4 | 16 | 32 | 0.005 | 80 | — | — | — | — |
| F Anchor | MobileNetV1 | 2×T4 | 8 | 16 | 0.0025 | 100 | — | — | — | — |
| 蒸馏 | MV1(S)+R34(T) | 2×T4 | 8 | 16 | 0.0025 | 80 | — | — | — | — |
| QAT量化 | MobileNetV1 | 1×T4 | 8 | 8 | 0.0001 | 20 | — | — | — | — |
| 剪枝 | MobileNetV1 | 1×T4 | 8 | 8 | 0.00125 | 50 | — | — | — | — |

> 训练完成后在此表中填入实际结果，方便横向对比。

---

## 附录 C: 常用命令速查

```bash
# GPU 监控
watch -n 1 nvidia-smi

# 查看训练进程
ps aux | grep train.py

# 杀掉训练进程
kill -9 $(pgrep -f "tools/train.py")

# 查看磁盘空间
df -h /hy-tmp

# 清理旧实验（谨慎操作）
# rm -rf output/A_smoke_test
```
