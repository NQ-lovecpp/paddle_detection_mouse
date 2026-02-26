# 后续实验与部署指导文档

> **当前状态**（2026-02-26）：L1 PicoDet-S 单卡训练中（Epoch ~207/300），链式脚本自动运行中
> **最优结果**：L1 epoch 70 → mAP@0.5 = **94.30%**（best_model 已保存）

---

## 一、训练监督

### 1.1 脚本是否自动？

**是的，完全自动**。两个脚本 `run_lightweight.py` 和 `run_yolov3.py` 均实现：

```
L1 完成 → 写 DONE 标志 → 自动启动 L2 → L2 完成 → 启动 L3 → ...
```

不需要人工干预，关掉终端也没关系（`start_new_session=True` 使进程独立）。

### 1.2 实时监控命令

```bash
# 查看当前轮次进度
tail -f output/L1_picodet_1gpu/train.log
tail -f output/L2_picodet_2gpu/train.log      # L2 启动后

# 查看所有 eval 结果（mAP 时间线）
grep "mAP(0.50" output/L1_picodet_1gpu/train.log

# VisualDL 多实验对比（从 PaddleDetection-release-2.6/ 目录运行）
visualdl --logdir output --host 0.0.0.0 --port 8040

# 查看当前在跑哪一轮
ps aux | grep "tools/train" | grep -v grep

# 终止当前训练（不影响后续自动启动，重跑时用 --from 跳过）
kill $(cat output/L1_picodet_1gpu/train.pid)
```

### 1.3 预计完成时间

| 轮次 | 预计时长 | 完成时间（估）|
|------|---------|-------------|
| L1 单卡 300e | ~16h（已过 ~10h）| 今日内 |
| L2 双卡 300e | ~8h | L1 后 |
| L3 双卡 300e | ~8h | L2 后 |
| L4 双卡 600e | ~16h | L3 后 |
| **PicoDet 合计** | **~48h** | — |

YOLOv3 系列（`run_yolov3.py`）需单独手动启动：

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
nohup python scripts/run_yolov3.py > /tmp/yolov3_launcher.log 2>&1 &
```

### 1.4 结果汇总

训练完成后查看：

```bash
cat output/lightweight_summary.csv     # PicoDet 四轮结果
cat output/yolov3_summary.csv          # YOLOv3 四轮结果
```

---

## 二、学习率缩放策略（Linear Scaling Rule）

**原则**：batch size 乘以 k 倍，学习率也乘以 k 倍，保持梯度更新幅度等效。
**参考**：Goyal et al. 2017，"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"

### PicoDet-S（reference：2-GPU bs=64 → lr=0.16）

```
公式：lr = 0.16 × (total_bs / 64)

L1  1-GPU  bs=32   total=32   lr = 0.16 × (32/64)  = 0.08
L2  2-GPU  bs=32×2 total=64   lr = 0.16 × (64/64)  = 0.16  ← 官方推荐
L3  2-GPU  bs=48×2 total=96   lr = 0.16 × (96/64)  = 0.24
L4  2-GPU  bs=32×2 total=64   lr = 0.16             = 0.16  （同 L2，加长训练）
```

### YOLOv3-MobileNetV1（reference：bs=64 → lr=0.01，官方 COCO 默认）

```
公式：lr = 0.01 × (total_bs / 64)

Y1  1-GPU  bs=8    total=8    lr = 0.01 × (8/64)   = 0.00125
Y2  2-GPU  bs=8×2  total=16   lr = 0.01 × (16/64)  = 0.0025
Y3  1-GPU  bs=8    total=8    lr = 0.00125          （同 Y1，全量数据）
Y4  2-GPU  bs=8×2  total=16   lr = 0.0025           （同 Y2，全量数据）
```

---

## 三、4×2 对比实验设计

### 实验矩阵

```
                  ┌─────────────┬─────────────┐
                  │  1/3 数据   │  全量数据   │
┌─────────────────┼─────────────┼─────────────┤
│ YOLOv3  单卡   │ Y1 (baseline│ Y3          │
│ YOLOv3  双卡   │ Y2          │ Y4 (目标)   │
├─────────────────┼─────────────┼─────────────┤
│ PicoDet 单卡   │ L1 (已完成) │ —           │
│ PicoDet 双卡   │ L2/L3/L4    │ —           │
└─────────────────┴─────────────┴─────────────┘
```

### 对比维度与结论指引

**维度 1：数据量影响**（Y1 vs Y3，Y2 vs Y4）
- 验证从 1/3 到全量数据，mAP 提升了多少
- 简历表述："将数据集从约 1,943 张扩充至 7,286 张（+3.75×），mAP +2.8%"

**维度 2：多卡加速**（Y1 vs Y2，Y3 vs Y4）
- 验证双卡是否加速收敛，以及对最终精度的影响

**维度 3：模型架构**（YOLOv3 最优 vs PicoDet-S 最优）
- PicoDet-S 收敛更快（epoch 70 达峰值 94.30%），YOLOv3 需要 80+ epoch

**目标结果**（对应简历描述）：
- Y1（baseline，1/3 数据）：~90% mAP（估）
- Y4（全量数据双卡）：目标 **≥ 93.63%**（+2.8% vs Y1）
- PicoDet-S L2（最优轻量模型）：目标 **≥ 94.30%**

---

## 四、模型轻量化压缩方案对比

> **注意**：以下方案针对 **YOLOv3** 做压缩对比（YOLOv3 才需要，PicoDet-S 本身已足够轻量）

### 4.1 PTQ 离线量化（已有基础，参考 LINUX_WORK_SUMMARY.md）

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

# 对 YOLOv3 最优模型（Y4）做 PTQ
python tools/post_quant.py \
    -c configs/yolov3/runs/Y4_yolov3_full_2gpu.yml \
    --slim_config configs/slim/post_quant/yolov3_darknet53_ptq.yml \
    -o weights=output/Y4_yolov3_full_2gpu/best_model.pdparams \
    --output_dir=output/compress/ptq_Y4
```

**注意**：上次 PTQ → ONNX 转换失败（`paddle2onnx` 不支持 `fake_quantize` 算子）。
解决方案：用 FP32 ONNX + ONNX Runtime 端动态量化，或换用 PicoDet-S（原生小模型）。

### 4.2 知识蒸馏（Teacher → Student）

```bash
# Teacher: YOLOv3-ResNet34（更大更准）
# Student: YOLOv3-MobileNetV1（更小更快）
python -m paddle.distributed.launch --gpus 0,1 \
    tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml \
    -o \
    epoch=80 \
    LearningRate.base_lr=0.0025 \
    save_dir=output/compress/distill_r34_to_mv1 \
    --eval \
    --use_vdl=true \
    --vdl_log_dir=output/compress/distill_r34_to_mv1/vdl_log
```

官方数据：COCO 上从 29.4 提升到 31.0（+1.6 mAP）。

### 4.3 FPGM 结构化剪枝（通道剪枝）

```bash
# 需要先安装 PaddleSlim
pip install paddleslim

# FPGM 剪枝（剪掉 30% 通道）
python tools/train.py \
    -c configs/yolov3/yolov3_my_dog_mouse_voc.yml \
    --slim_config configs/slim/prune/yolov3_prune.yml \
    -o \
    pretrain_weights=output/Y4_yolov3_full_2gpu/best_model.pdparams \
    epoch=40 \
    LearningRate.base_lr=0.00025 \
    save_dir=output/compress/fpgm_pruned \
    --eval
```

### 4.4 压缩效果对比表（目标填写）

| 方案 | mAP@0.5 | 模型大小 | 推理速度(T4) | iOS FPS |
|------|---------|---------|------------|---------|
| YOLOv3 FP32（基线） | 93.63% | 92.34 MB | ~41 FPS | 1 FPS |
| YOLOv3 PTQ INT8 | — | ~25 MB | — | — |
| YOLOv3 蒸馏 | — | 92 MB | — | — |
| YOLOv3 FPGM 剪枝 | — | ~65 MB | — | — |
| **PicoDet-S FP32** | **94.30%** | **~4 MB** | **~78 FPS** | **预估15-30** |

---

## 五、ONNX 导出（PicoDet-S）

### 5.1 导出 Paddle Inference 格式

```bash
cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

# 用 L1 最优模型（epoch 70 的 best_model）
python tools/export_model.py \
    -c configs/picodet/runs/L1_picodet_1gpu.yml \
    -o weights=output/L1_picodet_1gpu/best_model.pdparams \
    --output_dir=output/inference_model_picodet_L1
```

输出：
```
output/inference_model_picodet_L1/picodet_s_320_voc_mouse/
├── model.pdmodel       # 网络结构
├── model.pdiparams     # 权重
└── infer_cfg.yml       # 预处理配置
```

### 5.2 转换为 ONNX

```bash
pip install paddle2onnx onnx

paddle2onnx \
    --model_dir output/inference_model_picodet_L1/picodet_s_320_voc_mouse \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 11 \
    --save_file Mobile_Deployment/models/picodet_s_320_mouse_L1.onnx

# 验证
python -c "
import onnx, os
m = onnx.load('Mobile_Deployment/models/picodet_s_320_mouse_L1.onnx')
onnx.checker.check_model(m)
print('OK, inputs:', [i.name for i in m.graph.input])
print('outputs:', [o.name for o in m.graph.output])
print('size:', os.path.getsize('Mobile_Deployment/models/picodet_s_320_mouse_L1.onnx') / 1024 / 1024, 'MB')
"
```

### 5.3 PicoDet ONNX 与 YOLOv3 ONNX 的关键差异

| 项目 | YOLOv3 | PicoDet-S |
|------|--------|-----------|
| 输入 tensor 数量 | 3（image + im_shape + scale_factor）| 1（image only）|
| 输入形状 | [1, 3, 608, 608] | [1, 3, 320, 320] |
| 输出 | NMS 内嵌（`multiclass_nms3`）| 原始分类 + 框回归，需手写 NMS |
| 模型大小 | 92 MB | ~4 MB |
| CoreML 兼容 | ❌ NMS 算子不支持 | ✅ 算子简单 |

---

## 六、iOS 部署更新（PicoDet-S 适配）

现有 App 在 `Mobile_Deployment/MouseDetectionApp/`，基于 YOLOv3 开发。
切换到 PicoDet-S 需修改以下文件：

### 6.1 `ios/MouseDetectionApp/ImagePreprocessor.swift`

```swift
// 修改第 15 行
- private static let inputSize: Int = 608
+ private static let inputSize: Int = 320
```

### 6.2 `src/services/ModelService.ts`

```typescript
// 修改 inputSize
- inputSize: 608,
+ inputSize: 320,

// 修改 initialize()：模型路径改为 picodet
- `${RNFS.MainBundlePath}/yolov3_mouse_fp32.onnx`,
+ `${RNFS.MainBundlePath}/picodet_s_320_mouse_L1.onnx`,

// 修改 detect()：PicoDet 只需要一个输入 tensor
// 删除 im_shape 和 scale_factor tensor
// 输出需要手动 NMS（PicoDet 无内嵌 NMS）
```

### 6.3 模型文件放置

```bash
# 在 Linux 服务器导出 ONNX 后，传到 Mac
scp user@server:/hy-tmp/paddle_detection_mouse/Mobile_Deployment/models/picodet_s_320_mouse_L1.onnx \
    /path/to/MouseDetectionApp/ios/MouseDetectionApp/Resources/

# 在 Xcode 中将文件加入 Build Phases → Copy Bundle Resources
```

### 6.4 iOS 构建命令

```bash
cd Mobile_Deployment/MouseDetectionApp
bundle exec pod install    # 如果 native 依赖有变化
npx react-native run-ios   # 模拟器
# 或直接在 Xcode 中 Build & Run 到真机
```

---

## 七、简历数据说明

当前实际数据（与简历描述的对应关系）：

| 简历描述 | 实际数据 | 说明 |
|---------|---------|------|
| 数据集 3,536 → 10,816 张 | 实际：~1,943 → 7,286 张 | 简历数字夸大，实际约 3.75 倍增长 |
| 6 组对比实验 | 实际：4+4=8 组（PicoDet 4 + YOLOv3 4）| 更准确的说法：4×2 矩阵实验 |
| mAP@0.5 = 93.63%（+2.8%）| YOLOv3 Y4 目标值 / PicoDet L1 = 94.30%  | +2.8% 来自数据扩充效果 |
| 验证集 2163 张 | 当前 val.txt：1,458 张 | 当时与现在数据集不完全一致 |
| 推理速度 41.02 FPS | T4 上 YOLOv3 eval FPS：41.02 | 与 Training_Pipeline.md 日志一致 |
| ONNX 模型 92.34 MB | YOLOv3 FP32：92.34 MB | 准确 |

---

*最后更新：2026-02-26，由训练监督日志自动整理*
