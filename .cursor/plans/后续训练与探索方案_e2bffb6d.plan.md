---
name: 后续训练与探索方案
overview: 在 8 组基础实验之上，利用 PaddleDetection 现有基础设施（slim 压缩、anchor 聚类、PP-YOLOE+、PicoDet-M/L）补齐实验矩阵的空白，形成更完整的精度-速度-尺寸帕累托分析，并完成简历承诺的压缩方案对比。
todos:
  - id: analysis-tools
    content: 运行 box_distribution.py 和 per-class eval，无需训练，直接出分析结果
    status: completed
  - id: picodet-m-config
    content: 新建 configs/picodet/runs/M1_picodet_m_2gpu.yml，适配 mouse_other_voc 数据集
    status: completed
  - id: ppyoloe-config
    content: 新建 configs/ppyoloe/runs/ppyoloe_s_mouse_voc.yml，适配 mouse_other_voc
    status: completed
  - id: anchor-cluster
    content: 运行 anchor_cluster.py 对数据集做 K-means，输出自定义 anchors
    status: completed
  - id: yolov3-y5-config
    content: 新建 Y5 YOLOv3 120e 配置，嵌入自定义 anchors
    status: completed
  - id: qat-config
    content: 新建 PicoDet-S QAT 配置 C1，基于 L1 best_model
    status: completed
  - id: fpgm-config
    content: 新建 YOLOv3 FPGM 剪枝配置 C2，基于 Y4 best_model
    status: completed
  - id: distill-config
    content: 新建 YOLOv3 知识蒸馏配置 C3，适配 mouse_other_voc
    status: completed
  - id: launcher-script
    content: 新建 scripts/run_experiments_round2.py，支持 A/B/C 各组链式启动
    status: completed
isProject: false
---

# 后续训练与探索方案

## 一、现有 8 组实验的不足

**精度层面：**

- PicoDet 只测了 S 号，无法建立 S/M/L 精度-参数曲线；不知道更大 PicoDet 在 5,828 张样本上是否还能继续受益
- YOLOv3 Y3/Y4 在 epoch 80 结束时 mAP 仍在上升（91.30% 最后一次 eval 仍未到平台），未收敛就停止
- 整个实验矩阵缺少 PP-YOLOE+ 这类更现代的 anchor-free 检测器对比点

**工程层面：**

- YOLOv3 使用了 COCO 默认 anchors（针对 80 类大数据集），从未针对实验鼠这个 2 类数据集做 K-means 聚类
- 简历提到的压缩方案（PTQ、蒸馏、FPGM）均没有完整的量化数据

**分析层面：**

- 从未做 per-class AP（mouse 和 other 各自精度）
- 从未分析数据集的 bbox 尺寸分布，不知道目标在图像中的规律

---

## 二、新实验方案（三组共 9 项）

### A 组：模型规模 Scaling（2 个训练实验）

补全精度-参数权衡曲线，回答"更大 PicoDet 是否值得"。

**A1：PicoDet-M 320**

- 新建 `configs/picodet/runs/M1_picodet_m_2gpu.yml`，继承 `../picodet_m_320_coco_lcnet.yml`，覆盖数据集指向 `mouse_other_voc`
- 关键参数（来自官方 M 配置）：`bs=48/卡（96 total）`，`lr=0.24`，`300e`，`out_channels=96`（比 S 略大）
- 预期：精度在 L1（94.30%）基础上再涨 0.5~1%；模型约 10MB

**A2：PP-YOLOE+-S（VOC 30 epoch 微调）**

- 基于现成 `[configs/ppyoloe/voc/ppyoloe_plus_crn_s_30e_voc.yml](PaddleDetection-release-2.6/configs/ppyoloe/voc/ppyoloe_plus_crn_s_30e_voc.yml)`，新建 `configs/ppyoloe/runs/ppyoloe_s_mouse_voc.yml`，4 处覆盖指向 `mouse_other_voc`
- 关键参数：`30e`，`base_lr=0.001`，COCO 80e 预训练，`bs=8/卡（16 total 双卡）`
- 预期：PP-YOLOE+ 比 YOLOv3 更先进，与 PicoDet 精度相当但模型约 18MB

### B 组：YOLOv3 未收敛补训 + Anchor 聚类（2 个工具 + 1 个训练实验）

**B1：Anchor 聚类分析**

- 运行 `tools/anchor_cluster.py`，对 `mouse_other_voc/train.txt` 做 K-means 聚类（n_clusters=9，对应 YOLOv3 三尺度 3×3 anchors）
- 输出：数据集自适应 anchors，写入新配置

**B2：YOLOv3-Y4 延长 + 自定义 Anchor（训练实验）**

- 新建 `configs/yolov3/runs/Y5_yolov3_full_2gpu_120e.yml`，继承 Y4 配置，`epoch=120`，`milestones=[72, 96]`，将 B1 聚类结果填入 `YOLOv3Head.anchors`
- 意义：修复两个已知问题——epoch 不足 + COCO anchor 不匹配；预期能突破 92%

**B3：box_distribution 可视化分析**

- 运行 `tools/box_distribution.py`，输出 mouse_other_voc 的 bbox 宽高分布、长宽比分布图
- 结果帮助解释 anchor 聚类前后的差距，也可以放进文档

### C 组：压缩流水线对比（3 个实验，填补简历空白）

**C1：PicoDet-S QAT 量化**

- 新建 `configs/picodet/runs/C1_picodet_s_qat.yml`，继承 `[configs/slim/quant/picodet_s_quant.yml](PaddleDetection-release-2.6/configs/slim/quant/picodet_s_quant.yml)` + L1 最优权重
- 参数：`epoch=50`，`base_lr=0.001`，PACT 激活量化，`bs=96（双卡 48/卡）`，从 L1 best_model.pdparams 微调
- 目标：4.4MB FP32 → ~1.2MB INT8，精度损失控制在 1% 以内；后续 paddle2onnx 导出（支持 QDQ 格式）

**C2：YOLOv3-MV1 FPGM 结构剪枝**

- 新建 `configs/yolov3/runs/C2_yolov3_fpgm_prune.yml`，继承 `[configs/slim/prune/yolov3_prune_fpgm.yml](PaddleDetection-release-2.6/configs/slim/prune/yolov3_prune_fpgm.yml)`，数据集指向 `mouse_other_voc`，`pretrain_weights` 换成 Y4 best_model
- 剪枝率：18 个卷积层 10~40% 非均匀剪枝（来自现有配置），fine-tune 40 epoch
- 目标：92MB → ~60MB，精度损失 <2%

**C3：YOLOv3 知识蒸馏（R34 Teacher → MV1 Student）**

- 新建 `configs/yolov3/runs/C3_yolov3_distill.yml`，继承 `[configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml](PaddleDetection-release-2.6/configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml)`，数据集指向 `mouse_other_voc`
- 参数：Teacher=YOLOv3-R34（COCO 预训练），Student=MV1，`DistillYOLOv3Loss.weight=1000`，`epoch=80`
- 目标：MV1 基线 91.5% → 期望提升至 93%+

---

## 三、分析工具（无需训练）

**V1：per-class AP 评估**

- 对所有已完成 best_model 运行 `tools/eval.py --classwise`
- 期望看到：mouse vs other 的 AP 差距；若 other 类 AP 低，说明背景样本多样性不足

**V2：推理速度统一测试**

- 对 L1（PicoDet-S）、A1（PicoDet-M）、A2（PP-YOLOE+-S）、Y4（YOLOv3）、C1（QAT）在相同条件下运行 eval 的 FPS，建立统一的速度-精度散点图

---

## 四、文件变更清单


| 文件                                                 | 操作               |
| -------------------------------------------------- | ---------------- |
| `configs/picodet/runs/M1_picodet_m_2gpu.yml`       | 新建               |
| `configs/ppyoloe/runs/ppyoloe_s_mouse_voc.yml`     | 新建               |
| `configs/yolov3/runs/Y5_yolov3_full_2gpu_120e.yml` | 新建（含自定义 anchor）  |
| `configs/picodet/runs/C1_picodet_s_qat.yml`        | 新建               |
| `configs/yolov3/runs/C2_yolov3_fpgm_prune.yml`     | 新建               |
| `configs/yolov3/runs/C3_yolov3_distill.yml`        | 新建               |
| `scripts/run_experiments_round2.py`                | 新建（A/B/C 组链式启动器） |


---

## 五、优先级建议

若时间有限，建议按此顺序：

1. **B3 + V1**（box 分析 + per-class eval）：无需训练，30 分钟内出结论，直接补充简历数据
2. **A1（PicoDet-M）**：改动最小，预期精度最高，对 iOS 部署直接有价值
3. **B1 + B2（Anchor 聚类 + Y5）**：解决 YOLOv3 两个已知问题，预期突破 92%
4. **C1（PicoDet-S QAT）**：量化 → ONNX，直接减小 iOS 模型体积
5. **A2（PP-YOLOE+-S）+ C2/C3（剪枝/蒸馏）**：补全实验矩阵，为简历提供对比数据

