---
name: 复现训练脚本方案
overview: 针对当前 7,286 张实际数据集，先创建 1/3 子数据集，再用独立 YAML 配置文件定义各轮超参，最后编写两个支持后台持久化运行和日志重定向的 Python 启动脚本。
todos:
  - id: prep-dataset
    content: 编写 scripts/prepare_1of3_dataset.py，创建 mouse_other_voc_1of3 目录（含 train.txt/val.txt/label_list.txt + 图像软链接）
    status: completed
  - id: write-dataset-yml
    content: 新建 configs/datasets/mouse_other_voc_1of3.yml（数据集配置文件）
    status: completed
  - id: write-picodet-ymls
    content: 新建 configs/picodet/runs/ 下 4 个运行配置 YAML（L1~L4）
    status: completed
  - id: write-yolov3-ymls
    content: 新建 configs/yolov3/runs/ 下 4 个运行配置 YAML（Y1~Y4）
    status: completed
  - id: write-run-lightweight
    content: 编写 scripts/run_lightweight.py（读取 4 个 YAML 顺序执行 + 后台持久化 + eval 汇总）
    status: completed
  - id: write-run-yolov3
    content: 编写 scripts/run_yolov3.py（读取 4 个 YAML 顺序执行 + 后台持久化 + eval 汇总）
    status: completed
isProject: false
---

# 复现训练脚本方案

## 关键发现与数字修正

**数据集实际情况**（mouse_other_voc）：

- train.txt: 5,828 行，val.txt: 1,458 行，合计 **7,286 张**
- 1/3 训练集: 5,828 ÷ 3 = **1,943 行**（对应简历里"原始数据集"的 baseline）
- 简历数字建议修改为："将数据集从约 **1,943 张**扩充至 **7,286 张**，增长约 **3.75 倍**"
- 简历里的验证集 2,163 张是旧数据，当前 val.txt 为 1,458 张

---

## 整体文件结构

```
PaddleDetection-release-2.6/
├── dataset/
│   ├── mouse_other_voc/              原有（全量）
│   └── mouse_other_voc_1of3/         新建（1/3 数据集）
│       ├── images/                   软链接 → ../mouse_other_voc/images/
│       ├── annotations/              软链接 → ../mouse_other_voc/annotations/
│       ├── train.txt                 取全量 train.txt 前 1,943 行
│       ├── val.txt                   与全量相同（1,458 行，mAP 可直接对比）
│       └── label_list.txt            与全量相同
├── configs/
│   ├── datasets/
│   │   ├── mouse_other_voc.yml       原有
│   │   └── mouse_other_voc_1of3.yml  新建
│   ├── picodet/
│   │   ├── picodet_s_320_voc_mouse.yml  原有（公共基础配置）
│   │   └── runs/
│   │       ├── L1_picodet_1gpu.yml
│   │       ├── L2_picodet_2gpu.yml
│   │       ├── L3_picodet_bs96.yml
│   │       └── L4_picodet_600e.yml
│   └── yolov3/
│       ├── yolov3_my_dog_mouse_voc.yml  原有（公共基础配置）
│       └── runs/
│           ├── Y1_yolov3_1of3_1gpu.yml
│           ├── Y2_yolov3_1of3_2gpu.yml
│           ├── Y3_yolov3_full_1gpu.yml
│           └── Y4_yolov3_full_2gpu.yml
└── scripts/
    ├── prepare_1of3_dataset.py    一次性执行，生成 1/3 数据集目录
    ├── run_lightweight.py         PicoDet-S 4 轮启动器
    └── run_yolov3.py              YOLOv3 4 轮启动器
```

---

## Step 0: 数据集准备脚本

`[scripts/prepare_1of3_dataset.py](PaddleDetection-release-2.6/scripts/prepare_1of3_dataset.py)` 做以下事情（一次性运行）：

1. 创建 `dataset/mouse_other_voc_1of3/` 目录
2. 用 `os.symlink` 创建 `images/` 和 `annotations/` 软链接到全量目录
3. 取 `mouse_other_voc/train.txt` 前 1,943 行写入 `mouse_other_voc_1of3/train.txt`
4. 复制 `val.txt` 和 `label_list.txt`
5. 打印统计：train 行数 / val 行数确认

---

## Step 1: YAML 配置文件设计

用 `_BASE_` 继承代替命令行 `-o` 覆盖，每个运行有独立 yml，超参一目了然。

`**configs/datasets/mouse_other_voc_1of3.yml**`（与全量同结构，只改 `dataset_dir`）：

```yaml
metric: VOC
map_type: integral
num_classes: 2

TrainDataset:
  name: VOCDataSet
  dataset_dir: dataset/mouse_other_voc_1of3
  anno_path: train.txt
  label_list: label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

EvalDataset:
  name: VOCDataSet
  dataset_dir: dataset/mouse_other_voc_1of3
  anno_path: val.txt
  label_list: label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

TestDataset:
  name: ImageFolder
  anno_path: dataset/mouse_other_voc_1of3/label_list.txt
```

`**configs/picodet/runs/L1_picodet_1gpu.yml**`（示例，其余 3 个同结构只改超参）：

```yaml
_BASE_: ['../picodet_s_320_voc_mouse.yml']

epoch: 300
save_dir: output/L1_picodet_1gpu

TrainReader:
  batch_size: 32

LearningRate:
  base_lr: 0.08
```

`**configs/yolov3/runs/Y1_yolov3_1of3_1gpu.yml**`（1/3 数据集，直接覆盖 TrainDataset/EvalDataset）：

```yaml
_BASE_: ['../yolov3_my_dog_mouse_voc.yml']

epoch: 80
save_dir: output/Y1_yolov3_1of3_1gpu

LearningRate:
  base_lr: 0.00125

TrainDataset:
  name: VOCDataSet
  dataset_dir: dataset/mouse_other_voc_1of3
  anno_path: train.txt
  label_list: label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']

EvalDataset:
  name: VOCDataSet
  dataset_dir: dataset/mouse_other_voc_1of3
  anno_path: val.txt
  label_list: label_list.txt
  data_fields: ['image', 'gt_bbox', 'gt_class', 'difficult']
```

---

## Step 2: Python 启动脚本持久化架构

目标：启动后可关闭终端，训练继续；重启脚本自动跳过已完成轮次。

```
run_lightweight.py / run_yolov3.py
  ├── RUNS = [("L1", "configs/.../L1.yml", gpus="0"), ...]
  ├── for run in RUNS:
  │   ├── 检查 output/{name}/DONE 标志 → 已存在则 skip
  │   ├── mkdir output/{name}/
  │   ├── 打开 output/{name}/train.log（写入模式）
  │   ├── 构造命令：
  │   │   单卡: [python, tools/train.py, -c, yml, --eval, --use_vdl=true, --vdl_log_dir=...]
  │   │   双卡: [python, -m, paddle.distributed.launch, --gpus, 0,1, tools/train.py, ...]
  │   ├── subprocess.Popen(cmd,
  │   │     stdout=log_file, stderr=STDOUT,
  │   │     start_new_session=True,    ← 父进程死亡不影响子进程
  │   │     env={...NCCL_IB_DISABLE=1, CUDA_VISIBLE_DEVICES=...})
  │   ├── 写 output/{name}/train.pid
  │   ├── proc.wait()   ← 阻塞等待；父被 kill 后子进程独立存活
  │   ├── exit code == 0 → 写 DONE 标志
  │   └── 调用 tools/eval.py，追加 mAP 到 output/summary.csv
  └── 最终打印全部结果对比表
```

监控方式（脚本启动后随时可用）：

```bash
tail -f output/L1_picodet_1gpu/train.log
cat  output/L1_picodet_1gpu/train.pid
kill $(cat output/L1_picodet_1gpu/train.pid)
```

---

## 各轮次参数汇总

**PicoDet-S（全量数据）**：

- L1: 单卡，bs=32，lr=0.08，300 epoch
- L2: 双卡，bs/卡=32（total 64），lr=0.16，300 epoch（官方推荐）
- L3: 双卡，bs/卡=48（total 96），lr=0.24，300 epoch（大 bs 探索）
- L4: 双卡，bs/卡=32（total 64），lr=0.16，600 epoch（加长训练）

**YOLOv3 MobileNetV1（2×2 矩阵）**：

- Y1: 1/3 数据，单卡，lr=0.00125，80 epoch（简历 baseline）
- Y2: 1/3 数据，双卡，lr=0.0025，80 epoch
- Y3: 全量数据，单卡，lr=0.00125，80 epoch
- Y4: 全量数据，双卡，lr=0.0025，80 epoch

---

## Training_Pipeline.md 保留策略

原 A-F 6 组方案不修改，保留方案 A 中那段真实训练日志（mAP=93.63%、FPS=41.02）。新脚本和配置文件独立存在，文档无需改动。