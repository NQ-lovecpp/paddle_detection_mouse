#!/usr/bin/env python3
"""
YOLOv3-MobileNetV1 四轮训练启动器
===================================
四轮配置（2×2 矩阵：数据量 × 卡数）：
  Y1  1/3 数据  单卡  lr=0.00125  80 epoch  ← 简历 baseline
  Y2  1/3 数据  双卡  lr=0.0025   80 epoch
  Y3  全量数据  单卡  lr=0.00125  80 epoch
  Y4  全量数据  双卡  lr=0.0025   80 epoch

前置条件：
  先运行 python scripts/prepare_1of3_dataset.py 生成 1/3 数据集

特性：
  - 训练日志写入 output/{run_name}/train.log
  - VisualDL 日志写入 output/{run_name}/vdl_log/
  - start_new_session=True：父进程被 kill 后子进程独立存活
  - train.pid 记录 PID，方便手动监控 / 终止
  - DONE 标志文件：重启脚本自动跳过已完成轮次
  - 每轮结束后自动运行 eval 并写入 output/yolov3_summary.csv

用法：
  cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
  python scripts/prepare_1of3_dataset.py   # 仅首次需要
  python scripts/run_yolov3.py             # 顺序跑完所有轮次
  python scripts/run_yolov3.py --from Y3   # 从 Y3 开始跑

监控（启动后可关闭本终端）：
  tail -f output/Y1_yolov3_1of3_1gpu/train.log
  visualdl --logdir output/Y1_yolov3_1of3_1gpu/vdl_log --host 0.0.0.0 --port 8040
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────
# 路径与环境
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # PaddleDetection-release-2.6/
PYTHON = sys.executable
SUMMARY_CSV = BASE_DIR / "output" / "yolov3_summary.csv"

# ──────────────────────────────────────────────
# 轮次定义
# ──────────────────────────────────────────────
RUNS = [
    {
        "id":          "Y1",
        "name":        "Y1_yolov3_1of3_1gpu",
        "cfg":         "configs/yolov3/runs/Y1_yolov3_1of3_1gpu.yml",
        "gpus":        "0",
        "distributed": False,
        "dataset":     "1of3",
        "desc":        "1/3数据 单卡 lr=0.00125 80e（baseline）",
    },
    {
        "id":          "Y2",
        "name":        "Y2_yolov3_1of3_2gpu",
        "cfg":         "configs/yolov3/runs/Y2_yolov3_1of3_2gpu.yml",
        "gpus":        "0,1",
        "distributed": True,
        "dataset":     "1of3",
        "desc":        "1/3数据 双卡 lr=0.0025  80e",
    },
    {
        "id":          "Y3",
        "name":        "Y3_yolov3_full_1gpu",
        "cfg":         "configs/yolov3/runs/Y3_yolov3_full_1gpu.yml",
        "gpus":        "0",
        "distributed": False,
        "dataset":     "full",
        "desc":        "全量数据 单卡 lr=0.00125 80e",
    },
    {
        "id":          "Y4",
        "name":        "Y4_yolov3_full_2gpu",
        "cfg":         "configs/yolov3/runs/Y4_yolov3_full_2gpu.yml",
        "gpus":        "0,1",
        "distributed": True,
        "dataset":     "full",
        "desc":        "全量数据 双卡 lr=0.0025  80e",
    },
]


# ──────────────────────────────────────────────
# 前置检查
# ──────────────────────────────────────────────
def check_prerequisites():
    dataset_1of3 = BASE_DIR / "dataset" / "mouse_other_voc_1of3" / "train.txt"
    if not dataset_1of3.exists():
        print("[ERROR] 1/3 数据集不存在，请先运行：")
        print(f"  python scripts/prepare_1of3_dataset.py")
        sys.exit(1)
    n = len(dataset_1of3.read_text().splitlines())
    print(f"[✓] 1/3 数据集已就绪：{n} 行训练样本")


# ──────────────────────────────────────────────
# 命令构造
# ──────────────────────────────────────────────
def build_train_cmd(run: dict) -> list:
    name = run["name"]
    cfg  = run["cfg"]
    vdl  = f"output/{name}/vdl_log"

    if run["distributed"]:
        cmd = [
            PYTHON, "-m", "paddle.distributed.launch",
            "--gpus", run["gpus"],
            "tools/train.py",
            "-c", cfg,
            "--eval",
            "--use_vdl=true",
            f"--vdl_log_dir={vdl}",
        ]
    else:
        cmd = [
            PYTHON, "tools/train.py",
            "-c", cfg,
            "--eval",
            "--use_vdl=true",
            f"--vdl_log_dir={vdl}",
        ]
    return cmd


def build_eval_cmd(run: dict) -> list:
    name = run["name"]
    cfg  = run["cfg"]
    weights = f"output/{name}/best_model.pdparams"
    return [
        PYTHON, "tools/eval.py",
        "-c", cfg,
        "-o", f"weights={weights}",
        "--classwise",
    ]


# ──────────────────────────────────────────────
# 日志解析
# ──────────────────────────────────────────────
def extract_metrics(log_path: Path) -> tuple:
    """从 train.log 或分布式 workerlog.0 中提取 best_mAP（%）和 eval FPS。"""
    best_ap = None
    fps = None
    workerlog = BASE_DIR / "log" / "workerlog.0"
    candidates = [log_path, workerlog]
    for candidate in candidates:
        try:
            with open(candidate, "r", errors="replace") as f:
                for line in f:
                    m = re.search(r"Best test bbox ap is ([\d]+\.[\d]+)", line)
                    if m:
                        best_ap = float(m.group(1))
                    m = re.search(r"average FPS: ([\d.]+)", line)
                    if m:
                        fps = float(m.group(1))
        except FileNotFoundError:
            pass
    best_map = round(best_ap * 100, 2) if best_ap is not None else None
    return best_map, fps


# ──────────────────────────────────────────────
# 结果持久化
# ──────────────────────────────────────────────
def append_summary(run: dict, best_map, fps, duration_h: float, exit_code: int):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "name", "model", "dataset",
                "distributed", "desc",
                "best_mAP_50", "eval_FPS",
                "train_duration_h", "exit_code",
                "timestamp",
            ])
        writer.writerow([
            run["id"], run["name"], "yolov3_mobilenetv1", run["dataset"],
            run["distributed"], run["desc"],
            best_map, fps,
            round(duration_h, 2), exit_code,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])


# ──────────────────────────────────────────────
# 单轮训练
# ──────────────────────────────────────────────
def run_one(run: dict) -> bool:
    name      = run["name"]
    run_dir   = BASE_DIR / "output" / name
    done_flag = run_dir / "DONE"
    log_path  = run_dir / "train.log"
    pid_path  = run_dir / "train.pid"

    if done_flag.exists():
        print(f"\n[SKIP] {name}（已完成，删除 {done_flag} 可重新运行）")
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_train_cmd(run)
    cmd_str = " ".join(cmd)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = run["gpus"]
    env["NCCL_IB_DISABLE"] = "1"

    print(f"\n{'='*64}")
    print(f"[START] {name}  ({run['desc']})")
    print(f"  配置:  {run['cfg']}")
    print(f"  GPU:   {run['gpus']}")
    print(f"  数据:  {run['dataset']}")
    print(f"  日志:  tail -f {log_path}")
    print(f"  VDL:   visualdl --logdir {run_dir}/vdl_log --host 0.0.0.0 --port 8040")
    print(f"{'='*64}")

    t0 = time.time()

    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"# 启动时间:  {datetime.now().isoformat()}\n")
    log_file.write(f"# 轮次:      {name}\n")
    log_file.write(f"# 描述:      {run['desc']}\n")
    log_file.write(f"# 命令:      {cmd_str}\n")
    log_file.write(f"# {'='*58}\n\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        start_new_session=True,
        env=env,
    )

    pid_path.write_text(str(proc.pid))
    print(f"  PID:  {proc.pid}  (kill: kill {proc.pid})")

    ret = proc.wait()
    duration_h = (time.time() - t0) / 3600
    log_file.close()

    best_map, fps = extract_metrics(log_path)
    append_summary(run, best_map, fps, duration_h, ret)

    if ret == 0:
        done_flag.touch()
        print(f"\n[DONE] {name}")
        print(f"  mAP@0.5 = {best_map}%  |  FPS = {fps}  |  耗时 = {duration_h:.2f}h")
        return True
    else:
        print(f"\n[FAIL] {name}  exit_code={ret}  耗时={duration_h:.2f}h")
        print(f"  查看日志: tail -n 40 {log_path}")
        return False


# ──────────────────────────────────────────────
# 打印汇总表
# ──────────────────────────────────────────────
def print_summary():
    if not SUMMARY_CSV.exists():
        return
    print(f"\n{'='*64}")
    print("YOLOv3 训练结果汇总（数据量 × 卡数 对比矩阵）")
    print(f"{'='*64}")
    print(f"{'轮次':<6} {'数据':>6} {'卡':>4} {'mAP@0.5':>8} {'FPS':>7} {'耗时(h)':>8}")
    print("-" * 64)
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist_str = "双卡" if row.get("distributed", "").lower() in ("true", "1") else "单卡"
            print(
                f"{row['run_id']:<6} "
                f"{row['dataset']:>6} "
                f"{dist_str:>4} "
                f"{row['best_mAP_50']:>7}% "
                f"{row['eval_FPS']:>7} "
                f"{row['train_duration_h']:>8}"
            )
    print(f"\n完整 CSV: {SUMMARY_CSV}")
    print(f"VDL 多实验对比: visualdl --logdir output --host 0.0.0.0 --port 8040")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="YOLOv3 四轮训练启动器")
    parser.add_argument(
        "--from", dest="start_from", default=None,
        choices=[r["id"] for r in RUNS],
        help="从指定轮次开始（例：--from Y3）",
    )
    args = parser.parse_args()

    print(f"\nYOLOv3-MobileNetV1 四轮训练启动器")
    print(f"时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {BASE_DIR}")

    check_prerequisites()

    skip_ids = set()
    if args.start_from:
        for r in RUNS:
            if r["id"] == args.start_from:
                break
            skip_ids.add(r["id"])

    print(f"\n轮次计划:")
    for r in RUNS:
        if r["id"] in skip_ids:
            status = "跳过（--from）"
        elif (BASE_DIR / "output" / r["name"] / "DONE").exists():
            status = "已完成"
        else:
            status = "待运行"
        print(f"  {r['id']}: {r['desc']:38}  [{status}]")

    print()
    for run in RUNS:
        if run["id"] in skip_ids:
            print(f"[SKIP] {run['name']}（--from {args.start_from} 指定跳过）")
            continue
        success = run_one(run)
        if not success:
            print(f"\n训练失败，中止后续轮次。")
            break

    print_summary()


if __name__ == "__main__":
    main()
