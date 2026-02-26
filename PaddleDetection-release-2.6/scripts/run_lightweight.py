#!/usr/bin/env python3
"""
PicoDet-S 四轮训练启动器
========================
四轮配置（全量数据，2×2 矩阵：卡数 × batch size）：
  L1  单卡  bs=32  lr=0.08  300 epoch
  L2  双卡  bs=64  lr=0.16  300 epoch  ← 官方推荐
  L3  双卡  bs=96  lr=0.24  300 epoch  ← 大 bs 探索
  L4  双卡  bs=64  lr=0.16  600 epoch  ← 加长训练

特性：
  - 训练日志写入 output/{run_name}/train.log（开头附带完整启动命令）
  - VisualDL 日志写入 output/{run_name}/vdl_log/
  - start_new_session=True：父进程被 kill 后子进程独立存活
  - train.pid 记录 PID，方便手动监控 / 终止
  - DONE 标志文件：重启脚本自动跳过已完成轮次
  - 每轮结束后自动运行 eval 并写入 output/lightweight_summary.csv

用法：
  cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
  python scripts/run_lightweight.py              # 顺序跑完所有轮次
  python scripts/run_lightweight.py --from L3    # 从 L3 开始跑

监控（启动后可关闭本终端）：
  tail -f output/L1_picodet_1gpu/train.log
  visualdl --logdir output/L1_picodet_1gpu/vdl_log --host 0.0.0.0 --port 8040
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
SUMMARY_CSV = BASE_DIR / "output" / "lightweight_summary.csv"

# ──────────────────────────────────────────────
# 轮次定义
# ──────────────────────────────────────────────
RUNS = [
    {
        "id":          "L1",
        "name":        "L1_picodet_1gpu",
        "cfg":         "configs/picodet/runs/L1_picodet_1gpu.yml",
        "gpus":        "0",
        "distributed": False,
        "desc":        "单卡 bs=32 lr=0.08 300e",
    },
    {
        "id":          "L2",
        "name":        "L2_picodet_2gpu",
        "cfg":         "configs/picodet/runs/L2_picodet_2gpu.yml",
        "gpus":        "0,1",
        "distributed": True,
        "desc":        "双卡 bs=64 lr=0.16 300e（官方推荐）",
    },
    {
        "id":          "L3",
        "name":        "L3_picodet_bs96",
        "cfg":         "configs/picodet/runs/L3_picodet_bs96.yml",
        "gpus":        "0,1",
        "distributed": True,
        "desc":        "双卡 bs=96 lr=0.24 300e（大 bs）",
    },
    {
        "id":          "L4",
        "name":        "L4_picodet_600e",
        "cfg":         "configs/picodet/runs/L4_picodet_600e.yml",
        "gpus":        "0,1",
        "distributed": True,
        "desc":        "双卡 bs=64 lr=0.16 600e（加长）",
    },
]


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
    # 分布式训练时实际日志写入 log/workerlog.0，非分布式写入 train.log
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
            run["id"], run["name"], "picodet_s_320", "full",
            run["distributed"], run["desc"],
            best_map, fps,
            round(duration_h, 2), exit_code,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])


# ──────────────────────────────────────────────
# 单轮训练
# ──────────────────────────────────────────────
def run_one(run: dict) -> bool:
    name     = run["name"]
    run_dir  = BASE_DIR / "output" / name
    done_flag = run_dir / "DONE"
    log_path  = run_dir / "train.log"
    pid_path  = run_dir / "train.pid"

    # 已完成则跳过
    if done_flag.exists():
        print(f"\n[SKIP] {name}（已完成，删除 {done_flag} 可重新运行）")
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_train_cmd(run)
    cmd_str = " ".join(cmd)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = run["gpus"]
    env["NCCL_IB_DISABLE"] = "1"          # TCP fallback 静默

    print(f"\n{'='*64}")
    print(f"[START] {name}  ({run['desc']})")
    print(f"  配置: {run['cfg']}")
    print(f"  GPU:  {run['gpus']}")
    print(f"  日志: tail -f {log_path}")
    print(f"  VDL:  visualdl --logdir {run_dir}/vdl_log --host 0.0.0.0 --port 8040")
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
        start_new_session=True,    # 子进程加入独立 session，父进程死亡不影响它
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
    print("训练结果汇总")
    print(f"{'='*64}")
    print(f"{'轮次':<6} {'描述':<28} {'mAP@0.5':>8} {'FPS':>7} {'耗时(h)':>8}")
    print("-" * 64)
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model", "") != "picodet_s_320":
                continue
            print(
                f"{row['run_id']:<6} "
                f"{row['desc']:<28} "
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
    parser = argparse.ArgumentParser(description="PicoDet-S 四轮训练启动器")
    parser.add_argument(
        "--from", dest="start_from", default=None,
        choices=[r["id"] for r in RUNS],
        help="从指定轮次开始（例：--from L3）",
    )
    args = parser.parse_args()

    print(f"\nPicoDet-S 四轮训练启动器")
    print(f"时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {BASE_DIR}")
    print(f"\n轮次计划:")
    skip = args.start_from is not None
    for r in RUNS:
        if r["id"] == args.start_from:
            skip = False
        status = "跳过（--from）" if skip else ("已完成" if (BASE_DIR / "output" / r["name"] / "DONE").exists() else "待运行")
        print(f"  {r['id']}: {r['desc']:28}  [{status}]")
        if r["id"] == args.start_from:
            skip = False  # 第一次处理完再置 False 无效，上面已经处理

    # 重新决定跳过逻辑
    skip_ids = set()
    if args.start_from:
        for r in RUNS:
            if r["id"] == args.start_from:
                break
            skip_ids.add(r["id"])

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
