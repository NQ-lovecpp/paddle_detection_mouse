#!/usr/bin/env python3
"""
第二轮实验启动器（A/B/C 组）
============================
实验组定义：
  A1  PicoDet-M 320  双卡  bs=96  lr=0.24  300e  （补全精度-参数曲线）
  A2  PP-YOLOE+-S   双卡  bs=16  lr=0.001  30e   （anchor-free 对比点）
  B2  YOLOv3-Y5     双卡  bs=16  lr=0.0025 120e  （补训 + 自定义 anchors）
  C1  PicoDet-S QAT 双卡  bs=96  lr=0.001  50e   （量化 → INT8）
  C2  YOLOv3 FPGM   单卡  bs=8   lr=0.000625 40e （结构剪枝）
  C3  YOLOv3 蒸馏   双卡  bs=16  lr=0.0025  80e  （知识蒸馏 R34→MV1）

特性：
  - 训练日志写入 output/{run_name}/train.log
  - VisualDL 日志写入 output/{run_name}/vdl_log/
  - DONE 标志文件：重启脚本自动跳过已完成轮次
  - 每轮结束后自动运行 eval --classwise，结果追加至 output/round2_summary.csv
  - start_new_session=True：父进程退出子进程继续运行

用法：
  cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6

  # 运行全部 A/B/C 组
  python scripts/run_experiments_round2.py

  # 只运行 A 组
  python scripts/run_experiments_round2.py --groups A

  # 只运行 B 和 C 组
  python scripts/run_experiments_round2.py --groups B,C

  # 从指定实验开始（跳过前面已完成的）
  python scripts/run_experiments_round2.py --from B2

  # 运行 B1 anchor 聚类（不训练，仅分析）
  python scripts/run_experiments_round2.py --anchor-cluster

  # 运行 B3 box 分布分析（不训练，仅分析）
  python scripts/run_experiments_round2.py --box-dist

  # per-class eval（对所有已完成的实验运行 eval --classwise）
  python scripts/run_experiments_round2.py --classwise-eval

监控：
  tail -f output/M1_picodet_m_2gpu/train.log
  visualdl --logdir output --host 0.0.0.0 --port 8040
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

BASE_DIR = Path(__file__).resolve().parent.parent   # PaddleDetection-release-2.6/
PYTHON = sys.executable
SUMMARY_CSV = BASE_DIR / "output" / "round2_summary.csv"

# ──────────────────────────────────────────────
# 实验定义
# ──────────────────────────────────────────────
RUNS = [
    {
        "id":          "A1",
        "group":       "A",
        "name":        "M1_picodet_m_2gpu",
        "cfg":         "configs/picodet/runs/M1_picodet_m_2gpu.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        False,
        "desc":        "PicoDet-M 双卡 bs=96 lr=0.24 300e",
        "weights_path": "output/M1_picodet_m_2gpu/M1_picodet_m_2gpu/best_model.pdparams",
    },
    {
        "id":          "A2",
        "group":       "A",
        "name":        "A2_ppyoloe_s_mouse_voc",
        "cfg":         "configs/ppyoloe/runs/ppyoloe_s_mouse_voc.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        False,
        "desc":        "PP-YOLOE+-S 双卡 bs=16 lr=0.001 30e",
        "weights_path": "output/A2_ppyoloe_s_mouse_voc/best_model.pdparams",
    },
    {
        "id":          "B2",
        "group":       "B",
        "name":        "Y5_yolov3_full_2gpu_120e",
        "cfg":         "configs/yolov3/runs/Y5_yolov3_full_2gpu_120e.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        False,
        "desc":        "YOLOv3-MV1 双卡 bs=16 lr=0.0025 120e+自定义anchors（Y5，已失败）",
        "weights_path": "output/Y5_yolov3_full_2gpu_120e/Y5_yolov3_full_2gpu_120e/best_model.pdparams",
    },
    {
        "id":          "B3",
        "group":       "B",
        "name":        "Y6_yolov3_custom_anchor_300e",
        "cfg":         "configs/yolov3/runs/Y6_yolov3_custom_anchor_300e.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        False,
        "desc":        "YOLOv3 backbone-only pretrain + 自定义anchor 300e（修复Y5）",
        "weights_path": "output/Y6_yolov3_custom_anchor_300e/Y6_yolov3_custom_anchor_300e/best_model.pdparams",
    },
    {
        "id":          "C1",
        "group":       "C",
        "name":        "C1_picodet_s_qat",
        "cfg":         "configs/picodet/runs/C1_picodet_s_qat.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        True,
        # 自定义 slim_cfg：覆盖 pretrain_weights 为 L1 best_model（非 COCO 权重）
        "slim_cfg":    "configs/slim/quant/C1_qat_mouse_overrides.yml",
        "desc":        "PicoDet-S QAT 量化 双卡 bs=96 lr=0.001 50e（从L1微调）",
        "weights_path": "output/C1_picodet_s_qat/C1_picodet_s_qat/best_model.pdparams",
    },
    {
        "id":          "C2",
        "group":       "C",
        "name":        "C2_yolov3_fpgm_prune",
        "cfg":         "configs/yolov3/runs/C2_yolov3_fpgm_prune.yml",
        "gpus":        "0",
        "distributed": False,
        "slim":        True,
        # 自定义 slim_cfg：覆盖 pretrain_weights 为 Y4 best_model
        "slim_cfg":    "configs/slim/prune/C2_fpgm_mouse_overrides.yml",
        "desc":        "YOLOv3 FPGM 剪枝 单卡 40e fine-tune（从Y4微调）",
        "weights_path": "output/C2_yolov3_fpgm_prune/best_model.pdparams",
    },
    {
        "id":          "C3",
        "group":       "C",
        "name":        "C3_yolov3_distill",
        "cfg":         "configs/yolov3/runs/C3_yolov3_distill.yml",
        "gpus":        "0,1",
        "distributed": True,
        "slim":        True,
        # 官方蒸馏 slim_cfg：R34 Teacher COCO 权重
        "slim_cfg":    "configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml",
        "desc":        "YOLOv3 蒸馏 R34→MV1 双卡 bs=16 80e",
        "weights_path": "output/C3_yolov3_distill/C3_yolov3_distill/best_model.pdparams",
    },
]

# per-class eval 目标：包含第一轮已完成实验 + 第二轮实验
EVAL_TARGETS = [
    {"id": "L1", "cfg": "configs/picodet/runs/L1_picodet_1gpu.yml",
     "weights": "output/L1_picodet_1gpu/L1_picodet_1gpu/best_model.pdparams"},
    {"id": "Y4", "cfg": "configs/yolov3/runs/Y4_yolov3_full_2gpu.yml",
     "weights": "output/Y4_yolov3_full_2gpu/Y4_yolov3_full_2gpu/best_model.pdparams"},
] + [{"id": r["id"], "cfg": r["cfg"], "weights": r["weights_path"]}
     for r in RUNS if r["id"] != "B2"]  # 跳过 Y5（已确认失败）


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

    # slim 实验通过 --slim_config 传入，而非在主 yml 中写 slim 键
    # （PaddleDetection trainer 通过 build_slim_model(cfg, slim_cfg) 初始化）
    if run.get("slim") and run.get("slim_cfg"):
        cmd += ["--slim_config", run["slim_cfg"]]

    return cmd


def build_eval_cmd(run: dict, classwise: bool = True) -> list:
    cfg     = run["cfg"]
    weights = run.get("weights_path", run.get("weights", ""))
    cmd = [
        PYTHON, "tools/eval.py",
        "-c", cfg,
        "-o", f"weights={weights}",
    ]
    if classwise:
        cmd.append("--classwise")
    return cmd


# ──────────────────────────────────────────────
# 日志解析
# ──────────────────────────────────────────────
def extract_metrics(log_path: Path) -> tuple:
    """从训练日志提取 best_mAP（%）和 eval FPS。"""
    best_ap = None
    fps = None
    workerlog = BASE_DIR / "log" / "workerlog.0"
    for candidate in [log_path, workerlog]:
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
                "run_id", "group", "name", "desc",
                "best_mAP_50", "eval_FPS",
                "train_duration_h", "exit_code", "timestamp",
            ])
        writer.writerow([
            run["id"], run.get("group", ""), run["name"], run["desc"],
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
    print(f"[START] {run['id']} - {name}  ({run['desc']})")
    print(f"  配置: {run['cfg']}")
    print(f"  GPU:  {run['gpus']}")
    print(f"  日志: tail -f {log_path}")
    print(f"  VDL:  visualdl --logdir {run_dir}/vdl_log --host 0.0.0.0 --port 8040")
    print(f"{'='*64}")

    t0 = time.time()
    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"# 启动时间:  {datetime.now().isoformat()}\n")
    log_file.write(f"# 轮次:      {run['id']} - {name}\n")
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
# B1 Anchor 聚类
# ──────────────────────────────────────────────
def run_anchor_cluster():
    """B1：在 mouse_other_voc 训练集上运行 K-means 聚类，输出自定义 anchors。"""
    print("\n" + "="*64)
    print("[B1] Anchor K-means 聚类 (n=9, size=608×608)")
    print("="*64)
    cmd = [
        PYTHON, "tools/anchor_cluster.py",
        "-c", "configs/yolov3/runs/Y5_yolov3_full_2gpu_120e.yml",
        "-n", "9",
        "--iters", "1000",
        "--size", "608,608",
        "--method", "v2",
    ]
    print(f"  命令: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(BASE_DIR))
    if ret.returncode == 0:
        print("\n[B1] 聚类完成！请将输出 anchors 按面积升序排列后，")
        print("  填入 configs/yolov3/runs/Y5_yolov3_full_2gpu_120e.yml 的 YOLOv3Head.anchors")
    else:
        print(f"\n[B1] 聚类失败，exit_code={ret.returncode}")
    return ret.returncode == 0


# ──────────────────────────────────────────────
# B3 Box 分布分析
# ──────────────────────────────────────────────
def run_box_distribution():
    """B3：分析 mouse_other_voc 的 bbox 宽高分布（VOC XML 格式）。"""
    print("\n" + "="*64)
    print("[B3] BBox 宽高分布分析")
    print("="*64)
    out_img = "output/box_distribution_voc.jpg"
    cmd = [
        PYTHON, "scripts/box_distribution_voc.py",
        "--anno_dir", "dataset/mouse_other_voc/annotations",
        "--train_txt", "dataset/mouse_other_voc/train.txt",
        "--out_img", out_img,
        "--eval_size", "320",
        "--small_stride", "8",
    ]
    print(f"  命令: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(BASE_DIR))
    if ret.returncode == 0:
        print(f"\n[B3] 完成！分布图：output/box_distribution_voc.jpg")
    else:
        print(f"\n[B3] 失败，exit_code={ret.returncode}")
    return ret.returncode == 0


# ──────────────────────────────────────────────
# V1 Per-class Eval
# ──────────────────────────────────────────────
def run_classwise_eval():
    """V1：对所有已完成实验运行 eval --classwise，输出 per-class AP。"""
    print("\n" + "="*64)
    print("[V1] Per-class AP 评估（--classwise）")
    print("="*64)
    results = []
    for target in EVAL_TARGETS:
        weights_path = BASE_DIR / target["weights"]
        if not weights_path.exists():
            print(f"\n  [{target['id']}] 权重不存在，跳过: {weights_path}")
            continue
        print(f"\n  [{target['id']}] 评估中... 权重: {target['weights']}")
        cmd = [
            PYTHON, "tools/eval.py",
            "-c", target["cfg"],
            "-o", f"weights={target['weights']}",
            "--classwise",
        ]
        ret = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=False)
        results.append((target["id"], "OK" if ret.returncode == 0 else f"FAIL({ret.returncode})"))

    print("\n[V1] 评估完成汇总：")
    for run_id, status in results:
        print(f"  {run_id}: {status}")
    return True


# ──────────────────────────────────────────────
# 打印汇总表
# ──────────────────────────────────────────────
def print_summary():
    if not SUMMARY_CSV.exists():
        return
    print(f"\n{'='*64}")
    print("第二轮实验结果汇总")
    print(f"{'='*64}")
    print(f"{'ID':<5} {'组':<4} {'描述':<35} {'mAP@0.5':>8} {'FPS':>7} {'耗时(h)':>8}")
    print("-" * 64)
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(
                f"{row['run_id']:<5} "
                f"{row['group']:<4} "
                f"{row['desc']:<35} "
                f"{str(row.get('best_mAP_50','?')):>7}% "
                f"{str(row.get('eval_FPS','?')):>7} "
                f"{str(row.get('train_duration_h','?')):>8}"
            )
    print(f"\n完整 CSV: {SUMMARY_CSV}")
    print("VDL 多实验对比: visualdl --logdir output --host 0.0.0.0 --port 8040")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="第二轮实验启动器（A/B/C 组）")
    parser.add_argument(
        "--groups", default=None,
        help="指定运行组，多个组用逗号分隔，例：--groups A 或 --groups B,C（默认全部）",
    )
    parser.add_argument(
        "--from", dest="start_from", default=None,
        choices=[r["id"] for r in RUNS],
        help="从指定实验开始（例：--from B2）",
    )
    parser.add_argument(
        "--anchor-cluster", action="store_true",
        help="运行 B1 anchor K-means 聚类（不训练）",
    )
    parser.add_argument(
        "--box-dist", action="store_true",
        help="运行 B3 box 分布分析（不训练）",
    )
    parser.add_argument(
        "--classwise-eval", action="store_true",
        help="对所有已完成实验运行 per-class eval（不训练）",
    )
    args = parser.parse_args()

    print(f"\n第二轮实验启动器（A/B/C 组）")
    print(f"时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {BASE_DIR}")

    # 纯工具模式
    if args.anchor_cluster:
        run_anchor_cluster()
        return
    if args.box_dist:
        run_box_distribution()
        return
    if args.classwise_eval:
        run_classwise_eval()
        return

    # 过滤实验组
    allowed_groups = None
    if args.groups:
        allowed_groups = {g.strip().upper() for g in args.groups.split(",")}

    # 过滤起始实验
    skip_ids = set()
    if args.start_from:
        for r in RUNS:
            if r["id"] == args.start_from:
                break
            skip_ids.add(r["id"])

    print("\n实验计划：")
    for r in RUNS:
        if allowed_groups and r["group"] not in allowed_groups:
            status = "跳过（--groups 过滤）"
        elif r["id"] in skip_ids:
            status = f"跳过（--from {args.start_from}）"
        elif (BASE_DIR / "output" / r["name"] / "DONE").exists():
            status = "已完成"
        else:
            status = "待运行"
        print(f"  {r['id']} [{r['group']}]: {r['desc']:<35}  [{status}]")

    print()
    for run in RUNS:
        if allowed_groups and run["group"] not in allowed_groups:
            continue
        if run["id"] in skip_ids:
            print(f"[SKIP] {run['name']}（--from 跳过）")
            continue
        success = run_one(run)
        if not success:
            print(f"\n训练失败，中止后续轮次。（如需继续，使用 --from {run['id']}）")
            break

    print_summary()


if __name__ == "__main__":
    main()
