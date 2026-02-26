#!/usr/bin/env python3
"""
一次性脚本：从全量数据集 mouse_other_voc 中取前 1/3 训练样本，
构建新数据集目录 mouse_other_voc_1of3。

目录结构（运行后）：
  dataset/mouse_other_voc_1of3/
  ├── images/       -> 软链接指向 ../mouse_other_voc/images/
  ├── annotations/  -> 软链接指向 ../mouse_other_voc/annotations/
  ├── train.txt     取全量 train.txt 前 1,943 行
  ├── val.txt       与全量相同（1,458 行），保证 mAP 可直接横向对比
  └── label_list.txt 与全量相同

用法：
  cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
  python scripts/prepare_1of3_dataset.py
"""

import os
import shutil
from pathlib import Path

# ──────────────────────────────────────────────
# 路径定义（相对于 PaddleDetection 根目录运行）
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent               # PaddleDetection-release-2.6/

SRC = BASE_DIR / "dataset" / "mouse_other_voc"
DST = BASE_DIR / "dataset" / "mouse_other_voc_1of3"

TRAIN_RATIO = 1 / 3                        # 取前 1/3 训练行


def main():
    print("=" * 60)
    print("mouse_other_voc_1of3 数据集构建脚本")
    print("=" * 60)

    # ── 1. 检查源数据集 ──────────────────────────────────────
    for p in [SRC / "train.txt", SRC / "val.txt", SRC / "label_list.txt"]:
        if not p.exists():
            raise FileNotFoundError(f"源文件不存在：{p}")

    # ── 2. 创建目标目录 ──────────────────────────────────────
    DST.mkdir(parents=True, exist_ok=True)
    print(f"\n[✓] 创建目录：{DST}")

    # ── 3. 软链接 images/ 和 annotations/ ────────────────────
    for subdir in ["images", "annotations"]:
        link = DST / subdir
        target = SRC / subdir
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target.resolve(), link)
        print(f"[✓] 软链接：{link} -> {target.resolve()}")

    # ── 4. 生成 train.txt（前 1/3 行）────────────────────────
    src_train = SRC / "train.txt"
    all_lines = src_train.read_text().splitlines()
    # 去掉空行
    all_lines = [l for l in all_lines if l.strip()]
    total = len(all_lines)
    n_1of3 = total // 3
    selected = all_lines[:n_1of3]

    dst_train = DST / "train.txt"
    dst_train.write_text("\n".join(selected) + "\n")
    print(f"\n[✓] train.txt：从 {total} 行中取前 {n_1of3} 行 → {dst_train}")

    # ── 5. 复制 val.txt ───────────────────────────────────────
    dst_val = DST / "val.txt"
    shutil.copy2(SRC / "val.txt", dst_val)
    val_lines = len(dst_val.read_text().splitlines())
    print(f"[✓] val.txt：复制 {val_lines} 行 → {dst_val}")

    # ── 6. 复制 label_list.txt ────────────────────────────────
    dst_label = DST / "label_list.txt"
    shutil.copy2(SRC / "label_list.txt", dst_label)
    print(f"[✓] label_list.txt → {dst_label}")

    # ── 7. 统计汇报 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("数据集构建完成")
    print(f"  全量 train:  {total} 张")
    print(f"  1/3  train:  {n_1of3} 张（{n_1of3/total*100:.1f}%）")
    print(f"  val  张数:   {val_lines} 张（两个数据集共用，mAP 可直接对比）")
    print(f"  目录:        {DST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
