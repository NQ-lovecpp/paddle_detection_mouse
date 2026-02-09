#!/usr/bin/env python3
"""
数据集集中整理脚本
=================
目标：将三个数据源合并为一个统一的 mouse + other 二分类数据集

数据源：
  1. dataset/dog_mouse_other_voc (现有)
     - dog_NNN  -> 改标签为 other，改名合并
     - mouse_NNN -> 保留
     - other_NNN -> 保留
  2. RawData/wb-img
     - mouseNNNN (无下划线) -> 保留为 mouse
  3. RawData/dog_mouse_other_voc
     - mouseNNNN (无下划线) -> 保留为 mouse
     - other_NNN -> 保留

操作：
  - 只保留 图片+标注 都存在的配对
  - 统一命名为 mouse_NNNNN / other_NNNNN
  - XML 内部的 <filename>, <folder>, <path>, <name> 全部更新
  - 生成 train.txt / val.txt / label_list.txt

输出目录：dataset/mouse_other_voc/
"""

import os
import sys
import shutil
import glob
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

# ============================================================
# 路径定义
# ============================================================
PROJECT_ROOT = "/hy-tmp/paddle_detection_mouse"
PADDLE_DIR = os.path.join(PROJECT_ROOT, "PaddleDetection-release-2.6")

# 数据源路径
SRC_DATASET = os.path.join(PADDLE_DIR, "dataset/dog_mouse_other_voc")
SRC_WB_IMG = os.path.join(PROJECT_ROOT, "RawData/wb-img")
SRC_RAW_VOC = os.path.join(PROJECT_ROOT, "RawData/dog_mouse_other_voc")

# 输出目录（先写到临时目录，最后重命名）
OUT_DIR = os.path.join(PADDLE_DIR, "dataset/mouse_other_voc")
OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_ANNOTATIONS = os.path.join(OUT_DIR, "annotations")

# ============================================================
# 工具函数
# ============================================================

def find_paired_files(img_dir, ann_dir, img_pattern, ann_ext=".xml"):
    """找到同时有图片和标注的配对文件，返回 [(img_path, ann_path, basename_no_ext), ...]"""
    pairs = []
    img_files = glob.glob(os.path.join(img_dir, img_pattern))
    for img_path in sorted(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, basename + ann_ext)
        if os.path.exists(ann_path):
            pairs.append((img_path, ann_path, basename))
    return pairs


def find_paired_files_same_dir(directory, img_pattern, ann_ext=".xml"):
    """图片和标注在同一目录（如 wb-img）"""
    pairs = []
    img_files = glob.glob(os.path.join(directory, img_pattern))
    for img_path in sorted(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(directory, basename + ann_ext)
        if os.path.exists(ann_path):
            pairs.append((img_path, ann_path, basename))
    return pairs


def update_xml(ann_path, new_filename, new_label=None):
    """
    读取 XML 标注，更新文件名和标签，返回修改后的 XML 字符串。
    - <filename> -> new_filename (如 mouse_00001.jpg)
    - <folder> -> images
    - <path> -> 清除硬编码路径
    - <name> -> new_label (如果指定)
    """
    tree = ET.parse(ann_path)
    root = tree.getroot()

    # 更新 <filename>
    fn_elem = root.find("filename")
    if fn_elem is not None:
        fn_elem.text = new_filename

    # 更新 <folder>
    folder_elem = root.find("folder")
    if folder_elem is not None:
        folder_elem.text = "images"

    # 清除 <path> 中的硬编码路径
    path_elem = root.find("path")
    if path_elem is not None:
        path_elem.text = new_filename

    # 更新所有 <object><name> 标签
    if new_label is not None:
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None:
                name_elem.text = new_label

    return tree


def collect_data():
    """
    收集所有三个数据源中的配对文件，返回：
    {
        "mouse": [(img_path, ann_path, original_basename), ...],
        "other": [(img_path, ann_path, original_basename), ...],
    }
    """
    data = defaultdict(list)

    # ----------------------------------------------------------
    # 源1: dataset/dog_mouse_other_voc (现有数据集)
    # ----------------------------------------------------------
    ds_img = os.path.join(SRC_DATASET, "images")
    ds_ann = os.path.join(SRC_DATASET, "annotations")

    # mouse_NNN -> mouse
    pairs = find_paired_files(ds_img, ds_ann, "mouse_*.jpg")
    print(f"[源1-dataset] mouse 配对: {len(pairs)}")
    for p in pairs:
        data["mouse"].append((*p, "dataset"))

    # other_NNN -> other
    pairs = find_paired_files(ds_img, ds_ann, "other_*.jpg")
    print(f"[源1-dataset] other 配对: {len(pairs)}")
    for p in pairs:
        data["other"].append((*p, "dataset"))

    # dog_NNN -> 改成 other
    pairs = find_paired_files(ds_img, ds_ann, "dog_*.jpg")
    print(f"[源1-dataset] dog->other 配对: {len(pairs)}")
    for p in pairs:
        data["other"].append((*p, "dataset-dog"))

    # ----------------------------------------------------------
    # 源2: RawData/wb-img (图片和标注在同一目录)
    # ----------------------------------------------------------
    pairs = find_paired_files_same_dir(SRC_WB_IMG, "mouse*.jpg")
    print(f"[源2-wb-img] mouse 配对: {len(pairs)}")
    for p in pairs:
        data["mouse"].append((*p, "wb-img"))

    # ----------------------------------------------------------
    # 源3: RawData/dog_mouse_other_voc
    # ----------------------------------------------------------
    raw_img = os.path.join(SRC_RAW_VOC, "images")
    raw_ann = os.path.join(SRC_RAW_VOC, "annotations")

    # mouseNNNN -> mouse
    pairs = find_paired_files(raw_img, raw_ann, "mouse*.jpg")
    print(f"[源3-RawData] mouse 配对: {len(pairs)}")
    for p in pairs:
        data["mouse"].append((*p, "rawdata"))

    # other_NNN -> other
    pairs = find_paired_files(raw_img, raw_ann, "other_*.jpg")
    print(f"[源3-RawData] other 配对: {len(pairs)}")
    for p in pairs:
        data["other"].append((*p, "rawdata"))

    return data


def main():
    print("=" * 60)
    print("数据集集中整理脚本")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: 收集所有数据
    # ----------------------------------------------------------
    print("\n[Step 1] 收集所有数据源...")
    data = collect_data()

    print(f"\n汇总:")
    print(f"  mouse 总计: {len(data['mouse'])} 对")
    print(f"  other 总计: {len(data['other'])} 对")
    print(f"  合计: {len(data['mouse']) + len(data['other'])} 对")

    # ----------------------------------------------------------
    # Step 2: 创建输出目录
    # ----------------------------------------------------------
    print(f"\n[Step 2] 创建输出目录: {OUT_DIR}")
    if os.path.exists(OUT_DIR):
        print(f"  警告: 输出目录已存在，将清空重建")
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_ANNOTATIONS, exist_ok=True)

    # ----------------------------------------------------------
    # Step 3: 复制并重命名文件
    # ----------------------------------------------------------
    print("\n[Step 3] 复制并重命名文件...")

    all_entries = []  # [(new_basename, label), ...]

    for label in ["mouse", "other"]:
        items = data[label]
        print(f"\n  处理 [{label}]: {len(items)} 个文件")

        for idx, (img_path, ann_path, orig_name, source) in enumerate(items, start=1):
            # 统一命名: mouse_00001 / other_00001
            new_basename = f"{label}_{idx:05d}"
            new_img_name = f"{new_basename}.jpg"
            new_ann_name = f"{new_basename}.xml"

            # 复制图片
            dst_img = os.path.join(OUT_IMAGES, new_img_name)
            shutil.copy2(img_path, dst_img)

            # 处理 XML: 更新文件名和标签
            # 对于 dog->other 的情况，需要把 XML 里的 <name>dog</name> 改成 <name>other</name>
            if source == "dataset-dog":
                tree = update_xml(ann_path, new_img_name, new_label="other")
            else:
                tree = update_xml(ann_path, new_img_name, new_label=label)

            dst_ann = os.path.join(OUT_ANNOTATIONS, new_ann_name)
            tree.write(dst_ann, encoding="utf-8", xml_declaration=True)

            all_entries.append((new_basename, label))

            if idx % 500 == 0:
                print(f"    已处理 {idx}/{len(items)} ...")

        print(f"    {label} 完成: {len(items)} 个文件")

    # ----------------------------------------------------------
    # Step 4: 生成 train.txt / val.txt
    # ----------------------------------------------------------
    print("\n[Step 4] 生成数据集划分文件...")

    random.seed(42)
    random.shuffle(all_entries)

    total = len(all_entries)
    train_ratio = 0.8
    val_ratio = 0.2
    train_count = int(total * train_ratio)

    train_entries = all_entries[:train_count]
    val_entries = all_entries[train_count:]

    # 统计各划分中的类别分布
    def count_labels(entries):
        counts = defaultdict(int)
        for _, label in entries:
            counts[label] += 1
        return dict(counts)

    print(f"  总计: {total}")
    print(f"  训练集: {len(train_entries)} {count_labels(train_entries)}")
    print(f"  验证集: {len(val_entries)} {count_labels(val_entries)}")

    # 写 train.txt (格式: ./images/xxx.jpg ./annotations/xxx.xml)
    train_path = os.path.join(OUT_DIR, "train.txt")
    with open(train_path, "w") as f:
        for basename, _ in sorted(train_entries):
            f.write(f"./images/{basename}.jpg ./annotations/{basename}.xml\n")
    print(f"  写入: {train_path}")

    # 写 val.txt
    val_path = os.path.join(OUT_DIR, "val.txt")
    with open(val_path, "w") as f:
        for basename, _ in sorted(val_entries):
            f.write(f"./images/{basename}.jpg ./annotations/{basename}.xml\n")
    print(f"  写入: {val_path}")

    # 写 label_list.txt
    label_path = os.path.join(OUT_DIR, "label_list.txt")
    with open(label_path, "w") as f:
        f.write("mouse\n")
        f.write("other\n")
    print(f"  写入: {label_path}")

    # ----------------------------------------------------------
    # Step 5: 汇总报告
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("整理完成！最终数据集统计：")
    print("=" * 60)

    final_imgs = len(glob.glob(os.path.join(OUT_IMAGES, "*.jpg")))
    final_anns = len(glob.glob(os.path.join(OUT_ANNOTATIONS, "*.xml")))
    final_mouse = len(glob.glob(os.path.join(OUT_IMAGES, "mouse_*.jpg")))
    final_other = len(glob.glob(os.path.join(OUT_IMAGES, "other_*.jpg")))

    print(f"  输出目录: {OUT_DIR}")
    print(f"  图片总数: {final_imgs}")
    print(f"  标注总数: {final_anns}")
    print(f"  mouse: {final_mouse}")
    print(f"  other: {final_other}")
    print(f"  train.txt: {len(train_entries)} 条")
    print(f"  val.txt: {len(val_entries)} 条")
    print(f"  label_list.txt: mouse, other")
    print(f"\n下一步: 更新 configs/datasets/ 中的配置文件路径")


if __name__ == "__main__":
    main()
