#!/usr/bin/env python3
"""
B3: 从 VOC XML 注解中分析 bbox 宽高分布
用法：
  cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
  python scripts/box_distribution_voc.py \
    --anno_dir dataset/mouse_other_voc/annotations \
    --train_txt dataset/mouse_other_voc/train.txt \
    --out_img output/box_distribution_voc.jpg
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='VOC bbox 宽高分布分析')
    parser.add_argument('--anno_dir', type=str,
                        default='dataset/mouse_other_voc/annotations',
                        help='VOC XML 注解目录')
    parser.add_argument('--train_txt', type=str,
                        default='dataset/mouse_other_voc/train.txt',
                        help='训练集 train.txt 路径（每行格式: ./images/xxx.jpg ./annotations/xxx.xml）')
    parser.add_argument('--out_img', type=str,
                        default='output/box_distribution_voc.jpg',
                        help='输出分布图路径')
    parser.add_argument('--eval_size', type=int, default=320,
                        help='推理时的输入分辨率（用于估算小目标建议 reg_range）')
    parser.add_argument('--small_stride', type=int, default=8,
                        help='最小步长（PicoDet=8，YOLOv3=8）')
    return parser.parse_args()


def load_xml_paths_from_txt(train_txt, anno_dir):
    """从 train.txt 解析出对应的 XML 文件路径列表"""
    base_dir = os.path.dirname(train_txt)
    xml_paths = []
    with open(train_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                xml_rel = parts[1]  # ./annotations/xxx.xml
                xml_abs = os.path.join(base_dir, xml_rel)
            else:
                # 尝试从图像路径推断
                img_rel = parts[0]
                stem = os.path.splitext(os.path.basename(img_rel))[0]
                xml_abs = os.path.join(anno_dir, stem + '.xml')
            xml_paths.append(xml_abs)
    return xml_paths


def analyze_distribution(xml_paths, eval_size, small_stride):
    ratio_w, ratio_h = [], []
    abs_w, abs_h = [], []
    per_class = {}
    missing = 0

    for xml_path in tqdm(xml_paths, desc='解析 XML'):
        if not os.path.exists(xml_path):
            missing += 1
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_node = root.find('size')
        if size_node is None:
            continue
        img_w = float(size_node.find('width').text)
        img_h = float(size_node.find('height').text)
        if img_w == 0 or img_h == 0:
            continue

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            bnd = obj.find('bndbox')
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)
            bw = xmax - xmin
            bh = ymax - ymin
            if bw <= 0 or bh <= 0:
                continue
            abs_w.append(bw)
            abs_h.append(bh)
            ratio_w.append(bw / img_w)
            ratio_h.append(bh / img_h)
            per_class.setdefault(cls_name, []).append((bw / img_w, bh / img_h))

    if missing > 0:
        print(f'[警告] 缺失 XML 文件数量：{missing}')

    return ratio_w, ratio_h, abs_w, abs_h, per_class


def draw_and_print(ratio_w, ratio_h, abs_w, abs_h, per_class,
                   eval_size, small_stride, out_img):
    ratio_w = np.array(ratio_w)
    ratio_h = np.array(ratio_h)
    abs_w = np.array(abs_w)
    abs_h = np.array(abs_h)

    print(f'\n===== BBox 统计 =====')
    print(f'总 bbox 数量      : {len(ratio_w)}')
    print(f'相对宽度  均值/中值: {ratio_w.mean():.4f} / {np.median(ratio_w):.4f}')
    print(f'相对高度  均值/中值: {ratio_h.mean():.4f} / {np.median(ratio_h):.4f}')
    print(f'绝对宽度（像素）均值: {abs_w.mean():.1f}  范围: [{abs_w.min():.0f}, {abs_w.max():.0f}]')
    print(f'绝对高度（像素）均值: {abs_h.mean():.1f}  范围: [{abs_h.min():.0f}, {abs_h.max():.0f}]')

    for cls_name, boxes in sorted(per_class.items()):
        ws = [b[0] for b in boxes]
        hs = [b[1] for b in boxes]
        print(f'\n  类别 [{cls_name}]  数量={len(boxes)}')
        print(f'    相对宽度 均值={np.mean(ws):.4f}  中值={np.median(ws):.4f}')
        print(f'    相对高度 均值={np.mean(hs):.4f}  中值={np.median(hs):.4f}')

    # 估算建议 reg_range
    all_ratios = np.concatenate([ratio_w, ratio_h])
    reg_ratios = np.where(all_ratios < 0.2, all_ratios,
                 np.where(all_ratios < 0.4, all_ratios / 2, all_ratios / 4))
    max_ratio = np.percentile(reg_ratios, 95)
    reg_max = round(max_ratio * eval_size / small_stride)
    print(f'\n建议 PicoDet reg_range[1] = {reg_max + 1}  (eval_size={eval_size}, small_stride={small_stride})')

    # 绘制分布图
    os.makedirs(os.path.dirname(out_img) if os.path.dirname(out_img) else '.', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Mouse Dataset BBox Distribution', fontsize=14)

    axes[0, 0].hist(ratio_w * 100, bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_xlabel('Relative Width (%)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Relative Width Distribution')

    axes[0, 1].hist(ratio_h * 100, bins=50, color='darkorange', edgecolor='white')
    axes[0, 1].set_xlabel('Relative Height (%)')
    axes[0, 1].set_title('Relative Height Distribution')

    axes[1, 0].hist(abs_w, bins=50, color='green', edgecolor='white')
    axes[1, 0].set_xlabel('Absolute Width (px)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Absolute Width Distribution')

    axes[1, 1].scatter(ratio_w * 100, ratio_h * 100, alpha=0.3, s=5, color='purple')
    axes[1, 1].set_xlabel('Relative Width (%)')
    axes[1, 1].set_ylabel('Relative Height (%)')
    axes[1, 1].set_title('Width vs Height Scatter')

    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    print(f'\n分布图已保存：{out_img}')


def main():
    args = parse_args()
    xml_paths = load_xml_paths_from_txt(args.train_txt, args.anno_dir)
    print(f'读取到 {len(xml_paths)} 个训练样本（来自 {args.train_txt}）')
    ratio_w, ratio_h, abs_w, abs_h, per_class = analyze_distribution(
        xml_paths, args.eval_size, args.small_stride)
    draw_and_print(ratio_w, ratio_h, abs_w, abs_h, per_class,
                   args.eval_size, args.small_stride, args.out_img)


if __name__ == '__main__':
    main()
