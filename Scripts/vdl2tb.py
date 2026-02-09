#!/usr/bin/env python3
"""
VDL → TensorBoard 日志转换脚本
===============================
VisualDL 日志格式与 TensorBoard 不兼容，此脚本做桥接转换。

用法:
    python Scripts/vdl2tb.py output/A_smoke_test/vdl_log
    python Scripts/vdl2tb.py output/B_baseline_1gpu/vdl_log output/C_baseline_2gpu/vdl_log

转换后在同级目录生成 tb_log/，用 TensorBoard 查看:
    conda activate vdl
    tensorboard --logdir output/ --host 0.0.0.0 --port 8040
"""
import os
import sys
import glob

from visualdl import LogReader
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def convert_one(vdl_dir):
    """将一个 vdl_log 目录转为 tb_log"""
    tb_dir = os.path.join(os.path.dirname(vdl_dir), 'tb_log')
    os.makedirs(tb_dir, exist_ok=True)

    vdl_files = sorted(glob.glob(os.path.join(vdl_dir, 'vdlrecords.*')))
    if not vdl_files:
        print(f"  [SKIP] No vdlrecords found in {vdl_dir}")
        return

    total_points = 0
    writer = EventFileWriter(tb_dir)

    for vf in vdl_files:
        reader = LogReader(file_path=vf)
        tags = reader.get_tags()

        for tag in tags.get('scalar', []):
            data = reader.get_data('scalar', tag)
            for item in data:
                summary = Summary(value=[Summary.Value(tag=tag, simple_value=item.value)])
                event = Event(summary=summary, step=item.id, wall_time=item.timestamp)
                writer.add_event(event)
            total_points += len(data)

    writer.flush()
    writer.close()
    print(f"  [OK] {vdl_dir} → {tb_dir}  ({total_points} points)")


def main():
    if len(sys.argv) < 2:
        # 自动扫描 output/ 下所有 vdl_log
        base = 'output'
        if not os.path.isdir(base):
            print("Usage: python Scripts/vdl2tb.py <vdl_log_dir> [vdl_log_dir2] ...")
            sys.exit(1)
        dirs = sorted(glob.glob(os.path.join(base, '*/vdl_log')))
        if not dirs:
            print(f"No vdl_log directories found under {base}/")
            sys.exit(1)
        print(f"Auto-detected {len(dirs)} vdl_log directories:")
    else:
        dirs = sys.argv[1:]

    for d in dirs:
        print(f"Converting: {d}")
        convert_one(d)

    print("\nAll done! Start TensorBoard:")
    print("  conda activate vdl")
    print("  tensorboard --logdir output/ --host 0.0.0.0 --port 8040")


if __name__ == '__main__':
    main()
