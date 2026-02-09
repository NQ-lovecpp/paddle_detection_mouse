#!/usr/bin/env python3
"""
VOC 标注查看器 + 模型推理 Web 服务
====================================
零依赖（标准库），浏览器中查看 VOC 标注 + 可视化模型推理结果。

用法:
    cd /hy-tmp/paddle_detection_mouse/PaddleDetection-release-2.6
    python ../Scripts/voc_viewer/server.py [--port 8765]

功能:
    1. 浏览 VOC 格式数据集 (images/ + annotations/)
    2. 选择已训练模型，对数据集图片 / 上传图片执行推理
    3. 所有 tools/infer.py 参数均可通过 Web 界面配置
"""

import os
import sys
import json
import glob
import argparse
import mimetypes
import subprocess
import tempfile
import shutil
import time
import socket
import xml.etree.ElementTree as ET
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, unquote
from pathlib import Path
import threading
import io

# ──────────── 目录配置 ────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认 PaddleDetection 根目录（server.py 在 Scripts/voc_viewer/ 下）
PADDLE_DET_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../PaddleDetection-release-2.6"))
DATASET_ROOT = os.path.join(PADDLE_DET_ROOT, "dataset")
OUTPUT_ROOT = os.path.join(PADDLE_DET_ROOT, "output")
HTML_FILE = os.path.join(SCRIPT_DIR, "index.html")

# 推理结果临时目录
INFER_OUTPUT_DIR = os.path.join(PADDLE_DET_ROOT, "output", "_web_infer_vis")
os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

# 上传图片临时目录
UPLOAD_DIR = os.path.join(PADDLE_DET_ROOT, "output", "_web_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {"filename": "", "size": {"width": 0, "height": 0, "depth": 3}, "objects": []}
    fn = root.find("filename")
    if fn is not None:
        result["filename"] = fn.text
    size = root.find("size")
    if size is not None:
        for k in ("width", "height", "depth"):
            el = size.find(k)
            if el is not None:
                result["size"][k] = int(el.text or 0)
    for obj in root.findall("object"):
        o = {"name": "", "difficult": 0, "bbox": {}}
        name = obj.find("name")
        if name is not None:
            o["name"] = name.text
        diff = obj.find("difficult")
        if diff is not None:
            o["difficult"] = int(diff.text or 0)
        bbox = obj.find("bndbox")
        if bbox is not None:
            o["bbox"] = {k: int(float(bbox.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")}
        result["objects"].append(o)
    return result


def scan_dataset(dataset_path):
    """扫描数据集目录"""
    img_dir = os.path.join(dataset_path, "images")
    ann_dir = os.path.join(dataset_path, "annotations")
    if not os.path.isdir(img_dir):
        return {"error": f"images/ 目录不存在: {img_dir}", "items": []}
    items = []
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_extensions:
            continue
        base = os.path.splitext(fname)[0]
        xml_name = base + ".xml"
        has_annotation = os.path.exists(os.path.join(ann_dir, xml_name))
        items.append({"image": fname, "annotation": xml_name if has_annotation else None, "basename": base})
    labels = {}
    annotated = 0
    for item in items:
        if item["annotation"]:
            annotated += 1
            try:
                data = parse_voc_xml(os.path.join(ann_dir, item["annotation"]))
                for obj in data["objects"]:
                    labels[obj["name"]] = labels.get(obj["name"], 0) + 1
            except:
                pass
    return {"path": dataset_path, "total_images": len(items), "annotated": annotated, "labels": labels, "items": items}


def list_datasets():
    """列出 dataset/ 下的数据集"""
    datasets = []
    if not os.path.isdir(DATASET_ROOT):
        return {"error": f"目录不存在: {DATASET_ROOT}", "datasets": []}
    for name in sorted(os.listdir(DATASET_ROOT)):
        full = os.path.join(DATASET_ROOT, name)
        if os.path.isdir(full) and os.path.isdir(os.path.join(full, "images")):
            img_count = len([f for f in os.listdir(os.path.join(full, "images"))
                             if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
            datasets.append({"name": name, "image_count": img_count})
    return {"root": DATASET_ROOT, "datasets": datasets}


def list_models():
    """扫描 output/ 下的模型权重"""
    models = []
    if not os.path.isdir(OUTPUT_ROOT):
        return {"models": []}
    for root, dirs, files in os.walk(OUTPUT_ROOT):
        for f in files:
            if f.endswith(".pdparams"):
                rel = os.path.relpath(os.path.join(root, f), PADDLE_DET_ROOT)
                # 只保留 best_model 和 model_final，以及独立 epoch 权重
                models.append({"path": rel, "name": rel})
    models.sort(key=lambda x: x["path"])
    return {"models": models}


def list_configs():
    """扫描 configs/ 下的 yml 配置"""
    configs = []
    cfg_root = os.path.join(PADDLE_DET_ROOT, "configs")
    for root, dirs, files in os.walk(cfg_root):
        for f in files:
            if f.endswith(".yml") and "_base_" not in root:
                rel = os.path.relpath(os.path.join(root, f), PADDLE_DET_ROOT)
                configs.append({"path": rel, "name": rel})
    configs.sort(key=lambda x: x["path"])
    return {"configs": configs}


def list_slim_configs():
    """扫描 configs/slim/ 下的 slim 配置"""
    slims = [{"path": "", "name": "（无）"}]
    slim_root = os.path.join(PADDLE_DET_ROOT, "configs", "slim")
    if os.path.isdir(slim_root):
        for root, dirs, files in os.walk(slim_root):
            for f in files:
                if f.endswith(".yml"):
                    rel = os.path.relpath(os.path.join(root, f), PADDLE_DET_ROOT)
                    slims.append({"path": rel, "name": rel})
    return {"slim_configs": slims}


# ──────────── 推理任务管理 ────────────
infer_lock = threading.Lock()
infer_status = {"running": False, "log": "", "result_image": None, "exit_code": None}


def run_inference(params):
    """在子进程中执行 tools/infer.py"""
    global infer_status
    with infer_lock:
        infer_status = {"running": True, "log": "Starting inference...\n", "result_image": None, "exit_code": None}

    # 构建命令
    cmd = [sys.executable, "tools/infer.py"]
    cmd += ["-c", params["config"]]

    # -o 参数: 每个 key=value 必须作为独立的 list 元素传给 -o
    opt_items = []
    if params.get("weights"):
        opt_items.append(f"weights={params['weights']}")
    if params.get("extra_opts"):
        # 用户输入如 "use_gpu=True num_classes=2"，按空格拆成多个独立项
        for part in params["extra_opts"].strip().split():
            if part:
                opt_items.append(part)
    if opt_items:
        cmd += ["-o"] + opt_items

    # 推理目标
    if params.get("infer_img"):
        cmd += ["--infer_img", params["infer_img"]]
    elif params.get("infer_dir"):
        cmd += ["--infer_dir", params["infer_dir"]]

    # 输出目录（空字符串时用默认值）
    output_dir = params.get("output_dir") or INFER_OUTPUT_DIR
    cmd += ["--output_dir", output_dir]

    # 其他参数
    if params.get("draw_threshold"):
        cmd += ["--draw_threshold", str(params["draw_threshold"])]
    if params.get("slim_config"):
        cmd += ["--slim_config", params["slim_config"]]
    if params.get("save_results"):
        cmd += ["--save_results", "True"]
    if params.get("slice_infer"):
        cmd += ["--slice_infer"]
        if params.get("slice_size"):
            cmd += ["--slice_size"] + params["slice_size"].split()
        if params.get("overlap_ratio"):
            cmd += ["--overlap_ratio"] + params["overlap_ratio"].split()
        if params.get("combine_method"):
            cmd += ["--combine_method", params["combine_method"]]
        if params.get("match_threshold"):
            cmd += ["--match_threshold", str(params["match_threshold"])]
        if params.get("match_metric"):
            cmd += ["--match_metric", params["match_metric"]]
    if params.get("visualize") == "false":
        cmd += ["--visualize", "False"]

    with infer_lock:
        infer_status["log"] += f"CMD: {' '.join(cmd)}\n\n"

    try:
        proc = subprocess.Popen(
            cmd, cwd=PADDLE_DET_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            with infer_lock:
                infer_status["log"] += line
        proc.wait()
        with infer_lock:
            infer_status["exit_code"] = proc.returncode
            infer_status["log"] += f"\n[Exit code: {proc.returncode}]\n"

            # 找到输出图片
            if proc.returncode == 0:
                result_imgs = sorted(glob.glob(os.path.join(output_dir, "*.jpg")) +
                                     glob.glob(os.path.join(output_dir, "*.png")),
                                     key=os.path.getmtime, reverse=True)
                if result_imgs:
                    infer_status["result_image"] = result_imgs[0]
    except Exception as e:
        with infer_lock:
            infer_status["log"] += f"\n[ERROR] {e}\n"
            infer_status["exit_code"] = -1
    finally:
        with infer_lock:
            infer_status["running"] = False


class ViewerHandler(SimpleHTTPRequestHandler):
    """自定义 HTTP 请求处理器"""

    def log_message(self, format, *args):
        pass  # 静默日志

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # ── 标注查看 API ──
        if path == "/api/datasets":
            return self._json(list_datasets())
        if path == "/api/scan":
            ds = params.get("path", [""])[0]
            if not ds:
                return self._json({"error": "缺少 path 参数"})
            full_path = os.path.join(DATASET_ROOT, ds)
            if not os.path.isdir(full_path):
                return self._json({"error": f"目录不存在: {ds}"})
            return self._json(scan_dataset(full_path))
        if path == "/api/annotation":
            ds = params.get("dataset", [""])[0]
            xml_name = params.get("file", [""])[0]
            xml_path = os.path.join(DATASET_ROOT, ds, "annotations", xml_name)
            if os.path.exists(xml_path):
                return self._json(parse_voc_xml(xml_path))
            return self._json({"error": f"文件不存在: {xml_path}"})

        # ── 推理 API ──
        if path == "/api/models":
            return self._json(list_models())
        if path == "/api/configs":
            return self._json(list_configs())
        if path == "/api/slim_configs":
            return self._json(list_slim_configs())
        if path == "/api/infer/status":
            with infer_lock:
                return self._json(dict(infer_status))
        if path == "/api/infer/result_image":
            with infer_lock:
                img_path = infer_status.get("result_image")
            if img_path and os.path.exists(img_path):
                return self._serve_file(img_path, no_store=True)
            self.send_error(404, "No result image")
            return

        # ── 图片服务 ──
        if path.startswith("/images/"):
            ds = params.get("dataset", [""])[0]
            img_name = unquote(path[len("/images/"):])
            img_path = os.path.join(DATASET_ROOT, ds, "images", img_name)
            if os.path.exists(img_path):
                return self._serve_file(img_path)
            self.send_error(404)
            return

        # ── 推理结果图片 ──
        if path.startswith("/infer_vis/"):
            fname = unquote(path[len("/infer_vis/"):])
            fpath = os.path.join(INFER_OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                return self._serve_file(fpath)
            self.send_error(404)
            return

        # 首页
        if path == "/" or path == "/index.html":
            return self._serve_file(HTML_FILE)

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        if path == "/api/infer/run":
            try:
                params = json.loads(body.decode('utf-8'))
            except:
                return self._json({"error": "Invalid JSON"})

            with infer_lock:
                if infer_status["running"]:
                    return self._json({"error": "推理正在进行中，请等待完成"})

            # 清理旧结果
            for f in glob.glob(os.path.join(INFER_OUTPUT_DIR, "*")):
                try:
                    os.remove(f)
                except:
                    pass

            t = threading.Thread(target=run_inference, args=(params,), daemon=True)
            t.start()
            return self._json({"status": "started"})

        if path == "/api/upload":
            # 简单的图片上传
            # Content-Type: application/octet-stream, filename in query
            fname = parse_qs(urlparse(self.path).query).get("filename", ["upload.jpg"])[0]
            save_path = os.path.join(UPLOAD_DIR, fname)
            with open(save_path, "wb") as f:
                f.write(body)
            return self._json({"path": save_path, "filename": fname})

        self.send_error(404)

    def _json(self, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath, no_store=False):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.send_header("Cache-Control", "no-store" if no_store else "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_error(404)


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


def main():
    parser = argparse.ArgumentParser(description="VOC 标注查看器 + 模型推理")
    parser.add_argument("--port", type=int, default=18765, help="端口号 (默认 18765)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--paddle-root", type=str, default=None,
                        help="PaddleDetection 根目录 (默认自动检测)")
    args = parser.parse_args()

    if args.paddle_root:
        global PADDLE_DET_ROOT, DATASET_ROOT, OUTPUT_ROOT, INFER_OUTPUT_DIR, UPLOAD_DIR
        PADDLE_DET_ROOT = os.path.abspath(args.paddle_root)
        DATASET_ROOT = os.path.join(PADDLE_DET_ROOT, "dataset")
        OUTPUT_ROOT = os.path.join(PADDLE_DET_ROOT, "output")
        INFER_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "_web_infer_vis")
        UPLOAD_DIR = os.path.join(OUTPUT_ROOT, "_web_uploads")
        os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    print("=" * 56)
    print("  VOC 标注查看器 + 模型推理")
    print(f"  PaddleDetection: {PADDLE_DET_ROOT}")
    print(f"  数据集目录:      {DATASET_ROOT}")
    print(f"  模型输出目录:    {OUTPUT_ROOT}")
    print(f"  服务地址:        http://{args.host}:{args.port}")
    print("=" * 56)

    server = ReusableHTTPServer((args.host, args.port), ViewerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()
