import argparse
import os
from typing import Dict, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for YOLOv3 ONNX (PaddleDetection export)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input FP32 ONNX model path, e.g. models/yolov3_mouse_fp32.onnx",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output INT8 ONNX model path, e.g. models/yolov3_mouse_int8_qdq.onnx",
    )
    parser.add_argument(
        "--calib_dir",
        required=True,
        help="Calibration image directory (jpg/png).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of calibration samples to use (default: 200).",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=608,
        help="Model input size (default: 608).",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=[0.485, 0.456, 0.406],
        help="Mean normalization values.",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=[0.229, 0.224, 0.225],
        help="Std normalization values.",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Enable per-channel weight quantization (recommended).",
    )
    parser.add_argument(
        "--calib_method",
        choices=["minmax", "entropy"],
        default="minmax",
        help="Calibration method (default: minmax).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run onnxruntime graph optimization before quantization.",
    )
    parser.add_argument(
        "--shape_infer",
        action="store_true",
        help="Run symbolic shape inference before quantization.",
    )
    return parser.parse_args()


def _list_images(calib_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    images = []
    for root, _, files in os.walk(calib_dir):
        for name in files:
            if name.lower().endswith(exts):
                images.append(os.path.join(root, name))
    images.sort()
    return images


def _preprocess_image(
    image_path: str,
    input_size: int,
    mean: List[float],
    std: List[float],
) -> Dict[str, np.ndarray]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing Pillow. Install with: pip install pillow") from exc

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((input_size, input_size), Image.BILINEAR)
        img_np = np.asarray(img).astype("float32") / 255.0

    mean_arr = np.array(mean, dtype="float32").reshape(1, 1, 3)
    std_arr = np.array(std, dtype="float32").reshape(1, 1, 3)
    img_np = (img_np - mean_arr) / std_arr

    # HWC -> CHW, add batch
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    scale_y = float(input_size) / float(orig_h)
    scale_x = float(input_size) / float(orig_w)

    return {
        "image": img_np.astype("float32"),
        "im_shape": np.array([[input_size, input_size]], dtype="float32"),
        "scale_factor": np.array([[scale_y, scale_x]], dtype="float32"),
    }


class ImageCalibrationDataReader:
    def __init__(
        self,
        image_paths: List[str],
        input_size: int,
        mean: List[float],
        std: List[float],
    ):
        self.image_paths = image_paths
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self._iter = iter(self.image_paths)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        try:
            image_path = next(self._iter)
        except StopIteration:
            return None
        return _preprocess_image(image_path, self.input_size, self.mean, self.std)


def _maybe_optimize(input_path: str) -> str:
    try:
        from onnxruntime.tools import optimizer
    except Exception:
        return input_path

    optimized_path = os.path.splitext(input_path)[0] + ".opt.onnx"
    opt_model = optimizer.optimize_model(input_path)
    opt_model.save_model_to_file(optimized_path)
    return optimized_path


def _maybe_shape_infer(input_path: str) -> str:
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        import onnx
    except Exception:
        return input_path

    inferred_path = os.path.splitext(input_path)[0] + ".shape.onnx"
    model = onnx.load(input_path)
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True)
    onnx.save(inferred, inferred_path)
    return inferred_path


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input model not found: {args.input}")
    if not os.path.isdir(args.calib_dir):
        raise FileNotFoundError(f"Calibration dir not found: {args.calib_dir}")

    image_paths = _list_images(args.calib_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in calib_dir: {args.calib_dir}")
    image_paths = image_paths[: args.num_samples]

    model_path = _maybe_optimize(args.input) if args.optimize else args.input
    if args.shape_infer:
        model_path = _maybe_shape_infer(model_path)

    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    calib_method = (
        CalibrationMethod.Entropy
        if args.calib_method == "entropy"
        else CalibrationMethod.MinMax
    )

    data_reader = ImageCalibrationDataReader(
        image_paths=image_paths,
        input_size=args.input_size,
        mean=args.mean,
        std=args.std,
    )

    quantize_static(
        model_path,
        args.output,
        data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        calibrate_method=calib_method,
    )

    print(f"✅ Quantized model saved: {args.output}")


if __name__ == "__main__":
    main()
