import os
import sys
import numpy as np


def main():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov3_mouse_fp32.onnx")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print("Missing model file:", model_path)
        print("Please copy yolov3_mouse_fp32.onnx into Mobile_Deployment/models/")
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in session.get_inputs()}
    print("Inputs:", inputs)

    input_size = 608
    image = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    im_shape = np.array([[input_size, input_size]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

    outputs = session.run(
        None,
        {
            "image": image,
            "im_shape": im_shape,
            "scale_factor": scale_factor,
        },
    )

    boxes, num_boxes = outputs[0], outputs[1]
    print("boxes shape:", boxes.shape)
    print("num_boxes:", num_boxes)


if __name__ == "__main__":
    main()
