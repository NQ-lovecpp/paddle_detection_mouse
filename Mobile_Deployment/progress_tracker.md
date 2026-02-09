## Progress Tracker

- [x] Reviewed deployment requirements and model specs
- [x] Bootstrapped React Native app structure under `Mobile_Deployment/MouseDetectionApp`
- [x] Implemented ONNX model loading and inference flow
- [x] Implemented image preprocessing from base64 (resize + normalize + CHW)
- [x] Added simple UI for image selection and detection results
- [x] Added local ONNX inference sanity test script

### Pending
- [ ] Add `yolov3_mouse_fp32.onnx` to `Mobile_Deployment/models/`
- [ ] Integrate model files into iOS bundle via Xcode
- [ ] Run `tools/onnx_infer_test.py` to validate ONNX inference
- [ ] Run on-device test (iOS simulator or device)
