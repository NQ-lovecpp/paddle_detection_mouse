import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Detection, ModelConfig, PreprocessResult } from '../types';

export class ModelService {
  private session: InferenceSession | null = null;
  private labels: string[] = [];
  private config: ModelConfig = {
    inputSize: 608,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    confidenceThreshold: 0.5,
  };

  async initialize(): Promise<void> {
    // Try multiple possible paths for the model file
    const possibleModelPaths = [
      `${RNFS.MainBundlePath}/yolov3_mouse_int8_qdq.onnx`,
      `${RNFS.MainBundlePath}/Resources/yolov3_mouse_int8_qdq.onnx`,
      `${RNFS.MainBundlePath}/yolov3_mouse_int8.onnx`,
      `${RNFS.MainBundlePath}/Resources/yolov3_mouse_int8.onnx`,
      `${RNFS.MainBundlePath}/yolov3_mouse_fp32.onnx`,
      `${RNFS.MainBundlePath}/Resources/yolov3_mouse_fp32.onnx`,
    ];
    const possibleLabelPaths = [
      `${RNFS.MainBundlePath}/label_list.txt`,
      `${RNFS.MainBundlePath}/Resources/label_list.txt`,
    ];

    let modelPath: string | null = null;
    let labelPath: string | null = null;

    for (const p of possibleModelPaths) {
      if (await RNFS.exists(p)) {
        modelPath = p;
        break;
      }
    }
    for (const p of possibleLabelPaths) {
      if (await RNFS.exists(p)) {
        labelPath = p;
        break;
      }
    }

    if (!modelPath) {
      // List bundle contents for debugging
      let bundleFiles: string[] = [];
      try {
        bundleFiles = await RNFS.readDir(RNFS.MainBundlePath).then(items =>
          items.map(i => i.name)
        );
      } catch (_) {}
      throw new Error(
        `Model file not found. Bundle path: ${RNFS.MainBundlePath}\nBundle files: ${bundleFiles.join(', ')}`
      );
    }

    try {
      console.log('Loading model from:', modelPath);
      this.session = await InferenceSession.create(modelPath);
      console.log('Model loaded successfully');
      console.log(
        'Input names:',
        this.session.inputNames,
        'Output names:',
        this.session.outputNames
      );

      if (labelPath) {
        const labelContent = await RNFS.readFile(labelPath, 'utf8');
        this.labels = labelContent.split('\n').map(l => l.trim()).filter(Boolean);
        console.log('Labels loaded:', this.labels);
      } else {
        this.labels = ['mouse'];
        console.log('Label file not found, using default: mouse');
      }
    } catch (error) {
      console.error('Model initialization failed:', error);
      throw error;
    }
  }

  async detect(preprocess: PreprocessResult): Promise<{ all: Detection[]; debugInfo: string }> {
    if (!this.session) {
      throw new Error('Model not initialized');
    }

    const { imageData, originalHeight, originalWidth, scaleX, scaleY } = preprocess;
    const { inputSize } = this.config;

    // Build input tensors matching PaddleDetection's preprocessing:
    // image: [1, 3, 608, 608] - normalized image in CHW
    const imageTensor = new Tensor('float32', imageData, [1, 3, inputSize, inputSize]);

    // im_shape: [1, 2] - shape AFTER resize (which is [608, 608] since keep_ratio=false)
    // PaddleDetection Resize updates im_shape to resized dimensions
    const imShapeTensor = new Tensor(
      'float32',
      new Float32Array([inputSize, inputSize]),
      [1, 2]
    );

    // scale_factor: [1, 2] - [scale_y, scale_x] = [targetH/origH, targetW/origW]
    const scaleTensor = new Tensor(
      'float32',
      new Float32Array([scaleY, scaleX]),
      [1, 2]
    );

    let debugInfo = `inputs: im_shape=[${inputSize},${inputSize}], scale_factor=[${scaleY.toFixed(4)},${scaleX.toFixed(4)}], orig=[${originalWidth}x${originalHeight}]`;

    const results = await this.session.run({
      image: imageTensor,
      im_shape: imShapeTensor,
      scale_factor: scaleTensor,
    });

    // Debug: log output tensor info
    const outputKeys = Object.keys(results);
    debugInfo += ` | outputs: ${outputKeys.join(', ')}`;

    const parsed = this.parseOutput(results, debugInfo);
    return parsed;
  }

  private parseOutput(
    results: Record<string, Tensor>,
    debugPrefix: string
  ): { all: Detection[]; debugInfo: string } {
    const detections: Detection[] = [];
    const boxes = results['multiclass_nms3_0.tmp_0'];
    const numBoxes = results['multiclass_nms3_0.tmp_2'];

    let debugInfo = debugPrefix;

    if (!boxes || !numBoxes) {
      debugInfo += ' | ⚠️ Missing output tensors!';
      const keys = Object.keys(results);
      debugInfo += ` Available keys: [${keys.join(', ')}]`;
      return { all: detections, debugInfo };
    }

    const boxData = boxes.data as Float32Array;
    const numBoxData = numBoxes.data as Float32Array | Int32Array;
    const count = Math.round(Number(numBoxData[0]));

    debugInfo += ` | boxes.shape=[${boxes.dims}] count=${count} boxData.len=${boxData.length}`;

    // Log first few raw box values for debugging
    if (boxData.length > 0) {
      const rawSample = [];
      for (let i = 0; i < Math.min(18, boxData.length); i++) {
        rawSample.push(boxData[i].toFixed(3));
      }
      debugInfo += ` | raw[0..${rawSample.length - 1}]=[${rawSample.join(',')}]`;
    }

    // PaddleDetection output format: [class_id, score, x1, y1, x2, y2]
    // class_id == -1 means invalid detection
    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      if (offset + 5 >= boxData.length) {
        break;
      }

      const classId = Math.round(boxData[offset]);
      const confidence = boxData[offset + 1];

      // Skip invalid detections (class_id == -1)
      if (classId < 0) {
        continue;
      }

      const x1 = boxData[offset + 2];
      const y1 = boxData[offset + 3];
      const x2 = boxData[offset + 4];
      const y2 = boxData[offset + 5];

      // Store ALL detections (don't filter by threshold here, let the UI handle it)
      detections.push({
        classId,
        className: this.labels[classId] ?? `class_${classId}`,
        confidence,
        bbox: { x1, y1, x2, y2 },
      });
    }

    debugInfo += ` | total_valid=${detections.length}`;

    return { all: detections, debugInfo };
  }

  setConfidenceThreshold(threshold: number): void {
    this.config.confidenceThreshold = threshold;
  }

  isReady(): boolean {
    return this.session !== null;
  }

  getConfig(): ModelConfig {
    return this.config;
  }
}
