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
    const modelPath = `${RNFS.MainBundlePath}/yolov3_mouse_fp32.onnx`;
    const labelPath = `${RNFS.MainBundlePath}/label_list.txt`;

    try {
      this.session = await InferenceSession.create(modelPath);

      const labelContent = await RNFS.readFile(labelPath, 'utf8');
      this.labels = labelContent.split('\n').map(l => l.trim()).filter(Boolean);
    } catch (error) {
      console.error('Model initialization failed:', error);
      throw error;
    }
  }

  async detect(preprocess: PreprocessResult): Promise<Detection[]> {
    if (!this.session) {
      throw new Error('Model not initialized');
    }

    const { imageData, originalHeight, originalWidth, scale } = preprocess;
    const { inputSize } = this.config;

    const imageTensor = new Tensor('float32', imageData, [1, 3, inputSize, inputSize]);
    const imShapeTensor = new Tensor(
      'float32',
      new Float32Array([originalHeight, originalWidth]),
      [1, 2]
    );
    const scaleTensor = new Tensor('float32', new Float32Array([scale, scale]), [1, 2]);

    const results = await this.session.run({
      image: imageTensor,
      im_shape: imShapeTensor,
      scale_factor: scaleTensor,
    });

    return this.parseOutput(results);
  }

  private parseOutput(results: Record<string, Tensor>): Detection[] {
    const detections: Detection[] = [];
    const boxes = results['multiclass_nms3_0.tmp_0'];
    const numBoxes = results['multiclass_nms3_0.tmp_2'];

    if (!boxes || !numBoxes) {
      return detections;
    }

    const boxData = boxes.data as Float32Array;
    const count = (numBoxes.data as Int32Array | Float32Array)[0];

    for (let i = 0; i < count; i++) {
      const offset = i * 6;
      const classId = Math.round(boxData[offset]);
      const confidence = boxData[offset + 1];

      if (confidence < this.config.confidenceThreshold) {
        continue;
      }

      const x1 = boxData[offset + 2];
      const y1 = boxData[offset + 3];
      const x2 = boxData[offset + 4];
      const y2 = boxData[offset + 5];

      detections.push({
        classId,
        className: this.labels[classId] ?? `class_${classId}`,
        confidence,
        bbox: { x1, y1, x2, y2 },
      });
    }

    return detections;
  }

  getConfig(): ModelConfig {
    return this.config;
  }
}
