import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { Detection, ModelConfig, PreprocessResult } from '../types';

// PicoDet-S 320×320, 4 FPN levels (stride 8/16/32/64):
//   40×40 + 20×20 + 10×10 + 5×5 = 2125 anchors
const NUM_ANCHORS = 2125;
const NUM_CLASSES = 2;
const INPUT_SIZE = 320;

// NMS IoU threshold (matches infer_cfg.yml nms_threshold)
const NMS_IOU_THRESHOLD = 0.5;
// Minimum score to enter NMS (matches infer_cfg.yml score_threshold)
const NMS_SCORE_THRESHOLD = 0.3;

export class ModelService {
  private session: InferenceSession | null = null;
  private labels: string[] = [];
  private config: ModelConfig = {
    inputSize: INPUT_SIZE,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
    confidenceThreshold: 0.5,
  };

  async initialize(): Promise<void> {
    const possibleModelPaths = [
      `${RNFS.MainBundlePath}/picodet_s_320_mouse_L1_nonms.onnx`,
      `${RNFS.MainBundlePath}/Resources/picodet_s_320_mouse_L1_nonms.onnx`,
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
      console.log('Loading PicoDet model from:', modelPath);
      // Enable CoreML Execution Provider for hardware-accelerated inference on iOS.
      // Falls back to CPU automatically if CoreML is unavailable.
      this.session = await InferenceSession.create(modelPath, {
        executionProviders: ['coreml', 'cpu'],
      });
      console.log('Model loaded. Inputs:', this.session.inputNames, 'Outputs:', this.session.outputNames);

      if (labelPath) {
        const labelContent = await RNFS.readFile(labelPath, 'utf8');
        this.labels = labelContent.split('\n').map(l => l.trim()).filter(Boolean);
        console.log('Labels:', this.labels);
      } else {
        this.labels = ['mouse', 'other'];
      }
    } catch (error) {
      console.error('Model init failed:', error);
      throw error;
    }
  }

  async detect(preprocess: PreprocessResult): Promise<{ all: Detection[]; debugInfo: string }> {
    if (!this.session) {
      throw new Error('Model not initialized');
    }

    const { imageData, originalHeight, originalWidth, scaleX, scaleY } = preprocess;

    // PicoDet nonms model: single input 'image' [1, 3, 320, 320]
    // scale_factor was folded away by onnxsim — not needed
    const imageTensor = new Tensor('float32', imageData, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    let debugInfo = `img=[1,3,${INPUT_SIZE},${INPUT_SIZE}] orig=[${originalWidth}x${originalHeight}] scale=[${scaleX.toFixed(3)},${scaleY.toFixed(3)}]`;

    const results = await this.session.run({ image: imageTensor });

    debugInfo += ` | out=[${Object.keys(results).join(',')}]`;

    return this.parseOutput(results, scaleX, scaleY, debugInfo);
  }

  private parseOutput(
    results: Record<string, Tensor>,
    scaleX: number,
    scaleY: number,
    debugPrefix: string
  ): { all: Detection[]; debugInfo: string } {
    let debugInfo = debugPrefix;

    // boxes:  [1, 2125, 4]  — xyxy coords in 320px input space
    // scores: [1, 2, 2125]  — sigmoid class scores, channels-first
    const boxesTensor  = results['boxes'];
    const scoresTensor = results['scores'];

    if (!boxesTensor || !scoresTensor) {
      const available = Object.keys(results).join(', ');
      debugInfo += ` | ⚠️ missing tensors, got: [${available}]`;
      return { all: [], debugInfo };
    }

    const boxData   = boxesTensor.data  as Float32Array; // length: 2125 * 4
    const scoreData = scoresTensor.data as Float32Array; // length: 2 * 2125

    debugInfo += ` | boxes=${boxesTensor.dims} scores=${scoresTensor.dims}`;

    // Collect candidates above minimum score threshold
    const candidates: Detection[] = [];

    for (let i = 0; i < NUM_ANCHORS; i++) {
      // scores layout: [class0_anchor0 .. class0_anchor2124, class1_anchor0 .. class1_anchor2124]
      let bestClass = -1;
      let bestScore = NMS_SCORE_THRESHOLD;

      for (let c = 0; c < NUM_CLASSES; c++) {
        const s = scoreData[c * NUM_ANCHORS + i];
        if (s > bestScore) {
          bestScore = s;
          bestClass = c;
        }
      }

      if (bestClass < 0) {
        continue;
      }

      // Convert boxes from 320px space → original image coordinates
      const x1 = boxData[i * 4 + 0] / scaleX;
      const y1 = boxData[i * 4 + 1] / scaleY;
      const x2 = boxData[i * 4 + 2] / scaleX;
      const y2 = boxData[i * 4 + 3] / scaleY;

      candidates.push({
        classId:   bestClass,
        className: this.labels[bestClass] ?? `class_${bestClass}`,
        confidence: bestScore,
        bbox: { x1, y1, x2, y2 },
      });
    }

    debugInfo += ` | pre_nms=${candidates.length}`;

    // Per-class greedy NMS
    const detections = nmsPerClass(candidates, NMS_IOU_THRESHOLD);

    debugInfo += ` | post_nms=${detections.length}`;

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

// ─── NMS helpers ────────────────────────────────────────────────────────────

function iou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number }
): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const intersection = iw * ih;
  if (intersection === 0) { return 0; }
  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / (aArea + bArea - intersection);
}

function nmsPerClass(dets: Detection[], iouThreshold: number): Detection[] {
  // Group by class
  const byClass = new Map<number, Detection[]>();
  for (const d of dets) {
    if (!byClass.has(d.classId)) { byClass.set(d.classId, []); }
    byClass.get(d.classId)!.push(d);
  }

  const kept: Detection[] = [];
  for (const [, classDets] of byClass) {
    // Sort descending by confidence
    classDets.sort((a, b) => b.confidence - a.confidence);
    const suppressed = new Array(classDets.length).fill(false);

    for (let i = 0; i < classDets.length; i++) {
      if (suppressed[i]) { continue; }
      kept.push(classDets[i]);
      for (let j = i + 1; j < classDets.length; j++) {
        if (!suppressed[j] && iou(classDets[i].bbox, classDets[j].bbox) > iouThreshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  return kept;
}
