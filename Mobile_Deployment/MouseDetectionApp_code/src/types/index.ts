export interface Detection {
  classId: number;
  className: string;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
}

export interface ModelConfig {
  inputSize: number;
  mean: number[];
  std: number[];
  confidenceThreshold: number;
}

export interface PreprocessResult {
  imageData: Float32Array;
  originalWidth: number;
  originalHeight: number;
  scale: number;
}
