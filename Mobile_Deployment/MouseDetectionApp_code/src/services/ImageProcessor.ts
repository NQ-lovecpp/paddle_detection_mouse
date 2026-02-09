import jpeg from 'jpeg-js';
import { Buffer } from 'buffer';
import { ModelConfig, PreprocessResult } from '../types';

export class ImageProcessor {
  static preprocessFromBase64(
    base64: string,
    config: ModelConfig
  ): PreprocessResult {
    const normalized = base64.includes(',') ? base64.split(',')[1] : base64;
    let raw;
    try {
      raw = jpeg.decode(Buffer.from(normalized, 'base64'), { useTArray: true });
    } catch (error) {
      throw new Error(
        '当前仅支持JPEG图片，请将相机格式设置为“最兼容”(JPEG)或选择JPG图片。'
      );
    }
    if (!raw || !raw.data) {
      throw new Error('图片解析失败，请更换图片重试。');
    }
    const originalWidth = raw.width;
    const originalHeight = raw.height;
    const { inputSize, mean, std } = config;

    const scale = inputSize / Math.max(originalWidth, originalHeight);
    const resizedWidth = Math.round(originalWidth * scale);
    const resizedHeight = Math.round(originalHeight * scale);

    const normalized = ImageProcessor.resizeAndNormalize(
      raw.data,
      originalWidth,
      originalHeight,
      resizedWidth,
      resizedHeight,
      inputSize,
      mean,
      std
    );

    return {
      imageData: normalized,
      originalWidth,
      originalHeight,
      scale,
    };
  }

  private static resizeAndNormalize(
    rgba: Uint8Array,
    srcWidth: number,
    srcHeight: number,
    resizedWidth: number,
    resizedHeight: number,
    targetSize: number,
    mean: number[],
    std: number[]
  ): Float32Array {
    const chw = new Float32Array(3 * targetSize * targetSize);
    const [meanR, meanG, meanB] = mean;
    const [stdR, stdG, stdB] = std;

    const fillR = (0 - meanR) / stdR;
    const fillG = (0 - meanG) / stdG;
    const fillB = (0 - meanB) / stdB;

    const planeSize = targetSize * targetSize;
    for (let i = 0; i < planeSize; i++) {
      chw[i] = fillR;
      chw[planeSize + i] = fillG;
      chw[planeSize * 2 + i] = fillB;
    }

    for (let y = 0; y < resizedHeight; y++) {
      const srcY = Math.min(srcHeight - 1, Math.floor((y / resizedHeight) * srcHeight));
      for (let x = 0; x < resizedWidth; x++) {
        const srcX = Math.min(srcWidth - 1, Math.floor((x / resizedWidth) * srcWidth));
        const srcIndex = (srcY * srcWidth + srcX) * 4;
        const r = rgba[srcIndex];
        const g = rgba[srcIndex + 1];
        const b = rgba[srcIndex + 2];

        const normR = (r / 255 - meanR) / stdR;
        const normG = (g / 255 - meanG) / stdG;
        const normB = (b / 255 - meanB) / stdB;

        const dstIndex = y * targetSize + x;
        chw[dstIndex] = normR;
        chw[planeSize + dstIndex] = normG;
        chw[planeSize * 2 + dstIndex] = normB;
      }
    }

    return chw;
  }
}
