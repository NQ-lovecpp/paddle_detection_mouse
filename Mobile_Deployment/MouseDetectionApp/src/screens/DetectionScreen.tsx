import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  NativeModules,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { BoundingBoxOverlay } from '../components/BoundingBoxOverlay';
import { DebugConsole, DebugLog } from '../components/DebugConsole';
import { ThresholdSlider } from '../components/ThresholdSlider';
import { ModelService } from '../services/ModelService';
import { Detection, PreprocessResult } from '../types';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';

const { ImagePreprocessor } = NativeModules;

const INFERENCE_INTERVAL_MS = 500;
const MAX_DEBUG_LOGS = 200;

// Remote log server URL - sends logs to Mac terminal for debugging
// Use the Mac's local IP (same network as iPhone via USB)
const LOG_SERVER_URL = 'http://192.168.71.40:8082/log';

// Send log to remote server for terminal visibility
function sendRemoteLog(message: string, type: string, time: string) {
  fetch(LOG_SERVER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, type, time }),
  }).catch(() => {}); // Silently ignore network errors
}

export const DetectionScreen: React.FC = () => {
  const modelService = useMemo(() => new ModelService(), []);
  const cameraRef = useRef<Camera>(null);
  const inferringRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [modelReady, setModelReady] = useState(false);
  const [status, setStatus] = useState('正在加载模型...');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [allDetections, setAllDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [debugLogs, setDebugLogs] = useState<DebugLog[]>([]);
  const [showDebug, setShowDebug] = useState(true);

  // Camera preview layout
  const [previewLayout, setPreviewLayout] = useState({ width: 0, height: 0 });
  // Image dimensions from last inference
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });

  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  // Debug log helper - also sends to remote server
  const addLog = useCallback((message: string, type: DebugLog['type'] = 'info') => {
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now
      .getMinutes()
      .toString()
      .padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now
      .getMilliseconds()
      .toString()
      .padStart(3, '0')}`;
    setDebugLogs(prev => {
      const next = [...prev, { time, message, type }];
      return next.length > MAX_DEBUG_LOGS ? next.slice(-MAX_DEBUG_LOGS) : next;
    });
    // Send to remote log server
    sendRemoteLog(message, type, time);
  }, []);

  // Initialize model
  useEffect(() => {
    let mounted = true;
    addLog('开始加载模型...');
    modelService
      .initialize()
      .then(() => {
        if (mounted) {
          setModelReady(true);
          setStatus('模型就绪 - 点击"开始检测"');
          addLog('✅ 模型加载成功');
          addLog(`配置: inputSize=${modelService.getConfig().inputSize}, threshold=${modelService.getConfig().confidenceThreshold}`);
        }
      })
      .catch(error => {
        if (mounted) {
          const msg = String(error);
          setStatus(`模型加载失败: ${msg}`);
          addLog(`❌ 模型加载失败: ${msg}`, 'error');
        }
      });
    return () => {
      mounted = false;
    };
  }, [modelService, addLog]);

  // Request camera permission
  useEffect(() => {
    if (!hasPermission) {
      addLog('请求相机权限...');
      requestPermission();
    } else {
      addLog('✅ 相机权限已授权');
    }
  }, [hasPermission, requestPermission, addLog]);

  // Handle threshold change
  const handleThresholdChange = useCallback(
    (val: number) => {
      setThreshold(val);
      modelService.setConfidenceThreshold(val);
      // Re-filter existing detections immediately
      const filtered = allDetections.filter(d => d.confidence >= val);
      setDetections(filtered);
      addLog(`阈值调整为 ${(val * 100).toFixed(0)}%, 当前显示 ${filtered.length}/${allDetections.length} 个检测`, 'info');
    },
    [modelService, allDetections, addLog]
  );

  // Single inference cycle
  const runInference = useCallback(async () => {
    if (inferringRef.current || !cameraRef.current || !modelService.isReady()) {
      return;
    }

    inferringRef.current = true;
    const startTime = Date.now();

    try {
      // Use takeSnapshot for faster capture from video pipeline
      const photo = await cameraRef.current.takeSnapshot({
        quality: 50, // Slightly higher quality for better detection, still fast
      });

      const snapTime = Date.now() - startTime;

      const filePath = photo.path;

      // === Native preprocessing (Swift/Accelerate) ===
      // This replaces: RNFS.readFile + jpeg-js decode + JS resize/normalize
      // All done natively: JPEG decode → Resize 608×608 → Normalize → CHW
      const nativeResult = await ImagePreprocessor.preprocess(filePath);
      const preprocessTime = Date.now() - startTime;

      // Decode base64 Float32 data from native module
      const floatBuffer = Buffer.from(nativeResult.data, 'base64');
      const imageData = new Float32Array(
        floatBuffer.buffer,
        floatBuffer.byteOffset,
        floatBuffer.byteLength / 4
      );

      const preprocess: PreprocessResult = {
        imageData,
        originalWidth: nativeResult.originalWidth,
        originalHeight: nativeResult.originalHeight,
        scaleX: nativeResult.scaleX,
        scaleY: nativeResult.scaleY,
      };

      setImageDims({
        width: preprocess.originalWidth,
        height: preprocess.originalHeight,
      });

      // Run detection (returns all detections + debug info)
      const { all: results, debugInfo } = await modelService.detect(preprocess);
      const inferTime = Date.now() - startTime;

      // Store all raw detections and filter by threshold
      setAllDetections(results);
      const filtered = results.filter(d => d.confidence >= threshold);
      setDetections(filtered);

      const elapsed = Date.now() - startTime;
      setFps(Math.round(1000 / elapsed));
      setStatus(`检测中 | ${filtered.length}/${results.length} 个目标 | ${elapsed}ms`);

      // Debug log with timing breakdown
      addLog(
        `snap=${snapTime}ms native_pre=${preprocessTime - snapTime}ms(${nativeResult.timings}) infer=${inferTime - preprocessTime}ms total=${elapsed}ms`,
        'info'
      );

      // Log model debug info (tensor shapes, raw output)
      addLog(debugInfo, 'info');

      // Log each detection with bbox details
      if (results.length > 0) {
        results.forEach((det, i) => {
          const aboveThresh = det.confidence >= threshold ? '✅' : '⬇️';
          addLog(
            `  ${aboveThresh} [${i}] ${det.className} conf=${(det.confidence * 100).toFixed(1)}% bbox=(${det.bbox.x1.toFixed(1)}, ${det.bbox.y1.toFixed(1)}, ${det.bbox.x2.toFixed(1)}, ${det.bbox.y2.toFixed(1)}) size=${(det.bbox.x2 - det.bbox.x1).toFixed(0)}x${(det.bbox.y2 - det.bbox.y1).toFixed(0)}`,
            'detection'
          );
        });
      } else {
        addLog('  无检测结果 (count=0 或所有 classId<0)', 'warn');
      }

      // Clean up temp file
      try {
        await RNFS.unlink(filePath);
      } catch (_) {}
    } catch (error) {
      const msg = String(error);
      addLog(`推理错误: ${msg}`, 'error');
      console.error('Inference error:', error);
    } finally {
      inferringRef.current = false;
    }
  }, [modelService, addLog, threshold]);

  // Start/stop continuous detection
  const toggleDetection = useCallback(() => {
    if (isRunning) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setIsRunning(false);
      setStatus('已暂停');
      setDetections([]);
      setAllDetections([]);
      addLog('⏹ 检测已停止');
    } else {
      setIsRunning(true);
      setStatus('检测中...');
      addLog('▶ 开始持续检测, 间隔=' + INFERENCE_INTERVAL_MS + 'ms');
      runInference();
      timerRef.current = setInterval(runInference, INFERENCE_INTERVAL_MS);
    }
  }, [isRunning, runInference, addLog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const onPreviewLayout = useCallback((e: LayoutChangeEvent) => {
    setPreviewLayout({
      width: e.nativeEvent.layout.width,
      height: e.nativeEvent.layout.height,
    });
  }, []);

  // Render states
  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>需要相机权限才能使用实时检测</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>授权相机</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>未找到可用摄像头</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerRow}>
          <Text style={styles.title}>🐭 老鼠检测</Text>
          <TouchableOpacity onPress={() => setShowDebug(v => !v)}>
            <Text style={styles.debugToggle}>
              {showDebug ? '隐藏日志' : '显示日志'}
            </Text>
          </TouchableOpacity>
        </View>
        <View style={styles.headerRow}>
          <Text style={styles.statusText}>{status}</Text>
          {isRunning && fps > 0 && (
            <Text style={styles.fpsText}>{fps} FPS</Text>
          )}
        </View>
      </View>

      {/* Threshold Slider */}
      <ThresholdSlider value={threshold} onValueChange={handleThresholdChange} />

      {/* Camera Preview + Bounding Boxes */}
      <View style={styles.cameraContainer} onLayout={onPreviewLayout}>
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={true}
          photo={true}
          video={true}
        />
        <BoundingBoxOverlay
          detections={detections}
          previewWidth={previewLayout.width}
          previewHeight={previewLayout.height}
          imageWidth={imageDims.width}
          imageHeight={imageDims.height}
        />
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        {!modelReady ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator size="small" color="#fff" />
            <Text style={styles.loadingText}>模型加载中...</Text>
          </View>
        ) : (
          <TouchableOpacity
            style={[styles.button, isRunning && styles.buttonStop]}
            onPress={toggleDetection}>
            <Text style={styles.buttonText}>
              {isRunning ? '⏹ 停止检测' : '▶ 开始检测'}
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Debug Console */}
      {showDebug && <DebugConsole logs={debugLogs} maxHeight={160} />}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 20,
  },
  message: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 16,
  },
  header: {
    paddingTop: 4,
    paddingBottom: 4,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(0,0,0,0.8)',
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    color: '#fff',
  },
  debugToggle: {
    color: '#bd93f9',
    fontSize: 12,
    fontWeight: '600',
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  statusText: {
    color: '#aaa',
    fontSize: 11,
    marginTop: 1,
  },
  fpsText: {
    color: '#0f0',
    fontSize: 12,
    fontWeight: '700',
  },
  cameraContainer: {
    flex: 1,
    overflow: 'hidden',
  },
  controls: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(0,0,0,0.7)',
    alignItems: 'center',
  },
  button: {
    backgroundColor: '#22c55e',
    paddingHorizontal: 32,
    paddingVertical: 12,
    borderRadius: 12,
    minWidth: 200,
    alignItems: 'center',
  },
  buttonStop: {
    backgroundColor: '#ef4444',
  },
  buttonText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 16,
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  loadingText: {
    color: '#aaa',
    fontSize: 14,
  },
});
