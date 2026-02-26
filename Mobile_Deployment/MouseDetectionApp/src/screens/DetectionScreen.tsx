import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  NativeModules,
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
import { Detection } from '../types';
import RNFS from 'react-native-fs';

// Full native pipeline: image → ORT C API + CoreML → NMS → boxes
// No 1.2 MB float data over the bridge
const { MouseDetector } = NativeModules;

const MAX_DEBUG_LOGS = 200;
// How many ms to wait between frames (0 = run as fast as possible)
const INTER_FRAME_DELAY_MS = 0;

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
  const cameraRef = useRef<Camera>(null);
  const loopActiveRef = useRef(false);
  const logFrameCountRef = useRef(0);

  const [modelReady, setModelReady] = useState(false);
  const [status, setStatus] = useState('正在加载模型...');
  const [detections, setDetections] = useState<Detection[]>([]);
  const [allDetections, setAllDetections] = useState<Detection[]>([]);
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });
  const [fps, setFps] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [debugLogs, setDebugLogs] = useState<DebugLog[]>([]);
  const [showDebug, setShowDebug] = useState(true);

  // Camera preview layout
  const [previewLayout, setPreviewLayout] = useState({ width: 0, height: 0 });

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

  // Initialize native model
  useEffect(() => {
    let mounted = true;
    addLog('开始加载模型 (native MouseDetector)...');

    // Locate model in app bundle
    (async () => {
      let modelPath: string | null = null;
      const candidates = [
        `${RNFS.MainBundlePath}/picodet_s_320_mouse_L1_nonms.onnx`,
        `${RNFS.MainBundlePath}/Resources/picodet_s_320_mouse_L1_nonms.onnx`,
      ];
      for (const p of candidates) {
        if (await RNFS.exists(p)) { modelPath = p; break; }
      }
      if (!modelPath) {
        const msg = `Model not found in bundle (${RNFS.MainBundlePath})`;
        if (mounted) { setStatus(`模型加载失败: ${msg}`); addLog(`❌ ${msg}`, 'error'); }
        return;
      }
      addLog(`Model path: ${modelPath}`);
      try {
        await MouseDetector.initialize(modelPath);
        if (mounted) {
          setModelReady(true);
          setStatus('模型就绪 - 点击"开始检测"');
          addLog('✅ 原生模型加载成功 (CoreML EP)');
        }
      } catch (err) {
        const msg = String(err);
        if (mounted) { setStatus(`模型加载失败: ${msg}`); addLog(`❌ ${msg}`, 'error'); }
      }
    })();

    return () => { mounted = false; };
  }, [addLog]);

  // Request camera permission
  useEffect(() => {
    if (!hasPermission) {
      addLog('请求相机权限...');
      requestPermission();
    } else {
      addLog('✅ 相机权限已授权');
    }
  }, [hasPermission, requestPermission, addLog]);

  // Handle threshold change — re-filter already-collected detections immediately
  const handleThresholdChange = useCallback(
    (val: number) => {
      setThreshold(val);
      const filtered = allDetections.filter(d => d.confidence >= val);
      setDetections(filtered);
      addLog(`阈值调整为 ${(val * 100).toFixed(0)}%, 当前显示 ${filtered.length}/${allDetections.length} 个检测`, 'info');
    },
    [allDetections, addLog]
  );

  // Single inference cycle — one native round-trip: snap → C++ ORT + CoreML → boxes
  const runInference = useCallback(async () => {
    if (!cameraRef.current || !modelReady) { return; }

    const startTime = Date.now();

    try {
      const photo = await cameraRef.current.takeSnapshot({ quality: 50 });
      const snapTime = Date.now() - startTime;
      const filePath = photo.path;

      // Single native call: resize + normalize + ORT inference + NMS — all in C++/Swift
      // Only the detection boxes (tiny JSON) cross the RN bridge
      const nativeResult = await MouseDetector.detect(filePath, threshold);
      const elapsed = Date.now() - startTime;

      const rawBoxes: Array<{
        classId: number; className: string; confidence: number;
        x1: number; y1: number; x2: number; y2: number;
      }> = nativeResult.detections;

      const results: Detection[] = rawBoxes.map(b => ({
        classId:    b.classId,
        className:  b.className,
        confidence: b.confidence,
        bbox: { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 },
      }));

      setImageDims({ width: nativeResult.originalWidth, height: nativeResult.originalHeight });
      setAllDetections(results);
      setDetections(results); // threshold already applied inside native module
      setFps(Math.round(1000 / elapsed));
      setStatus(`检测中 | ${results.length} 个目标 | ${elapsed}ms`);

      // Throttle remote logs every 5 frames
      logFrameCountRef.current += 1;
      if (logFrameCountRef.current % 5 === 1) {
        addLog(
          `snap=${snapTime}ms ${nativeResult.timings} js=${elapsed - snapTime - parseInt(nativeResult.timings.match(/total=(\d+)/)?.[1] ?? '0', 10)}ms total=${elapsed}ms`,
          'info'
        );
        if (results.length > 0) {
          results.forEach((det, i) => {
            addLog(
              `  ✅ [${i}] ${det.className} conf=${(det.confidence * 100).toFixed(1)}% bbox=(${det.bbox.x1.toFixed(0)},${det.bbox.y1.toFixed(0)},${det.bbox.x2.toFixed(0)},${det.bbox.y2.toFixed(0)})`,
              'detection'
            );
          });
        } else {
          addLog('  无检测结果', 'warn');
        }
      }

      try { await RNFS.unlink(filePath); } catch (_) {}
    } catch (error) {
      addLog(`推理错误: ${String(error)}`, 'error');
      console.error('Inference error:', error);
    }
  }, [modelReady, addLog, threshold]);

  const toggleDetection = useCallback(() => {
    if (isRunning) {
      loopActiveRef.current = false;
      setIsRunning(false);
      setStatus('已暂停');
      setDetections([]);
      setAllDetections([]);
      addLog('⏹ 检测已停止');
    } else {
      loopActiveRef.current = true;
      logFrameCountRef.current = 0;
      setIsRunning(true);
      setStatus('检测中...');
      addLog('▶ 开始持续检测 (全原生推理，无固定间隔)');

      (async () => {
        while (loopActiveRef.current) {
          await runInference();
          if (INTER_FRAME_DELAY_MS > 0) {
            await new Promise(r => setTimeout(r, INTER_FRAME_DELAY_MS));
          }
        }
      })();
    }
  }, [isRunning, runInference, addLog]);

  // Stop loop on unmount
  useEffect(() => {
    return () => {
      loopActiveRef.current = false;
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
