import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  LayoutChangeEvent,
  NativeModules,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  unstable_batchedUpdates,
  useWindowDimensions,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  useCameraPermission,
} from 'react-native-vision-camera';
import { BoundingBoxOverlay } from '../components/BoundingBoxOverlay';
import { DebugConsole, DebugLog } from '../components/DebugConsole';
import { ThresholdSlider } from '../components/ThresholdSlider';
import { Detection } from '../types';
import RNFS from 'react-native-fs';

const { MouseDetector } = NativeModules;

const MAX_DEBUG_LOGS = 200;
const INTER_FRAME_DELAY_MS = 0;
const LOG_SERVER_URL = 'http://192.168.71.85:8082/log';

function sendRemoteLog(message: string, type: string, time: string) {
  fetch(LOG_SERVER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, type, time }),
  }).catch(() => {});
}

// ─── Model registry — add more models here when available ────────────────────
const MODELS = [
  {
    id: 'picodet',
    label: 'PicoDet-S 320',
    filename: 'picodet_s_320_mouse_L1_nonms.onnx',
  },
] as const;

type ModelId = (typeof MODELS)[number]['id'];

// ─── Main screen ─────────────────────────────────────────────────────────────

export const DetectionScreen: React.FC = () => {
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  const cameraRef = useRef<Camera>(null);
  const loopActiveRef = useRef(false);
  const logFrameCountRef = useRef(0);
  // Pipeline: pre-started snapshot runs in parallel with current inference
  const pendingSnapRef = useRef<ReturnType<Camera['takeSnapshot']> | null>(null);

  // Camera
  const [cameraPos, setCameraPos] = useState<'back' | 'front'>('back');
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice(cameraPos);

  // Prefer a low-resolution format at high FPS to keep takeSnapshot fast.
  // 720p@60fps gives half the pixels to decode vs 1080p@30fps,
  // reducing snap latency from ~60ms (thermal) to ~15ms.
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1280, height: 720 } },
    { fps: 60 },
  ]);

  // Model
  const [activeModel] = useState<ModelId>('picodet');
  const [modelReady, setModelReady] = useState(false);

  // Detection state
  const [detections, setDetections] = useState<Detection[]>([]);
  // allDetections is only read in the threshold callback, not rendered directly;
  // keep it as a ref to avoid triggering an extra re-render per frame.
  const allDetectionsRef = useRef<Detection[]>([]);
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });
  const [fps, setFps] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [threshold, setThreshold] = useState(0.5);
  const [status, setStatus] = useState('正在加载模型...');

  // UI
  const [debugLogs, setDebugLogs] = useState<DebugLog[]>([]);
  const [showDebug, setShowDebug] = useState(true);
  const [previewLayout, setPreviewLayout] = useState({ width: 0, height: 0 });

  // ── Logging ────────────────────────────────────────────────────────────────

  const addLog = useCallback((message: string, type: DebugLog['type'] = 'info') => {
    const now = new Date();
    const time = [
      now.getHours().toString().padStart(2, '0'),
      now.getMinutes().toString().padStart(2, '0'),
      now.getSeconds().toString().padStart(2, '0'),
    ].join(':') + '.' + now.getMilliseconds().toString().padStart(3, '0');

    setDebugLogs(prev => {
      const next = [...prev, { time, message, type }];
      return next.length > MAX_DEBUG_LOGS ? next.slice(-MAX_DEBUG_LOGS) : next;
    });
    sendRemoteLog(message, type, time);
  }, []);

  // ── Model initialization ───────────────────────────────────────────────────

  useEffect(() => {
    let mounted = true;
    const modelFile = MODELS.find(m => m.id === activeModel)!.filename;
    addLog(`加载模型: ${modelFile}`);

    (async () => {
      let modelPath: string | null = null;
      for (const base of [RNFS.MainBundlePath, `${RNFS.MainBundlePath}/Resources`]) {
        const p = `${base}/${modelFile}`;
        if (await RNFS.exists(p)) { modelPath = p; break; }
      }
      if (!modelPath) {
        if (mounted) {
          setStatus('模型文件未找到');
          addLog(`❌ 找不到 ${modelFile}`, 'error');
        }
        return;
      }
      try {
        await MouseDetector.initialize(modelPath);
        if (mounted) {
          setModelReady(true);
          setStatus('就绪 — 点击开始');
          addLog('✅ 模型加载成功 (CoreML ANE)');
        }
      } catch (err) {
        if (mounted) {
          setStatus('模型加载失败');
          addLog(`❌ ${String(err)}`, 'error');
        }
      }
    })();

    return () => { mounted = false; };
  }, [activeModel, addLog]);

  // ── Camera permission ──────────────────────────────────────────────────────

  useEffect(() => {
    if (!hasPermission) { requestPermission(); }
  }, [hasPermission, requestPermission]);

  // ── Threshold ──────────────────────────────────────────────────────────────

  const handleThresholdChange = useCallback((val: number) => {
    setThreshold(val);
    const filtered = allDetectionsRef.current.filter(d => d.confidence >= val);
    setDetections(filtered);
  }, []);

  // ── Camera switch ──────────────────────────────────────────────────────────

  const toggleCamera = useCallback(() => {
    // Stop detection before switching to avoid snapshot-on-wrong-camera issues
    loopActiveRef.current = false;
    setIsRunning(false);
    setDetections([]);
    allDetectionsRef.current = [];
    setCameraPos(p => (p === 'back' ? 'front' : 'back'));
    addLog(`切换到${cameraPos === 'back' ? '前置' : '后置'}摄像头`);
  }, [cameraPos, addLog]);

  // ── Inference loop ─────────────────────────────────────────────────────────
  //
  // Pipeline: while MouseDetector.detect() runs on frame N (C++/CoreML, ~50ms),
  // the next takeSnapshot() is already in flight in parallel.
  // Effective throughput = max(snap_time, infer_time) instead of snap + infer.

  const runInference = useCallback(async () => {
    if (!cameraRef.current || !modelReady) { return; }
    const t0 = Date.now();
    try {
      // Use a pre-started snapshot if available; otherwise start one now.
      const photoPromise = pendingSnapRef.current
        ?? cameraRef.current.takeSnapshot({ quality: 25 });

      // Immediately kick off the next snapshot — runs concurrently with detect().
      pendingSnapRef.current = cameraRef.current.takeSnapshot({ quality: 25 });

      const photo = await photoPromise;
      const snapMs = Date.now() - t0;

      const native = await MouseDetector.detect(photo.path, threshold);
      const totalMs = Date.now() - t0;

      // Fire-and-forget: don't block the next frame on disk cleanup
      RNFS.unlink(photo.path).catch(() => {});

      const results: Detection[] = (native.detections as any[]).map((b: any) => ({
        classId: b.classId,
        className: b.className,
        confidence: b.confidence,
        bbox: { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 },
      }));

      allDetectionsRef.current = results;
      logFrameCountRef.current += 1;
      const frameCount = logFrameCountRef.current;

      // Batch all state updates into a single React render pass.
      // FPS + status only update every 3 frames to halve the re-render rate
      // for those counters without affecting detection smoothness.
      unstable_batchedUpdates(() => {
        setImageDims({ width: native.originalWidth, height: native.originalHeight });
        setDetections(results);
        if (frameCount % 3 === 1) {
          setFps(Math.round(1000 / totalMs));
          setStatus(`${results.length} 个目标 | ${totalMs}ms`);
        }
      });

      if (frameCount % 5 === 1) {
        addLog(`snap=${snapMs}ms ${native.timings} total=${totalMs}ms`, 'info');
        results.forEach((d, i) =>
          addLog(
            `  ✅[${i}] ${d.className} ${(d.confidence * 100).toFixed(1)}% (${d.bbox.x1.toFixed(0)},${d.bbox.y1.toFixed(0)},${d.bbox.x2.toFixed(0)},${d.bbox.y2.toFixed(0)})`,
            'detection',
          ),
        );
        if (results.length === 0) addLog('  无检测结果', 'warn');
      }
    } catch (err) {
      addLog(`推理错误: ${String(err)}`, 'error');
      pendingSnapRef.current = null; // reset pipeline on error
    }
  }, [modelReady, threshold, addLog]);

  const toggleDetection = useCallback(() => {
    if (isRunning) {
      loopActiveRef.current = false;
      setIsRunning(false);
      setStatus('已暂停');
      setDetections([]);
      allDetectionsRef.current = [];
      // Clean up any pre-started snapshot that won't be used
      if (pendingSnapRef.current) {
        pendingSnapRef.current
          .then(p => RNFS.unlink(p.path).catch(() => {}))
          .catch(() => {});
        pendingSnapRef.current = null;
      }
      addLog('⏹ 停止');
    } else {
      loopActiveRef.current = true;
      logFrameCountRef.current = 0;
      setIsRunning(true);
      setStatus('推理中...');
      addLog('▶ 开始');
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

  useEffect(() => () => { loopActiveRef.current = false; }, []);

  const onPreviewLayout = useCallback((e: LayoutChangeEvent) => {
    setPreviewLayout({
      width: e.nativeEvent.layout.width,
      height: e.nativeEvent.layout.height,
    });
  }, []);

  // ── Permission gate ────────────────────────────────────────────────────────

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.gateText}>需要相机权限</Text>
        <TouchableOpacity style={styles.btnPrimary} onPress={requestPermission}>
          <Text style={styles.btnText}>授权相机</Text>
        </TouchableOpacity>
      </View>
    );
  }
  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.gateText}>未找到摄像头</Text>
      </View>
    );
  }

  // ── Shared sidebar content (controls + debug) ──────────────────────────────

  const sidebarContent = (
    <>
      {/* Title row */}
      <View style={styles.titleRow}>
        <Text style={styles.title}>🐭 鼠检测</Text>
        <Text style={[styles.fpsChip, !isRunning && { opacity: 0 }]}>
          {fps} FPS
        </Text>
      </View>

      {/* Status */}
      <Text style={styles.statusText} numberOfLines={2}>{status}</Text>

      <View style={styles.divider} />

      {/* Threshold slider */}
      <Text style={styles.sectionLabel}>
        置信度阈值 {(threshold * 100).toFixed(0)}%
      </Text>
      <ThresholdSlider value={threshold} onValueChange={handleThresholdChange} />

      <View style={styles.divider} />

      {/* Start / stop */}
      {!modelReady ? (
        <View style={styles.loadingRow}>
          <ActivityIndicator size="small" color="#60a5fa" />
          <Text style={styles.loadingText}>加载模型中...</Text>
        </View>
      ) : (
        <TouchableOpacity
          style={[styles.btnPrimary, isRunning && styles.btnStop]}
          onPress={toggleDetection}
          activeOpacity={0.8}>
          <Text style={styles.btnText}>
            {isRunning ? '⏹ 停止检测' : '▶ 开始检测'}
          </Text>
        </TouchableOpacity>
      )}

      {/* Camera switch */}
      <TouchableOpacity
        style={styles.btnCamera}
        onPress={toggleCamera}
        activeOpacity={0.8}>
        <Text style={styles.btnCameraText}>
          {cameraPos === 'back' ? '🤳 切换前置' : '📷 切换后置'}
        </Text>
      </TouchableOpacity>

      <View style={styles.divider} />

      {/* Debug toggle */}
      <TouchableOpacity onPress={() => setShowDebug(v => !v)} style={styles.debugToggleRow}>
        <Text style={styles.debugToggleText}>
          {showDebug ? '▾ 隐藏日志' : '▸ 显示日志'}
        </Text>
      </TouchableOpacity>

      {/* Debug console (only in sidebar in landscape) */}
      {showDebug && isLandscape && (
        <DebugConsole logs={debugLogs} maxHeight={999} flex />
      )}
    </>
  );

  // ── Camera + overlay ───────────────────────────────────────────────────────

  const cameraView = (
    <View style={styles.cameraContainer} onLayout={onPreviewLayout}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
        video={true}
        format={format}
        fps={format?.maxFps ?? 30}
      />
      <BoundingBoxOverlay
        detections={detections}
        previewWidth={previewLayout.width}
        previewHeight={previewLayout.height}
        imageWidth={imageDims.width}
        imageHeight={imageDims.height}
      />
      {/* Inline FPS badge on camera */}
      {isRunning && fps > 0 && (
        <View style={styles.fpsBadge}>
          <Text style={styles.fpsBadgeText}>{fps} FPS</Text>
        </View>
      )}
    </View>
  );

  // ── Landscape layout ───────────────────────────────────────────────────────

  if (isLandscape) {
    return (
      <View style={styles.landscapeRoot}>
        {/* Left sidebar */}
        <ScrollView
          style={styles.sidebar}
          contentContainerStyle={styles.sidebarContent}
          showsVerticalScrollIndicator={false}>
          {sidebarContent}
        </ScrollView>

        {/* Camera fills remaining space */}
        {cameraView}
      </View>
    );
  }

  // ── Portrait layout ────────────────────────────────────────────────────────

  return (
    <View style={styles.portraitRoot}>
      {/* Top bar */}
      <View style={styles.portraitHeader}>
        <View style={styles.titleRow}>
          <Text style={styles.title}>🐭 老鼠检测</Text>
          <TouchableOpacity onPress={() => setShowDebug(v => !v)}>
            <Text style={styles.debugToggleText}>
              {showDebug ? '隐藏日志' : '显示日志'}
            </Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.statusText}>{status}</Text>
        <ThresholdSlider value={threshold} onValueChange={handleThresholdChange} />
      </View>

      {/* Camera */}
      {cameraView}

      {/* Bottom controls */}
      <View style={styles.portraitControls}>
        {!modelReady ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator size="small" color="#60a5fa" />
            <Text style={styles.loadingText}>加载中...</Text>
          </View>
        ) : (
          <View style={styles.portraitBtnRow}>
            <TouchableOpacity
              style={[styles.btnPrimary, styles.btnFlex, isRunning && styles.btnStop]}
              onPress={toggleDetection}
              activeOpacity={0.8}>
              <Text style={styles.btnText}>
                {isRunning ? '⏹ 停止' : '▶ 开始'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.btnCamera, styles.btnCameraCompact]}
              onPress={toggleCamera}
              activeOpacity={0.8}>
              <Text style={styles.btnCameraText}>
                {cameraPos === 'back' ? '🤳' : '📷'}
              </Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

      {/* Debug console (portrait: bottom sheet) */}
      {showDebug && <DebugConsole logs={debugLogs} maxHeight={180} />}
    </View>
  );
};

// ─── Styles ──────────────────────────────────────────────────────────────────

const SIDEBAR_WIDTH = 220;

const styles = StyleSheet.create({
  // ── Layout roots
  landscapeRoot: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#0a0a0f',
  },
  portraitRoot: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: '#0a0a0f',
  },
  center: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0a0f',
    gap: 16,
  },
  gateText: {
    color: '#e2e8f0',
    fontSize: 17,
    fontWeight: '500',
  },

  // ── Sidebar (landscape)
  sidebar: {
    width: SIDEBAR_WIDTH,
    backgroundColor: '#111118',
    borderRightWidth: 1,
    borderRightColor: '#1e1e2e',
  },
  sidebarContent: {
    padding: 14,
    paddingTop: 52, // safe area top
    flexGrow: 1,
  },

  // ── Portrait header
  portraitHeader: {
    paddingTop: 52,
    paddingHorizontal: 14,
    paddingBottom: 6,
    backgroundColor: '#111118',
    borderBottomWidth: 1,
    borderBottomColor: '#1e1e2e',
  },

  // ── Camera
  cameraContainer: {
    flex: 1,
    overflow: 'hidden',
    backgroundColor: '#000',
  },

  // ── Portrait controls bar
  portraitControls: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: '#111118',
    borderTopWidth: 1,
    borderTopColor: '#1e1e2e',
  },
  portraitBtnRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
  },
  btnFlex: {
    flex: 1,
  },
  btnCameraCompact: {
    minWidth: 48,
    paddingHorizontal: 0,
  },

  // ── Typography
  titleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
    color: '#f1f5f9',
    letterSpacing: 0.3,
  },
  fpsChip: {
    fontSize: 11,
    fontWeight: '700',
    color: '#4ade80',
    backgroundColor: 'rgba(74,222,128,0.12)',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
  },
  statusText: {
    color: '#94a3b8',
    fontSize: 11,
    marginBottom: 4,
    lineHeight: 16,
  },
  sectionLabel: {
    color: '#64748b',
    fontSize: 11,
    fontWeight: '600',
    marginBottom: 2,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },

  // ── Divider
  divider: {
    height: 1,
    backgroundColor: '#1e1e2e',
    marginVertical: 10,
  },

  // ── Buttons
  btnPrimary: {
    backgroundColor: '#22c55e',
    paddingHorizontal: 16,
    paddingVertical: 11,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 8,
  },
  btnStop: {
    backgroundColor: '#ef4444',
  },
  btnText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
  },
  btnCamera: {
    backgroundColor: '#1e293b',
    borderWidth: 1,
    borderColor: '#334155',
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 4,
  },
  btnCameraText: {
    color: '#94a3b8',
    fontWeight: '600',
    fontSize: 13,
  },

  // ── Loading
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 4,
  },
  loadingText: {
    color: '#64748b',
    fontSize: 13,
  },

  // ── Debug toggle
  debugToggleRow: {
    paddingVertical: 2,
    marginBottom: 6,
  },
  debugToggleText: {
    color: '#7c3aed',
    fontSize: 12,
    fontWeight: '600',
  },

  // ── FPS badge on camera
  fpsBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  fpsBadgeText: {
    color: '#4ade80',
    fontSize: 12,
    fontWeight: '700',
  },
});
