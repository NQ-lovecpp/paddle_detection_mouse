/**
 * MouseDetector.mm
 *
 * Full native inference pipeline for PicoDet-S 320×320 mouse detector.
 * Does everything in one native round-trip:
 *   image file → resize 320×320 → normalize → ORT C API + CoreML → NMS → detection boxes
 *
 * Advantages over the JS approach:
 *  - No 1.2 MB Float32 bridge transfer per frame
 *  - Proper CoreML EP via ORT C API (not the RN JS wrapper)
 *  - All post-processing (NMS) runs natively
 *  - Only tiny JSON array of boxes crosses the RN bridge
 */

#import <React/RCTBridgeModule.h>
#import <UIKit/UIKit.h>
#import <Accelerate/Accelerate.h>
#import <onnxruntime/onnxruntime_c_api.h>
#import <onnxruntime/coreml_provider_factory.h>
#include <vector>
#include <algorithm>

// ─── PicoDet-S 320×320 constants ────────────────────────────────────────────

static const int   kInputSize        = 320;
static const int   kNumAnchors       = 2125;   // 40×40 + 20×20 + 10×10 + 5×5
static const int   kNumClasses       = 2;
static const float kNmsIouThreshold  = 0.5f;

// ImageNet normalization (matches PaddleDetection preprocessing)
static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

// ─── ORT error check helper ──────────────────────────────────────────────────

#define ORT_CHECK(status) do { \
  if (status) { \
    const char* _msg = _api->GetErrorMessage(status); \
    NSString* _errStr = [NSString stringWithUTF8String:_msg]; \
    _api->ReleaseStatus(status); \
    reject(@"E_ORT", _errStr, nil); \
    return; \
  } \
} while(0)

// ─── Candidate struct for NMS ────────────────────────────────────────────────

struct Candidate {
  float x1, y1, x2, y2, score;
  int   classId;
};

// ─── Interface ───────────────────────────────────────────────────────────────

@interface MouseDetector : NSObject <RCTBridgeModule>
@end

// ─── Implementation ──────────────────────────────────────────────────────────

@implementation MouseDetector {
  const OrtApi*   _api;
  OrtEnv*         _env;
  OrtSession*     _session;
  OrtMemoryInfo*  _memInfo;
  NSString*       _labels[kNumClasses];
}

RCT_EXPORT_MODULE()

- (instancetype)init {
  if (self = [super init]) {
    _api      = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    _env      = nullptr;
    _session  = nullptr;
    _memInfo  = nullptr;
    _labels[0] = @"mouse";
    _labels[1] = @"other";
  }
  return self;
}

// ── initialize(modelPath) ─────────────────────────────────────────────────────

RCT_EXPORT_METHOD(initialize:(NSString*)modelPath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    OrtStatus* status;

    // ORT environment (created once)
    if (!self->_env) {
      status = self->_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "MouseDetector", &self->_env);
      ORT_CHECK(status);
    }

    // Session options
    OrtSessionOptions* opts;
    status = self->_api->CreateSessionOptions(&opts);
    ORT_CHECK(status);

    self->_api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

    // CoreML EP — MLProgram format for better ANE utilisation on iOS 15+.
    // Ops not supported by CoreML are automatically delegated to CPU.
    OrtSessionOptionsAppendExecutionProvider_CoreML(opts, COREML_FLAG_CREATE_MLPROGRAM);

    // Create inference session
    const char* path = [modelPath UTF8String];
    status = self->_api->CreateSession(self->_env, path, opts, &self->_session);
    self->_api->ReleaseSessionOptions(opts);
    ORT_CHECK(status);

    // Reusable CPU memory info
    if (!self->_memInfo) {
      status = self->_api->CreateMemoryInfo(
        "Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &self->_memInfo);
      ORT_CHECK(status);
    }

    resolve(@"ok");
  });
}

// ── detect(imagePath, threshold) ──────────────────────────────────────────────

RCT_EXPORT_METHOD(detect:(NSString*)imagePath
                  threshold:(float)threshold
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    if (!self->_session) {
      reject(@"E_NOT_INIT", @"Model not initialized — call initialize() first", nil);
      return;
    }

    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();

    // ── 1. Load image ────────────────────────────────────────────────────────
    NSURL* url = [imagePath hasPrefix:@"file://"]
      ? [NSURL URLWithString:imagePath]
      : [NSURL fileURLWithPath:imagePath];

    NSData*  imgData = [NSData dataWithContentsOfURL:url];
    UIImage* uiImage = imgData ? [UIImage imageWithData:imgData] : nil;
    if (!uiImage || !uiImage.CGImage) {
      reject(@"E_IMG", @"Cannot load / decode image", nil);
      return;
    }

    int origW = (int)(uiImage.size.width  * uiImage.scale);
    int origH = (int)(uiImage.size.height * uiImage.scale);

    // ── 2. Resize to 320×320 via CoreGraphics ────────────────────────────────
    const int sz = kInputSize;
    std::vector<uint8_t> pixels(sz * sz * 4, 0);

    CGColorSpaceRef cs  = CGColorSpaceCreateDeviceRGB();
    CGContextRef    ctx = CGBitmapContextCreate(
      pixels.data(), sz, sz, 8, sz * 4, cs,
      kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    CGColorSpaceRelease(cs);

    if (!ctx) {
      reject(@"E_CTX", @"CGBitmapContextCreate failed", nil);
      return;
    }
    CGContextSetInterpolationQuality(ctx, kCGInterpolationMedium);
    CGContextDrawImage(ctx, CGRectMake(0, 0, sz, sz), uiImage.CGImage);
    CGContextRelease(ctx);

    CFAbsoluteTime tResize = CFAbsoluteTimeGetCurrent();

    // ── 3. Normalize + CHW permute (Accelerate) ──────────────────────────────
    const int planeSize = sz * sz;
    std::vector<float> chw(3 * planeSize);
    float* dst = chw.data();

    // Extract each channel with stride-4 and convert UInt8→Float in one vDSP pass
    vDSP_vfltu8(pixels.data() + 0, 4, dst,                1, (vDSP_Length)planeSize);
    vDSP_vfltu8(pixels.data() + 1, 4, dst + planeSize,    1, (vDSP_Length)planeSize);
    vDSP_vfltu8(pixels.data() + 2, 4, dst + 2 * planeSize, 1, (vDSP_Length)planeSize);

    // Normalize: pixel = pixel / (255 * std) - mean / std
    for (int c = 0; c < 3; c++) {
      float scale  = 1.0f / (255.0f * kStd[c]);
      float offset = -kMean[c] / kStd[c];
      vDSP_vsmsa(dst + c * planeSize, 1, &scale, &offset,
                 dst + c * planeSize, 1, (vDSP_Length)planeSize);
    }

    CFAbsoluteTime tNorm = CFAbsoluteTimeGetCurrent();

    // ── 4. ORT Inference ─────────────────────────────────────────────────────
    OrtStatus* status;

    int64_t inputShape[] = {1, 3, sz, sz};
    OrtValue* inputTensor = nullptr;
    status = self->_api->CreateTensorWithDataAsOrtValue(
      self->_memInfo,
      chw.data(), chw.size() * sizeof(float),
      inputShape, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &inputTensor);

    if (status) {
      NSString* msg = [NSString stringWithUTF8String:self->_api->GetErrorMessage(status)];
      self->_api->ReleaseStatus(status);
      reject(@"E_TENSOR", msg, nil);
      return;
    }

    const char* inputNames[]  = {"image"};
    const char* outputNames[] = {"boxes", "scores"};
    OrtValue*   outputs[2]    = {nullptr, nullptr};

    status = self->_api->Run(
      self->_session, nullptr,
      inputNames, (const OrtValue* const*)&inputTensor, 1,
      outputNames, 2, outputs);

    self->_api->ReleaseValue(inputTensor);

    if (status) {
      NSString* msg = [NSString stringWithUTF8String:self->_api->GetErrorMessage(status)];
      self->_api->ReleaseStatus(status);
      reject(@"E_RUN", msg, nil);
      return;
    }

    CFAbsoluteTime tInfer = CFAbsoluteTimeGetCurrent();

    // ── 5. Parse output tensors ──────────────────────────────────────────────
    // boxes  shape: [1, 2125, 4]  — xyxy coords in 320px input space
    // scores shape: [1, 2, 2125]  — class scores (channels first)

    float* boxData   = nullptr;
    float* scoreData = nullptr;
    self->_api->GetTensorMutableData(outputs[0], (void**)&boxData);
    self->_api->GetTensorMutableData(outputs[1], (void**)&scoreData);

    // Convert scale: 320px → original image coordinates
    float invScaleX = (float)origW / sz;
    float invScaleY = (float)origH / sz;

    std::vector<Candidate> candidates;
    candidates.reserve(64);

    for (int i = 0; i < kNumAnchors; i++) {
      float bestScore = threshold;
      int   bestClass = -1;
      for (int c = 0; c < kNumClasses; c++) {
        float s = scoreData[c * kNumAnchors + i];
        if (s > bestScore) { bestScore = s; bestClass = c; }
      }
      if (bestClass < 0) { continue; }

      candidates.push_back({
        boxData[i * 4 + 0] * invScaleX,
        boxData[i * 4 + 1] * invScaleY,
        boxData[i * 4 + 2] * invScaleX,
        boxData[i * 4 + 3] * invScaleY,
        bestScore,
        bestClass
      });
    }

    self->_api->ReleaseValue(outputs[0]);
    self->_api->ReleaseValue(outputs[1]);

    // ── 6. Per-class greedy NMS ──────────────────────────────────────────────
    NSMutableArray<NSDictionary*>* detections = [NSMutableArray array];

    for (int cls = 0; cls < kNumClasses; cls++) {
      std::vector<Candidate*> cc;
      for (auto& cand : candidates) {
        if (cand.classId == cls) { cc.push_back(&cand); }
      }
      std::sort(cc.begin(), cc.end(), [](const Candidate* a, const Candidate* b) {
        return a->score > b->score;
      });

      std::vector<bool> suppressed(cc.size(), false);
      for (size_t i = 0; i < cc.size(); i++) {
        if (suppressed[i]) { continue; }
        const Candidate* a = cc[i];
        [detections addObject:@{
          @"classId":    @(cls),
          @"className":  self->_labels[cls],
          @"confidence": @(a->score),
          @"x1": @(a->x1), @"y1": @(a->y1),
          @"x2": @(a->x2), @"y2": @(a->y2),
        }];
        for (size_t j = i + 1; j < cc.size(); j++) {
          if (suppressed[j]) { continue; }
          const Candidate* b = cc[j];
          float ix1 = MAX(a->x1, b->x1), iy1 = MAX(a->y1, b->y1);
          float ix2 = MIN(a->x2, b->x2), iy2 = MIN(a->y2, b->y2);
          float iw = MAX(0.0f, ix2 - ix1);
          float ih = MAX(0.0f, iy2 - iy1);
          float inter  = iw * ih;
          float aArea  = (a->x2 - a->x1) * (a->y2 - a->y1);
          float bArea  = (b->x2 - b->x1) * (b->y2 - b->y1);
          float unionA = aArea + bArea - inter;
          if (unionA > 0 && inter / unionA > kNmsIouThreshold) {
            suppressed[j] = true;
          }
        }
      }
    }

    CFAbsoluteTime tEnd = CFAbsoluteTimeGetCurrent();

    NSString* timings = [NSString stringWithFormat:
      @"resize=%.0fms norm=%.0fms infer=%.0fms nms=%.0fms total=%.0fms",
      (tResize - t0)    * 1000.0,
      (tNorm   - tResize) * 1000.0,
      (tInfer  - tNorm)   * 1000.0,
      (tEnd    - tInfer)  * 1000.0,
      (tEnd    - t0)      * 1000.0];

    resolve(@{
      @"detections":     detections,
      @"originalWidth":  @(origW),
      @"originalHeight": @(origH),
      @"timings":        timings,
    });
  });
}

+ (BOOL)requiresMainQueueSetup { return NO; }

@end
