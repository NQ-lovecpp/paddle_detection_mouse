#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(ImagePreprocessor, NSObject)

RCT_EXTERN_METHOD(preprocess:(NSString *)filePath
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
