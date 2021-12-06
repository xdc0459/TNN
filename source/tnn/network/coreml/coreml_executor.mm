#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "coreml_executor.h"

#include <fstream>
#include <iostream>

namespace {

NSURL* createTemporaryFile() {
    NSURL* temporaryDirectoryURL = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
    NSString* temporaryFilename = [[NSProcessInfo processInfo] globallyUniqueString];
    NSURL* temporaryFileURL = [temporaryDirectoryURL URLByAppendingPathComponent:temporaryFilename];
    return temporaryFileURL;
    }
}

@implementation CoreMLExecutor

- (TNN_NS::Status)cleanup {
    NSError* error = nil;
    [[NSFileManager defaultManager] removeItemAtPath:_mlModelFilePath error:&error];
    if (error != nil) {
        LOGE("Failed cleaning up model: %@.\n", [error localizedDescription]);
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_CLEAN_ERROR, "Error: Failed cleaning up model.");
    }
    [[NSFileManager defaultManager] removeItemAtPath:_compiledModelFilePath error:&error];
    if (error != nil) {
        LOGE("Failed cleaning up compiled model: %@.\n", [error localizedDescription]);
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_CLEAN_ERROR, "Error: Failed cleaning up compiled model.");
    }
    return TNN_NS::TNN_OK;
}

- (NSURL*)saveModel:(CoreML__Specification__Model*)model {
    NSURL* modelUrl = createTemporaryFile();
    NSString* modelPath = [modelUrl path];
    
    if (model->specificationversion == 3) {
        _coreMlVersion = 2;
    } else if (model->specificationversion == 4) {
        _coreMlVersion = 3;
    } else {
        LOGE("Only Core ML models with specification version 3 or 4 are supported.\n");
        return nil;
    }
    size_t modelSize = core_ml__specification__model__get_packed_size(model);
    std::unique_ptr<uint8_t> writeBuffer(new uint8_t[modelSize]);
    core_ml__specification__model__pack(model, writeBuffer.get());
    std::ofstream file_stream([modelPath UTF8String], std::ios::out | std::ios::binary);
    if (!file_stream || !file_stream.is_open() || !file_stream.good()) {
        file_stream.close();
        LOGE("CoreML models file can not be written.\n");
    }
    const char* ptr = reinterpret_cast<const char*>(writeBuffer.get());
    if (ptr) {
        file_stream.write(ptr, modelSize);
    } else {
        LOGE("CoreML models file is empty.\n");
    }
    file_stream.close();
    return modelUrl;
}

- (TNN_NS::Status)build:(NSURL*)modelUrl {
    NSError* error = nil;
    NSURL* compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];
    if (error != nil) {
        LOGE("Error compiling model %@.\n", [error localizedDescription]);
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed compiling model.");
    }
    _mlModelFilePath = [modelUrl path];
    _compiledModelFilePath = [compileUrl path];

    if (@available(iOS 12.0, *)) {
        MLModelConfiguration* config = [MLModelConfiguration alloc];
        config.computeUnits = MLComputeUnitsAll;
        _model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:&error];
    } else {
        _model = [MLModel modelWithContentsOfURL:compileUrl error:&error];
    }
    if (error != NULL) {
        LOGE("Error Creating MLModel %@.\n", [error localizedDescription]);
        return TNN_NS::Status(TNN_NS::TNNERR_ANE_COMPILE_MODEL_ERROR, "Error: Failed Creating MLModel.");
    }
    return TNN_NS::TNN_OK;
}

- (NSString*) getMLModelFilePath{
    return _compiledModelFilePath;
}

@end
