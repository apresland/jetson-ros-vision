#pragma once

#include <memory>
#include <NvInfer.h>
#include "cuda_runtime.h"

constexpr size_t MAX_WORKSPACE_SIZE = 32 << 20;
constexpr char* model_path = "data/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff";
constexpr char* cached_model_path = "data/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff.engine";

class InferenceEngine
{

    public:
    static nvinfer1::ICudaEngine* Create();

    private:
    static nvinfer1::ICudaEngine* LoadFromCache(const char* cache_path);

    private:
    static nvinfer1::ICudaEngine* LoadFromUFF(const char* file_path);

    private:
    static nvinfer1::IBuilder* CreateBuilder();

    private:
    static void Serialize(nvinfer1::ICudaEngine* engine);

    private:
    static size_t fileSize( const std::string& path );
};