#pragma once

#include <memory>
#include <map>
#include <NvInfer.h>
#include "cuda_runtime.h"
#include "rclcpp/rclcpp.hpp"

#include "engine.h"

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

constexpr uint32_t kMaxBatchSize = 1;

class Network {

    struct LayerBinding
    {
        std::string name;
        uint32_t index;
        uint32_t size;
        nvinfer1::Dims3 dims;
        float* CPU;
        float* CUDA;
    };

    public:
	struct Detection
	{
		float Confidence;
		float Left;
		float Right;
		float Top;
		float Bottom;


		/**< Calculate the width of the object */
		inline float Width() const { return Right - Left; }

		/**< Calculate the height of the object */
		inline float Height() const	{ return Bottom - Top; }
    };

    public:
    Network(rclcpp::Node *node);
    ~Network();

    public:
    bool Initialize();
    bool LoadNetwork();

    public:
    bool Detect(
        uchar3* img_data,
        uint32_t img_width,
        uint32_t img_height,
        Detection** detections,
        uint32_t& numDetections);

    bool RegisterBinding(
        LayerBinding& binding,
        const std::string& name);

    bool CreateBindings(
        nvinfer1::ICudaEngine* engine);

    bool AllocDetections();

    bool Overlay( 
        void* input, void* output, uint32_t width, uint32_t height,
        Detection* detections, uint32_t numDetections);

    private:
    rclcpp::Node *node_;

	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	void**  bindings_;
	
    LayerBinding input_binding_;
    LayerBinding output_binding_;
    LayerBinding output_count_binding_;

    private:
    Detection* mDetectionSets[2];
	uint32_t   mDetectionSet;
	uint32_t   mMaxDetections;

	static const uint32_t mNumDetectionSets = 16;
};