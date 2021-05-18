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
    Network(rclcpp::Node *node);

    public:
    static Network* Create(
        rclcpp::Node *node);
    
    void LoadNetwork();

    int Detect(
        uchar3* img_data,
        uint32_t img_width,
        uint32_t img_height );

    LayerBinding RegisterBinding(
        const std::string& input);

    void CreateBindings(
        nvinfer1::ICudaEngine* engine);

    private:
    rclcpp::Node *node_;

	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	void**  bindings_;
	
    LayerBinding input_binding_;
    LayerBinding output_binding_;
    LayerBinding output_count_binding_;
};