#include <memory>

//#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"

#include "cudamappedmemory.h"
#include "tensorconvert.h"
#include "network.h"
#include "logger.h"

static inline nvinfer1::Dims validateDims( const nvinfer1::Dims& dims )
{
	if( dims.nbDims == nvinfer1::Dims::MAX_DIMS )
		return dims;
	
	nvinfer1::Dims dims_out = dims;

	// TRT doesn't set the higher dims, so make sure they are 1
	for( int n=dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
		dims_out.d[n] = 1;

	return dims_out;
}

Network* Network::Create(rclcpp::Node *node)
{
	const char* class_labels = "/home/andy/code/jetson-ros-vision/data/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt";

    assert( initLibNvInferPlugins(&gLogger, "") );

    Network* network = new Network(node);
    network->LoadNetwork();

	return network;
}  

Network::Network(rclcpp::Node *node)
 : node_(node)
 {
	this->engine_   = nullptr;
	this->context_  = nullptr;
	this->bindings_ = nullptr;
 }

Network::LayerBinding Network::RegisterBinding( const std::string& name)
{
	RCLCPP_INFO(node_->get_logger(), "registering binding for tensor, %s", name.c_str());

	if( nullptr == bindings_ )
	{
		RCLCPP_ERROR(node_->get_logger(), "cannot register binding - no memory allocated operation");
	}

	const int index = engine_->getBindingIndex(name.c_str());

	if( index < 0 )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to get tensor binding index from CUDA engine");
	}

	nvinfer1::Dims dimensions = validateDims(engine_->getBindingDimensions(index));
	size_t size = kMaxBatchSize * DIMS_C(dimensions) * DIMS_H(dimensions) * DIMS_W(dimensions) * sizeof(float);

	void* CPU  = nullptr;
	void* CUDA = nullptr;

	if( !cudaAllocMapped((void**)&CPU, (void**)&CUDA, size) )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to alloc CUDA mapped memory for tensor input, %zu bytes", size);
	}

	LayerBinding binding;

	binding.name 	= name;
	binding.index 	= index;
	binding.size 	= size;	
	
	DIMS_W(binding.dims) = DIMS_W(dimensions);
	DIMS_H(binding.dims) = DIMS_H(dimensions);
	DIMS_C(binding.dims) = DIMS_C(dimensions);

	binding.CPU  = (float*)CPU;
	binding.CUDA = (float*)CUDA;

	bindings_[binding.index] = binding.CUDA;

	return binding;
}

void Network::CreateBindings(nvinfer1::ICudaEngine* engine)
{
	const int bytes = engine->getNbBindings() * sizeof(void*);

	bindings_ = (void**)malloc(bytes);
	memset(bindings_, 0, bytes);

	this->input_binding_ 		= RegisterBinding("Input");
	this->output_binding_ 		= RegisterBinding("NMS");
	this->output_count_binding_ = RegisterBinding("NMS_1");
}

void Network::LoadNetwork()
{
	/*
	 * setup TensorRT engine
	 */
	engine_ = InferenceEngine::Create();
    RCLCPP_INFO(node_->get_logger(), "finished creating engine");

	/*
	 * setup TensorRT execution context
	 */
	context_ = engine_->createExecutionContext();
	RCLCPP_INFO(node_->get_logger(), "created execution context");

	/*
	 * setup tensor bindings
	 */
	this->CreateBindings(engine_);

    RCLCPP_INFO(node_->get_logger(), "finished defining bindings");
}

int Network::Detect( uchar3* img_data, uint32_t img_width, uint32_t img_height) {

	if( cudaSuccess != cudaTensorNormBGR(
		img_data, img_width, img_height, 
		input_binding_.CUDA, DIMS_W(input_binding_.dims), DIMS_H(input_binding_.dims),
		make_float2(-1.0f, 1.0f), NULL))
	{
		RCLCPP_ERROR(node_->get_logger(), "cudaTensorNormBGR() failed");
		return -1;
	}

	if( ! context_->execute(1, bindings_) )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to execute TensorRT context");
		return -1;
	}

	const int rawDetections = *(int*)output_count_binding_.CPU;
	const int rawParameters = DIMS_W(output_binding_.dims);

	for( int n=0; n < rawDetections; n++ )
	{
		float* object_data = output_binding_.CPU + n * rawParameters;

		if( object_data[2] < 0.5 /** coverage threshold */)
			continue;

		RCLCPP_INFO(node_->get_logger(), "Detection --- class: %i confidence %f:", (uint32_t)object_data[1], object_data[2]);
	}

	return 0;
}