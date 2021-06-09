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

Network::Network(rclcpp::Node *node)
 : node_(node)
 {
	this->engine_   = nullptr;
	this->context_  = nullptr;
	this->bindings_ = nullptr;

	mDetectionSets[0] = NULL; // cpu ptr
	mDetectionSets[1] = NULL; // gpu ptr
	mDetectionSet     = 0;
	mMaxDetections    = 0;
 }

Network::~Network() {

	if( mDetectionSets != NULL )
	{
		cudaFreeHost(mDetectionSets[0]);
		
		mDetectionSets[0] = NULL;
		mDetectionSets[1] = NULL;
	}
}

bool Network::Initialize() {
    assert( initLibNvInferPlugins(&gLogger, "") 
		&& "Network::Initialize -- failed to initialize NVIDIA infererence plugins");

	return LoadNetwork();
} 

bool Network::RegisterBinding( Network::LayerBinding &binding, const std::string& name)
{
	if( nullptr == bindings_ )
	{
		RCLCPP_ERROR(node_->get_logger(),
			"cannot register binding - no memory allocated operation");
		return false;
	}

	const int index = engine_->getBindingIndex(name.c_str());

	if( index < 0 )
	{
		RCLCPP_ERROR(node_->get_logger(),
			"failed to get tensor binding index from CUDA engine");
		return false;
	}

	nvinfer1::Dims dimensions = 
		validateDims(engine_->getBindingDimensions(index));

	size_t size = kMaxBatchSize 
		* DIMS_C(dimensions)
		* DIMS_H(dimensions)
		* DIMS_W(dimensions)
		* sizeof(float);

	void* CPU  = nullptr;
	void* CUDA = nullptr;

	if( ! cudaAllocMapped((void**)&CPU, (void**)&CUDA, size) )
	{
		RCLCPP_ERROR(node_->get_logger(),
			"Network::RegisterBinding -- failed to allocate %zu bytes", size);
		return false;
	}

	binding.name 	= name;
	binding.index 	= index;
	binding.size 	= size;	
	
	DIMS_W(binding.dims) = DIMS_W(dimensions);
	DIMS_H(binding.dims) = DIMS_H(dimensions);
	DIMS_C(binding.dims) = DIMS_C(dimensions);

	binding.CPU  = (float*)CPU;
	binding.CUDA = (float*)CUDA;

	bindings_[binding.index] = binding.CUDA;

	RCLCPP_INFO(node_->get_logger(),
		"Network::RegisterBinding -- registered  binding for tensor, %s", name.c_str());

	return true;
}

bool Network::CreateBindings(nvinfer1::ICudaEngine* engine)
{
	const int bytes = engine->getNbBindings() * sizeof(void*);
	bindings_ = (void**)malloc(bytes);
	memset(bindings_, 0, bytes);

	if ( ! RegisterBinding(input_binding_, "Input") ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Network::CreatBindings - failed to register layer [Input]");
		return false;
			}

	if ( ! RegisterBinding(output_binding_, "NMS") ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Network::CreatBindings - failed to register layer [NMS]");
		return false;
			}

	if ( ! RegisterBinding(output_count_binding_, "NMS_1") ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Network::CreatBindings - failed to register layer [NMS_1]");
		return false;
			}

	return true;
}

bool Network::LoadNetwork()
{
	/*
	 * setup TensorRT engine
	 */
	engine_ = InferenceEngine::Create();
	if ( nullptr == engine_ ) {
		RCLCPP_INFO(node_->get_logger(),
			"Network::LoadNetwork -- finished creating engine");
		return false;
			}

	/*
	 * setup TensorRT execution context
	 */
	context_ = engine_->createExecutionContext();
	if ( nullptr == context_ ) {
		RCLCPP_INFO(node_->get_logger(),
			"Network::LoadNetwork -- created execution context");
		return false;
			}

	/*
	 * setup tensor bindings
	 */
	if ( ! CreateBindings(engine_) ) {
    	RCLCPP_ERROR(node_->get_logger(),
			"Network::LoadNetwork -- failed to create bindings");
		return false;
			}

	/*
	 * setup CUDA dections buffer
	 */
	if ( ! AllocDetections() ) {
		RCLCPP_ERROR(node_->get_logger(),
			"CUDA detection buffer allocation failed");
		return false;
			}

	return true;
}

bool Network::AllocDetections() {

	mMaxDetections = DIMS_H(output_binding_.dims) * DIMS_C(output_binding_.dims);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;	
}

bool Network::Detect( uchar3* img_data, uint32_t img_width, uint32_t img_height, 
	Detection** detections, uint32_t& numDetections) {

	if( cudaSuccess != cudaTensorNormBGR(
		img_data, img_width, img_height, 
		input_binding_.CUDA, DIMS_W(input_binding_.dims), DIMS_H(input_binding_.dims),
		make_float2(-1.0f, 1.0f), NULL))
	{
		RCLCPP_ERROR(node_->get_logger(), "cudaTensorNormBGR() failed");
		return false;
	}

	if( ! context_->execute(1, bindings_) )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to execute TensorRT context");
		return false;
	}

	Detection* detection = mDetectionSets[0] + mDetectionSet * mMaxDetections;

	if( detections != NULL )
		*detections = detection;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;


	uint32_t numDets = 0;
	const int rawDetections = *(int*)output_count_binding_.CPU;
	const int rawParameters = DIMS_W(output_binding_.dims);

	for( int n=0; n < rawDetections; n++ )
	{
		float* object_data = output_binding_.CPU + n * rawParameters;

		if ( ! contains( node_->get_parameter("target_classes").as_integer_array() /*target_classes_*/, (int)object_data[1]))
			continue;

		if( object_data[2] < 0.66 /** detection threshold */)
			continue;	

		detection[numDets].ClassId 	  = (uint32_t)object_data[1];
		detection[numDets].Confidence = object_data[2];
		detection[numDets].Left       = object_data[3] * img_width;
		detection[numDets].Top        = object_data[4] * img_height;
		detection[numDets].Right      = object_data[5] * img_width;
		detection[numDets].Bottom	  = object_data[6] * img_height;

		++numDets;
	}

	numDetections = numDets;

	return true;
}