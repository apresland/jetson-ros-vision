#include <cstdlib>
#include <ctime>
#include <vector>
#include <map>
#include "cudamappedmemory.h"
#include "overlay.h"

// from overlay.cu
cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, 
	Network::Detection* detections, uint32_t numDetections, float4* colors,
	std::map<int,int> &mapping);

Overlay::Overlay(rclcpp::Node *node)
	:  node_(node) {
	mClassColors[0]   = NULL; // cpu ptr
	mClassColors[1]   = NULL; // gpu ptr
}

Overlay::~Overlay() {

	if( mClassColors != NULL )
	{
		cudaFreeHost(mClassColors[0]);
		
		mClassColors[0] = NULL;
		mClassColors[1] = NULL;
	}	   
}

bool Overlay::Initialize() {

	std::vector<int64_t> targets = node_->get_parameter("target_classes").as_integer_array();
	const uint32_t numClasses = targets.size();

	if ( ! cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Initialize -- failed to allocate CUDA buffer");
		return false;
			}

	std::srand(std::time(nullptr));
	for (int n = 0 ; n < targets.size(); ++n) {
		unsigned int rgba = std::rand();
		mClassColors[0][n*4+0] = (float)(std::rand() % 256);	// r
		mClassColors[0][n*4+1] = (float)(std::rand() % 256);	// g
		mClassColors[0][n*4+2] = (float)(std::rand() % 256);	// b
		mClassColors[0][n*4+3] = 75.0f;	// a
	}

	color_dictionary.clear();
	for( auto& target : targets ) {
		int idx = std::distance(targets.begin(), std::find(targets.begin(), targets.end(), target));
		color_dictionary.insert({target, idx});
	}

	font_ = new Details(node_);
	font_->init();

	return true;
}

bool Overlay::Render( uchar3* input, uchar3* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections) {

	const size_t size  =(width * height * sizeof(uchar3) * 8) / 8;

	if( cudaSuccess != cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice) ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Render -- failed to copy to CUDA buffer");
		return false;
			}

	if( cudaSuccess != cudaDetectionOverlay(input, output, width, height, detections, numDetections, (float4*)mClassColors[1], color_dictionary)) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Render -- failed to overlay detections");	
		return false;
			}

	if ( cudaSuccess != font_->cudaDetectionLabelOverlay(output, width, height, detections, numDetections) ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Render -- failed to overlay labels");	
		return false;		
			}

	return true;
}