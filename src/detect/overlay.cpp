#include "cudamappedmemory.h"
#include "overlay.h"

// from overlay.cu
cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections, float4* colors);

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

	if ( ! cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], sizeof(float4)) ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Initialize -- failed to allocate CUDA buffer");
		return false;
			}

	mClassColors[0][4+0] = 0.0f;	// r
	mClassColors[0][4+1] = 255.0f;	// g
	mClassColors[0][4+2] = 175.0f;	// b
	mClassColors[0][4+3] = 75.0f;	// a

	return true;
}

bool Overlay::Render( uchar3* input, uchar3* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections) {

	const size_t size  =(width * height * sizeof(uchar3) * 8) / 8;

	if( cudaSuccess != cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice) ) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Render -- failed to copy to CUDA buffer");
		return false;
			}

	if( cudaSuccess != cudaDetectionOverlay(input, output, width, height, detections, numDetections, (float4*)mClassColors[1])) {
		RCLCPP_ERROR(node_->get_logger(),
			"Overlay::Render -- failed to overlay detections");	
		return false;
			}
			
	return true;
}