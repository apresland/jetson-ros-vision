#include "overlay.h"

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

template<typename T>
__global__ void gpuDetectionOverlayBox( 
	T* input, T* output, 
	int imgWidth, int imgHeight, 
	int x0, int y0, 
	int boxWidth, int boxHeight, 
	const float4 color) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight )
		return;

	T px = input[ y * imgWidth + x ];

	float alpha = color.w / 255.0f;
	if (box_x < 240 && box_y < 30) {
		alpha = 1.0f;
	}
	if ( box_x < 2 || box_y < 2) {
		alpha = 1.0f;
	}
	if ( boxWidth - box_x < 2 || boxHeight - box_y < 2) {
		alpha = 1.0f;
	}

	const float ialph = 1.0f - alpha;

	px.x = alpha * color.x + ialph * px.x;
	px.y = alpha * color.y + ialph * px.y;
	px.z = alpha * color.z + ialph * px.z;
	
	output[y * imgWidth + x] = px;
}

template<typename T>
cudaError_t launchDetectionOverlay( 
	T* input, T* output, 
	uint32_t width, uint32_t height, 
	Network::Detection* detections, 
	uint32_t numDetections, 
	float4* colors,
	std::map<int,int> &colormap)
{
	if( !input || !output || width == 0 || height == 0 )
		return cudaErrorInvalidValue;
			
	for( int n=0; n < numDetections; n++ )
	{
		const int boxWidth = (int)detections[n].Width();
		const int boxHeight = (int)detections[n].Height();

		const float4 color = colors[
			colormap.find(detections[n].ClassId)
				->second
			];

		const dim3 blockDim(8, 8);
		const dim3 gridDim(
			iDivUp(boxWidth, blockDim.x),
			iDivUp(boxHeight,blockDim.y));

		gpuDetectionOverlayBox<T><<<gridDim, blockDim>>>(input, output, width, height, (int)detections[n].Left, (int)detections[n].Top, boxWidth, boxHeight, color); 
	}

	return cudaGetLastError();
}

cudaError_t cudaDetectionOverlay( 
	void* input, void* output, uint32_t width, uint32_t height, 
	Network::Detection* detections, uint32_t numDetections, float4* colors,
	std::map<int,int> &colormap)
{
    return launchDetectionOverlay<uchar3>((uchar3*)input, (uchar3*)output, width, height, detections, numDetections, colors, colormap); 
}