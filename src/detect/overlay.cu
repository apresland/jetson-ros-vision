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

	const float alpha = color.w / 255.0f;
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
	float4* colors)
{
	if( !input || !output || width == 0 || height == 0 )
		return cudaErrorInvalidValue;
			
	for( int n=0; n < numDetections; n++ )
	{
		const int boxWidth = (int)detections[n].Width();
		const int boxHeight = (int)detections[n].Height();

		const dim3 blockDim(8, 8);
		const dim3 gridDim(
			iDivUp(boxWidth, blockDim.x),
			iDivUp(boxHeight,blockDim.y));

		gpuDetectionOverlayBox<T><<<gridDim, blockDim>>>(input, output, width, height, (int)detections[n].Left, (int)detections[n].Top, boxWidth, boxHeight, colors[1]); 
	}

	return cudaGetLastError();
}

cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections, float4* colors)
{
    return launchDetectionOverlay<uchar3>((uchar3*)input, (uchar3*)output, width, height, detections, numDetections, colors); 
}