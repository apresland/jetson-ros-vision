#include "tensorconvert.h"

inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }

// gpuTensorNorm
template<typename T, bool isBGR>
__global__ void gpuTensorNorm( float2 scale, T* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = rgb.x * multiplier + min_value;
	output[n * 1 + m] = rgb.y * multiplier + min_value;
	output[n * 2 + m] = rgb.z * multiplier + min_value;
}

// cudaTensorNormBGR
cudaError_t cudaTensorNormBGR( void* input, size_t inputWidth, size_t inputHeight,
    float* output, size_t outputWidth, size_t outputHeight,
    const float2& range, cudaStream_t stream )
{
    return launchTensorNorm<true>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, range, stream);
}

template<bool isBGR>
cudaError_t launchTensorNorm( void* input, size_t inputWidth, size_t inputHeight,
						float* output, size_t outputWidth, size_t outputHeight, 
						const float2& range, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuTensorNorm<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar3*)input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);


	return cudaGetLastError();
}
