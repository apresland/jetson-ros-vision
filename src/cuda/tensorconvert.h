#ifndef __CUDA_TENSOR_PREPROCESSING_H__
#define __CUDA_TENSOR_PREPROCESSING_H__

cudaError_t cudaTensorNormBGR(
    void* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float2& range, cudaStream_t stream );

template<bool isBGR>
cudaError_t launchTensorNorm( void* input, size_t inputWidth, size_t inputHeight,
						float* output, size_t outputWidth, size_t outputHeight, 
						const float2& range, cudaStream_t stream );

#endif