#ifndef __CUDA_MAPPED_MEMORY_H_
#define __CUDA_MAPPED_MEMORY_H_

#include <cstring>
#include <cuda_runtime.h>

inline bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( ! cpuPtr || ! gpuPtr || size == 0 )
		return false;

	if( cudaSuccess != cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped) )
		return false;

	if( cudaSuccess != cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0) )
		return false;

	memset(*cpuPtr, 0, size);
	//LogDebug(LOG_CUDA "cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
	return true;
}

inline bool cudaAllocMapped( void** ptr, size_t size )
{
	void* cpuPtr = NULL;
	void* gpuPtr = NULL;

	if( ! ptr || size == 0 )
		return false;

	if( ! cudaAllocMapped(&cpuPtr, &gpuPtr, size) )
		return false;

	if( cpuPtr != gpuPtr )
	{
		//LogError(LOG_CUDA "cudaAllocMapped() - addresses of CPU and GPU pointers don't match\n");
		return false;
	}

	*ptr = gpuPtr;
	return true;
}

#endif