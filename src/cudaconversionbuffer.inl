#ifndef __COLOR_CONVERSION_BUFFER_INLINE_H_
#define __COLOR_CONVERSION_BUFFER_INLINE_H_

#include "cudamappedmemory.h"
#include "cudaconversionbuffer.h"

CUDAColorConversionBuffer::CUDAColorConversionBuffer() {
	threaded_ = true;
	buffers_ = NULL;
	size_ = 0;
	read_index_ = 0;
	write_index_ = 0;
    read_once_ = false;
}

CUDAColorConversionBuffer::~CUDAColorConversionBuffer() {

	MFree();
	
	if( buffers_ != NULL ) {
		free(buffers_);
		buffers_ = NULL;
	}
}

inline bool CUDAColorConversionBuffer::Allocate(size_t size)
{
	if( size == size_ )
		return true;
	
	MFree();
	
	if( buffers_ != NULL )
	{
		free(buffers_);
		buffers_ = NULL;
	}

	if( buffers_ == NULL )
	{
		const size_t bufferListSize = number_of_buffers * sizeof(void*);
		buffers_ = (void**)malloc(bufferListSize);
		memset(buffers_, 0, bufferListSize);
	}

	for( uint32_t n=0; n < number_of_buffers; n++ )
	{
        if( ! cudaAllocMapped(&buffers_[n], size) )
        {
            //LogError("VideoBuffer -- failed to allocate zero-copy buffer of %zu bytes\n", size);
            return false;
        }

	}
		
	//LogVerbose("RingBuffer -- allocated %u buffers (%zu bytes each, %zu bytes total)\n", numBuffers, size, size * numBuffers);
	size_ = size;
	
	return true;
}

inline void CUDAColorConversionBuffer::MFree()
{
	if( !buffers_ )
		return;
	
	for( uint32_t n=0; n < number_of_buffers; n++ )
	{
		if(cudaSuccess != cudaFreeHost(buffers_[n]))
		{
			// log error
		}
		buffers_[n] = NULL;
	}
}

inline void* CUDAColorConversionBuffer::Peek()
{
	if( !buffers_ )
	{
		//LogError("RingBuffer::Peek() -- error, must call RingBuffer::Alloc() first\n");
		return NULL;
	}

	if( threaded_ )
		mutex_.lock();

	int bufferIndex = (write_index_ + 1) % number_of_buffers;
	
	if( threaded_ )
		mutex_.unlock();

	if( bufferIndex < 0 )
	{
		//LogError("RingBuffer::Peek() -- error, invalid flags (must be Write or Read flags)\n");
		return NULL;
	}

	return buffers_[bufferIndex];
}

inline void* CUDAColorConversionBuffer::Read()
{
	if( !buffers_ )
	{
		//LogError("RingBuffer::Next() -- error, must call RingBuffer::Alloc() first\n");
		return NULL;
	}

    if( read_once_ )
	{
		if( threaded_ )
			mutex_.unlock();

		return NULL;
	}

	int index = -1;	
    read_index_ = (read_index_ + 1) % number_of_buffers;
    index = read_index_;
    read_once_   = true;

	if( threaded_ )
		mutex_.unlock();

	if( index < 0 )
	{
		//LogError("RingBuffer::Next() -- error, invalid flags (must be Write or Read flags)\n");
		return NULL;
	}

	return buffers_[index];
}

inline void* CUDAColorConversionBuffer::Write()
{
	if( !buffers_ )
	{
		//LogError("RingBuffer::Next() -- error, must call RingBuffer::Alloc() first\n");
		return NULL;
	}

	if( threaded_ )
		mutex_.lock();

	int index = -1;
    write_index_ = (write_index_ + 1) % number_of_buffers;
    index  = write_index_;
    read_once_ = false;

	if( threaded_ )
		mutex_.unlock();

	if( index < 0 )
	{
		//LogError("RingBuffer::Next() -- error, invalid flags (must be Write or Read flags)\n");
		return NULL;
	}

	return buffers_[index];
}

#endif