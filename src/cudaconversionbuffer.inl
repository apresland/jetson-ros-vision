#ifndef __COLOR_CONVERSION_BUFFER_INLINE_H_
#define __COLOR_CONVERSION_BUFFER_INLINE_H_

#include "cudamappedmemory.h"
#include "cudaconversionbuffer.h"

CUDAColorConversionBuffer::CUDAColorConversionBuffer() {
	threaded_ = true;
	buffers_ = nullptr;
	size_ = 0;
	read_index_ = 0;
	write_index_ = 0;
    read_once_ = false;
}

CUDAColorConversionBuffer::~CUDAColorConversionBuffer() {

	MFree();
	
	if( buffers_ != nullptr ) {
		free(buffers_);
		buffers_ = nullptr;
	}
}

inline bool CUDAColorConversionBuffer::Allocate(size_t size)
{
	if( size == size_ )
		return true;
	
	MFree();
	
	if( buffers_ != nullptr )
	{
		free(buffers_);
		buffers_ = nullptr;
	}

	if( buffers_ == nullptr )
	{
		const size_t bufferListSize = number_of_buffers * sizeof(void*);
		buffers_ = (void**)malloc(bufferListSize);
		memset(buffers_, 0, bufferListSize);
	}

	for( uint32_t n=0; n < number_of_buffers; n++ )
	{
        if( ! cudaAllocMapped(&buffers_[n], size) )
        {

            return false;
        }

	}
		
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
		buffers_[n] = nullptr;
	}
}

inline void* CUDAColorConversionBuffer::Peek()
{
	if( !buffers_ )
	{
		return nullptr;
	}

	if( threaded_ )
		mutex_.lock();

	int bufferIndex = (write_index_ + 1) % number_of_buffers;
	
	if( threaded_ )
		mutex_.unlock();

	if( bufferIndex < 0 )
	{
		return nullptr;
	}

	return buffers_[bufferIndex];
}

inline void* CUDAColorConversionBuffer::Read()
{
	if( !buffers_ )
	{
		return nullptr;
	}

    if( read_once_ )
	{
		if( threaded_ )
			mutex_.unlock();

		return nullptr;
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
		return nullptr;
	}

	return buffers_[index];
}

inline void* CUDAColorConversionBuffer::Write()
{
	if( !buffers_ )
	{
		//LogError("RingBuffer::Next() -- error, must call RingBuffer::Alloc() first\n");
		return nullptr;
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
		return nullptr;
	}

	return buffers_[index];
}

#endif