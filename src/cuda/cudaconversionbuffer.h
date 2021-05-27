#ifndef __COLOR_CONVERSION_BUFFER_H_
#define __COLOR_CONVERSION_BUFFER_H_

#include <mutex>

class CUDAColorConversionBuffer {

uchar3** buffers_;
static const uint32_t number_of_buffers = 4;
size_t size_;

uint32_t read_index_;
uint32_t write_index_;
bool read_once_;

std::mutex mutex_;

public:
bool threaded_;

public:
inline CUDAColorConversionBuffer();
inline ~CUDAColorConversionBuffer();

public:
inline uchar3* Peek();
inline uchar3* Read();
inline uchar3* Write();

public:
inline bool Allocate( size_t size);
inline void MFree();

};

#include "cudaconversionbuffer.inl"

#endif
