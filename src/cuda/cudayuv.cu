/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

 #include "cudayuv.h"
 #include "cudavector.h"
 
 #define COLOR_COMPONENT_MASK            0x3FF
 #define COLOR_COMPONENT_BIT_SIZE        10
 
 #define FIXED_DECIMAL_POINT             24
 #define FIXED_POINT_MULTIPLIER          1.0f
 #define FIXED_COLOR_COMPONENT_MASK      0xffffffff
 
 inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }
 
 //-----------------------------------------------------------------------------------
 // YUV to RGB colorspace conversion
 //-----------------------------------------------------------------------------------
 static inline __device__ float clamp( float x )	{ return fminf(fmaxf(x, 0.0f), 255.0f); }
 
 // YUV2RGB
 template<typename T>
 static inline __device__ T YUV2RGB(const uint3& yuvi)
 {
     const float luma = float(yuvi.x);
     const float u    = float(yuvi.y) - 512.0f;
     const float v    = float(yuvi.z) - 512.0f;
     const float s    = 1.0f / 1024.0f * 255.0f;	// TODO clamp for uchar output?
 
 #if 1
     return make_vec<T>(clamp((luma + 1.402f * v) * s),
                     clamp((luma - 0.344f * u - 0.714f * v) * s),
                     clamp((luma + 1.772f * u) * s), 255);
 #else
     return make_vec<T>(clamp((luma + 1.140f * v) * s),
                     clamp((luma - 0.395f * u - 0.581f * v) * s),
                     clamp((luma + 2.032f * u) * s), 255);
 #endif
 }
 
 
 //-----------------------------------------------------------------------------------
 // NV12 to RGB
 //-----------------------------------------------------------------------------------
 template<typename T>
 __global__ void NV12ToRGB(uint32_t* srcImage, size_t nSourcePitch,
                           T* dstImage,        size_t nDestPitch,
                           uint32_t width,     uint32_t height)
 {
     int x, y;
     uint32_t yuv101010Pel[2];
     uint32_t processingPitch = ((width) + 63) & ~63;
     uint8_t *srcImageU8     = (uint8_t *)srcImage;
 
     processingPitch = nSourcePitch;
 
     // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
     x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
     y = blockIdx.y *  blockDim.y       +  threadIdx.y;
 
     if( x >= width )
         return; //x = width - 1;
 
     if( y >= height )
         return; // y = height - 1;
 
     // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
     // if we move to texture we could read 4 luminance values
     yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]) << 2;
     yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;
 
     uint32_t chromaOffset    = processingPitch * height;
     int y_chroma = y >> 1;
 
     if (y & 1)  // odd scanline ?
     {
         uint32_t chromaCb;
         uint32_t chromaCr;
 
         chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x    ];
         chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];
 
         if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
         {
             chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x    ] + 1) >> 1;
             chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
         }
 
         yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
         yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
 
         yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
         yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
     }
     else
     {
         yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
         yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
 
         yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
         yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
     }
 
     // this steps performs the color conversion
     const uint3 yuvi_0 = make_uint3((yuv101010Pel[0] &   COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));
   
     const uint3 yuvi_1 = make_uint3((yuv101010Pel[1] &   COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK),
                                    ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK));
                                    
     // YUV to RGB transformation conversion
     dstImage[y * width + x]     = YUV2RGB<T>(yuvi_0);
     dstImage[y * width + x + 1] = YUV2RGB<T>(yuvi_1);
 }
 
 
 template<typename T> 
 static cudaError_t launchNV12ToRGB( void* srcDev, T* dstDev, size_t width, size_t height )
 {
     if( !srcDev || !dstDev )
         return cudaErrorInvalidDevicePointer;
 
     if( width == 0 || height == 0 )
         return cudaErrorInvalidValue;
 
     const size_t srcPitch = width * sizeof(uint8_t);
     const size_t dstPitch = width * sizeof(T);
     
     const dim3 blockDim(32,8,1);
     const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);
 
     NV12ToRGB<T><<<gridDim, blockDim>>>( (uint32_t*)srcDev, srcPitch, dstDev, dstPitch, width, height );
     
     return cudaGetLastError();
 }
 
 // cudaNV12ToRGB
 cudaError_t cudaNV12ToRGB( uchar3* srcDev, uchar3* destDev, size_t width, size_t height )
 {
     return launchNV12ToRGB<uchar3>(srcDev, destDev, width, height);
 }
 