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

 #include "cudargb.h"
 #include "cudavector.h"
 
 inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }
 
 //-----------------------------------------------------------------------------------
 // RGB <-> BGR
 //-----------------------------------------------------------------------------------
 template<typename T>
 __global__ void RGBToBGR(T* srcImage, T* dstImage, int width, int height)
 {
     const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
     const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
     
     const int pixel = y * width + x;
 
     if( x >= width )
        return; 
 
     if( y >= height )
        return;
 
     const T px = srcImage[pixel];
     
     dstImage[pixel] = make_vec<T>(px.z, px.y, px.x, alpha(px));
 }
 
 template<typename T> 
 static cudaError_t launchRGBToBGR( T* srcDev, T* dstDev, size_t width, size_t height )
 {
     if( !srcDev || !dstDev )
         return cudaErrorInvalidDevicePointer;
 
     const dim3 blockDim(32,8,1);
     const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);
 
     RGBToBGR<T><<<gridDim, blockDim>>>(srcDev, dstDev, width, height);
     
     return cudaGetLastError();
 }
 
 cudaError_t cudaRGB8ToBGR8( uchar3* input, uchar3* output, size_t width, size_t height )
 {
     return launchRGBToBGR<uchar3>(input, output, width, height);
 }


 
