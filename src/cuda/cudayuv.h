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

#ifndef __CUDA_YUV_CONVERT_H
#define __CUDA_YUV_CONVERT_H

#include "cuda.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV NV12 4:2:0 to RGB
/// @see cudaConvertColor() from cudaColorspace.h for automated format conversion
/// @ingroup colorspace
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to RGB uchar3 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGB( uchar3* input, uchar3* output, size_t width, size_t height );

#endif