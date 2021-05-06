/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef __CUDA_HELPER_MATH_H_
#define __CUDA_HELPER_MATH_H_

#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////
/// @name Vector Math
/// @internal
/// @ingroup cuda
////////////////////////////////////////////////////////////////////////////////

///@{

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 make_float3(float3 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(uchar3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uchar4 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}


inline __host__ __device__ uchar3 make_uchar3(uchar3 a)
{
    return make_uchar3(a.x, a.y, a.z);
}
inline __host__ __device__ uchar3 make_uchar3(uchar4 a)
{
    return make_uchar3(a.x, a.y, a.z);
}
inline __host__ __device__ uchar3 make_uchar3(float3 a)
{
    return make_uchar3(a.x, a.y, a.z);
}
inline __host__ __device__ uchar3 make_uchar3(float4 a)
{
    return make_uchar3(a.x, a.y, a.z);
}

inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float4 a)
{
    return make_float4(a.x, a.y, a.z, a.w);
}

inline __host__ __device__ float4 make_float4(uchar3 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), 0.0f);
}
inline __host__ __device__ float4 make_float4(uchar4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ uchar4 make_uchar4(uchar3 a)
{
    return make_uchar4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uchar4 make_uchar4(uchar4 a)
{
    return make_uchar4(a.x, a.y, a.z, a.w);
}
inline __host__ __device__ uchar4 make_uchar4(float3 a)
{
    return make_uchar4(a.x, a.y, a.z, 0);
}

inline __host__ __device__ uchar4 make_uchar4(float4 a)
{
    return make_uchar4(a.x, a.y, a.z, a.w);
}

#endif