/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __CUDA_VECTOR_TEMPLATES_H__
#define __CUDA_VECTOR_TEMPLATES_H__

#include <stdint.h>

// vector overloads
#include "cudamath.h"

// static assertion
#include <type_traits>


//////////////////////////////////////////////////////////////////////////////////
/// @name Vector Templates
/// @internal
/// @ingroup cuda
//////////////////////////////////////////////////////////////////////////////////

///@{

// get base type (uint8 or float) from vector
template<class T> struct cudaVectorTypeInfo;

//template<> struct cudaVectorTypeInfo<uchar>  { typedef uint8_t Base; };
template<> struct cudaVectorTypeInfo<uchar3> { typedef uint8_t Base; };

// static compile-time assertion
template<typename T> struct cuda_assert_false : std::false_type { };

// make_vec<T> templates
template<typename T> inline __host__ __device__ T make_vec( typename cudaVectorTypeInfo<T>::Base x, typename cudaVectorTypeInfo<T>::Base y, typename cudaVectorTypeInfo<T>::Base z, typename cudaVectorTypeInfo<T>::Base w )	{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<> inline __host__ __device__ uchar3 make_vec( uint8_t x, uint8_t y, uint8_t z, uint8_t w )	{ return make_uchar3(x,y,z); }

// cast_vec<T> templates
template<typename T> inline __host__ __device__ T cast_vec( const uchar3& a )				{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<> inline __host__ __device__ uchar3 cast_vec( const uchar3& a )					{ return make_uchar3(a); }

// extract alpha color component
template<typename T> inline __device__ typename cudaVectorTypeInfo<T>::Base alpha( T vec, typename cudaVectorTypeInfo<T>::Base default_alpha=255 )	{ static_assert(cuda_assert_false<T>::value, "invalid vector type - supported types are uchar3, uchar4, float3, float4");  }
template<> inline __host__ __device__ uint8_t alpha( uchar3 vec, uint8_t default_alpha )		{ return default_alpha; }

///@}

#endif