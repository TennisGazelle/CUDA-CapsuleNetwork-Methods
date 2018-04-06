//
// Created by daniellopez on 4/6/18.
//

#ifndef NEURALNETS_CUDACLIONHELPER_H
#define NEURALNETS_CUDACLIONHELPER_H

/**
 * Stuff to help CLion give proper syntax highlighting
 *
 * source: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion
 */
#ifdef __JETBRAINS_IDE__
#define __CUDACC__ 1
#define __host__
#define __device__
#define __global__
#define __forceinline__
#define __shared__
inline void __syncthreads() {}
inline void __threadfence_block() {}
template<class T> inline T __clz(const T val) { return val; }
struct __cuda_fake_struct { int x; int y; int z; };
extern __cuda_fake_struct blockDim;
extern __cuda_fake_struct threadIdx;
extern __cuda_fake_struct blockIdx;
extern __cuda_fake_struct gridDim;

// math helpers
inline float sqrtf(float val) { return val; }
inline float powf(float x, float power) {
    float temp = 1;
    while(power-- > 0) {
        temp *= x;
    }
    return temp;
}
//inline bool isnan(float x) { return x != x; }

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__
#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>

#endif

#endif //NEURALNETS_CUDACLIONHELPER_H
