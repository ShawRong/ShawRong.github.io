---
title: "Tensor Core GEMM"
date: 2025-07-22T11:05:42.205Z
draft: false
tags: []
---

This is a simple tensor core gemm of int8
```cpp
#include "cublasLt.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <vector>

using namespace nvcuda;

// m, n, k, k is the inner dimmension
// C is in shape m x n
// we use setting 16x16x16 to do mma
// we know a warp will result in a 16x16 in C.
// reformating the C into [(m + 15) / 16, (n + 15) / 16]
// so we can get c_row and c_col for which warp to deal with
__global__ void int8_gemm_tensor_core_kernel(const int8_t *A, const int8_t *B,
                                             int32_t *C, int M, int N, int K,
                                             int lda, int ldb, int ldc,
                                             int32_t alpha, int32_t beta) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) /
                32; // suppose we do not have y dim.
  // we need to know which row to deal with A.
  int how_many_col_in_tiles = (N + 15) / 16; // can be optimized
  int c_row =
      (warp_id / how_many_col_in_tiles) * 16; // warp num / reformated index
  int c_col = (warp_id % how_many_col_in_tiles) * 16;

  if (c_row * 16 >= M || c_col * 16 >= N)
    return;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;
  wmma::fill_fragment(c_frag, 0);

  for (int k = 0; k < K; k += 16) {
    const int8_t *a_ptr = A + c_row * lda + k; // row by c_row, k by k
    const int8_t *b_ptr = B + k * ldb + c_col;

    wmma::load_matrix_sync(a_frag, a_ptr, lda);
    wmma::load_matrix_sync(b_frag, b_ptr, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // c_ptr refer the the start of c matrix fragment
  int32_t *c_ptr = C + c_row * ldc + c_col;

  // load original C fragment
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_original_frag;
  wmma::load_matrix_sync(c_original_frag, c_ptr, ldc, wmma::mem_row_major);

  // incorporate alpha, beta
  for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = alpha * c_frag.x[i] + beta * c_original_frag.x[i];
  }

  wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
}

```