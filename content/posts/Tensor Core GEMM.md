---
title: "Tensor Core GEMM"
date: 2025-07-22T12:20:54.008Z
draft: false
tags: []
---

This is a simple tensor core gemm of int8
```cpp
#include "device_launch_parameters.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <numeric>
#include <random>
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

  if (c_row >= M || c_col >= N)
    return;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;
  wmma::fill_fragment(c_frag, 0);

  for (int k = 0; k < K; k += 16) {
    const int8_t *a_ptr = A + c_row * lda + k; // row by c_row, k by k
    const int8_t *b_ptr = B + c_col * ldb + k;

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

// -- CPU reference GEMM Function --
void cpu_gemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K,
              int lda, int ldb, int ldc, int32_t alpha, int32_t beta) {
  std::vector<int32_t> C_temp(M * N);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int32_t sum = 0;
      for (int k = 0; k < K; k++) {
        sum += static_cast<int32_t>(A[i * lda + k]) *
               static_cast<int32_t>(B[j * ldb + k]);
      }
      C_temp[i * N + j] = sum;
    }
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int32_t original_c_val = C[i * ldc + j];
      C[i * ldc + j] = alpha * C_temp[i * N + j] + beta * original_c_val;
    }
  }
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Main function
int main() {
  // --- Matrix Dimensions (adjust as needed, K should be multiple of 16 for
  // this kernel) ---
  const int M = 256; // Rows of A, Rows of C
  const int N = 512; // Cols of B, Cols of C
  const int K =
      128; // Cols of A, Rows of B (inner dimension, must be multiple of 16)

  // Leading dimensions (assume M, N, K are actual dimensions)
  const int lda = K; // A is M x K (row-major)
  const int ldb = K; // B is K x N (column-major)
  const int ldc = N; // C is M x N (row-major)

  // Alpha and Beta values
  const int32_t alpha = 1; // Example: simple matrix multiplication
  const int32_t beta = 0;  // Example: C is initialized to 0, or overwritten

  // --- Host Memory Allocation ---
  std::vector<int8_t> h_A(M * K);
  std::vector<int8_t> h_B(K * N);
  std::vector<int32_t> h_C(M * N);     // For GPU result
  std::vector<int32_t> h_C_ref(M * N); // For CPU reference

  // --- Data Initialization ---
  // Seed random number generator
  std::mt19937 gen(0); // Use a fixed seed for reproducibility
  std::uniform_int_distribution<> distrib(-128, 127); // int8_t range

  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<int8_t>(distrib(gen));
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<int8_t>(distrib(gen));
  }
  // Initialize C to some non-zero values for beta test, or zeros
  for (int i = 0; i < M * N; ++i) {
    h_C[i] = static_cast<int32_t>(distrib(gen));
    h_C_ref[i] = h_C[i]; // Copy initial C for reference
  }

  // --- Device Memory Allocation ---
  int8_t *d_A, *d_B;
  int32_t *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(int8_t)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(int32_t)));

  // --- Data Transfer (Host to Device) ---
  CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(int32_t),
                        cudaMemcpyHostToDevice)); // Copy initial C

  // --- Kernel Launch Configuration ---
  // A warp processes a 16x16 tile of C.
  // Total number of 16x16 tiles in C: (M/16) * (N/16) (assuming M, N are
  // multiples of 16) For general M, N: ceil(M/16) * ceil(N/16)
  int num_tiles_m = (M + 15) / 16;
  int num_tiles_n = (N + 15) / 16;
  int total_warps = num_tiles_m * num_tiles_n;

  // We'll use 8 warps per block (256 threads) for good occupancy.
  const int WARPS_PER_BLOCK = 8;
  dim3 blockDim(WARPS_PER_BLOCK * 32); // 256 threads per block

  // Calculate grid dimension based on total_warps
  dim3 gridDim((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  std::cout << "Launching kernel with:" << std::endl;
  std::cout << "  Grid Dim: (" << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << ")" << std::endl;
  std::cout << "  Block Dim: (" << blockDim.x << ", " << blockDim.y << ", "
            << blockDim.z << ")" << std::endl;
  std::cout << "  Total Warps: " << total_warps << std::endl;

  // --- Kernel Execution ---
  int8_gemm_tensor_core_kernel<<<gridDim, blockDim>>>(
      d_A, d_B, d_C, M, N, K, lda, ldb, ldc, alpha, beta);

  CHECK_CUDA(cudaGetLastError());      // Check for errors during kernel launch
  CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to complete

  std::cout << "Kernel execution complete." << std::endl;

  // --- Data Transfer (Device to Host) ---
  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  // --- CPU Reference Computation ---
  cpu_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K, lda, ldb, ldc,
           alpha, beta);

  // --- Verification ---
  bool success = true;
  for (int i = 0; i < M * N; ++i) {
    if (h_C[i] != h_C_ref[i]) {
      std::cerr << "Mismatch at C[" << i / N << "][" << i % N
                << "]: GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << std::endl;
      success = false;
      // Print a few mismatches, then break
      if (++i % 10 == 0)
        break; // Limit error output
    }
  }

  if (success) {
    std::cout << "Verification PASSED! 🎉" << std::endl;
  } else {
    std::cerr << "Verification FAILED! 💔" << std::endl;
  }

  // --- Memory Deallocation ---
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  std::cout << "Memory freed. Exiting." << std::endl;

  return 0;
}
```