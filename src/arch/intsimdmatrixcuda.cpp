///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixcuda.cpp
// Description: CUDA implementation of 8-bit int SIMD matrix operations.
// Author:      AI Assistant
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#if defined(HAVE_CUDA)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "intsimdmatrix.h"
#include "genericvector.h"
#include "tesstypes.h"
#include "tprintf.h"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <climits>

namespace tesseract {

// CUDA kernel for matrix-vector multiplication with int8 weights
__global__ void matmul_int8_kernel(
    const int8_t* weights,
    const float* input,
    float* output,
    const float* scales,
    int num_inputs,
    int num_outputs,
    int weight_width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= num_outputs) return;
    
    float sum = 0.0f;
    const int8_t* w_row = weights + out_idx * weight_width;
    
    // Compute dot product (num_inputs excludes bias)
    for (int i = 0; i < num_inputs; i++) {
        sum += static_cast<float>(w_row[i]) * input[i];
    }
    
    // Add bias (at index num_inputs)
    sum += static_cast<float>(w_row[num_inputs]) * 127.0f;  // INT8_MAX
    
    // Apply scale factor
    output[out_idx] = sum * scales[out_idx];
}

// CUDA context and resources
struct CUDAContext {
  int device_id = 0;
  bool initialized = false;
  
  bool init() {
    if (initialized) return true;
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      tprintf("CUDA: No devices available\n");
      return false;
    }
    
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      tprintf("CUDA: Failed to set device\n");
      return false;
    }
    
    initialized = true;
    return true;
  }
};

static CUDAContext cuda_ctx;

// CUDA implementation of matrix-vector multiplication
static void MatrixDotVectorCUDA(int dim1, int dim2, const int8_t* wi,
                               const TFloat* scales, const int8_t* u,
                               TFloat* v) {
  if (!cuda_ctx.init()) {
    // Fall back to generic CPU implementation
    for (int i = 0; i < dim1; ++i) {
      const int8_t* w_row = wi + i * dim2;
      int total = 0;
      for (int j = 0; j < dim2 - 1; ++j) {
        total += w_row[j] * u[j];
      }
      // Add bias (last element)
      total += w_row[dim2 - 1] * INT8_MAX;
      v[i] = static_cast<TFloat>(total) * scales[i];
    }
    return;
  }
  
  // Allocate device memory
  int8_t* d_weights = nullptr;
  float* d_input = nullptr;
  float* d_output = nullptr;
  float* d_scales = nullptr;
  
  size_t weights_size = dim1 * dim2 * sizeof(int8_t);
  size_t input_size = (dim2 - 1) * sizeof(float);
  size_t output_size = dim1 * sizeof(float);
  size_t scales_size = dim1 * sizeof(float);
  
  cudaError_t err;
  
  err = cudaMalloc(&d_weights, weights_size);
  if (err != cudaSuccess) {
    // Fall back to CPU implementation
    for (int i = 0; i < dim1; ++i) {
      const int8_t* w_row = wi + i * dim2;
      int total = 0;
      for (int j = 0; j < dim2 - 1; ++j) {
        total += w_row[j] * u[j];
      }
      total += w_row[dim2 - 1] * INT8_MAX;
      v[i] = static_cast<TFloat>(total) * scales[i];
    }
    return;
  }
  
  err = cudaMalloc(&d_input, input_size);
  if (err != cudaSuccess) {
    cudaFree(d_weights);
    // Fall back to CPU implementation
    for (int i = 0; i < dim1; ++i) {
      const int8_t* w_row = wi + i * dim2;
      int total = 0;
      for (int j = 0; j < dim2 - 1; ++j) {
        total += w_row[j] * u[j];
      }
      total += w_row[dim2 - 1] * INT8_MAX;
      v[i] = static_cast<TFloat>(total) * scales[i];
    }
    return;
  }
  
  err = cudaMalloc(&d_output, output_size);
  if (err != cudaSuccess) {
    cudaFree(d_weights);
    cudaFree(d_input);
    // Fall back to CPU implementation
    for (int i = 0; i < dim1; ++i) {
      const int8_t* w_row = wi + i * dim2;
      int total = 0;
      for (int j = 0; j < dim2 - 1; ++j) {
        total += w_row[j] * u[j];
      }
      total += w_row[dim2 - 1] * INT8_MAX;
      v[i] = static_cast<TFloat>(total) * scales[i];
    }
    return;
  }
  
  err = cudaMalloc(&d_scales, scales_size);
  if (err != cudaSuccess) {
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    // Fall back to CPU implementation
    for (int i = 0; i < dim1; ++i) {
      const int8_t* w_row = wi + i * dim2;
      int total = 0;
      for (int j = 0; j < dim2 - 1; ++j) {
        total += w_row[j] * u[j];
      }
      total += w_row[dim2 - 1] * INT8_MAX;
      v[i] = static_cast<TFloat>(total) * scales[i];
    }
    return;
  }
  
  // Convert int8 input to float for CUDA kernel
  std::vector<float> u_float(dim2 - 1);  // Exclude bias
  for (int i = 0; i < dim2 - 1; i++) {
    u_float[i] = static_cast<float>(u[i]);
  }
  
  // Copy data to device
  cudaMemcpy(d_weights, wi, weights_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, u_float.data(), input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scales, scales, scales_size, cudaMemcpyHostToDevice);
  
  // Launch kernel
  int block_size = 256;
  int num_blocks = (dim1 + block_size - 1) / block_size;
  int num_inputs = dim2 - 1;  // Exclude bias from input count
  matmul_int8_kernel<<<num_blocks, block_size>>>(
      d_weights, d_input, d_output, d_scales, num_inputs, dim1, dim2);
  
  // Copy result back to host
  std::vector<float> v_float(dim1);
  cudaMemcpy(v_float.data(), d_output, output_size, cudaMemcpyDeviceToHost);
  
  // Convert back to TFloat
  for (int i = 0; i < dim1; i++) {
    v[i] = static_cast<TFloat>(v_float[i]);
  }
  
  // Cleanup
  cudaFree(d_weights);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scales);
}

const IntSimdMatrix IntSimdMatrix::intSimdMatrixCUDA = {
  // MatrixDotVectorFunction
  MatrixDotVectorCUDA,
  // num_outputs_per_register
  4,
  // max_output_registers
  8,
  // num_inputs_per_register
  16,
  // num_inputs_per_group
  4
};

} // namespace tesseract

#endif // HAVE_CUDA
