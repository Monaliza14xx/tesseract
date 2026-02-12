///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixopencl.cpp
// Description: OpenCL implementation of 8-bit int SIMD matrix operations.
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

#if defined(HAVE_OPENCL)

#  ifdef __APPLE__
#    include <OpenCL/opencl.h>
#  else
#    include <CL/cl.h>
#  endif

#include "intsimdmatrix.h"
#include "genericvector.h"
#include "tesstypes.h"
#include "tprintf.h"

#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstring>
#include <climits>

namespace tesseract {

// OpenCL kernel for matrix-vector multiplication with int8 weights
static const char* opencl_kernel_source = R"(
__kernel void matmul_int8(
    __global const char* weights,
    __global const float* input,
    __global float* output,
    __global const float* scales,
    const int num_inputs,
    const int num_outputs
) {
    int out_idx = get_global_id(0);
    if (out_idx >= num_outputs) return;
    
    float sum = 0.0f;
    __global const char* w_row = weights + out_idx * (num_inputs + 1);
    
    // Compute dot product
    for (int i = 0; i < num_inputs; i++) {
        sum += (float)w_row[i] * input[i];
    }
    
    // Add bias (last element in weight row)
    sum += (float)w_row[num_inputs];
    
    // Apply scale factor
    output[out_idx] = sum * scales[out_idx];
}
)";

// OpenCL context and resources
struct OpenCLContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  bool initialized = false;
  
  ~OpenCLContext() {
    cleanup();
  }
  
  void cleanup() {
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    kernel = nullptr;
    program = nullptr;
    queue = nullptr;
    context = nullptr;
    initialized = false;
  }
  
  bool init() {
    if (initialized) return true;
    
    cl_int err;
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to get platform ID\n");
      return false;
    }
    
    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
      // Try CPU if GPU not available
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
      if (err != CL_SUCCESS) {
        tprintf("OpenCL: Failed to get device ID\n");
        return false;
      }
    }
    
    // Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create context\n");
      return false;
    }
    
    // Create command queue
#ifdef CL_VERSION_2_0
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create command queue\n");
      cleanup();
      return false;
    }
    
    // Create program from kernel source
    size_t source_len = strlen(opencl_kernel_source);
    program = clCreateProgramWithSource(context, 1, &opencl_kernel_source, 
                                       &source_len, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create program\n");
      cleanup();
      return false;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_size;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                           0, nullptr, &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                           log_size, log.data(), nullptr);
      tprintf("OpenCL: Build failed:\n%s\n", log.data());
      cleanup();
      return false;
    }
    
    // Create kernel
    kernel = clCreateKernel(program, "matmul_int8", &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create kernel\n");
      cleanup();
      return false;
    }
    
    initialized = true;
    return true;
  }
};

static OpenCLContext opencl_ctx;

// OpenCL implementation of matrix-vector multiplication
static void MatrixDotVectorOpenCL(int dim1, int dim2, const int8_t* wi,
                                 const TFloat* scales, const int8_t* u,
                                 TFloat* v) {
  if (!opencl_ctx.init()) {
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
  
  cl_int err;
  
  // Create buffers
  size_t weights_size = dim1 * (dim2 + 1) * sizeof(int8_t);
  size_t input_size = dim2 * sizeof(TFloat);
  size_t output_size = dim1 * sizeof(TFloat);
  size_t scales_size = dim1 * sizeof(TFloat);
  
  cl_mem weights_buf = clCreateBuffer(opencl_ctx.context, 
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     weights_size, const_cast<int8_t*>(wi), &err);
  if (err != CL_SUCCESS) {
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
  
  // Convert int8 input to float for OpenCL kernel
  std::vector<TFloat> u_float(dim2);
  for (int i = 0; i < dim2; i++) {
    u_float[i] = static_cast<TFloat>(u[i]);
  }
  
  cl_mem input_buf = clCreateBuffer(opencl_ctx.context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   input_size, u_float.data(), &err);
  cl_mem output_buf = clCreateBuffer(opencl_ctx.context,
                                    CL_MEM_WRITE_ONLY,
                                    output_size, nullptr, &err);
  cl_mem scales_buf = clCreateBuffer(opencl_ctx.context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    scales_size, const_cast<TFloat*>(scales), &err);
  
  // Set kernel arguments
  clSetKernelArg(opencl_ctx.kernel, 0, sizeof(cl_mem), &weights_buf);
  clSetKernelArg(opencl_ctx.kernel, 1, sizeof(cl_mem), &input_buf);
  clSetKernelArg(opencl_ctx.kernel, 2, sizeof(cl_mem), &output_buf);
  clSetKernelArg(opencl_ctx.kernel, 3, sizeof(cl_mem), &scales_buf);
  clSetKernelArg(opencl_ctx.kernel, 4, sizeof(int), &dim2);
  clSetKernelArg(opencl_ctx.kernel, 5, sizeof(int), &dim1);
  
  // Execute kernel
  size_t global_work_size = dim1;
  err = clEnqueueNDRangeKernel(opencl_ctx.queue, opencl_ctx.kernel, 1,
                               nullptr, &global_work_size, nullptr,
                               0, nullptr, nullptr);
  
  // Read results
  clEnqueueReadBuffer(opencl_ctx.queue, output_buf, CL_TRUE, 0,
                     output_size, v, 0, nullptr, nullptr);
  
  // Cleanup
  clReleaseMemObject(weights_buf);
  clReleaseMemObject(input_buf);
  clReleaseMemObject(output_buf);
  clReleaseMemObject(scales_buf);
}

const IntSimdMatrix IntSimdMatrix::intSimdMatrixOpenCL = {
  // MatrixDotVectorFunction
  MatrixDotVectorOpenCL,
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

#endif // HAVE_OPENCL
