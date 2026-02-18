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

// Define target OpenCL version before including headers
// Using OpenCL 1.2 (version 120) for maximum compatibility
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif

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
    const int num_outputs,
    const int weight_width
) {
    int out_idx = get_global_id(0);
    if (out_idx >= num_outputs) return;
    
    float sum = 0.0f;
    __global const char* w_row = weights + out_idx * weight_width;
    
    // Compute dot product (num_inputs excludes bias)
    for (int i = 0; i < num_inputs; i++) {
        sum += (float)w_row[i] * input[i];
    }
    
    // Add bias (at index num_inputs)
    sum += (float)w_row[num_inputs] * 127.0f;  // INT8_MAX
    
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
  
  // Buffer cache to avoid repeated allocation/deallocation
  cl_mem cached_weights_buf = nullptr;
  cl_mem cached_input_buf = nullptr;
  cl_mem cached_output_buf = nullptr;
  cl_mem cached_scales_buf = nullptr;
  size_t cached_weights_size = 0;
  size_t cached_input_size = 0;
  size_t cached_output_size = 0;
  size_t cached_scales_size = 0;
  
  // Debug counters
  int call_count = 0;
  bool verbose_mode = false;
  
  ~OpenCLContext() {
    cleanup();
  }
  
  void cleanup() {
    if (cached_weights_buf) clReleaseMemObject(cached_weights_buf);
    if (cached_input_buf) clReleaseMemObject(cached_input_buf);
    if (cached_output_buf) clReleaseMemObject(cached_output_buf);
    if (cached_scales_buf) clReleaseMemObject(cached_scales_buf);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    cached_weights_buf = nullptr;
    cached_input_buf = nullptr;
    cached_output_buf = nullptr;
    cached_scales_buf = nullptr;
    kernel = nullptr;
    program = nullptr;
    queue = nullptr;
    context = nullptr;
    initialized = false;
  }
  
  bool init() {
    if (initialized) return true;
    
    // Check for verbose mode via environment variable
    const char* verbose_env = getenv("TESSERACT_OPENCL_VERBOSE");
    verbose_mode = (verbose_env != nullptr && verbose_env[0] == '1');
    
    cl_int err;
    
    if (verbose_mode) {
      tprintf("OpenCL: Initialization starting...\n");
    }
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to get platform ID (error %d)\n", err);
      return false;
    }
    
    if (verbose_mode) {
      char platform_name[256] = "";
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
      tprintf("OpenCL: Platform: %s\n", platform_name);
    }
    
    // Check environment variable for device preference
    const char* device_env = getenv("TESSERACT_OPENCL_DEVICE");
    cl_device_type preferred_type = CL_DEVICE_TYPE_GPU;
    if (device_env) {
      if (strstr(device_env, "CPU")) {
        preferred_type = CL_DEVICE_TYPE_CPU;
        if (verbose_mode) {
          tprintf("OpenCL: Environment requests CPU device\n");
        }
      } else {
        if (verbose_mode) {
          tprintf("OpenCL: Environment requests GPU device\n");
        }
      }
    }
    
    // Get device - try preferred type first
    err = clGetDeviceIDs(platform, preferred_type, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
      if (verbose_mode) {
        tprintf("OpenCL: Preferred device type not available, trying fallback\n");
      }
      // Try opposite type as fallback
      cl_device_type fallback_type = (preferred_type == CL_DEVICE_TYPE_GPU) 
                                     ? CL_DEVICE_TYPE_CPU 
                                     : CL_DEVICE_TYPE_GPU;
      err = clGetDeviceIDs(platform, fallback_type, 1, &device, nullptr);
      if (err != CL_SUCCESS) {
        tprintf("OpenCL: Failed to get any device (error %d)\n", err);
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
    
    // Get device name and info for success logging
    char device_name[256] = "Unknown";
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    
    // Get device global memory size
    cl_ulong mem_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
    
    // Determine if GPU or CPU
    cl_device_type dev_type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr);
    const char* dev_type_str = (dev_type == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
    
    // Log success with device information
    tprintf("OpenCL: Successfully initialized on %s device: %s (%.1f GB)\n",
            dev_type_str,
            device_name,
            mem_size / (1024.0 * 1024.0 * 1024.0));
    
    return true;
  }
};

static OpenCLContext opencl_ctx;

// OpenCL implementation of matrix-vector multiplication with buffer caching
static void MatrixDotVectorOpenCL(int dim1, int dim2, const int8_t* wi,
                                 const TFloat* scales, const int8_t* u,
                                 TFloat* v) {
  // Increment call counter
  opencl_ctx.call_count++;
  
  // Log first few calls for debugging
  bool log_this_call = (opencl_ctx.call_count <= 3) || opencl_ctx.verbose_mode;
  
  if (log_this_call) {
    tprintf("OpenCL: MatrixDotVector call #%d (dim1=%d, dim2=%d)\n",
            opencl_ctx.call_count, dim1, dim2);
  }
  
  if (!opencl_ctx.init()) {
    if (log_this_call) {
      tprintf("OpenCL: Initialization failed, falling back to CPU\n");
    }
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
  
  // Calculate buffer sizes
  size_t weights_size = dim1 * dim2 * sizeof(int8_t);
  size_t input_size = (dim2 - 1) * sizeof(TFloat);
  size_t output_size = dim1 * sizeof(TFloat);
  size_t scales_size = dim1 * sizeof(TFloat);
  
  if (log_this_call) {
    tprintf("OpenCL: Buffer sizes: weights=%zu, input=%zu, output=%zu, scales=%zu bytes\n",
            weights_size, input_size, output_size, scales_size);
  }
  
  // Reuse or create weights buffer
  if (opencl_ctx.cached_weights_size != weights_size) {
    if (log_this_call) {
      tprintf("OpenCL: Creating new weights buffer (%zu -> %zu bytes)\n",
              opencl_ctx.cached_weights_size, weights_size);
    }
    if (opencl_ctx.cached_weights_buf) {
      clReleaseMemObject(opencl_ctx.cached_weights_buf);
    }
    opencl_ctx.cached_weights_buf = clCreateBuffer(opencl_ctx.context, 
                                                   CL_MEM_READ_ONLY,
                                                   weights_size, nullptr, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create weights buffer (error %d)\n", err);
      opencl_ctx.cached_weights_buf = nullptr;
      opencl_ctx.cached_weights_size = 0;
      // Fall back to CPU
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
    opencl_ctx.cached_weights_size = weights_size;
  } else if (log_this_call) {
    tprintf("OpenCL: Reusing weights buffer (%zu bytes)\n", weights_size);
  }
  
  // Reuse or create input buffer
  if (opencl_ctx.cached_input_size != input_size) {
    if (opencl_ctx.cached_input_buf) {
      clReleaseMemObject(opencl_ctx.cached_input_buf);
    }
    opencl_ctx.cached_input_buf = clCreateBuffer(opencl_ctx.context,
                                                 CL_MEM_READ_ONLY,
                                                 input_size, nullptr, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create input buffer\n");
      opencl_ctx.cached_input_buf = nullptr;
      opencl_ctx.cached_input_size = 0;
      // Fall back to CPU
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
    opencl_ctx.cached_input_size = input_size;
  }
  
  // Reuse or create output buffer
  if (opencl_ctx.cached_output_size != output_size) {
    if (opencl_ctx.cached_output_buf) {
      clReleaseMemObject(opencl_ctx.cached_output_buf);
    }
    opencl_ctx.cached_output_buf = clCreateBuffer(opencl_ctx.context,
                                                  CL_MEM_WRITE_ONLY,
                                                  output_size, nullptr, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create output buffer\n");
      opencl_ctx.cached_output_buf = nullptr;
      opencl_ctx.cached_output_size = 0;
      // Fall back to CPU
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
    opencl_ctx.cached_output_size = output_size;
  }
  
  // Reuse or create scales buffer
  if (opencl_ctx.cached_scales_size != scales_size) {
    if (opencl_ctx.cached_scales_buf) {
      clReleaseMemObject(opencl_ctx.cached_scales_buf);
    }
    opencl_ctx.cached_scales_buf = clCreateBuffer(opencl_ctx.context,
                                                  CL_MEM_READ_ONLY,
                                                  scales_size, nullptr, &err);
    if (err != CL_SUCCESS) {
      tprintf("OpenCL: Failed to create scales buffer\n");
      opencl_ctx.cached_scales_buf = nullptr;
      opencl_ctx.cached_scales_size = 0;
      // Fall back to CPU
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
    opencl_ctx.cached_scales_size = scales_size;
  }
  
  // Convert int8 input to float for OpenCL kernel
  std::vector<TFloat> u_float(dim2 - 1);
  for (int i = 0; i < dim2 - 1; i++) {
    u_float[i] = static_cast<TFloat>(u[i]);
  }
  
  if (log_this_call) {
    tprintf("OpenCL: Transferring data to GPU...\n");
  }
  
  // Transfer data to GPU
  err = clEnqueueWriteBuffer(opencl_ctx.queue, opencl_ctx.cached_weights_buf, CL_FALSE, 0,
                      weights_size, wi, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: Failed to write weights buffer (error %d)\n", err);
    return;
  }
  
  err = clEnqueueWriteBuffer(opencl_ctx.queue, opencl_ctx.cached_input_buf, CL_FALSE, 0,
                      input_size, u_float.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: Failed to write input buffer (error %d)\n", err);
    return;
  }
  
  err = clEnqueueWriteBuffer(opencl_ctx.queue, opencl_ctx.cached_scales_buf, CL_FALSE, 0,
                      scales_size, scales, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: Failed to write scales buffer (error %d)\n", err);
    return;
  }
  
  if (log_this_call) {
    tprintf("OpenCL: Setting kernel arguments and executing...\n");
  }
  
  // Set kernel arguments
  int num_inputs = dim2 - 1;
  clSetKernelArg(opencl_ctx.kernel, 0, sizeof(cl_mem), &opencl_ctx.cached_weights_buf);
  clSetKernelArg(opencl_ctx.kernel, 1, sizeof(cl_mem), &opencl_ctx.cached_input_buf);
  clSetKernelArg(opencl_ctx.kernel, 2, sizeof(cl_mem), &opencl_ctx.cached_output_buf);
  clSetKernelArg(opencl_ctx.kernel, 3, sizeof(cl_mem), &opencl_ctx.cached_scales_buf);
  clSetKernelArg(opencl_ctx.kernel, 4, sizeof(int), &num_inputs);
  clSetKernelArg(opencl_ctx.kernel, 5, sizeof(int), &dim1);
  clSetKernelArg(opencl_ctx.kernel, 6, sizeof(int), &dim2);
  
  // Execute kernel
  size_t global_work_size = dim1;
  err = clEnqueueNDRangeKernel(opencl_ctx.queue, opencl_ctx.kernel, 1,
                               nullptr, &global_work_size, nullptr,
                               0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: Kernel execution failed (error %d)\n", err);
    return;
  }
  
  if (log_this_call) {
    tprintf("OpenCL: Reading results back from GPU...\n");
  }
  
  // Read results back
  err = clEnqueueReadBuffer(opencl_ctx.queue, opencl_ctx.cached_output_buf, CL_FALSE, 0,
                     output_size, v, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: Failed to read output buffer (error %d)\n", err);
    return;
  }
  
  // CRITICAL: Wait for all operations to complete
  // Without this, GPU work may not finish before next iteration
  err = clFinish(opencl_ctx.queue);
  if (err != CL_SUCCESS) {
    tprintf("OpenCL: clFinish failed (error %d)\n", err);
    return;
  }
  
  if (log_this_call) {
    tprintf("OpenCL: Operation completed successfully\n");
  }
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
