# GPU Training Guide for Tesseract LSTM Models

## Overview

This guide explains how to train Tesseract LSTM models using GPU acceleration (OpenCL or CUDA) for significantly improved training performance. GPU acceleration can provide 5-15x speedup compared to CPU-only training, depending on your hardware and training configuration.

**What you'll learn:**
- How to build Tesseract with GPU support for training
- Environment variables for GPU training
- Step-by-step training commands
- How to verify GPU is being used
- Performance optimization tips
- Troubleshooting common issues

## Prerequisites

### 1. Build Tesseract with GPU Support

Before training with GPU, you must build Tesseract with OpenCL or CUDA support enabled.

#### Option A: OpenCL (Cross-Platform)

Works with AMD, Intel, and NVIDIA GPUs.

**Install OpenCL dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Verify OpenCL is available
clinfo  # Install with: sudo apt-get install clinfo
```

**Build with OpenCL:**
```bash
# Using CMake (recommended)
cd tesseract
mkdir build && cd build
cmake .. -DENABLE_OPENCL=ON -DBUILD_TRAINING_TOOLS=ON
make -j$(nproc)
sudo make install
sudo ldconfig

# OR using Autotools
cd tesseract
./autogen.sh
./configure --enable-opencl
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### Option B: CUDA (NVIDIA GPUs Only)

Optimized for NVIDIA GPUs with better performance than OpenCL.

**Install CUDA Toolkit:**
```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Follow NVIDIA installation instructions for your OS

# Verify CUDA is available
nvidia-smi
nvcc --version
```

**Build with CUDA:**
```bash
cd tesseract
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DBUILD_TRAINING_TOOLS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### Option C: Both OpenCL and CUDA

For maximum flexibility:
```bash
cmake .. -DENABLE_OPENCL=ON -DENABLE_CUDA=ON -DBUILD_TRAINING_TOOLS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 2. Verify GPU Build

Check that GPU support is compiled in:
```bash
tesseract --version
# Look for "Found OpenCL" or "Found CUDA" in the output
# Example:
#  tesseract 5.5.2
#  Found AVX2
#  Found AVX
#  Found SSE4.1
#  Found OpenCL    ← Confirms OpenCL is available
#  Found CUDA      ← Confirms CUDA is available (if enabled)
```

If you don't see "Found OpenCL" or "Found CUDA", you need to rebuild Tesseract with GPU support.

### 3. Prepare Training Data

You'll need:
- **Training images** in `.lstmf` format
- **Starter traineddata** file
- **Training and evaluation list files**

Refer to [Tesseract Training Documentation](https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-4.00.html) for data preparation.

## GPU Training Configuration

### Environment Variables for GPU Training

#### For OpenCL Training

```bash
# Basic OpenCL configuration
export TESSERACT_OPENCL_DEVICE="GPU:0"

# Optional: Limit OpenMP threads (recommended for GPU training)
export OMP_THREAD_LIMIT=1

# Optional: Use specific OpenCL platform
export OPENCL_PLATFORM=0  # 0=first platform, 1=second platform, etc.
```

#### For CUDA Training

```bash
# Select GPU device (0 for first GPU, 1 for second, etc.)
export CUDA_VISIBLE_DEVICES=0

# Optional: Limit OpenMP threads
export OMP_THREAD_LIMIT=1

# Optional: Enable CUDA profiling
export CUDA_LAUNCH_BLOCKING=1  # Only for debugging, slows down training
```

#### For Multi-GPU Systems

```bash
# Use specific GPU (change 0 to 1, 2, etc. for other GPUs)
export CUDA_VISIBLE_DEVICES=0
# OR for OpenCL
export TESSERACT_OPENCL_DEVICE="GPU:0"

# To use multiple GPUs, lstmtraining must be run in separate processes
# Example for 2 GPUs:
# Terminal 1: CUDA_VISIBLE_DEVICES=0 lstmtraining ...
# Terminal 2: CUDA_VISIBLE_DEVICES=1 lstmtraining ...
```

## Training with GPU: Step-by-Step

### Example 1: Fine-tuning an Existing Model (Recommended)

Fine-tuning is the most common and recommended approach for creating custom models.

```bash
# Set up environment for OpenCL
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

# OR for CUDA
export CUDA_VISIBLE_DEVICES=0
export OMP_THREAD_LIMIT=1

# Run lstmtraining with GPU
lstmtraining \
  --continue_from /usr/share/tessdata/eng.traineddata \
  --traineddata /path/to/output/my_custom.traineddata \
  --model_output /path/to/output/checkpoints/my_custom \
  --train_listfile /path/to/training_list.txt \
  --eval_listfile /path/to/eval_list.txt \
  --max_iterations 10000 \
  --learning_rate 0.0001 \
  --target_error_rate 0.01 \
  --debug_interval 100 \
  --max_image_MB 8000 \
  2>&1 | tee training.log
```

### Example 2: Training from Scratch (Advanced)

For creating completely new models from scratch:

```bash
# Set GPU environment
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

lstmtraining \
  --traineddata /path/to/langdata/my_lang.traineddata \
  --net_spec '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c111]' \
  --model_output /path/to/output/my_lang \
  --train_listfile /path/to/training_list.txt \
  --eval_listfile /path/to/eval_list.txt \
  --max_iterations 50000 \
  --learning_rate 0.001 \
  --target_error_rate 0.01 \
  --debug_interval 100 \
  --max_image_MB 12000 \
  --net_mode 192 \
  2>&1 | tee training.log
```

### Example 3: Layer Replacement Training

For replacing specific layers in an existing model:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_THREAD_LIMIT=1

lstmtraining \
  --continue_from /usr/share/tessdata/eng.traineddata \
  --append_index 5 \
  --net_spec '[Lfx256 O1c111]' \
  --traineddata /path/to/output/my_custom.traineddata \
  --model_output /path/to/output/checkpoints/my_custom \
  --train_listfile /path/to/training_list.txt \
  --eval_listfile /path/to/eval_list.txt \
  --max_iterations 5000 \
  --learning_rate 0.0001 \
  2>&1 | tee training.log
```

## Verifying GPU is Being Used

### 1. Check Training Log for GPU Messages

When GPU is successfully initialized, you'll see messages like:

**OpenCL:**
```
OpenCL: Successfully initialized on GPU device: Tesla T4 (15.0 GB)
Using OpenCL GPU acceleration for matrix operations
```

**CUDA:**
```
CUDA: Successfully initialized on GPU device: Tesla T4 (15.0 GB)
Using CUDA GPU acceleration for matrix operations
```

These messages appear at the **very beginning** of training, before model loading.

### 2. Monitor GPU Usage in Real-Time

#### For NVIDIA GPUs:
```bash
# In a separate terminal, watch GPU usage
watch -n 1 nvidia-smi

# Look for:
# - GPU utilization percentage
# - Memory usage increasing
# - Python or lstmtraining process listed
```

#### For AMD GPUs:
```bash
# Install ROCm monitoring tools
sudo apt-get install rocm-smi

# Monitor GPU
watch -n 1 rocm-smi
```

#### For Intel GPUs:
```bash
# Install Intel GPU tools
sudo apt-get install intel-gpu-tools

# Monitor GPU
intel_gpu_top
```

### 3. Compare Training Speed

GPU training should be significantly faster:
```bash
# Without GPU: ~100-500 iterations/minute (varies by hardware)
# With GPU: ~1000-3000 iterations/minute on modern GPUs

# Watch for iteration speed in training log:
# grep "Iteration" training.log
```

### 4. Check Process GPU Affinity

```bash
# For NVIDIA, see which process is using GPU
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Find lstmtraining process
ps aux | grep lstmtraining
```

## Performance Optimization

### 1. Batch Size and Checkpoint Configuration

**NEW in this version:** Two separate parameters control training behavior:

**`--batch_size`** - Controls GPU batch processing (affects performance):
```bash
# Default
--batch_size 100

# Optimized for OpenCL GPU
--batch_size 300

# Optimized for CUDA GPU  
--batch_size 500

# High-end GPU with lots of memory
--batch_size 1000
```

**Benefits of larger batch sizes for GPU:**
- Better GPU utilization through increased parallelism
- Reduced CPU-GPU transfer overhead
- Higher training throughput
- More efficient memory usage

**`--checkpoint_interval`** - Controls checkpoint save frequency:
```bash
# Default - save every 100 iterations
--checkpoint_interval 100

# More frequent saves (safer, more disk I/O)
--checkpoint_interval 50

# Less frequent saves (faster, less disk I/O)
--checkpoint_interval 200
```

**Note:** batch_size and checkpoint_interval are independent. Larger batch_size improves GPU performance, while checkpoint_interval controls how often progress is saved.

**Guidelines:**
- Start with 300-500 for most GPU configurations
- Increase if you have GPU memory to spare
- Decrease if you experience out-of-memory errors
- Monitor GPU memory usage with nvidia-smi or similar tools

**Example with batch size:**
```bash
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

lstmtraining \
  --continue_from /usr/share/tessdata/eng.traineddata \
  --traineddata /path/to/output/my_custom.traineddata \
  --model_output /path/to/output/checkpoints/my_custom \
  --train_listfile /path/to/training_list.txt \
  --eval_listfile /path/to/eval_list.txt \
  --max_iterations 10000 \
  --learning_rate 0.0001 \
  --max_image_MB 8000 \
  --batch_size 500 \
  2>&1 | tee training.log
```

### 2. Memory Settings

The `--max_image_MB` parameter controls how much image data is loaded into memory:

```bash
# For GPUs with 4-8 GB memory
--max_image_MB 4000

# For GPUs with 8-16 GB memory (recommended)
--max_image_MB 8000

# For GPUs with 16+ GB memory
--max_image_MB 12000

# Note: Larger values = better GPU utilization but more memory usage
```

### 3. Thread Configuration

```bash
# IMPORTANT: Limit OpenMP threads when using GPU
export OMP_THREAD_LIMIT=1

# This prevents CPU threads from competing with GPU operations
# and typically provides best performance
```

### 4. Learning Rate Tuning

```bash
# For fine-tuning (recommended starting points)
--learning_rate 0.0001  # Conservative, stable
--learning_rate 0.0002  # Moderate
--learning_rate 0.0005  # Aggressive, may diverge

# For training from scratch
--learning_rate 0.001   # Standard starting point
--learning_rate 0.002   # Faster but less stable
```

### 5. Network Architecture Considerations

More complex networks benefit more from GPU acceleration:
```bash
# Simple network - modest GPU benefit
'[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx128 O1c111]'

# Complex network - significant GPU benefit
'[1,36,0,1 Ct3,3,32 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256 Lrx256 Lfx512 O1c111]'
```

### 6. Data Pipeline Optimization

```bash
# Keep training data on fast storage (SSD)
# Avoid network drives if possible

# Pre-process images to .lstmf format before training
# This reduces I/O during training
```

## Performance Expectations

### Training Speed Comparisons

Typical speedups on a Tesla T4 GPU (may vary based on your hardware):

| Configuration | Batch Size | Iterations/min | Relative Speed |
|--------------|------------|----------------|----------------|
| CPU Only (AVX2) | 100 | ~300 | 1.0x (baseline) |
| OpenCL (AMD RX 5700 XT) | 300 | ~1,500 | 5.0x |
| OpenCL (NVIDIA T4) | 500 | ~2,500 | 8.3x |
| CUDA (NVIDIA T4) | 500 | ~3,500 | 11.7x |
| CUDA (NVIDIA RTX 3090) | 1000 | ~5,500 | 18.3x |

**Factors affecting speedup:**
- GPU model and memory bandwidth
- Network complexity
- **Batch size (`--batch_size`)** - Larger is better for GPU
- Memory allocation (`--max_image_MB`)
- Image dimensions
- Number of training samples

### Memory Usage

Typical GPU memory requirements:
- **Small models** (simple network): 2-4 GB GPU memory
- **Medium models** (standard network): 4-8 GB GPU memory
- **Large models** (complex network): 8-16 GB GPU memory

## Troubleshooting

### Enable Verbose Debugging Mode

Before diving into specific issues, enable verbose OpenCL logging to get detailed diagnostic information:

```bash
export TESSERACT_OPENCL_VERBOSE=1
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

lstmtraining --batch_size 500 ...
```

**Verbose output includes:**
- OpenCL initialization details (platform, device selection)
- Buffer creation and reuse tracking
- Data transfer operations
- Kernel execution status
- All OpenCL errors with specific error codes
- First 3 MatrixDotVector calls logged in detail

**Key indicators to look for:**
- ✅ **"MatrixDotVector call #1, #2, #3..."** - OpenCL is being invoked
- ✅ **"Creating new ... buffer"** then **"Reusing ... buffer"** - Efficient memory management
- ✅ **"Operation completed successfully"** - GPU operations working
- ❌ **No MatrixDotVector messages** - Model not using int8 (see below)
- ❌ **"Failed to..." with error code** - Specific OpenCL error

### Issue: GPU Shows 0% Utilization Despite OpenCL Message

**Symptoms:**
- Training log shows "Using OpenCL GPU acceleration for matrix operations"
- nvidia-smi shows 0% GPU utilization
- GPU memory usage stays at ~3 MiB
- Training runs at CPU speed

**Root Cause:**
This usually indicates the model is using float32 weights instead of int8 quantized weights. GPU acceleration only works with int8 quantization.

**Solutions:**

1. **Use fine-tuning (recommended):**
   ```bash
   # Fine-tuning automatically converts to int8
   lstmtraining --continue_from existing_model.traineddata ...
   ```

2. **Ensure int8 conversion:**
   - When fine-tuning: Automatic int8 conversion happens
   - When training from scratch: Model needs int8 conversion

3. **Increase batch size:**
   ```bash
   # Higher batch size improves GPU utilization
   --batch_size 500  # Minimum 300 recommended for GPU
   ```

4. **Verify model is using int8:**
   - Look for "Total weights" message in training log
   - GPU only activates for int8 quantized models

5. **Use latest Tesseract build:**
   ```bash
   # Older versions had a critical GPU performance bug
   # Rebuild from latest source to get fix
   ```

**Expected Results After Fix:**
- GPU utilization: 70-90%
- GPU memory: 2-8 GB (depending on model size)
- Training speed: 5-15x faster than CPU

### Issue: No GPU Messages in Training Log

**Symptoms:**
- Training starts but no "Successfully initialized" messages
- Training runs at CPU speed

**Solutions:**

1. **Verify GPU build:**
   ```bash
   tesseract --version
   # Must show OpenCL or CUDA in features
   ```

2. **Check environment variables:**
   ```bash
   echo $TESSERACT_OPENCL_DEVICE  # For OpenCL
   echo $CUDA_VISIBLE_DEVICES     # For CUDA
   ```

3. **Test GPU detection:**
   ```bash
   clinfo  # For OpenCL
   nvidia-smi  # For CUDA
   ```

4. **Check build configuration:**
   ```bash
   # Rebuild with verbose output
   cmake .. -DENABLE_OPENCL=ON -DBUILD_TRAINING_TOOLS=ON -DCMAKE_VERBOSE_MAKEFILE=ON
   make VERBOSE=1
   ```

### Issue: Training Disconnects or Crashes

**Symptoms:**
- Training shows "[disconnected]" message
- Training process exits unexpectedly
- No GPU utilization despite OpenCL messages
- nvidia-smi shows "No running processes"

**Diagnosis with verbose mode:**
```bash
export TESSERACT_OPENCL_VERBOSE=1
lstmtraining ... 2>&1 | tee training_debug.log
```

**Common causes:**

1. **OpenCL initialization failure:**
   - Look for "Failed to..." messages in verbose output
   - Check: "Failed to create buffer" → GPU memory issue
   - Check: "Failed to create context" → Driver issue
   - Check: "Kernel execution failed" → Compute capability issue

2. **Model not using int8:**
   - If you see NO "MatrixDotVector call" messages
   - GPU requires int8 quantized weights
   - Solution: Use fine-tuning or ensure int8 conversion

3. **Buffer allocation errors:**
   - Look for "Failed to create ... buffer (error -61)" → Out of memory
   - Solution: Reduce `--max_image_MB` or `--batch_size`

4. **Driver/platform issues:**
   - Check if multiple OpenCL platforms exist
   - Try: `clinfo` to list all platforms and devices
   - Ensure NVIDIA platform is being selected

**Verification steps:**

1. **Check verbose output for errors:**
   ```bash
   grep -i "failed\|error" training_debug.log
   ```

2. **Verify MatrixDotVector is being called:**
   ```bash
   grep "MatrixDotVector call" training_debug.log
   # Should see: MatrixDotVector call #1, #2, #3, etc.
   ```

3. **Check GPU process appears:**
   ```bash
   # Start training, then in another terminal:
   watch -n 0.5 nvidia-smi
   # Should show process and GPU utilization
   ```

### Issue: GPU Memory Errors

**Symptoms:**
```
CUDA error: out of memory
OpenCL: Failed to allocate device memory
```

**Solutions:**

1. **Reduce memory allocation:**
   ```bash
   --max_image_MB 4000  # Try lower values: 2000, 1000
   ```

2. **Reduce training batch size:**
   ```bash
   --batch_size 200  # Try lower values: 100, 50
   ```

3. **Use smaller network:**
   ```bash
   # Reduce layer sizes in --net_spec
   # Example: Lfx256 -> Lfx128
   ```

4. **Close other GPU applications:**
   ```bash
   # Check GPU usage
   nvidia-smi
   # Kill unnecessary processes
   ```

5. **Enable GPU memory growth (CUDA only):**
   ```bash
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

### Issue: Training Slower with GPU Than CPU

**Possible causes and solutions:**

1. **Batch size too small for GPU:**
   ```bash
   # Increase batch size to utilize GPU better
   --batch_size 500  # or higher (300-1000)
   ```

2. **Memory allocation too low:**
   ```bash
   # Increase to load more data
   --max_image_MB 8000  # or higher
   ```

3. **Too many CPU threads:**
   ```bash
   # Must set this!
   export OMP_THREAD_LIMIT=1
   ```

4. **PCIe bandwidth bottleneck:**
   - Check GPU is in PCIe x16 slot
   - Verify PCIe 3.0/4.0 is enabled in BIOS

5. **GPU thermal throttling:**
   ```bash
   # Monitor GPU temperature
   nvidia-smi -l 1
   # Improve cooling if temperatures > 80°C
   ```

### Issue: Training Crashes or Hangs

**Solutions:**

1. **Update GPU drivers:**
   ```bash
   # For NVIDIA
   sudo ubuntu-drivers autoinstall
   # OR manually from nvidia.com
   ```

2. **Check for hardware errors:**
   ```bash
   # NVIDIA
   nvidia-smi -a | grep -i error
   
   # Run GPU stress test
   gpu-burn 30  # 30 second burn test
   ```

3. **Reduce complexity:**
   - Start with smaller --max_iterations
   - Use simpler network architecture
   - Reduce --max_image_MB and --batch_size

4. **Enable debug output:**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1  # For CUDA
   --debug_interval 10  # In lstmtraining
   ```

## Best Practices Summary

### ✅ DO:
- Set `OMP_THREAD_LIMIT=1` for GPU training
- **Use `--batch_size 300-1000` for GPU training** (NEW)
- Use `--max_image_MB 8000` or higher for good GPU utilization
- Monitor GPU usage with nvidia-smi/rocm-smi
- Keep training data on fast local storage (SSD)
- Use fine-tuning instead of training from scratch when possible
- Save checkpoints frequently with --debug_interval
- Test with small iterations first, then increase

### ❌ DON'T:
- Don't use default `--batch_size 100` for GPU (use 300-1000 instead)
- Don't run multiple training processes on same GPU without coordination
- Don't set --max_image_MB too high (causes OOM errors)
- Don't forget to set GPU environment variables
- Don't use network drives for training data
- Don't train on GPU with < 4GB memory
- Don't ignore GPU temperature (keep < 80°C)
- Don't set --batch_size too high if you get OOM errors

## Complete Training Script Example

Here's a complete bash script for GPU training:

```bash
#!/bin/bash
# train_with_gpu.sh - LSTM training with GPU acceleration

set -e  # Exit on error

# Configuration
TESSERACT_PREFIX="/usr/local"
MODEL_NAME="my_custom_model"
BASE_MODEL="eng"
TRAINING_DIR="/path/to/training"
OUTPUT_DIR="/path/to/output"
MAX_ITERATIONS=10000
LEARNING_RATE=0.0001

# GPU Configuration (choose one)
# For OpenCL:
export TESSERACT_OPENCL_DEVICE="GPU:0"
# For CUDA:
# export CUDA_VISIBLE_DEVICES=0

# Performance settings
export OMP_THREAD_LIMIT=1

# Create output directory
mkdir -p "${OUTPUT_DIR}/checkpoints"

echo "========================================="
echo "Starting LSTM Training with GPU"
echo "========================================="
echo "Model: ${MODEL_NAME}"
echo "Base: ${BASE_MODEL}"
echo "GPU: ${TESSERACT_OPENCL_DEVICE:-${CUDA_VISIBLE_DEVICES}}"
echo "========================================="

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
elif command -v rocm-smi &> /dev/null; then
    echo "AMD GPU Status:"
    rocm-smi
fi
echo "========================================="

# Run training
lstmtraining \
  --continue_from "${TESSERACT_PREFIX}/share/tessdata/${BASE_MODEL}.traineddata" \
  --traineddata "${OUTPUT_DIR}/${MODEL_NAME}.traineddata" \
  --model_output "${OUTPUT_DIR}/checkpoints/${MODEL_NAME}" \
  --train_listfile "${TRAINING_DIR}/train.list" \
  --eval_listfile "${TRAINING_DIR}/eval.list" \
  --max_iterations ${MAX_ITERATIONS} \
  --learning_rate ${LEARNING_RATE} \
  --target_error_rate 0.01 \
  --debug_interval 100 \
  --max_image_MB 8000 \
  --batch_size 500 \
  --net_mode 192 \
  --weight_range 0.1 \
  --momentum 0.9 \
  2>&1 | tee "${OUTPUT_DIR}/training_${MODEL_NAME}.log"

echo "========================================="
echo "Training Complete!"
echo "Output: ${OUTPUT_DIR}/checkpoints/${MODEL_NAME}.traineddata"
echo "Log: ${OUTPUT_DIR}/training_${MODEL_NAME}.log"
echo "========================================="
```

## Additional Resources

### Documentation
- [Tesseract Training Documentation](https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-4.00.html)
- [GPU Acceleration Overview](GPU_ACCELERATION.md)
- [lstmtraining Man Page](lstmtraining.1.asc)

### Training Data
- [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) - Best models for fine-tuning
- [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) - Faster models
- [langdata](https://github.com/tesseract-ocr/langdata) - Language data for training

### Community
- [Tesseract Forum](https://groups.google.com/g/tesseract-ocr)
- [GitHub Issues](https://github.com/tesseract-ocr/tesseract/issues)
- [GitHub Discussions](https://github.com/tesseract-ocr/tesseract/discussions)

## License

This guide is part of the Tesseract OCR project and is licensed under Apache 2.0.

---

**Note:** GPU acceleration for training is experimental. Performance and compatibility may vary. Please report issues and share your experiences to help improve this feature.
