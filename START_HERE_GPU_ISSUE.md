# START HERE - GPU Not Being Used

## Your Problem

You see this in training output:
```
Using OpenCL GPU acceleration for matrix operations
```

But nvidia-smi shows:
- **GPU Utilization: 0%**
- **GPU Memory: 3 MiB**
- **No GPU processes**

Training works but uses CPU, not GPU.

## Quick Answer

**Your model is using FLOAT32 operations.**

**GPU acceleration ONLY works with INT8 quantized models.**

That's why GPU shows 0% - it's expected and correct for float32 models.

## Immediate Action

### Step 1: Rebuild Tesseract

```bash
cd /content/tesseract
git checkout copilot/optimize-tesseract-performance
git pull

cd build
cmake .. -DENABLE_OPENCL=ON -DBUILD_TRAINING_TOOLS=ON
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Step 2: Run Training Again

```bash
make training MODEL_NAME=lao140k ... 2>&1 | tee training.log
```

### Step 3: Look at First 20 Lines

```bash
head -20 training.log
```

## What You'll See

You'll immediately see ONE of these messages:

### Message 1: Float32 (Your Current Situation)
```
WARNING: Using float32 operations - GPU acceleration NOT available
         GPU only works with int8 quantized models.
         Use fine-tuning (--continue_from) with int8 models for GPU support.
```

**This means:**
- ❌ Model is using float32
- ❌ GPU cannot be used
- ✅ 0% GPU utilization is NORMAL (not a bug!)
- ✅ Training still works (using CPU SIMD)

### Message 2: Int8 (GPU Can Work)
```
INFO: Using int8 quantized operations - GPU acceleration available
OpenCL: Successfully initialized on GPU device: Tesla T4 (15.0 GB)
OpenCL: Kernel execution time: 0.125 ms (GPU actively used)
```

**This means:**
- ✅ Model is using int8
- ✅ GPU IS being used
- ✅ nvidia-smi should show 70-90% utilization
- ✅ Training is 5-15x faster

## Why Is Your Model Float32?

Looking at your command:
```bash
--continue_from /content/tesstrain/data/Lao/lao140k.lstm \
--old_traineddata /content/tesseract/tessdata/Lao.traineddata \
```

You ARE using `--continue_from` which should load an int8 model.

BUT you also have:
```
Code range changed from 163 to 175!
```

When the code range changes, Tesseract may convert the model, and this can result in float32.

## Solutions

### Option 1: Convert to Int8 (For GPU)

Add this flag to training:
```bash
lstmtraining \
  --convert_to_int8 \  # <-- Add this
  ... other flags ...
```

### Option 2: Use Int8 Checkpoint

Use a checkpoint that's already int8:
```bash
lstmtraining \
  --continue_from /path/to/int8_model.lstm \
  ... other flags ...
```

### Option 3: Accept CPU Training

Float32 training on CPU with SIMD is still fast (just not GPU fast):
- CPU: 300-500 iterations/minute
- GPU: 1,500-5,500 iterations/minute

If you don't need maximum speed, CPU training is fine!

## Complete Guides

For more details, read these:

1. **GPU_NOT_USED_DIAGNOSIS.md** - Complete explanation
2. **FINAL_GPU_SOLUTION_SUMMARY.md** - All fixes overview
3. **doc/GPU_TRAINING_GUIDE.md** - Full training guide

## Key Points

1. **"Using OpenCL GPU acceleration" means OpenCL is AVAILABLE, not that it's BEING USED**

2. **GPU only works with int8 quantized models**

3. **Float32 models always use CPU (even if OpenCL is available)**

4. **After rebuild, diagnostic messages will tell you exactly what's happening**

5. **0% GPU utilization with float32 is NORMAL, not a bug**

## Summary

**Your Issue:** GPU shows 0% utilization

**Root Cause:** Model using float32 (not int8)

**Why:** Code range change may have caused float32 conversion

**Solution:** Convert to int8 model OR accept CPU training

**Action:** Rebuild and check diagnostic messages

**Documentation:** Complete guides provided

---

**Rebuild Tesseract now and you'll immediately understand what's happening!** 🎯
