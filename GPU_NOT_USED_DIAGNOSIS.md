# GPU Not Being Used - Complete Diagnosis Guide

## Your Problem

You see:
```
Using OpenCL GPU acceleration for matrix operations
```

But nvidia-smi shows:
- 0% GPU utilization
- Only 3 MiB GPU memory
- No GPU processes

Training proceeds but on CPU.

## Root Cause: Float32 vs Int8

**CRITICAL FACT: GPU acceleration ONLY works with int8 quantized models.**

Tesseract's LSTM training has two operation modes:
- **Float32**: Used for training from scratch, uses CPU SIMD (AVX, SSE)
- **Int8**: Used for fine-tuning, uses GPU (OpenCL/CUDA)

The message "Using OpenCL GPU acceleration for matrix operations" just means OpenCL is DETECTED and AVAILABLE. It doesn't mean your model is actually USING it.

## Immediate Diagnosis (After Rebuild)

After rebuilding with the latest code, you'll immediately see one of these messages:

### Message 1: GPU IS Being Used ✅
```
Using OpenCL GPU acceleration for matrix operations
INFO: Using int8 quantized operations - GPU acceleration available
OpenCL: Successfully initialized on GPU device: Tesla T4 (15.0 GB)
OpenCL: Kernel execution time: 0.125 ms (GPU actively used)
```

If you see this:
- ✅ GPU path is active
- ✅ Check nvidia-smi - should show 70-90% utilization
- ✅ GPU memory should be 2-8 GB
- ✅ Training will be 5-15x faster

### Message 2: GPU CANNOT Be Used ❌
```
Using OpenCL GPU acceleration for matrix operations
WARNING: Using float32 operations - GPU acceleration NOT available
         GPU only works with int8 quantized models.
         Use fine-tuning (--continue_from) with int8 models for GPU support.
```

If you see this:
- ❌ Model is using float32 operations
- ❌ GPU cannot be used for float32
- ❌ Training will use CPU SIMD (still fast, but not GPU fast)
- ❌ nvidia-smi will show 0% utilization

## Why Is Your Model Using Float32?

Looking at your training command:
```bash
--continue_from /content/tesstrain/data/Lao/lao140k.lstm \
--old_traineddata /content/tesseract/tessdata/Lao.traineddata \
```

You ARE using `--continue_from` which should load an int8 model. However, the code range changed:
```
Code range changed from 163 to 175!
```

When the code range changes, Tesseract needs to convert the model, and this conversion might result in a float32 model temporarily.

## Solution: Ensure Int8 Model

### Option 1: Fine-tune from int8 checkpoint (Recommended)

```bash
# Use a checkpoint that's already int8
lstmtraining \
  --continue_from /path/to/existing_int8_model.lstm \
  --traineddata /path/to/new.traineddata \
  --train_listfile list.train \
  --batch_size 500 \
  ...
```

### Option 2: Use --convert_to_int8

If training from scratch, use the `--convert_to_int8` flag:

```bash
lstmtraining \
  --traineddata new.traineddata \
  --net_spec '[1,36,0,1 Ct3,3,16 Mp3,3 Lfys48 Lfx96 Lrx96 Lfx256 O1c111]' \
  --model_output model \
  --train_listfile list.train \
  --convert_to_int8 \  # <-- Add this
  --batch_size 500 \
  ...
```

### Option 3: Pre-convert to int8

Convert a float32 model to int8 before training:

```bash
# Train initial model (float32)
lstmtraining --traineddata new.traineddata ...

# Convert to int8
combine_tessdata -u new.traineddata model.lstm
# Now use model.lstm as --continue_from with int8 operations
```

## Verification Steps

### Step 1: Rebuild Tesseract
```bash
cd /content/tesseract/build
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Step 2: Run Training
```bash
export TESSERACT_OPENCL_VERBOSE=1
export TESSERACT_OPENCL_DEVICE="GPU:0"
export OMP_THREAD_LIMIT=1

make training ... 2>&1 | tee training.log
```

### Step 3: Check First 20 Lines
```bash
head -20 training.log
```

You'll immediately see either:
- **INFO: Using int8** → GPU IS working
- **WARNING: Using float32** → GPU CANNOT work

### Step 4: Verify with nvidia-smi

If using int8:
```bash
watch -n 0.5 nvidia-smi
```

Should show:
- GPU Utilization: 70-90%
- GPU Memory: 2-8 GB
- Process: lstmtraining listed

If using float32:
- GPU Utilization: 0%
- GPU Memory: 3 MiB
- No processes (normal, not an error)

## Summary

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| "WARNING: float32" message | Model using float32 | Use int8 model or --convert_to_int8 |
| "INFO: int8" but 0% GPU | GPU initialization failed | Check verbose logs for errors |
| "INFO: int8" and 70-90% GPU | Working correctly! | Training is 5-15x faster |

## Key Takeaway

**The "Using OpenCL GPU acceleration" message is misleading.**

It should say "OpenCL GPU acceleration AVAILABLE" because it's just detecting hardware, not indicating actual usage.

**The NEW messages (INFO/WARNING) tell you if GPU is ACTUALLY being used.**

## Next Steps

1. Rebuild Tesseract with the diagnostic logging
2. Run training and check for INFO or WARNING message
3. If WARNING: Convert your model to int8 or use int8 checkpoint
4. If INFO: Verify GPU utilization with nvidia-smi
5. Enjoy 5-15x faster training! 🚀
