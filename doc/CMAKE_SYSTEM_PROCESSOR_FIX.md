# CMAKE_SYSTEM_PROCESSOR Detection Fix

## Problem

In some build environments (particularly containers, CI systems, or certain cross-compilation scenarios), CMake's automatic detection of `CMAKE_SYSTEM_PROCESSOR` can fail, resulting in an empty value. This causes several issues:

1. **SIMD Optimizations Disabled**: The code that checks processor architecture to enable AVX, AVX2, SSE, or NEON optimizations fails to match, falling through to the `else` block which disables all optimizations.
2. **GPU Code Issues**: GPU acceleration code added for OpenCL/CUDA relies on proper processor detection.
3. **Performance Impact**: Without SIMD optimizations, OCR performance can be 2-10x slower.

## Solution

Added a three-tier fallback mechanism to ensure `CMAKE_SYSTEM_PROCESSOR` is always set:

### Tier 1: Apple Cross-Compilation (Existing)
For Apple systems where CMake doesn't populate `CMAKE_SYSTEM_PROCESSOR` during cross-compilation:
- Uses `CMAKE_OSX_ARCHITECTURES` to determine processor
- Already existed in codebase

### Tier 2: Unix System Detection (NEW)
For Unix-like systems where `CMAKE_SYSTEM_PROCESSOR` is empty:
```cmake
execute_process(
  COMMAND uname -m
  OUTPUT_VARIABLE DETECTED_PROCESSOR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
```
- Executes `uname -m` to get processor architecture
- Common outputs: `x86_64`, `aarch64`, `armv7l`, etc.
- Works on Linux, macOS, BSD, etc.

### Tier 3: Compiler-Based Fallback (NEW)
When both CMake and uname fail (e.g., Windows without proper detection):
- Uses `CMAKE_SIZEOF_VOID_P` to determine 32-bit vs 64-bit
- 64-bit: assumes `x86_64` (or `AMD64` on Windows)
- 32-bit: assumes `i686` (or `x86` on Windows)

## Code Location

File: `CMakeLists.txt`
Lines: ~99-140 (after Apple cross-compilation handling)

## Testing

Verified with three test scenarios:

### Test 1: Normal Detection
```bash
cmake .. -DBUILD_TRAINING_TOOLS=OFF
```
Result: ✅ `CMAKE_SYSTEM_PROCESSOR=<x86_64>` (detected by CMake)

### Test 2: Forced Empty (Simulates Container/CI)
```bash
cmake .. -DCMAKE_SYSTEM_PROCESSOR="" -DBUILD_TRAINING_TOOLS=OFF
```
Result: ✅ `CMAKE_SYSTEM_PROCESSOR=<x86_64>` (fallback via uname)

### Test 3: SIMD Detection
After empty CMAKE_SYSTEM_PROCESSOR:
```
-- Performing Test HAVE_AVX - Success
-- Performing Test HAVE_AVX2 - Success
-- Performing Test HAVE_AVX512F - Success
-- Performing Test HAVE_SSE4_1 - Success
```
Result: ✅ All SIMD optimizations correctly enabled

## Messages

The fix provides informative status messages:

When fallback detection is needed:
```
-- CMAKE_SYSTEM_PROCESSOR is empty, attempting to detect...
-- CMAKE_SYSTEM_PROCESSOR set to 'x86_64' from uname
```

Or if uname fails:
```
-- CMAKE_SYSTEM_PROCESSOR is empty, attempting to detect...
-- CMAKE_SYSTEM_PROCESSOR set to 'x86_64' (64-bit assumed)
```

## Impact

- ✅ **No Breaking Changes**: Existing builds continue to work unchanged
- ✅ **Fixes Container/CI Builds**: Ensures proper detection in all environments
- ✅ **Enables Optimizations**: SIMD flags correctly set even when CMake detection fails
- ✅ **GPU Support**: GPU acceleration code receives correct architecture information
- ✅ **Better Diagnostics**: Clear messages help debug detection issues

## Related Issues

This fix addresses scenarios similar to:
- Docker containers without proper `/proc` access
- Minimal CI environments (GitHub Actions, GitLab CI, etc.)
- Cross-compilation toolchains with incomplete CMake support
- Windows Subsystem for Linux (WSL) edge cases

## Backward Compatibility

The fallback code only executes when `CMAKE_SYSTEM_PROCESSOR` is empty or not set. When CMake successfully detects the processor (the normal case), the fallback code is skipped entirely, ensuring zero impact on existing builds.
