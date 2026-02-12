# Tesseract OCR Quick Start Installation Guide

This guide will help you get Tesseract installed and running on your system quickly.

## Table of Contents
- [Quick Installation (Recommended for Most Users)](#quick-installation-recommended-for-most-users)
- [Building from Source](#building-from-source)
- [Verifying Installation](#verifying-installation)
- [First Steps](#first-steps)
- [Troubleshooting](#troubleshooting)

---

## Quick Installation (Recommended for Most Users)

For most users, installing pre-built binaries is the easiest and fastest option.

### Ubuntu/Debian

```bash
# Install Tesseract
sudo apt update
sudo apt install tesseract-ocr

# Install language data (English is included by default)
# For additional languages:
sudo apt install tesseract-ocr-[lang]
# Example: sudo apt install tesseract-ocr-fra  # French
```

### macOS

```bash
# Install using Homebrew
brew install tesseract

# For additional languages:
brew install tesseract-lang
```

### Windows

**Option 1: Using Installer (Recommended)**
1. Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer
3. Add Tesseract to your PATH (installer should do this automatically)

**Option 2: Using Chocolatey**
```powershell
choco install tesseract
```

### Fedora/RHEL/CentOS

```bash
sudo dnf install tesseract
```

### Arch Linux

```bash
sudo pacman -S tesseract tesseract-data-eng
```

---

## Building from Source

If you need the latest features, GPU acceleration, or want to customize the build, follow these steps.

### Prerequisites

**Ubuntu/Debian:**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  libleptonica-dev \
  libarchive-dev \
  libcurl4-openssl-dev \
  libpango1.0-dev \
  autoconf \
  automake \
  libtool
```

**macOS:**
```bash
# Install dependencies
brew install cmake pkg-config leptonica \
  autoconf automake libtool pango
```

**Windows:**
- Install Visual Studio 2019 or later
- Install CMake
- Install vcpkg for dependencies:
  ```powershell
  vcpkg install leptonica:x64-windows
  ```

**For detailed dependency information, see [doc/DEPENDENCIES.md](doc/DEPENDENCIES.md)**

### Step 1: Clone the Repository

```bash
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
```

### Step 2: Choose Your Build Method

#### Option A: CMake (Recommended, Cross-Platform)

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build (use -j$(nproc) to use all CPU cores on Linux/macOS)
make -j$(nproc)

# Install
sudo make install

# Update library cache (Linux only)
sudo ldconfig
```

**For GPU acceleration (OpenCL or CUDA):**
```bash
cmake .. -DENABLE_OPENCL=ON
# or
cmake .. -DENABLE_CUDA=ON
```

See [doc/GPU_ACCELERATION.md](doc/GPU_ACCELERATION.md) for details.

#### Option B: Autotools (Linux/Unix/macOS)

```bash
# Generate configure script
./autogen.sh

# Configure
./configure

# Build
make

# Install
sudo make install
sudo ldconfig

# Optional: Build training tools
make training
sudo make training-install
```

### Step 3: Download Language Data

Tesseract requires language data files to work. Download the files for your language(s):

**Quick method (English only):**
```bash
# Create tessdata directory
sudo mkdir -p /usr/local/share/tessdata

# Download English data
wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
sudo mv eng.traineddata /usr/local/share/tessdata/
```

**For multiple languages:**
```bash
# Clone the tessdata repository (warning: large download ~1.2 GB)
git clone https://github.com/tesseract-ocr/tessdata.git

# Copy desired language files
sudo cp tessdata/*.traineddata /usr/local/share/tessdata/
```

**Alternative tessdata options:**
- `tessdata` - Best quality, larger files
- `tessdata_best` - Highest quality, largest files  
- `tessdata_fast` - Faster but lower quality

Choose based on your needs for speed vs accuracy.

---

## Verifying Installation

After installation, verify that Tesseract is working:

### Check Version

```bash
tesseract --version
```

You should see output like:
```
tesseract 5.5.0
 leptonica-1.84.1
  libgif 5.2.2 : libjpeg 8d (libjpeg-turbo 2.1.5.1) : libpng 1.6.43 : libtiff 4.6.0 : zlib 1.3.1 : libwebp 1.4.0
```

### List Available Languages

```bash
tesseract --list-langs
```

You should see at least:
```
List of available languages (2):
eng
osd
```

### Test OCR on a Sample Image

```bash
# Create a simple test image with text (or use your own)
echo "Hello Tesseract!" | convert -pointsize 32 label:@- test.png

# Run OCR
tesseract test.png output

# View results
cat output.txt
```

The output should contain "Hello Tesseract!"

---

## First Steps

### Basic Usage

```bash
# OCR an image file
tesseract input.png output

# Specify language
tesseract input.png output -l eng

# Get PDF output instead of text
tesseract input.png output pdf
```

### Common Options

```bash
# Use specific OCR engine mode (0=Legacy, 1=LSTM, 2=Legacy+LSTM, 3=Default)
tesseract input.png output --oem 1

# Set page segmentation mode (6=single block, 3=auto)
tesseract input.png output --psm 6

# Multiple languages
tesseract input.png output -l eng+fra
```

### Getting Help

```bash
# Show all available options
tesseract --help

# Show extra help
tesseract --help-extra

# Show version and build info
tesseract --version
```

---

## Troubleshooting

### "tesseract: command not found"

**Solution:**
- Verify installation completed successfully
- Check if Tesseract is in your PATH:
  ```bash
  which tesseract  # Linux/macOS
  where tesseract  # Windows
  ```
- On Windows, add Tesseract directory to PATH environment variable

### "Error opening data file"

**Solution:**
- Download language data files (see Step 3 above)
- Set TESSDATA_PREFIX environment variable:
  ```bash
  export TESSDATA_PREFIX=/usr/local/share/tessdata/  # Linux/macOS
  set TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata  # Windows
  ```

### "Cannot find required library Leptonica"

**Solution:**
- Install Leptonica library:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libleptonica-dev
  
  # macOS
  brew install leptonica
  ```
- See [doc/DEPENDENCIES.md](doc/DEPENDENCIES.md) for detailed instructions

### Poor OCR Accuracy

**Solutions:**
- Use higher quality/resolution images (300 DPI recommended)
- Pre-process images (convert to grayscale, increase contrast, remove noise)
- Try different page segmentation modes (`--psm`)
- Use `tessdata_best` for higher accuracy (slower)
- Consider training custom data for specialized fonts/documents

### Build Errors

**Common solutions:**
- Ensure all dependencies are installed (see Prerequisites above)
- Clear build directory and try again: `rm -rf build && mkdir build`
- Check [doc/DEPENDENCIES.md](doc/DEPENDENCIES.md) for platform-specific notes
- Search existing issues: https://github.com/tesseract-ocr/tesseract/issues

---

## Next Steps

### Learn More

- **Command Line Usage**: https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
- **Improve OCR Quality**: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- **API Documentation**: See [README.md](README.md#for-developers)
- **Training Custom Models**: https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html

### Additional Resources

- **Official Documentation**: https://tesseract-ocr.github.io/tessdoc/
- **GitHub Repository**: https://github.com/tesseract-ocr/tesseract
- **User Forum**: https://groups.google.com/g/tesseract-ocr
- **Issue Tracker**: https://github.com/tesseract-ocr/tesseract/issues

### Advanced Features

- **GPU Acceleration**: See [doc/GPU_ACCELERATION.md](doc/GPU_ACCELERATION.md)
- **Building Training Tools**: See [INSTALL.GIT.md](INSTALL.GIT.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Basic OCR | `tesseract input.png output` |
| Specify language | `tesseract input.png output -l fra` |
| Multiple languages | `tesseract input.png output -l eng+fra` |
| PDF output | `tesseract input.png output pdf` |
| Check version | `tesseract --version` |
| List languages | `tesseract --list-langs` |
| Show help | `tesseract --help` |

---

**Need help?** Check the [troubleshooting section](#troubleshooting) above or visit the [official documentation](https://tesseract-ocr.github.io/tessdoc/).
