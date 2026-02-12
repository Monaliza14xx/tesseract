# Installing Dependencies for Tesseract

This guide explains how to install the required dependencies to build Tesseract from source.

## Required Dependencies

### Leptonica (Required)

Leptonica is an image processing library and is **required** to build Tesseract. You need version 1.74 or later.

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libleptonica-dev
```

#### macOS
```bash
brew install leptonica
```

#### Fedora/CentOS/RHEL
```bash
sudo dnf install leptonica-devel
```

#### Windows (vcpkg)
```bash
vcpkg install leptonica
```

#### Build from Source
If your package manager doesn't have Leptonica or you need a specific version:
```bash
git clone https://github.com/DanBloomberg/leptonica.git
cd leptonica
mkdir build && cd build
cmake ..
make
sudo make install
```

### Other Build Dependencies

#### Ubuntu/Debian
```bash
# Compiler and build tools
sudo apt-get install build-essential cmake pkg-config

# Optional dependencies for training tools
sudo apt-get install libarchive-dev libcurl4-openssl-dev
sudo apt-get install libpango1.0-dev
sudo apt-get install autoconf automake libtool

# For TIFF support
sudo apt-get install libtiff-dev

# For PNG support  
sudo apt-get install libpng-dev

# For JPEG support
sudo apt-get install libjpeg-dev
```

#### macOS
```bash
# Using Homebrew
brew install cmake pkg-config
brew install leptonica
brew install autoconf automake
brew install libarchive curl pango

# Optional: ICU for training tools
brew install icu4c
```

#### Fedora/CentOS/RHEL
```bash
sudo dnf install gcc-c++ cmake pkgconfig
sudo dnf install leptonica-devel
sudo dnf install libarchive-devel libcurl-devel
sudo dnf install pango-devel
```

## Building Tesseract

After installing dependencies, you can build Tesseract:

### Using CMake (Recommended)
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

### Using Autotools
```bash
./autogen.sh
./configure
make
sudo make install
sudo ldconfig
```

## Troubleshooting

### "Cannot find required library Leptonica"
This means Leptonica is not installed or CMake cannot find it. Install libleptonica-dev (Ubuntu/Debian) or leptonica (macOS/vcpkg).

### "pkg-config not found"
Install pkg-config:
- Ubuntu/Debian: `sudo apt-get install pkg-config`
- macOS: `brew install pkg-config`
- Fedora: `sudo dnf install pkgconfig`

### Leptonica found but wrong version
If you have an older version of Leptonica, you'll need to upgrade it or build from source. Tesseract requires Leptonica 1.74 or later.

### Custom installation path
If you installed Leptonica in a non-standard location, tell CMake where to find it:
```bash
cmake -DLeptonica_DIR=/path/to/leptonica/cmake ..
```

## For More Information

- [Official Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/)
- [Compiling Guide](https://tesseract-ocr.github.io/tessdoc/Compiling.html)
- [Leptonica Homepage](http://www.leptonica.org/)
- [Leptonica GitHub](https://github.com/DanBloomberg/leptonica)
