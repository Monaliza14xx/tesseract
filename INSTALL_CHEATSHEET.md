# Tesseract Installation Cheat Sheet

## One-Command Installation

### Ubuntu/Debian
```bash
sudo apt update && sudo apt install tesseract-ocr
```

### macOS
```bash
brew install tesseract
```

### Windows (Chocolatey)
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

## Verify Installation
```bash
tesseract --version
tesseract --list-langs
```

## Test OCR
```bash
tesseract input_image.png output_file
cat output_file.txt
```

## Common Commands
```bash
# Basic OCR
tesseract image.png output

# Specify language
tesseract image.png output -l eng

# Multiple languages
tesseract image.png output -l eng+fra

# PDF output
tesseract image.png output pdf

# Show help
tesseract --help
```

## Build from Source (Quick)

### Ubuntu/Debian
```bash
# Install dependencies
sudo apt-get install -y build-essential cmake pkg-config libleptonica-dev

# Clone and build
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig

# Download English language data
wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
sudo mkdir -p /usr/local/share/tessdata
sudo mv eng.traineddata /usr/local/share/tessdata/
```

### macOS
```bash
# Install dependencies
brew install cmake pkg-config leptonica

# Clone and build
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
sudo make install

# Download English language data
wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
sudo mkdir -p /usr/local/share/tessdata
sudo mv eng.traineddata /usr/local/share/tessdata/
```

## Troubleshooting

**Command not found?**
```bash
which tesseract  # Check if installed
```

**Missing language data?**
```bash
export TESSDATA_PREFIX=/usr/local/share/tessdata/
```

**Need more help?**
- Full Guide: [QUICKSTART.md](QUICKSTART.md)
- Dependencies: [doc/DEPENDENCIES.md](doc/DEPENDENCIES.md)
- Official Docs: https://tesseract-ocr.github.io/tessdoc/
