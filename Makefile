# Defensio Miner - Makefile
# =========================

# CUDA Configuration
NVCC := nvcc
CUDA_PATH ?= /usr/local/cuda
CUDA_ARCH ?= sm_70  # Default architecture (V100), change for your GPU

# Python
PYTHON := python3

# Directories
SRC_DIR := src
CUDA_DIR := cuda
BUILD_DIR := build
WALLET_DIR := wallets
LOG_DIR := logs

# CUDA compiler flags
NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -Xcompiler -fPIC --shared
NVCC_INCLUDES := -I$(CUDA_PATH)/include

# Library names
CUDA_LIB := $(CUDA_DIR)/libashmaize_cuda.so

# Default target
.PHONY: all
all: setup cuda

# Setup directories
.PHONY: setup
setup:
	@mkdir -p $(BUILD_DIR) $(WALLET_DIR) $(LOG_DIR)
	@echo "Directories created"

# Build CUDA library
.PHONY: cuda
cuda: setup
	@echo "Building CUDA Ashmaize kernel..."
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) -o $(CUDA_LIB) $(CUDA_DIR)/ashmaize_kernel.cu && \
		echo "CUDA library built: $(CUDA_LIB)"; \
	else \
		echo "Warning: NVCC not found. CUDA support disabled."; \
		echo "Install CUDA toolkit or use CPU-only mode."; \
	fi

# Build for different GPU architectures
.PHONY: cuda-sm50
cuda-sm50:
	@$(MAKE) cuda CUDA_ARCH=sm_50

.PHONY: cuda-sm60
cuda-sm60:
	@$(MAKE) cuda CUDA_ARCH=sm_60

.PHONY: cuda-sm70
cuda-sm70:
	@$(MAKE) cuda CUDA_ARCH=sm_70

.PHONY: cuda-sm75
cuda-sm75:
	@$(MAKE) cuda CUDA_ARCH=sm_75

.PHONY: cuda-sm80
cuda-sm80:
	@$(MAKE) cuda CUDA_ARCH=sm_80

.PHONY: cuda-sm86
cuda-sm86:
	@$(MAKE) cuda CUDA_ARCH=sm_86

.PHONY: cuda-sm89
cuda-sm89:
	@$(MAKE) cuda CUDA_ARCH=sm_89

# Install Python dependencies
.PHONY: install-deps
install-deps:
	@echo "Installing Python dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install requests pycardano mnemonic numpy
	@if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "Installing CuPy for GPU support..."; \
		$(PYTHON) -m pip install cupy-cuda12x || $(PYTHON) -m pip install cupy-cuda11x || echo "CuPy installation failed, using CPU-only mode"; \
	fi
	@echo "Dependencies installed"

# Run the miner
.PHONY: run
run:
	$(PYTHON) miner.py

# Run with specific options
.PHONY: run-cpu
run-cpu:
	$(PYTHON) miner.py --cpu-workers 4

.PHONY: run-gpu
run-gpu:
	$(PYTHON) miner.py --gpu 0 --cpu-workers 2

.PHONY: run-multi-gpu
run-multi-gpu:
	$(PYTHON) miner.py --gpu 0,1 --cpu-workers 2

# Generate wallets
.PHONY: generate-wallets
generate-wallets:
	$(PYTHON) miner.py --generate-wallets 10 --register-only

# Consolidate earnings
.PHONY: consolidate
consolidate:
	@if [ -z "$(ADDRESS)" ]; then \
		echo "Usage: make consolidate ADDRESS=addr1..."; \
		exit 1; \
	fi
	$(PYTHON) miner.py --consolidate-only --consolidate $(ADDRESS)

# Test CUDA
.PHONY: test-cuda
test-cuda:
	$(PYTHON) $(SRC_DIR)/cuda_solver.py

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(CUDA_LIB)
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	@echo "Cleaned build artifacts"

# Clean all (including wallets and logs - BE CAREFUL!)
.PHONY: clean-all
clean-all: clean
	@echo "WARNING: This will delete wallets and logs!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf $(WALLET_DIR) $(LOG_DIR); \
		echo "All data cleaned"; \
	else \
		echo "Aborted"; \
	fi

# Show help
.PHONY: help
help:
	@echo "Defensio Miner - Build System"
	@echo "=============================="
	@echo ""
	@echo "Build targets:"
	@echo "  make all           - Build everything (setup + CUDA)"
	@echo "  make setup         - Create directories"
	@echo "  make cuda          - Build CUDA library (default sm_70)"
	@echo "  make cuda-sm50     - Build for Maxwell (GTX 9xx)"
	@echo "  make cuda-sm60     - Build for Pascal (GTX 10xx)"
	@echo "  make cuda-sm70     - Build for Volta (V100, Titan V)"
	@echo "  make cuda-sm75     - Build for Turing (RTX 20xx)"
	@echo "  make cuda-sm80     - Build for Ampere (A100)"
	@echo "  make cuda-sm86     - Build for Ampere (RTX 30xx)"
	@echo "  make cuda-sm89     - Build for Ada Lovelace (RTX 40xx)"
	@echo ""
	@echo "Run targets:"
	@echo "  make run           - Run miner with defaults"
	@echo "  make run-cpu       - Run CPU-only (4 workers)"
	@echo "  make run-gpu       - Run with GPU 0 + 2 CPU workers"
	@echo "  make run-multi-gpu - Run with GPUs 0,1 + 2 CPU workers"
	@echo ""
	@echo "Utility targets:"
	@echo "  make install-deps       - Install Python dependencies"
	@echo "  make generate-wallets   - Generate and register 10 wallets"
	@echo "  make consolidate ADDRESS=addr1...  - Consolidate to address"
	@echo "  make test-cuda          - Test CUDA functionality"
	@echo ""
	@echo "Cleanup targets:"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make clean-all     - Remove all data (CAREFUL!)"
	@echo ""
	@echo "GPU Architecture Guide:"
	@echo "  sm_50: GTX 750, 9xx (Maxwell)"
	@echo "  sm_60: GTX 10xx, P100 (Pascal)"
	@echo "  sm_70: V100, Titan V (Volta)"
	@echo "  sm_75: RTX 20xx, T4 (Turing)"
	@echo "  sm_80: A100, A30 (Ampere)"
	@echo "  sm_86: RTX 30xx, A10, A40 (Ampere)"
	@echo "  sm_89: RTX 40xx, L4, L40 (Ada Lovelace)"
