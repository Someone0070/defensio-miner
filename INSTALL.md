# Defensio Miner - Complete Installation Guide

## A-Z Instructions: From GitHub Clone to Mining

This guide will take you from zero to mining in about 10 minutes.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone Repository](#2-clone-repository)
3. [Install Dependencies](#3-install-dependencies)
4. [Build GPU Support (Optional)](#4-build-gpu-support-optional)
5. [Configure Settings](#5-configure-settings)
6. [Generate Wallets](#6-generate-wallets)
7. [Start Mining](#7-start-mining)
8. [Monitor Progress](#8-monitor-progress)
9. [Consolidate Earnings](#9-consolidate-earnings)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (1GB per worker)
- **Internet**: Stable connection

### For GPU Mining (Optional but Recommended)
- **NVIDIA GPU**: GTX 10xx series or newer
- **CUDA Toolkit**: 11.x or 12.x
- **NVIDIA Drivers**: 470+ recommended

### Check Your System

```bash
# Check Python version (need 3.8+)
python3 --version

# Check if NVIDIA GPU available (Linux/macOS)
nvidia-smi

# Check CUDA version
nvcc --version
```

---

## 2. Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/defensio-miner.git

# Enter the directory
cd defensio-miner
```

---

## 3. Install Dependencies

### Option A: Using Make (Recommended)

```bash
# This installs all Python dependencies
make install-deps
```

### Option B: Manual Installation

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support, also install CuPy
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x
```

### Verify Installation

```bash
python3 -c "import requests, pycardano, mnemonic, numpy; print('Core deps OK')"
python3 -c "import rich; print('Dashboard OK')"
python3 -c "import flask; print('Web dashboard OK')"
```

---

## 4. Build GPU Support (Optional)

Skip this section if you only want CPU mining.

### Find Your GPU Architecture

| GPU Series | Architecture | Build Command |
|------------|--------------|---------------|
| GTX 9xx | Maxwell (sm_50) | `make cuda-sm50` |
| GTX 10xx | Pascal (sm_60) | `make cuda-sm60` |
| GTX 16xx / RTX 20xx | Turing (sm_75) | `make cuda-sm75` |
| RTX 30xx | Ampere (sm_86) | `make cuda-sm86` |
| RTX 40xx | Ada (sm_89) | `make cuda-sm89` |

### Build CUDA Kernel

```bash
# Check your GPU
nvidia-smi

# Build for your architecture (example: RTX 3080)
make cuda-sm86

# Verify build
ls -la cuda/libashmaize_cuda.so
```

### Test CUDA

```bash
python3 src/cuda_solver.py
```

---

## 5. Configure Settings

### Environment Variables (Optional)

Create a `.env` file or export variables:

```bash
# API Configuration
export DEFENSIO_API_BASE="https://mine.defensio.io/api"
export DEFENSIO_NETWORK="mainnet"

# Storage
export DEFENSIO_WALLET_DIR="./wallets"
export DEFENSIO_LOG_DIR="./logs"

# Mining Settings
export MAX_SOLUTIONS_PER_WALLET=100
export CHALLENGE_POLL_INTERVAL_MS=30000
export ASHMAIZE_THREADS=8
```

Or simply use command-line arguments (see Step 7).

---

## 6. Generate Wallets

### Option A: Auto-Generate on Start

The miner automatically generates wallets as needed. Just start mining!

### Option B: Pre-Generate Wallets

```bash
# Generate 100 wallets and register them
python3 miner.py --generate-wallets 100 --register-only

# Check generated wallets
python3 wallet_utils.py list
```

### Backup Your Wallets! ⚠️

```bash
# Your wallets are stored here - BACK THIS UP!
cp wallets/wallets.json ~/backup/wallets_backup.json

# Or export mnemonics for manual backup
python3 wallet_utils.py export-mnemonics -o ~/backup/mnemonics.txt
```

---

## 7. Start Mining

### Basic Usage

```bash
# CPU only (uses all cores)
python3 miner.py

# Specify number of CPU workers
python3 miner.py --cpu-workers 8

# Single GPU + CPU workers
python3 miner.py --gpu 0 --cpu-workers 4

# Multiple GPUs
python3 miner.py --gpu 0,1,2,3 --cpu-workers 2

# With web dashboard
python3 miner.py --gpu 0 --cpu-workers 4 --web-port 8080

# With earnings consolidation
python3 miner.py --gpu 0,1 --cpu-workers 4 --consolidate addr1qxy...
```

### Full Command Reference

```bash
python3 miner.py \
  --cpu-workers 8 \           # Number of CPU mining threads
  --gpu 0,1 \                 # GPU IDs to use (comma-separated)
  --web-port 8080 \           # Enable web dashboard on this port
  --consolidate addr1q... \   # Auto-consolidate to this address
  --max-solutions 100 \       # Switch wallet after N solutions
  --wallet-dir ./wallets \    # Wallet storage directory
  --api-base https://... \    # API endpoint (don't change unless needed)
  --debug                     # Enable debug logging
```

### Run in Background (Linux)

```bash
# Using nohup
nohup python3 miner.py --gpu 0,1 --cpu-workers 4 > miner.log 2>&1 &

# Using screen
screen -S defensio
python3 miner.py --gpu 0,1 --cpu-workers 4
# Press Ctrl+A, D to detach
# screen -r defensio to reattach

# Using systemd (see systemd section below)
```

---

## 8. Monitor Progress

### Terminal Dashboard

The terminal shows a live dashboard automatically:

```
Defensio V1.0 | Hybrid | 9 Workers | 8 CPU + 1 GPU | 1.1K Tokens

●   1  addr1q......na  ████ ████ ████ ████ ████ ████   5  Solved   23s
●   2  addr1q......jz  ████ ████ ████ ████ ██░░ ░░░░   4  Mining       
●   3  addr1q......ac  ████ ████ ████ ████ ████ ████   5  Solved    8s

Today: 151 Solved | Rate:151/h | Hash:246.5K/s | CPU:99% | Task:D09C14
```

### Web Dashboard

If you enabled `--web-port`:

```bash
# Open in browser
http://localhost:8080

# Or from another machine on your network
http://YOUR_IP:8080
```

### Check Statistics

```bash
# View all wallets
python3 wallet_utils.py list

# Get earnings stats from API
python3 wallet_utils.py stats

# View challenge status
python3 wallet_utils.py challenges
```

---

## 9. Consolidate Earnings

### What is Consolidation?

Consolidation transfers all your mining earnings from multiple wallets to a single destination address. This is important because:
- You're mining with many wallets
- You want all tokens in one place
- The Defensio API supports "donating" allocations

### How to Consolidate

```bash
# During mining (auto-consolidates on shutdown)
python3 miner.py --gpu 0 --consolidate addr1qYOUR_MAIN_ADDRESS...

# Manual consolidation (after mining)
python3 miner.py --consolidate-only --consolidate addr1qYOUR_MAIN_ADDRESS...

# Or using wallet utils
python3 wallet_utils.py consolidate addr1qYOUR_MAIN_ADDRESS...
```

### Export Keys for Wallet Import

To access your tokens in a Cardano wallet:

```bash
# Export signing keys
python3 wallet_utils.py export-skeys -o ./skeys

# Import into Eternl wallet:
# 1. Open Eternl
# 2. Add Wallet → More → CLI Signing Keys
# 3. Select the .skey files from ./skeys/
```

---

## 10. Troubleshooting

### Common Issues

#### "Module not found" errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### "CUDA not available"

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CuPy
pip uninstall cupy-cuda12x cupy-cuda11x
pip install cupy-cuda12x  # or cuda11x
```

#### "Connection refused" / API errors

```bash
# Check internet connection
curl https://mine.defensio.io/api/challenge

# Check API status on Defensio Discord
```

#### "No wallets found"

```bash
# Generate wallets first
python3 miner.py --generate-wallets 10 --register-only
```

#### Dashboard not displaying properly

```bash
# Install rich library
pip install rich

# Or disable dashboard
python3 miner.py --no-dashboard
```

#### High memory usage

```bash
# Reduce workers (each uses ~1GB RAM)
python3 miner.py --cpu-workers 4  # instead of 8
```

### Logs

Check logs for detailed error information:

```bash
# View latest log
tail -f logs/miner_*.log

# View all logs
ls -la logs/
```

---

## Advanced: Systemd Service (Linux)

Create `/etc/systemd/system/defensio-miner.service`:

```ini
[Unit]
Description=Defensio Miner
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/defensio-miner
ExecStart=/usr/bin/python3 miner.py --gpu 0,1 --cpu-workers 4 --consolidate addr1q...
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable defensio-miner
sudo systemctl start defensio-miner
sudo systemctl status defensio-miner
```

---

## Advanced: Docker Deployment

```bash
# Build image
docker build -t defensio-miner .

# Run with GPU
docker run -d \
  --name defensio \
  --gpus all \
  -v $(pwd)/wallets:/data/wallets \
  -p 8080:8080 \
  defensio-miner \
  --gpu 0 --cpu-workers 4 --web-port 8080

# View logs
docker logs -f defensio

# Stop
docker stop defensio
```

Or use Docker Compose:

```bash
docker-compose up -d miner-gpu0
docker-compose logs -f
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start mining (CPU) | `python3 miner.py` |
| Start mining (GPU) | `python3 miner.py --gpu 0` |
| Multi-GPU mining | `python3 miner.py --gpu 0,1,2,3` |
| With web dashboard | `python3 miner.py --web-port 8080` |
| Generate wallets | `python3 miner.py --generate-wallets 100 --register-only` |
| List wallets | `python3 wallet_utils.py list` |
| Check stats | `python3 wallet_utils.py stats` |
| Consolidate | `python3 wallet_utils.py consolidate addr1q...` |
| Export keys | `python3 wallet_utils.py export-skeys` |
| View logs | `tail -f logs/miner_*.log` |

---

## Support

- **Defensio Website**: https://defensio.io
- **Defensio Discord**: Join for community support
- **Issues**: Open a GitHub issue

---

## ⚠️ Important Reminders

1. **BACKUP YOUR WALLETS** - The `wallets/wallets.json` file contains your keys!
2. **Mining ends December 4, 2025** - The Scavenger Mine is time-limited
3. **Consolidate before claiming** - Make sure all earnings go to one address
4. **This is unofficial software** - Use at your own risk

Happy Mining! 🚀
