# Defensio Scavenger Mine - GPU+CPU Hybrid Miner

A high-performance mining solution for [Defensio.io](https://defensio.io) Scavenger Mine that supports:

- **GPU Mining** (CUDA) for NVIDIA GPUs
- **CPU Mining** with multi-threading
- **Multi-GPU setups** with automatic detection
- **Automatic wallet cycling** when solution limits are reached
- **Auto-generation of new wallets** as needed
- **Earnings consolidation** to a single address
- **Challenge backlog processing** (up to 24 hours)

## Requirements

### System Requirements
- Python 3.8 or higher
- Linux, macOS, or Windows

### For GPU Mining (Optional)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.x or 12.x
- NVIDIA drivers

### Python Dependencies
```
requests
pycardano
mnemonic
numpy
cupy (optional, for GPU support)
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <repo-url> defensio-miner
cd defensio-miner

# Create directories and install dependencies
make setup
make install-deps
```

### 2. Build CUDA Support (Optional)
```bash
# Detect your GPU and build
nvidia-smi  # Check your GPU

# Build for your GPU architecture
make cuda-sm86  # RTX 30xx series
# OR
make cuda-sm89  # RTX 40xx series
# OR
make cuda-sm70  # V100 / Titan V
```

### 3. Run the Miner
```bash
# Auto-detect GPU and run
python miner.py

# Specify workers
python miner.py --cpu-workers 4 --gpu 0

# Multi-GPU setup
python miner.py --cpu-workers 2 --gpu 0,1,2

# With consolidation address
python miner.py --gpu 0,1 --consolidate addr1qxy...
```

## Usage

### Basic Commands

```bash
# Run with defaults (auto-detect)
python miner.py

# CPU-only mode
python miner.py --cpu-workers 8

# Single GPU + CPU workers
python miner.py --gpu 0 --cpu-workers 4

# Multiple GPUs
python miner.py --gpu 0,1,2,3 --cpu-workers 2

# Pre-generate wallets
python miner.py --generate-wallets 100 --register-only

# Consolidate earnings to single address
python miner.py --consolidate-only --consolidate addr1qxy...
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cpu-workers N` | Number of CPU mining workers | CPU cores |
| `--gpu IDs` | Comma-separated GPU IDs (e.g., "0,1,2") | Auto-detect |
| `--consolidate ADDR` | Cardano address to consolidate earnings | None |
| `--generate-wallets N` | Pre-generate N wallets before mining | 0 |
| `--register-only` | Only register wallets, don't mine | False |
| `--consolidate-only` | Only consolidate, don't mine | False |
| `--wallet-dir PATH` | Directory for wallet storage | ./wallets |
| `--api-base URL` | API base URL | https://mine.defensio.io/api |
| `--max-solutions N` | Max solutions per wallet before switching | 100 |
| `--debug` | Enable debug logging | False |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFENSIO_API_BASE` | API base URL | https://mine.defensio.io/api |
| `DEFENSIO_NETWORK` | Cardano network (mainnet/testnet) | mainnet |
| `DEFENSIO_WALLET_DIR` | Wallet storage directory | ./wallets |
| `DEFENSIO_LOG_DIR` | Log file directory | ./logs |
| `CHALLENGE_POLL_INTERVAL_MS` | Challenge poll interval | 30000 |
| `MAX_SOLUTIONS_PER_WALLET` | Solutions before wallet switch | 100 |
| `ASHMAIZE_BATCH_SIZE` | Batch size for hashing | 16 |
| `ASHMAIZE_THREADS` | CPU threads for solving | CPU cores |

## Architecture

### How It Works

1. **Challenge Polling**: The miner polls the Defensio API for new hourly challenges
2. **Wallet Management**: Automatically generates and cycles through wallets
3. **Proof of Work**: Uses the Ashmaize algorithm (memory-hard hash) to find valid solutions
4. **Submission**: Submits solutions via API for each wallet/challenge pair
5. **Consolidation**: Optionally consolidates all earnings to a single address

### Ashmaize Algorithm

The Ashmaize algorithm is a memory-hard proof-of-work function:

1. Build a 1MB ROM table from the challenge seed
2. Compute initial BLAKE2b hash of input (seed + address + nonce)
3. Perform 64 rounds of memory-hard mixing with ROM lookups
4. Final BLAKE2b hash to produce result
5. Check if result has required leading zero bits (difficulty)

### Multi-GPU Support

The miner distributes work across multiple GPUs:
- Each GPU runs its own solver thread
- Wallets are assigned round-robin to GPUs
- ROM tables are cached per-GPU to minimize rebuilds

## Wallet Management

### Automatic Cycling

When a wallet reaches the solution limit (default: 100), the miner:
1. Marks the wallet as exhausted
2. Gets or creates a new wallet
3. Registers the new wallet with the API
4. Continues mining with the new wallet

### Generating Wallets in Advance

For large operations, pre-generate wallets:
```bash
# Generate and register 1000 wallets
python miner.py --generate-wallets 1000 --register-only
```

### Wallet Storage

Wallets are stored in `./wallets/wallets.json`:
```json
{
  "wallets": [
    {
      "id": 0,
      "mnemonic": "word1 word2 ...",
      "address": "addr1qxy...",
      "signing_key": "...",
      "verification_key": "...",
      "solutions_submitted": 42,
      "is_registered": true
    }
  ]
}
```

**IMPORTANT**: Back up your `wallets.json` file! It contains your wallet keys.

## Consolidation

### How Consolidation Works

Defensio uses a "donation" mechanism to consolidate earnings:
1. Sign a message authorizing transfer from donor to recipient
2. Submit signed authorization to API
3. All future earnings from donor go to recipient

### Consolidating Earnings

```bash
# Consolidate all wallet earnings to your main address
python miner.py --consolidate-only --consolidate addr1qxy...

# Or during mining (auto-consolidate on shutdown)
python miner.py --gpu 0,1 --consolidate addr1qxy...
```

## GPU Architecture Guide

| Architecture | GPUs | Make Target |
|--------------|------|-------------|
| Maxwell | GTX 750, 9xx | `make cuda-sm50` |
| Pascal | GTX 10xx, P100 | `make cuda-sm60` |
| Volta | V100, Titan V | `make cuda-sm70` |
| Turing | RTX 20xx, T4 | `make cuda-sm75` |
| Ampere | A100, A30 | `make cuda-sm80` |
| Ampere | RTX 30xx, A10, A40 | `make cuda-sm86` |
| Ada Lovelace | RTX 40xx, L4, L40 | `make cuda-sm89` |

## Performance Tuning

### CPU Mining
- Set `--cpu-workers` to number of CPU cores
- Each worker uses ~1GB RAM for ROM table

### GPU Mining
- Use latest CUDA drivers
- Build for correct architecture
- Ensure adequate GPU memory (>2GB recommended)

### Network
- Stable internet connection required
- Low latency helps with challenge submission

## Troubleshooting

### CUDA Not Available
```
CUDA library not found. Run 'make cuda' to build.
```
Solution: Install CUDA Toolkit and run `make cuda-sm<arch>`

### No GPUs Detected
```
No CUDA GPUs available, using CPU fallback
```
Solution: Check NVIDIA drivers with `nvidia-smi`

### API Errors
```
Error fetching challenge: Connection refused
```
Solution: Check internet connection and API status

### Registration Failed
```
Failed to register wallet
```
Solution: Wait and retry (rate limiting) or check API status

## File Structure

```
defensio-miner/
├── miner.py              # Main miner script
├── Makefile              # Build system
├── README.md             # This file
├── cuda/
│   └── ashmaize_kernel.cu    # CUDA kernel source
│   └── libashmaize_cuda.so   # Compiled CUDA library
├── src/
│   └── cuda_solver.py    # Python CUDA wrapper
├── wallets/
│   ├── wallets.json      # Wallet storage
│   └── challenges.json   # Challenge cache
└── logs/
    └── miner_*.log       # Log files
```

## License

MIT License - See LICENSE file

## Disclaimer

This is an unofficial mining tool. Use at your own risk. The tokens may have no value. Always backup your wallet files and never share your mnemonic phrases.

## Support

- Defensio: https://defensio.io
- Discord: Join Defensio Discord for community support
- Issues: Open a GitHub issue for bugs/features
