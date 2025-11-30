"""
CUDA Ashmaize Solver Wrapper
============================
Python wrapper for the CUDA-accelerated Ashmaize mining kernel.
"""

import ctypes
import os
import sys
import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Find the shared library
CUDA_LIB_PATHS = [
    Path(__file__).parent / "cuda" / "libashmaize_cuda.so",
    Path(__file__).parent.parent / "cuda" / "libashmaize_cuda.so",
    Path("/usr/local/lib/libashmaize_cuda.so"),
    Path("./libashmaize_cuda.so"),
]

def find_cuda_lib() -> Optional[Path]:
    """Find the CUDA library."""
    for path in CUDA_LIB_PATHS:
        if path.exists():
            return path
    return None

class CUDASolver:
    """CUDA-accelerated Ashmaize solver."""
    
    ROM_SIZE = 1024 * 1024  # 1MB
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._lib = None
        self._rom = None
        self._rom_seed = None
        
        # Try to load CUDA library
        lib_path = find_cuda_lib()
        if lib_path:
            try:
                self._lib = ctypes.CDLL(str(lib_path))
                self._setup_functions()
                print(f"CUDA library loaded: {lib_path}")
            except OSError as e:
                print(f"Failed to load CUDA library: {e}")
        else:
            print("CUDA library not found. Run 'make cuda' to build.")
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        if not self._lib:
            return
        
        # cuda_get_device_count
        self._lib.cuda_get_device_count.restype = ctypes.c_int
        self._lib.cuda_get_device_count.argtypes = []
        
        # cuda_get_device_info
        self._lib.cuda_get_device_info.restype = ctypes.c_int
        self._lib.cuda_get_device_info.argtypes = [
            ctypes.c_int,                    # device_id
            ctypes.c_char_p,                 # name
            ctypes.c_int,                    # name_len
            ctypes.POINTER(ctypes.c_size_t)  # total_mem
        ]
        
        # cuda_build_rom
        self._lib.cuda_build_rom.restype = ctypes.c_int
        self._lib.cuda_build_rom.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # rom_seed
            ctypes.c_int,                     # rom_seed_len
            ctypes.POINTER(ctypes.c_uint32),  # rom_out
            ctypes.c_int                      # device_id
        ]
        
        # cuda_mine
        self._lib.cuda_mine.restype = ctypes.c_int
        self._lib.cuda_mine.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),   # seed
            ctypes.c_int,                      # seed_len
            ctypes.POINTER(ctypes.c_uint8),   # address
            ctypes.c_int,                      # address_len
            ctypes.POINTER(ctypes.c_uint32),  # rom
            ctypes.c_int,                      # rom_size
            ctypes.c_uint64,                   # start_nonce
            ctypes.c_uint64,                   # max_nonces
            ctypes.c_int,                      # difficulty
            ctypes.c_int,                      # device_id
            ctypes.POINTER(ctypes.c_uint64),  # found_nonce
            ctypes.POINTER(ctypes.c_uint8)    # found_hash
        ]
    
    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._lib is not None
    
    def get_device_count(self) -> int:
        """Get number of CUDA devices."""
        if not self._lib:
            return 0
        return self._lib.cuda_get_device_count()
    
    def get_device_info(self, device_id: int) -> Optional[dict]:
        """Get information about a CUDA device."""
        if not self._lib:
            return None
        
        name = ctypes.create_string_buffer(256)
        total_mem = ctypes.c_size_t()
        
        result = self._lib.cuda_get_device_info(
            device_id,
            name,
            256,
            ctypes.byref(total_mem)
        )
        
        if result == 0:
            return {
                'id': device_id,
                'name': name.value.decode('utf-8'),
                'total_memory': total_mem.value,
                'total_memory_gb': total_mem.value / (1024**3)
            }
        return None
    
    def build_rom(self, rom_seed: bytes) -> np.ndarray:
        """Build ROM table from seed."""
        if self._rom_seed == rom_seed and self._rom is not None:
            return self._rom
        
        rom_words = self.ROM_SIZE // 4
        
        if self._lib:
            # Use CUDA to build ROM
            rom = np.zeros(rom_words, dtype=np.uint32)
            seed_arr = np.frombuffer(rom_seed, dtype=np.uint8)
            
            result = self._lib.cuda_build_rom(
                seed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                len(rom_seed),
                rom.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                self.device_id
            )
            
            if result == 0:
                self._rom = rom
                self._rom_seed = rom_seed
                return rom
        
        # Fallback to CPU
        return self._build_rom_cpu(rom_seed)
    
    def _build_rom_cpu(self, rom_seed: bytes) -> np.ndarray:
        """CPU fallback for ROM building."""
        import hashlib
        
        rom_words = self.ROM_SIZE // 4
        rom = np.zeros(rom_words, dtype=np.uint32)
        
        state = hashlib.blake2b(rom_seed, digest_size=64).digest()
        
        for i in range(0, rom_words, 16):
            state = hashlib.blake2b(state, digest_size=64).digest()
            chunk = np.frombuffer(state, dtype=np.uint32)
            end_idx = min(i + 16, rom_words)
            rom[i:end_idx] = chunk[:end_idx - i]
        
        self._rom = rom
        self._rom_seed = rom_seed
        return rom
    
    def mine(self, seed: bytes, address: str, rom: np.ndarray,
             difficulty: int, start_nonce: int = 0,
             max_nonces: int = 100_000_000) -> Optional[Tuple[int, bytes]]:
        """
        Mine for a valid nonce.
        
        Args:
            seed: Challenge seed bytes
            address: Wallet address string
            rom: ROM table (numpy array)
            difficulty: Required leading zero bits
            start_nonce: Starting nonce value
            max_nonces: Maximum nonces to try
            
        Returns:
            Tuple of (nonce, hash) if found, None otherwise
        """
        if not self._lib:
            return self._mine_cpu(seed, address, rom, difficulty, start_nonce, max_nonces)
        
        seed_arr = np.frombuffer(seed, dtype=np.uint8)
        address_bytes = address.encode('utf-8')
        address_arr = np.frombuffer(address_bytes, dtype=np.uint8)
        
        found_nonce = ctypes.c_uint64()
        found_hash = (ctypes.c_uint8 * 32)()
        
        result = self._lib.cuda_mine(
            seed_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(seed),
            address_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(address_bytes),
            rom.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            self.ROM_SIZE,
            start_nonce,
            max_nonces,
            difficulty,
            self.device_id,
            ctypes.byref(found_nonce),
            found_hash
        )
        
        if result == 1:
            return int(found_nonce.value), bytes(found_hash)
        return None
    
    def _mine_cpu(self, seed: bytes, address: str, rom: np.ndarray,
                  difficulty: int, start_nonce: int,
                  max_nonces: int) -> Optional[Tuple[int, bytes]]:
        """CPU fallback for mining."""
        import hashlib
        
        address_bytes = address.encode('utf-8')
        
        for nonce in range(start_nonce, start_nonce + max_nonces):
            # Build input
            nonce_bytes = struct.pack('<Q', nonce)
            input_data = seed + address_bytes + nonce_bytes
            
            # Hash
            hash_result = self._ashmaize_hash_cpu(input_data, rom)
            
            # Check difficulty
            if self._check_difficulty(hash_result, difficulty):
                return nonce, hash_result
            
            if nonce % 100000 == 0 and nonce > start_nonce:
                print(f"CPU: {nonce - start_nonce:,} attempts...")
        
        return None
    
    def _ashmaize_hash_cpu(self, data: bytes, rom: np.ndarray) -> bytes:
        """CPU implementation of Ashmaize hash."""
        import hashlib
        
        h = hashlib.blake2b(data, digest_size=32).digest()
        state = np.frombuffer(h, dtype=np.uint32).copy()
        
        rom_words = len(rom)
        
        for round_idx in range(64):
            idx0 = int(state[0] % (rom_words - 4))
            idx1 = int(state[2] % (rom_words - 4))
            idx2 = int(state[4] % (rom_words - 4))
            idx3 = int(state[6] % (rom_words - 4))
            
            state[0] ^= rom[idx0]
            state[1] = (state[1] + rom[idx1]) & 0xFFFFFFFF
            state[2] ^= rom[idx2]
            state[3] = (state[3] + rom[idx3]) & 0xFFFFFFFF
            state[4] ^= rom[idx0 + 1]
            state[5] = (state[5] + rom[idx1 + 1]) & 0xFFFFFFFF
            state[6] ^= rom[idx2 + 1]
            state[7] = (state[7] + rom[idx3 + 1]) & 0xFFFFFFFF
            
            state = np.roll(state, 1)
            
            if (round_idx + 1) % 8 == 0:
                h = hashlib.blake2b(state.tobytes(), digest_size=32).digest()
                state = np.frombuffer(h, dtype=np.uint32).copy()
        
        return hashlib.blake2b(state.tobytes(), digest_size=32).digest()
    
    def _check_difficulty(self, hash_result: bytes, difficulty: int) -> bool:
        """Check if hash meets difficulty requirement."""
        leading_zeros = 0
        for byte in hash_result:
            if byte == 0:
                leading_zeros += 8
            else:
                for i in range(7, -1, -1):
                    if byte & (1 << i):
                        break
                    leading_zeros += 1
                break
        return leading_zeros >= difficulty


class MultiGPUSolver:
    """Manager for multiple GPU solvers."""
    
    def __init__(self, gpu_ids: list = None):
        """
        Initialize multi-GPU solver.
        
        Args:
            gpu_ids: List of GPU device IDs to use, or None for auto-detect
        """
        self.solvers = {}
        
        # Detect available GPUs
        probe_solver = CUDASolver(0)
        num_gpus = probe_solver.get_device_count()
        
        if gpu_ids is None:
            gpu_ids = list(range(num_gpus))
        
        # Create solver for each GPU
        for gpu_id in gpu_ids:
            if gpu_id < num_gpus:
                solver = CUDASolver(gpu_id)
                if solver.is_available:
                    info = solver.get_device_info(gpu_id)
                    self.solvers[gpu_id] = solver
                    if info:
                        print(f"GPU {gpu_id}: {info['name']} ({info['total_memory_gb']:.1f}GB)")
        
        if not self.solvers:
            print("No CUDA GPUs available, using CPU fallback")
            self.solvers[0] = CUDASolver(0)  # CPU fallback
    
    def get_solver(self, gpu_id: int = 0) -> CUDASolver:
        """Get solver for a specific GPU."""
        if gpu_id in self.solvers:
            return self.solvers[gpu_id]
        # Return first available
        return list(self.solvers.values())[0]
    
    @property
    def num_gpus(self) -> int:
        """Get number of available GPUs."""
        return len(self.solvers)
    
    @property
    def gpu_ids(self) -> list:
        """Get list of available GPU IDs."""
        return list(self.solvers.keys())


def test_cuda():
    """Test CUDA functionality."""
    print("Testing CUDA Ashmaize Solver")
    print("=" * 50)
    
    solver = CUDASolver(0)
    
    print(f"CUDA available: {solver.is_available}")
    print(f"Device count: {solver.get_device_count()}")
    
    if solver.get_device_count() > 0:
        info = solver.get_device_info(0)
        if info:
            print(f"Device 0: {info['name']}")
            print(f"Memory: {info['total_memory_gb']:.2f} GB")
    
    # Test ROM building
    print("\nBuilding ROM...")
    rom_seed = b"test_rom_seed_for_defensio_mining_12345"
    rom = solver.build_rom(rom_seed)
    print(f"ROM size: {len(rom)} words ({len(rom) * 4 / 1024:.1f} KB)")
    
    # Test mining (with low difficulty for speed)
    print("\nTesting mining (difficulty=8)...")
    seed = b"test_challenge_seed"
    address = "addr1_test_address_12345"
    
    import time
    start = time.time()
    result = solver.mine(seed, address, rom, difficulty=8, max_nonces=1_000_000)
    elapsed = time.time() - start
    
    if result:
        nonce, hash_result = result
        print(f"Solution found!")
        print(f"Nonce: {nonce}")
        print(f"Hash: {hash_result.hex()}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Hash rate: {1_000_000 / elapsed / 1000:.1f} KH/s")
    else:
        print(f"No solution found in {elapsed:.2f}s")


if __name__ == "__main__":
    test_cuda()
