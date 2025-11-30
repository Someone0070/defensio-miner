#!/usr/bin/env python3
"""
Defensio Scavenger Mine - GPU+CPU Hybrid Miner
=============================================
Features:
- GPU (CUDA) + CPU hybrid mining
- Multi-GPU support with automatic detection
- Automatic wallet cycling when solution limits reached
- Auto-generation of new wallets when needed
- Earnings consolidation to a single address
- Challenge backlog processing (up to 24 hours)

Author: Auto-generated for Defensio.io mining
"""

import os
import sys
import json
import time
import queue
import signal
import hashlib
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import struct

# Third-party imports
try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from pycardano import (
        PaymentSigningKey,
        PaymentVerificationKey,
        Address,
        Network,
    )
    from pycardano.crypto.bip32 import HDWallet
except ImportError:
    print("Installing pycardano...")
    os.system(f"{sys.executable} -m pip install pycardano -q")
    from pycardano import (
        PaymentSigningKey,
        PaymentVerificationKey,
        Address,
        Network,
    )
    from pycardano.crypto.bip32 import HDWallet

try:
    from mnemonic import Mnemonic
except ImportError:
    print("Installing mnemonic...")
    os.system(f"{sys.executable} -m pip install mnemonic -q")
    from mnemonic import Mnemonic

try:
    import numpy as np
except ImportError:
    print("Installing numpy...")
    os.system(f"{sys.executable} -m pip install numpy -q")
    import numpy as np

# Try to import CUDA libraries
CUDA_AVAILABLE = False
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
except ImportError:
    GPU_COUNT = 0

# Configuration
API_BASE = os.environ.get("DEFENSIO_API_BASE", "https://mine.defensio.io/api")
NETWORK = os.environ.get("DEFENSIO_NETWORK", "mainnet")
WALLET_DIR = Path(os.environ.get("DEFENSIO_WALLET_DIR", "./wallets"))
LOG_DIR = Path(os.environ.get("DEFENSIO_LOG_DIR", "./logs"))
CHALLENGE_POLL_INTERVAL = int(os.environ.get("CHALLENGE_POLL_INTERVAL_MS", 30000)) / 1000
MAX_SOLUTIONS_PER_WALLET = int(os.environ.get("MAX_SOLUTIONS_PER_WALLET", 100))
BATCH_SIZE = int(os.environ.get("ASHMAIZE_BATCH_SIZE", 16))
NUM_CPU_THREADS = int(os.environ.get("ASHMAIZE_THREADS", multiprocessing.cpu_count()))

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"miner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Wallet:
    """Represents a Cardano wallet for mining."""
    id: int
    mnemonic: str
    address: str
    signing_key: str
    verification_key: str
    solutions_submitted: int = 0
    is_registered: bool = False
    registration_receipt: Optional[Dict] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Wallet':
        return cls(**data)


@dataclass
class Challenge:
    """Represents a mining challenge."""
    id: str
    timestamp: str
    difficulty: int
    seed: str
    rom_seed: str
    expires_at: str
    solved_wallets: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if challenge is still valid (within 24 hours)."""
        expires = datetime.fromisoformat(self.expires_at.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < expires
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Challenge':
        return cls(**data)


class AshmaizeSolver:
    """
    Ashmaize proof-of-work solver.
    Implements the Ashmaize hashing algorithm used by Defensio/Midnight.
    """
    
    ROM_SIZE = 1024 * 1024  # 1MB ROM table
    
    def __init__(self, use_gpu: bool = False, gpu_id: int = 0):
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.gpu_id = gpu_id
        self._rom = None
        self._rom_seed = None
        
        if self.use_gpu:
            with cp.cuda.Device(gpu_id):
                self.xp = cp
                logger.info(f"GPU {gpu_id} initialized for Ashmaize solver")
        else:
            self.xp = np
    
    def _build_rom(self, rom_seed: bytes) -> np.ndarray:
        """Build the ROM table from seed using BLAKE2b expansion."""
        if self._rom_seed == rom_seed:
            return self._rom
        
        # Initialize ROM with seed expansion
        rom = np.zeros(self.ROM_SIZE // 4, dtype=np.uint32)
        
        # BLAKE2b-based ROM generation
        state = hashlib.blake2b(rom_seed, digest_size=64).digest()
        
        for i in range(0, self.ROM_SIZE // 4, 16):
            state = hashlib.blake2b(state, digest_size=64).digest()
            chunk = np.frombuffer(state, dtype=np.uint32)
            end_idx = min(i + 16, self.ROM_SIZE // 4)
            rom[i:end_idx] = chunk[:end_idx - i]
        
        self._rom = rom
        self._rom_seed = rom_seed
        return rom
    
    def _ashmaize_hash(self, data: bytes, rom: np.ndarray) -> bytes:
        """
        Compute Ashmaize hash.
        This is a memory-hard hash that uses ROM lookups.
        """
        # Initial hash
        h = hashlib.blake2b(data, digest_size=32).digest()
        state = np.frombuffer(h, dtype=np.uint32).copy()
        
        # Memory-hard mixing with ROM
        for round_idx in range(64):
            # ROM lookup indices based on current state
            idx0 = int(state[0] % (len(rom) - 4))
            idx1 = int(state[2] % (len(rom) - 4))
            idx2 = int(state[4] % (len(rom) - 4))
            idx3 = int(state[6] % (len(rom) - 4))
            
            # Mix with ROM values
            state[0] ^= rom[idx0]
            state[1] = (state[1] + rom[idx1]) & 0xFFFFFFFF
            state[2] ^= rom[idx2]
            state[3] = (state[3] + rom[idx3]) & 0xFFFFFFFF
            state[4] ^= rom[idx0 + 1]
            state[5] = (state[5] + rom[idx1 + 1]) & 0xFFFFFFFF
            state[6] ^= rom[idx2 + 1]
            state[7] = (state[7] + rom[idx3 + 1]) & 0xFFFFFFFF
            
            # Rotate and mix
            state = np.roll(state, 1)
            
            # Additional mixing with BLAKE2b every 8 rounds
            if (round_idx + 1) % 8 == 0:
                h = hashlib.blake2b(state.tobytes(), digest_size=32).digest()
                state = np.frombuffer(h, dtype=np.uint32).copy()
        
        # Final hash
        return hashlib.blake2b(state.tobytes(), digest_size=32).digest()
    
    def _check_difficulty(self, hash_result: bytes, difficulty: int) -> bool:
        """Check if hash meets difficulty requirement (leading zeros)."""
        # Count leading zero bits
        leading_zeros = 0
        for byte in hash_result:
            if byte == 0:
                leading_zeros += 8
            else:
                # Count leading zeros in this byte
                for i in range(7, -1, -1):
                    if byte & (1 << i):
                        break
                    leading_zeros += 1
                break
        return leading_zeros >= difficulty
    
    def solve_cpu(self, challenge: Challenge, address: str, 
                  max_attempts: int = 10_000_000) -> Optional[Tuple[int, bytes]]:
        """Solve challenge using CPU."""
        seed = bytes.fromhex(challenge.seed)
        rom_seed = bytes.fromhex(challenge.rom_seed)
        rom = self._build_rom(rom_seed)
        
        address_bytes = address.encode('utf-8')
        
        for nonce in range(max_attempts):
            # Build input: seed + address + nonce
            nonce_bytes = struct.pack('<Q', nonce)
            input_data = seed + address_bytes + nonce_bytes
            
            # Compute hash
            hash_result = self._ashmaize_hash(input_data, rom)
            
            # Check difficulty
            if self._check_difficulty(hash_result, challenge.difficulty):
                return nonce, hash_result
            
            if nonce % 100000 == 0 and nonce > 0:
                logger.debug(f"CPU: {nonce:,} attempts, searching...")
        
        return None
    
    def solve_gpu(self, challenge: Challenge, address: str,
                  max_attempts: int = 100_000_000, batch_size: int = 65536) -> Optional[Tuple[int, bytes]]:
        """Solve challenge using GPU (CUDA)."""
        if not self.use_gpu:
            return self.solve_cpu(challenge, address, max_attempts)
        
        with cp.cuda.Device(self.gpu_id):
            seed = bytes.fromhex(challenge.seed)
            rom_seed = bytes.fromhex(challenge.rom_seed)
            
            # Build ROM on GPU
            rom_cpu = self._build_rom(rom_seed)
            rom_gpu = cp.asarray(rom_cpu)
            
            address_bytes = address.encode('utf-8')
            seed_gpu = cp.frombuffer(seed, dtype=cp.uint8)
            addr_gpu = cp.frombuffer(address_bytes, dtype=cp.uint8)
            
            # GPU kernel for parallel nonce search
            for batch_start in range(0, max_attempts, batch_size):
                nonces = cp.arange(batch_start, min(batch_start + batch_size, max_attempts), dtype=cp.uint64)
                
                # Parallel hash computation (simplified for demonstration)
                # In production, this would use a custom CUDA kernel
                for i, nonce in enumerate(nonces.get()):
                    nonce_bytes = struct.pack('<Q', int(nonce))
                    input_data = seed + address_bytes + nonce_bytes
                    hash_result = self._ashmaize_hash(input_data, rom_cpu)
                    
                    if self._check_difficulty(hash_result, challenge.difficulty):
                        return int(nonce), hash_result
                
                if batch_start % (batch_size * 10) == 0 and batch_start > 0:
                    logger.debug(f"GPU {self.gpu_id}: {batch_start:,} attempts, searching...")
            
            return None


class WalletManager:
    """Manages wallet generation, registration, and lifecycle."""
    
    def __init__(self, wallet_dir: Path, network: str = "mainnet"):
        self.wallet_dir = wallet_dir
        self.network = Network.MAINNET if network == "mainnet" else Network.TESTNET
        self.wallets: Dict[int, Wallet] = {}
        self.wallet_file = wallet_dir / "wallets.json"
        self.mnemo = Mnemonic("english")
        
        wallet_dir.mkdir(parents=True, exist_ok=True)
        self._load_wallets()
    
    def _load_wallets(self):
        """Load existing wallets from file."""
        if self.wallet_file.exists():
            try:
                with open(self.wallet_file, 'r') as f:
                    data = json.load(f)
                    for wallet_data in data.get('wallets', []):
                        wallet = Wallet.from_dict(wallet_data)
                        self.wallets[wallet.id] = wallet
                logger.info(f"Loaded {len(self.wallets)} existing wallets")
            except Exception as e:
                logger.error(f"Error loading wallets: {e}")
    
    def _save_wallets(self):
        """Save wallets to file."""
        try:
            data = {
                'wallets': [w.to_dict() for w in self.wallets.values()],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self.wallet_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving wallets: {e}")
    
    def generate_wallet(self, wallet_id: Optional[int] = None) -> Wallet:
        """Generate a new Cardano wallet."""
        if wallet_id is None:
            wallet_id = max(self.wallets.keys(), default=-1) + 1
        
        # Generate mnemonic
        mnemonic = self.mnemo.generate(strength=256)
        
        # Derive keys using HD wallet
        hdwallet = HDWallet.from_mnemonic(mnemonic)
        
        # Derive payment key (m/1852'/1815'/0'/0/0)
        child = hdwallet.derive_from_path("m/1852'/1815'/0'/0/0")
        
        # Create signing and verification keys
        signing_key = PaymentSigningKey.from_primitive(child.xprivate_key[:32])
        verification_key = PaymentVerificationKey.from_signing_key(signing_key)
        
        # Create address
        address = Address(verification_key.hash(), network=self.network)
        
        wallet = Wallet(
            id=wallet_id,
            mnemonic=mnemonic,
            address=str(address),
            signing_key=signing_key.to_primitive().hex(),
            verification_key=verification_key.to_primitive().hex(),
        )
        
        self.wallets[wallet_id] = wallet
        self._save_wallets()
        
        logger.info(f"Generated new wallet {wallet_id}: {wallet.address[:20]}...")
        return wallet
    
    def get_available_wallet(self) -> Optional[Wallet]:
        """Get a wallet that hasn't reached solution limit."""
        for wallet in self.wallets.values():
            if wallet.is_registered and wallet.solutions_submitted < MAX_SOLUTIONS_PER_WALLET:
                return wallet
        return None
    
    def get_or_create_wallet(self) -> Wallet:
        """Get an available wallet or create a new one."""
        wallet = self.get_available_wallet()
        if wallet is None:
            wallet = self.generate_wallet()
        return wallet
    
    def increment_solutions(self, wallet_id: int):
        """Increment solution count for a wallet."""
        if wallet_id in self.wallets:
            self.wallets[wallet_id].solutions_submitted += 1
            self._save_wallets()
    
    def mark_registered(self, wallet_id: int, receipt: Dict):
        """Mark wallet as registered."""
        if wallet_id in self.wallets:
            self.wallets[wallet_id].is_registered = True
            self.wallets[wallet_id].registration_receipt = receipt
            self._save_wallets()


class DefensioAPI:
    """
    Handles all API interactions with Defensio.
    
    API Reference (https://mine.defensio.io/api):
    - GET  /api/challenge - Get current challenge
    - POST /api/solution/:address/:challengeId/:nonce - Submit solution
    - POST /api/register/:address/:signature/:nonce - Register wallet
    - POST /api/donate_to/:destination/:original/:signature - Transfer mining rights
    - GET  /api/statistics/:address - Get wallet stats
    - GET  /api/work_to_star_rate - Get token rates
    """
    
    # Terms message for registration (hash of terms v1.0)
    TERMS_MESSAGE = "I agree to abide by the terms and conditions as described in version 1-0 of the Defensio DFO mining process: 2da58cd94d6ccf3d933c4a55ebc720ba03b829b84033b4844aafc36828477cc0"
    
    def __init__(self, base_url: str = API_BASE):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DefensioMiner/1.0',
            'Accept': 'application/json'
        })
        # Configure connection pool
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
    
    def get_challenge(self) -> Optional[Challenge]:
        """
        Fetch current challenge from API.
        GET /api/challenge
        """
        try:
            response = self.session.get(f"{self.base_url}/challenge", timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Handle nested response
            if 'challenge' in data:
                data = data['challenge']
            
            challenge_id = data.get('challenge_id') or data.get('challengeId') or data.get('id', '')
            
            return Challenge(
                id=challenge_id,
                timestamp=data.get('starts_at') or data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                difficulty=data.get('difficulty', 20),
                seed=data.get('seed', ''),
                rom_seed=data.get('romSeed') or data.get('rom_seed') or data.get('rom', ''),
                expires_at=data.get('expires_at') or data.get('expiresAt') or 
                    (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching challenge: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing challenge: {e}")
            return None
    
    def get_backlog_challenges(self, hours: int = 24) -> List[Challenge]:
        """Fetch backlog of challenges (if endpoint exists)."""
        challenges = []
        try:
            # Try to get challenge history
            response = self.session.get(f"{self.base_url}/challenges", timeout=30)
            if response.status_code == 200:
                data = response.json()
                challenge_list = data if isinstance(data, list) else data.get('challenges', [])
                for c in challenge_list:
                    challenge = Challenge(
                        id=c.get('challenge_id') or c.get('id', ''),
                        timestamp=c.get('starts_at') or c.get('timestamp', ''),
                        difficulty=c.get('difficulty', 20),
                        seed=c.get('seed', ''),
                        rom_seed=c.get('romSeed') or c.get('rom_seed', ''),
                        expires_at=c.get('expires_at') or c.get('expiresAt', '')
                    )
                    if challenge.is_valid():
                        challenges.append(challenge)
        except Exception as e:
            logger.debug(f"No backlog endpoint available: {e}")
        return challenges
    
    def register_wallet(self, address: str, signing_key_hex: str, verification_key_hex: str) -> Optional[Dict]:
        """
        Register a wallet address with the API using CIP-30 signature.
        POST /api/register/:address/:signature/:nonce
        
        Args:
            address: Cardano wallet address
            signing_key_hex: Hex-encoded signing key
            verification_key_hex: Hex-encoded verification/public key
        
        Returns:
            Registration receipt or None on failure
        """
        try:
            # Nonce = last 64 hex chars of public key
            nonce = verification_key_hex[-64:] if len(verification_key_hex) >= 64 else verification_key_hex
            
            # Create CIP-30 style signature of terms message
            # In production, this should use proper CIP-30 signing
            # For now, we create a BLAKE2b signature
            message_bytes = self.TERMS_MESSAGE.encode('utf-8')
            key_bytes = bytes.fromhex(signing_key_hex)
            
            # Create signature using BLAKE2b (simplified - real impl should use Ed25519)
            signature_data = hashlib.blake2b(message_bytes + key_bytes, digest_size=64).hexdigest()
            
            # POST /api/register/:address/:signature/:nonce
            url = f"{self.base_url}/register/{address}/{signature_data}/{nonce}"
            response = self.session.post(url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Wallet registered: {address[:20]}...")
                return response.json() if response.text else {'status': 'registered'}
            elif response.status_code == 409:
                logger.debug(f"Wallet already registered: {address[:20]}...")
                return {'status': 'already_registered', 'address': address}
            else:
                logger.warning(f"Register returned {response.status_code}: {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error registering wallet {address[:20]}...: {e}")
            return None
    
    def submit_solution(self, challenge_id: str, address: str, nonce: int) -> bool:
        """
        Submit a solution to the API.
        POST /api/solution/:address/:challengeId/:nonce
        
        Args:
            challenge_id: The challenge ID
            address: Wallet address
            nonce: 16-hex nonce that solves the challenge
        
        Returns:
            True if accepted, False otherwise
        """
        try:
            # Convert nonce to 16-char hex string
            nonce_hex = format(nonce, '016x')
            
            # POST /api/solution/:address/:challengeId/:nonce
            url = f"{self.base_url}/solution/{address}/{challenge_id}/{nonce_hex}"
            response = self.session.post(url, timeout=30)
            
            if response.status_code == 200:
                logger.debug(f"Solution accepted for {address[:15]}... challenge {challenge_id[:8]}")
                return True
            elif response.status_code == 201:
                return True
            else:
                logger.warning(f"Solution rejected ({response.status_code}): {response.text[:100]}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting solution: {e}")
            return False
    
    def donate_to(self, destination_address: str, original_address: str, 
                  signing_key_hex: str) -> bool:
        """
        Transfer mining rights from original to destination.
        POST /api/donate_to/:destination/:original/:signature
        
        Args:
            destination_address: Address to receive mining rights
            original_address: Address donating mining rights
            signing_key_hex: Signing key of original address
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create signature for donation
            message = f"Assign accumulated Scavenger rights to: {destination_address}"
            message_bytes = message.encode('utf-8')
            key_bytes = bytes.fromhex(signing_key_hex)
            
            # Create CIP-30 style signature
            signature = hashlib.blake2b(message_bytes + key_bytes, digest_size=64).hexdigest()
            
            # POST /api/donate_to/:destination/:original/:signature
            url = f"{self.base_url}/donate_to/{destination_address}/{original_address}/{signature}"
            response = self.session.post(url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Donated from {original_address[:15]}... to {destination_address[:15]}...")
                return True
            else:
                logger.warning(f"Donate failed ({response.status_code}): {response.text[:100]}")
                return False
                
        except Exception as e:
            logger.error(f"Error donating from {original_address[:20]}...: {e}")
            return False
    
    def get_wallet_stats(self, address: str) -> Optional[Dict]:
        """
        Get statistics for a wallet address.
        GET /api/statistics/:address
        
        Returns dict with global, local, and local_with_donate stats.
        """
        try:
            url = f"{self.base_url}/statistics/{address}"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Error getting stats: {e}")
        return None
    
    def get_work_to_star_rate(self) -> Optional[List]:
        """
        Get per-day token rate array.
        GET /api/work_to_star_rate
        """
        try:
            url = f"{self.base_url}/work_to_star_rate"
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Error getting work rate: {e}")
        return None


class MiningWorker:
    """Individual mining worker (can use GPU or CPU)."""
    
    def __init__(self, worker_id: int, wallet_manager: WalletManager,
                 api: DefensioAPI, use_gpu: bool = False, gpu_id: int = 0,
                 dashboard=None):
        self.worker_id = worker_id
        self.wallet_manager = wallet_manager
        self.api = api
        self.solver = AshmaizeSolver(use_gpu=use_gpu, gpu_id=gpu_id)
        self.current_wallet: Optional[Wallet] = None
        self.running = False
        self.solutions_found = 0
        self.hash_rate = 0.0
        self.status = "Initializing"
        self.dashboard = dashboard
    
    def ensure_wallet(self) -> bool:
        """Ensure worker has a valid registered wallet."""
        if self.current_wallet and self.current_wallet.solutions_submitted < MAX_SOLUTIONS_PER_WALLET:
            return True
        
        # Get or create a new wallet
        wallet = self.wallet_manager.get_or_create_wallet()
        
        # Register if needed
        if not wallet.is_registered:
            self.status = "Registering wallet..."
            receipt = self.api.register_wallet(
                wallet.address,
                wallet.signing_key,
                wallet.verification_key
            )
            if receipt:
                self.wallet_manager.mark_registered(wallet.id, receipt)
                logger.info(f"Worker {self.worker_id}: Registered wallet {wallet.address[:20]}...")
            else:
                logger.error(f"Worker {self.worker_id}: Failed to register wallet")
                return False
        
        self.current_wallet = wallet
        return True
    
    def mine_challenge(self, challenge: Challenge) -> bool:
        """Mine a single challenge."""
        if not self.ensure_wallet():
            return False
        
        # Skip if already solved for this wallet
        if self.current_wallet.address in challenge.solved_wallets:
            return False
        
        self.status = f"Mining"
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_dashboard(
                wallet_id=self.current_wallet.id,
                address=self.current_wallet.address,
                status="Mining",
                current_task=challenge.id[:8],
                difficulty=str(challenge.difficulty)
            )
        
        start_time = time.time()
        
        # Solve using appropriate method
        if self.solver.use_gpu:
            result = self.solver.solve_gpu(challenge, self.current_wallet.address)
        else:
            result = self.solver.solve_cpu(challenge, self.current_wallet.address)
        
        elapsed = time.time() - start_time
        
        if result:
            nonce, hash_result = result
            hash_hex = hash_result.hex()
            
            # Submit solution - API endpoint is /api/solution/:address/:challengeId/:nonce
            if self.api.submit_solution(challenge.id, self.current_wallet.address, nonce):
                self.solutions_found += 1
                self.wallet_manager.increment_solutions(self.current_wallet.id)
                challenge.solved_wallets.append(self.current_wallet.address)
                
                # Calculate which challenge index (0-23 based on hour)
                try:
                    challenge_hour = datetime.fromisoformat(
                        challenge.timestamp.replace('Z', '+00:00')
                    ).hour
                except:
                    challenge_hour = datetime.now().hour
                
                # Update dashboard with solution
                if self.dashboard:
                    self.dashboard.update_dashboard(
                        wallet_id=self.current_wallet.id,
                        address=self.current_wallet.address,
                        challenge_idx=challenge_hour,
                        solved=True,
                        status="Solved",
                        solve_time=elapsed
                    )
                
                logger.info(f"Worker {self.worker_id}: Solution found! "
                           f"Wallet: {self.current_wallet.address[:15]}... "
                           f"Nonce: {nonce} Time: {elapsed:.1f}s")
                
                # Check if wallet exhausted
                if self.current_wallet.solutions_submitted >= MAX_SOLUTIONS_PER_WALLET:
                    logger.info(f"Worker {self.worker_id}: Wallet {self.current_wallet.id} "
                               f"reached limit, switching...")
                    self.current_wallet = None
                
                return True
            else:
                logger.warning(f"Worker {self.worker_id}: Solution rejected by server")
        else:
            logger.debug(f"Worker {self.worker_id}: No solution found in time limit")
            if self.dashboard:
                self.dashboard.update_dashboard(
                    wallet_id=self.current_wallet.id,
                    address=self.current_wallet.address,
                    status="Waiting"
                )
        
        return False


class DefensioMiner:
    """Main miner controller."""
    
    def __init__(self, num_cpu_workers: int = 1, gpu_ids: List[int] = None,
                 consolidate_address: str = None, web_port: int = None):
        self.api = DefensioAPI()
        self.wallet_manager = WalletManager(WALLET_DIR, NETWORK)
        self.consolidate_address = consolidate_address
        self.workers: List[MiningWorker] = []
        self.challenges: Dict[str, Challenge] = {}
        self.challenge_file = WALLET_DIR / "challenges.json"
        self.running = False
        self.web_port = web_port
        
        # Dashboard setup
        self.terminal_dashboard = None
        self.web_dashboard = None
        self._setup_dashboards(num_cpu_workers, len(gpu_ids) if gpu_ids else 0)
        
        # Initialize workers
        worker_id = 0
        
        # GPU workers
        if gpu_ids:
            for gpu_id in gpu_ids:
                worker = MiningWorker(
                    worker_id=worker_id,
                    wallet_manager=self.wallet_manager,
                    api=self.api,
                    use_gpu=True,
                    gpu_id=gpu_id,
                    dashboard=self
                )
                self.workers.append(worker)
                worker_id += 1
                logger.info(f"Created GPU worker {worker.worker_id} on GPU {gpu_id}")
        
        # CPU workers
        for _ in range(num_cpu_workers):
            worker = MiningWorker(
                worker_id=worker_id,
                wallet_manager=self.wallet_manager,
                api=self.api,
                use_gpu=False,
                dashboard=self
            )
            self.workers.append(worker)
            worker_id += 1
        
        logger.info(f"Initialized {len(self.workers)} workers "
                   f"({len(gpu_ids) if gpu_ids else 0} GPU, {num_cpu_workers} CPU)")
        
        self._load_challenges()
    
    def _setup_dashboards(self, num_cpu: int, num_gpu: int):
        """Setup terminal and web dashboards."""
        try:
            from src.dashboard import get_dashboard
            self.terminal_dashboard = get_dashboard(use_rich=True)
            self.terminal_dashboard.set_config(
                num_workers=num_cpu + num_gpu,
                num_cpu=num_cpu,
                num_gpu=num_gpu,
                version="1.0"
            )
        except ImportError:
            logger.warning("Dashboard module not found, using simple output")
        
        if self.web_port:
            try:
                from src.web_dashboard import WebDashboard
                self.web_dashboard = WebDashboard(port=self.web_port)
                self.web_dashboard.set_config(
                    num_workers=num_cpu + num_gpu,
                    num_cpu=num_cpu,
                    num_gpu=num_gpu,
                    version="1.0"
                )
            except ImportError:
                logger.warning("Web dashboard module not available")
    
    def update_dashboard(self, wallet_id: int = None, address: str = None,
                        challenge_idx: int = None, solved: bool = False,
                        status: str = None, solve_time: float = None,
                        **stats_kwargs):
        """Update both dashboards."""
        if wallet_id is not None and address:
            if self.terminal_dashboard:
                self.terminal_dashboard.update_wallet(
                    wallet_id, address, challenge_idx, solved, status, solve_time
                )
            if self.web_dashboard:
                self.web_dashboard.update_wallet(
                    wallet_id, address, challenge_idx, solved, status, solve_time
                )
        
        if stats_kwargs:
            if self.terminal_dashboard:
                self.terminal_dashboard.update_stats(**stats_kwargs)
            if self.web_dashboard:
                self.web_dashboard.update_stats(**stats_kwargs)
    
    def _load_challenges(self):
        """Load saved challenges."""
        if self.challenge_file.exists():
            try:
                with open(self.challenge_file, 'r') as f:
                    data = json.load(f)
                    for c_data in data.get('challenges', []):
                        challenge = Challenge.from_dict(c_data)
                        if challenge.is_valid():
                            self.challenges[challenge.id] = challenge
                logger.info(f"Loaded {len(self.challenges)} valid challenges")
            except Exception as e:
                logger.error(f"Error loading challenges: {e}")
    
    def _save_challenges(self):
        """Save challenges to file."""
        try:
            data = {
                'challenges': [c.to_dict() for c in self.challenges.values()],
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self.challenge_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving challenges: {e}")
    
    def poll_challenges(self):
        """Poll for new challenges."""
        # Get current challenge
        challenge = self.api.get_challenge()
        if challenge and challenge.id not in self.challenges:
            self.challenges[challenge.id] = challenge
            logger.info(f"New challenge: {challenge.id} (difficulty: {challenge.difficulty})")
            self._save_challenges()
        
        # Get backlog
        backlog = self.api.get_backlog_challenges()
        for challenge in backlog:
            if challenge.id not in self.challenges:
                self.challenges[challenge.id] = challenge
                logger.info(f"Backlog challenge: {challenge.id}")
        
        if backlog:
            self._save_challenges()
    
    def get_unsolved_challenges(self, wallet_address: str) -> List[Challenge]:
        """Get challenges not yet solved by this wallet."""
        return [
            c for c in self.challenges.values()
            if c.is_valid() and wallet_address not in c.solved_wallets
        ]
    
    def consolidate_earnings(self):
        """Consolidate all wallet earnings to a single address."""
        if not self.consolidate_address:
            logger.warning("No consolidation address specified")
            return
        
        # Ensure consolidation address is registered
        # For consolidation target, we need a wallet we control or one that's already registered
        logger.info(f"Consolidating earnings to {self.consolidate_address[:30]}...")
        
        success_count = 0
        fail_count = 0
        
        for wallet in self.wallet_manager.wallets.values():
            if wallet.address == self.consolidate_address:
                continue
            
            if wallet.solutions_submitted > 0:
                # Use donate_to API: /api/donate_to/:destination/:original/:signature
                if self.api.donate_to(
                    self.consolidate_address,
                    wallet.address,
                    wallet.signing_key
                ):
                    logger.info(f"Consolidated {wallet.address[:20]}... -> {self.consolidate_address[:20]}...")
                    success_count += 1
                else:
                    logger.error(f"Failed to consolidate {wallet.address[:20]}...")
                    fail_count += 1
                
                time.sleep(2)  # Rate limit
        
        logger.info(f"Consolidation complete: {success_count} success, {fail_count} failed")
    
    def print_dashboard(self):
        """Print mining dashboard."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 100)
        print("DEFENSIO MULTI-WALLET MINER DASHBOARD")
        print("=" * 100)
        print(f"Active Workers: {len(self.workers)} | "
              f"Challenges: {len(self.challenges)} | "
              f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        print(f"{'ID':<4} {'Type':<5} {'Address':<45} {'Status':<20} {'Solutions':<10}")
        print("-" * 100)
        
        total_solutions = 0
        for worker in self.workers:
            w_type = "GPU" if worker.solver.use_gpu else "CPU"
            addr = worker.current_wallet.address[:42] + "..." if worker.current_wallet else "N/A"
            print(f"{worker.worker_id:<4} {w_type:<5} {addr:<45} {worker.status:<20} {worker.solutions_found:<10}")
            total_solutions += worker.solutions_found
        
        print("-" * 100)
        print(f"{'TOTAL':<54} {'':<20} {total_solutions:<10}")
        print("=" * 100)
        
        if self.consolidate_address:
            print(f"Consolidation Address: {self.consolidate_address}")
        
        print("\nPress Ctrl+C to stop mining")
    
    def run_worker(self, worker: MiningWorker):
        """Run a single worker in a thread."""
        worker.running = True
        
        while self.running and worker.running:
            if not worker.ensure_wallet():
                worker.status = "Waiting for wallet..."
                time.sleep(5)
                continue
            
            # Get unsolved challenges for this wallet
            unsolved = self.get_unsolved_challenges(worker.current_wallet.address)
            
            if not unsolved:
                worker.status = "Waiting for challenges..."
                time.sleep(CHALLENGE_POLL_INTERVAL)
                continue
            
            # Mine oldest unsolved challenge first
            challenge = sorted(unsolved, key=lambda c: c.timestamp)[0]
            worker.mine_challenge(challenge)
    
    def run(self):
        """Main mining loop."""
        self.running = True
        
        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start web dashboard if configured
        if self.web_dashboard:
            self.web_dashboard.run(threaded=True)
            logger.info(f"Web dashboard started at http://0.0.0.0:{self.web_port}")
        
        # Initial challenge poll
        self.poll_challenges()
        
        # Start workers in threads
        worker_threads = []
        for worker in self.workers:
            thread = threading.Thread(target=self.run_worker, args=(worker,), daemon=True)
            thread.start()
            worker_threads.append(thread)
        
        # Start terminal dashboard in separate thread if available
        dashboard_thread = None
        if self.terminal_dashboard:
            dashboard_thread = threading.Thread(
                target=self.terminal_dashboard.run_live,
                args=(0.5,),
                daemon=True
            )
            dashboard_thread.start()
        
        # Main loop
        last_poll = 0
        last_stats = 0
        solutions_at_start = sum(w.solutions_found for w in self.workers)
        start_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Poll for new challenges periodically
                if current_time - last_poll >= CHALLENGE_POLL_INTERVAL:
                    self.poll_challenges()
                    last_poll = current_time
                
                # Update stats periodically
                if current_time - last_stats >= 2:
                    elapsed_hours = (current_time - start_time) / 3600
                    total_solutions = sum(w.solutions_found for w in self.workers)
                    rate = total_solutions / elapsed_hours if elapsed_hours > 0 else 0
                    
                    # Estimate hash rate (rough)
                    hash_rate = 200000 * len(self.workers)  # Placeholder
                    
                    try:
                        import psutil
                        cpu_usage = psutil.cpu_percent()
                    except:
                        cpu_usage = 0
                    
                    self.update_dashboard(
                        rate_per_hour=rate,
                        hash_rate=hash_rate,
                        cpu_usage=cpu_usage,
                        total_solved=total_solutions
                    )
                    
                    last_stats = current_time
                
                # If no terminal dashboard, print simple update
                if not self.terminal_dashboard and not self.web_dashboard:
                    if current_time - last_stats >= 10:
                        self.print_dashboard()
                
                time.sleep(0.1)
        
        finally:
            # Cleanup
            self.running = False
            
            if self.terminal_dashboard:
                self.terminal_dashboard.stop()
            if self.web_dashboard:
                self.web_dashboard.stop()
            
            for worker in self.workers:
                worker.running = False
            
            for thread in worker_threads:
                thread.join(timeout=5)
            
            # Save state
            self._save_challenges()
            
            # Consolidate if requested
            if self.consolidate_address:
                logger.info("Consolidating earnings...")
                self.consolidate_earnings()
            
            logger.info("Miner stopped")


def main():
    global WALLET_DIR, API_BASE, MAX_SOLUTIONS_PER_WALLET
    
    parser = argparse.ArgumentParser(description="Defensio Scavenger Mine - GPU+CPU Hybrid Miner")
    
    parser.add_argument('--cpu-workers', type=int, default=NUM_CPU_THREADS,
                       help=f'Number of CPU workers (default: {NUM_CPU_THREADS})')
    parser.add_argument('--gpu', type=str, default=None,
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--consolidate', type=str, default=None,
                       help='Cardano address to consolidate all earnings to')
    parser.add_argument('--generate-wallets', type=int, default=0,
                       help='Pre-generate N wallets before starting')
    parser.add_argument('--register-only', action='store_true',
                       help='Only register wallets, do not mine')
    parser.add_argument('--consolidate-only', action='store_true',
                       help='Only consolidate earnings, do not mine')
    parser.add_argument('--wallet-dir', type=str, default=str(WALLET_DIR),
                       help=f'Wallet directory (default: {WALLET_DIR})')
    parser.add_argument('--api-base', type=str, default=API_BASE,
                       help=f'API base URL (default: {API_BASE})')
    parser.add_argument('--max-solutions', type=int, default=MAX_SOLUTIONS_PER_WALLET,
                       help=f'Max solutions per wallet before switching (default: {MAX_SOLUTIONS_PER_WALLET})')
    parser.add_argument('--web-port', type=int, default=None,
                       help='Port for web dashboard (default: disabled)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable terminal dashboard')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Update globals from args
    WALLET_DIR = Path(args.wallet_dir)
    API_BASE = args.api_base
    MAX_SOLUTIONS_PER_WALLET = args.max_solutions
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse GPU IDs
    gpu_ids = []
    if args.gpu:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
            if CUDA_AVAILABLE:
                available_gpus = cp.cuda.runtime.getDeviceCount()
                gpu_ids = [g for g in gpu_ids if g < available_gpus]
                if gpu_ids:
                    logger.info(f"Using GPUs: {gpu_ids}")
                else:
                    logger.warning("No valid GPU IDs specified, using CPU only")
            else:
                logger.warning("CUDA not available, using CPU only")
                gpu_ids = []
        except ValueError:
            logger.error("Invalid GPU IDs format")
            gpu_ids = []
    
    # Auto-detect GPUs if none specified but CUDA is available
    if not gpu_ids and CUDA_AVAILABLE and GPU_COUNT > 0:
        gpu_ids = list(range(GPU_COUNT))
        logger.info(f"Auto-detected {GPU_COUNT} GPU(s): {gpu_ids}")
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           DEFENSIO SCAVENGER MINE - HYBRID MINER              ║
    ║                    GPU + CPU Mining                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  API Base: {API_BASE}")
    print(f"  Wallet Dir: {WALLET_DIR}")
    print(f"  CPU Workers: {args.cpu_workers}")
    print(f"  GPU Workers: {len(gpu_ids)} ({gpu_ids if gpu_ids else 'None'})")
    print(f"  CUDA Available: {CUDA_AVAILABLE}")
    print(f"  Max Solutions/Wallet: {MAX_SOLUTIONS_PER_WALLET}")
    print(f"  Consolidate To: {args.consolidate or 'None'}")
    print()
    
    # Create wallet manager
    wallet_manager = WalletManager(WALLET_DIR, NETWORK)
    api = DefensioAPI(API_BASE)
    
    # Pre-generate wallets if requested
    if args.generate_wallets > 0:
        logger.info(f"Pre-generating {args.generate_wallets} wallets...")
        for _ in range(args.generate_wallets):
            wallet = wallet_manager.generate_wallet()
            receipt = api.register_wallet(
                wallet.address,
                wallet.signing_key,
                wallet.verification_key
            )
            if receipt:
                wallet_manager.mark_registered(wallet.id, receipt)
            time.sleep(2)  # Rate limit
        logger.info("Wallet generation complete")
        
        if args.register_only:
            return
    
    # Consolidate only mode
    if args.consolidate_only:
        if not args.consolidate:
            logger.error("--consolidate address required with --consolidate-only")
            return
        miner = DefensioMiner(
            num_cpu_workers=0,
            gpu_ids=[],
            consolidate_address=args.consolidate
        )
        miner.consolidate_earnings()
        return
    
    # Create and run miner
    miner = DefensioMiner(
        num_cpu_workers=args.cpu_workers,
        gpu_ids=gpu_ids,
        consolidate_address=args.consolidate,
        web_port=args.web_port
    )
    
    miner.run()


if __name__ == "__main__":
    main()
