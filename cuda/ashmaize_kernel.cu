/*
 * Defensio Miner - CUDA Ashmaize Kernel
 * =====================================
 * High-performance GPU implementation of the Ashmaize proof-of-work algorithm.
 * 
 * Compile with:
 *   nvcc -O3 -arch=sm_70 -shared -o libashmaize_cuda.so ashmaize_kernel.cu
 *
 * For newer GPUs (RTX 30/40 series):
 *   nvcc -O3 -arch=sm_86 -shared -o libashmaize_cuda.so ashmaize_kernel.cu
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

// Constants
#define ROM_SIZE (1024 * 1024)  // 1MB ROM
#define BLAKE2B_OUTBYTES 32
#define BLOCK_SIZE 256
#define MAX_NONCES_PER_KERNEL 1048576

// BLAKE2b constants
__constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

// Result structure
struct MiningResult {
    uint64_t nonce;
    uint8_t hash[32];
    int found;
};

// Device functions for BLAKE2b
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ void blake2b_compress(uint64_t *h, const uint64_t *m, uint64_t t, int f) {
    uint64_t v[16];
    
    // Initialize working vector
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = blake2b_IV[i];
    }
    
    v[12] ^= t;
    if (f) v[14] ^= 0xFFFFFFFFFFFFFFFFULL;
    
    // Mixing rounds
    for (int round = 0; round < 12; round++) {
        #define G(r, i, a, b, c, d) do { \
            a = a + b + m[blake2b_sigma[r][2*i]]; \
            d = rotr64(d ^ a, 32); \
            c = c + d; \
            b = rotr64(b ^ c, 24); \
            a = a + b + m[blake2b_sigma[r][2*i+1]]; \
            d = rotr64(d ^ a, 16); \
            c = c + d; \
            b = rotr64(b ^ c, 63); \
        } while(0)
        
        G(round, 0, v[0], v[4], v[8], v[12]);
        G(round, 1, v[1], v[5], v[9], v[13]);
        G(round, 2, v[2], v[6], v[10], v[14]);
        G(round, 3, v[3], v[7], v[11], v[15]);
        G(round, 4, v[0], v[5], v[10], v[15]);
        G(round, 5, v[1], v[6], v[11], v[12]);
        G(round, 6, v[2], v[7], v[8], v[13]);
        G(round, 7, v[3], v[4], v[9], v[14]);
        
        #undef G
    }
    
    // Finalize
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

__device__ void blake2b_hash(const uint8_t *input, int input_len, uint8_t *output, int output_len) {
    uint64_t h[8];
    uint64_t m[16];
    
    // Initialize state
    for (int i = 0; i < 8; i++) {
        h[i] = blake2b_IV[i];
    }
    h[0] ^= 0x01010000 ^ output_len;
    
    // Process full blocks
    int blocks = input_len / 128;
    uint64_t t = 0;
    
    for (int b = 0; b < blocks; b++) {
        for (int i = 0; i < 16; i++) {
            m[i] = ((uint64_t*)&input[b * 128])[i];
        }
        t += 128;
        blake2b_compress(h, m, t, 0);
    }
    
    // Process final block (with padding)
    int remaining = input_len - blocks * 128;
    memset(m, 0, 128);
    for (int i = 0; i < remaining; i++) {
        ((uint8_t*)m)[i] = input[blocks * 128 + i];
    }
    t += remaining;
    blake2b_compress(h, m, t, 1);
    
    // Output
    for (int i = 0; i < output_len; i++) {
        output[i] = ((uint8_t*)h)[i];
    }
}

// Ashmaize hash function with ROM lookups
__device__ void ashmaize_hash(
    const uint8_t *seed,
    int seed_len,
    const uint8_t *address,
    int address_len,
    uint64_t nonce,
    const uint32_t *rom,
    int rom_size,
    uint8_t *output
) {
    // Build input: seed + address + nonce
    uint8_t input[256];
    int input_len = 0;
    
    for (int i = 0; i < seed_len && input_len < 256; i++) {
        input[input_len++] = seed[i];
    }
    for (int i = 0; i < address_len && input_len < 256; i++) {
        input[input_len++] = address[i];
    }
    
    // Add nonce (little-endian)
    for (int i = 0; i < 8 && input_len < 256; i++) {
        input[input_len++] = (nonce >> (i * 8)) & 0xFF;
    }
    
    // Initial BLAKE2b hash
    uint8_t h[32];
    blake2b_hash(input, input_len, h, 32);
    
    // Memory-hard mixing with ROM
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = ((uint32_t*)h)[i];
    }
    
    int rom_words = rom_size / 4;
    
    for (int round = 0; round < 64; round++) {
        // ROM lookup indices based on current state
        int idx0 = state[0] % (rom_words - 4);
        int idx1 = state[2] % (rom_words - 4);
        int idx2 = state[4] % (rom_words - 4);
        int idx3 = state[6] % (rom_words - 4);
        
        // Mix with ROM values
        state[0] ^= rom[idx0];
        state[1] = (state[1] + rom[idx1]) & 0xFFFFFFFF;
        state[2] ^= rom[idx2];
        state[3] = (state[3] + rom[idx3]) & 0xFFFFFFFF;
        state[4] ^= rom[idx0 + 1];
        state[5] = (state[5] + rom[idx1 + 1]) & 0xFFFFFFFF;
        state[6] ^= rom[idx2 + 1];
        state[7] = (state[7] + rom[idx3 + 1]) & 0xFFFFFFFF;
        
        // Rotate state
        uint32_t tmp = state[0];
        for (int i = 0; i < 7; i++) {
            state[i] = state[i + 1];
        }
        state[7] = tmp;
        
        // Additional mixing with BLAKE2b every 8 rounds
        if ((round + 1) % 8 == 0) {
            blake2b_hash((uint8_t*)state, 32, h, 32);
            for (int i = 0; i < 8; i++) {
                state[i] = ((uint32_t*)h)[i];
            }
        }
    }
    
    // Final hash
    blake2b_hash((uint8_t*)state, 32, output, 32);
}

// Check if hash meets difficulty (leading zeros)
__device__ int check_difficulty(const uint8_t *hash, int difficulty) {
    int leading_zeros = 0;
    
    for (int i = 0; i < 32; i++) {
        if (hash[i] == 0) {
            leading_zeros += 8;
        } else {
            // Count leading zeros in this byte
            for (int bit = 7; bit >= 0; bit--) {
                if (hash[i] & (1 << bit)) {
                    return leading_zeros >= difficulty;
                }
                leading_zeros++;
            }
            break;
        }
    }
    
    return leading_zeros >= difficulty;
}

// Main mining kernel
__global__ void ashmaize_mine_kernel(
    const uint8_t *seed,
    int seed_len,
    const uint8_t *address,
    int address_len,
    const uint32_t *rom,
    int rom_size,
    uint64_t start_nonce,
    int difficulty,
    MiningResult *result
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = start_nonce + idx;
    
    // Early exit if solution already found
    if (result->found) return;
    
    // Compute hash
    uint8_t hash[32];
    ashmaize_hash(seed, seed_len, address, address_len, nonce, rom, rom_size, hash);
    
    // Check difficulty
    if (check_difficulty(hash, difficulty)) {
        // Atomic to prevent race condition
        if (atomicCAS(&result->found, 0, 1) == 0) {
            result->nonce = nonce;
            for (int i = 0; i < 32; i++) {
                result->hash[i] = hash[i];
            }
        }
    }
}

// ROM building kernel
__global__ void build_rom_kernel(
    const uint8_t *rom_seed,
    int rom_seed_len,
    uint32_t *rom,
    int rom_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int words_per_thread = 16;
    int start_word = idx * words_per_thread;
    
    if (start_word >= rom_words) return;
    
    // Build state for this chunk
    uint8_t state[64];
    
    // Initial hash of seed with index
    uint8_t input[72];
    for (int i = 0; i < rom_seed_len && i < 64; i++) {
        input[i] = rom_seed[i];
    }
    
    // Add chunk index
    int chunk_idx = idx;
    for (int i = 0; i < 4; i++) {
        input[64 + i] = (chunk_idx >> (i * 8)) & 0xFF;
    }
    
    blake2b_hash(input, rom_seed_len + 4, state, 64);
    
    // Write to ROM
    int end_word = min(start_word + words_per_thread, rom_words);
    for (int i = start_word; i < end_word; i++) {
        rom[i] = ((uint32_t*)state)[(i - start_word) % 16];
    }
}

// Host-side wrapper functions
extern "C" {

// Build ROM table on GPU
int cuda_build_rom(
    const uint8_t *rom_seed,
    int rom_seed_len,
    uint32_t *rom_out,
    int device_id
) {
    cudaError_t err;
    
    // Select device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return -1;
    
    int rom_words = ROM_SIZE / 4;
    
    // Allocate device memory
    uint8_t *d_rom_seed;
    uint32_t *d_rom;
    
    cudaMalloc(&d_rom_seed, rom_seed_len);
    cudaMalloc(&d_rom, ROM_SIZE);
    
    // Copy seed to device
    cudaMemcpy(d_rom_seed, rom_seed, rom_seed_len, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int total_threads = (rom_words + 15) / 16;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    build_rom_kernel<<<num_blocks, threads_per_block>>>(
        d_rom_seed, rom_seed_len, d_rom, rom_words
    );
    
    // Copy result back
    cudaMemcpy(rom_out, d_rom, ROM_SIZE, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_rom_seed);
    cudaFree(d_rom);
    
    return 0;
}

// Mine for a solution
int cuda_mine(
    const uint8_t *seed,
    int seed_len,
    const uint8_t *address,
    int address_len,
    const uint32_t *rom,
    int rom_size,
    uint64_t start_nonce,
    uint64_t max_nonces,
    int difficulty,
    int device_id,
    uint64_t *found_nonce,
    uint8_t *found_hash
) {
    cudaError_t err;
    
    // Select device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return -1;
    
    // Allocate device memory
    uint8_t *d_seed, *d_address;
    uint32_t *d_rom;
    MiningResult *d_result;
    
    cudaMalloc(&d_seed, seed_len);
    cudaMalloc(&d_address, address_len);
    cudaMalloc(&d_rom, rom_size);
    cudaMalloc(&d_result, sizeof(MiningResult));
    
    // Copy data to device
    cudaMemcpy(d_seed, seed, seed_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_address, address, address_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rom, rom, rom_size, cudaMemcpyHostToDevice);
    
    // Initialize result
    MiningResult h_result = {0, {0}, 0};
    cudaMemcpy(d_result, &h_result, sizeof(MiningResult), cudaMemcpyHostToDevice);
    
    // Mining loop
    int threads_per_block = BLOCK_SIZE;
    int blocks = 1024;
    uint64_t nonces_per_kernel = (uint64_t)threads_per_block * blocks;
    
    for (uint64_t offset = 0; offset < max_nonces && !h_result.found; offset += nonces_per_kernel) {
        ashmaize_mine_kernel<<<blocks, threads_per_block>>>(
            d_seed, seed_len,
            d_address, address_len,
            d_rom, rom_size,
            start_nonce + offset,
            difficulty,
            d_result
        );
        
        cudaDeviceSynchronize();
        
        // Check if solution found
        cudaMemcpy(&h_result, d_result, sizeof(MiningResult), cudaMemcpyDeviceToHost);
    }
    
    // Copy result if found
    if (h_result.found) {
        *found_nonce = h_result.nonce;
        for (int i = 0; i < 32; i++) {
            found_hash[i] = h_result.hash[i];
        }
    }
    
    // Cleanup
    cudaFree(d_seed);
    cudaFree(d_address);
    cudaFree(d_rom);
    cudaFree(d_result);
    
    return h_result.found ? 1 : 0;
}

// Get number of CUDA devices
int cuda_get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

// Get device info
int cuda_get_device_info(int device_id, char *name, int name_len, size_t *total_mem) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return -1;
    
    strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';
    *total_mem = prop.totalGlobalMem;
    
    return 0;
}

} // extern "C"
