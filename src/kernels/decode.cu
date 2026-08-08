#include <iostream>
#include <cmath>
#include <vector>
#include "../blk_mgr.h"

// decode attention for ONE query against a pre-filled KV cache
__global__ void decode_attention(const float* q, const float* K, const float* V, float* out, int seq_len, int d_k) {
    float scores[64]; // row of scores one query produces

    // np.matmul(q_row, k_cache.T)
    // for each cached token i, dot query with key
    for(int i{}; i < seq_len; i++) {
        float dot = 0.0f;
        for (int j{}; j < d_k; j++) {
            dot += q[j] * K[i * d_k + j];
        }
        scores[i] = dot / sqrtf((float)d_k);
    }

    // find max score for softmax
    float max_score = scores[0];
    for(int i = 1; i < seq_len; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // exp(score- max) / sum of exp(score - max)
    float sum = 0.0f;
    for(int i{}; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // output = weighted sum of value rows
    for(int i{}; i < d_k; i++) {
        float acc = 0.0f;
        for(int j{}; j < seq_len; j++) {
            float weight = scores[j] / sum; // softmax weight for token i
            acc += weight * V[j * d_k + i];
        }
        out[i] = acc;
    }
}

// given a logical token position 'i' and element 'j', return the flat index for K/V from the block table
__device__ int paged_index(const int* block_table, int i, int j, int block_size, int d_k) {
    int logical_block = i / block_size;
    int slot = i % block_size;
    int physical_block = block_table[logical_block];
    return physical_block * (block_size * d_k) + slot * d_k + j;
}

__global__ void decode_attention_paged(const float* q, const float* K, const float* V, float* out, int seq_len, int d_k, const int* block_table, int block_size) {
    float scores[64]; // row of scores one query produces

    // np.matmul(q_row, k_cache.T)
    // for each cached token i, dot query with key
    for(int i{}; i < seq_len; i++) {
        float dot = 0.0f;
        for (int j{}; j < d_k; j++) {
            dot += q[j] * K[paged_index(block_table, i, j, block_size, d_k)];
        }
        scores[i] = dot / sqrtf((float)d_k);
    }

    // find max score for softmax
    float max_score = scores[0];
    for(int i = 1; i < seq_len; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // exp(score- max) / sum of exp(score - max)
    float sum = 0.0f;
    for(int i{}; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // output = weighted sum of value rows
    for(int i{}; i < d_k; i++) {
        float acc = 0.0f;
        for(int j{}; j < seq_len; j++) {
            float weight = scores[j] / sum; // softmax weight for token i
            acc += weight * V[paged_index(block_table, j, i, block_size, d_k)];
        }
        out[i] = acc;
    }
}

int main() {
    const int seq_len = 3, d_k = 4, block_size = 2;

    // host values - naive
    float h_q[d_k] = {1, 0, 1, 0};
    float h_K[seq_len * d_k] = { 1,0,0,0,  0,1,0,0,  1,1,0,0 };
    float h_V[seq_len * d_k] = { 1,2,3,4,   5,6,7,8,   9,10,11,12 };
    float h_out[d_k] = {};

    // host values - paged, the block manager decides where tokens live now
    const int num_phys_blocks = 8;
    const int pool_floats = num_phys_blocks * block_size * d_k;

    Block_Manager mgr;
    mgr.init(num_phys_blocks, block_size);
    // this sequence steals blocks in between so seq 0's table comes out scattered
    for (int i{}; i < seq_len; i++) {
        mgr.append_token(0);
        mgr.append_token(1);
    }
    const std::vector<int>& h_block_table = mgr.get_block_table(0);

    std::cout << "Block table for seq 0 (logical -> physical): ";
    for(size_t i{}; i < h_block_table.size(); i++) {
        std::cout << i << "->" << h_block_table[i] << " ";
    }
    std::cout << '\n';

    // any read of an unallocated slot becomes N/A in the output
    float h_K_paged[pool_floats];
    float h_V_paged[pool_floats];
    for(int i{}; i < pool_floats; i++) {
        h_K_paged[i] = nanf("");
        h_V_paged[i] = nanf("");
    }

    // scatter each token into the slot the manager assigned it
    // this is the host-side mirror of paged_index(), minus j
    for(int i{}; i < seq_len; i++) {
        std::pair<int,int> loc = mgr.get_physical_location(i, 0); // {physical block, slot}
        int base = loc.first * (block_size * d_k) + loc.second * d_k;
        for (int j{}; j < d_k; j++) {
            h_K_paged[base + j] = h_K[i * d_k + j];
            h_V_paged[base + j] = h_V[i * d_k + j];
        }
    }

    float h_out_paged[d_k] {};

    // gpu pointers
    float *d_q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_q, d_k * sizeof(float));
    cudaMalloc(&d_K, d_k * seq_len * sizeof(float));
    cudaMalloc(&d_V, d_k * seq_len * sizeof(float));
    cudaMalloc(&d_out, d_k * sizeof(float));

    float *d_K_paged, *d_V_paged, *d_out_paged;
    int *d_block_table;
    cudaMalloc(&d_K_paged, pool_floats * sizeof(float));
    cudaMalloc(&d_V_paged, pool_floats * sizeof(float));
    cudaMalloc(&d_out_paged, d_k * sizeof(float));
    cudaMalloc(&d_block_table, h_block_table.size() * sizeof(int));

    // copy mem to gpu
    cudaMemcpy(d_q, h_q, d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, d_k * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, d_k * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_K_paged, h_K_paged, pool_floats * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_paged, h_V_paged, pool_floats * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_table, h_block_table.data(), h_block_table.size() * sizeof(int), cudaMemcpyHostToDevice);

    // just doing serially first lol
    decode_attention<<<1,1>>>(d_q, d_K, d_V, d_out, seq_len, d_k);
    decode_attention_paged<<<1,1>>>(d_q, d_K_paged, d_V_paged, d_out_paged, seq_len, d_k, d_block_table, block_size);
    cudaDeviceSynchronize();

    // copy mem to cpu
    cudaMemcpy(h_out, d_out, d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_paged, d_out_paged, d_k * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Naive output: ";
    for (int i{}; i < d_k; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << '\n';
    std::cout << "Paged output: ";
    for (int i{}; i < d_k; i++) {
        std::cout << h_out_paged[i] << " ";
    }
    std::cout << '\n';

    // paging is pure indirection, so these must agree
    float max_diff = 0.0f;
    for (int i{}; i < d_k; i++) {
        max_diff = fmaxf(max_diff, fabsf(h_out[i] - h_out_paged[i]));
    }
    std::cout << "Max abs diff = " << max_diff << '\n';

    cudaFree(d_q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    cudaFree(d_K_paged);
    cudaFree(d_V_paged);
    cudaFree(d_out_paged);
    cudaFree(d_block_table);

    if (!(max_diff < 1e-5f)) { // '!' so NaN counts as a failure
        std::cout << "PAGED DOES NOT MATCH NAIVE OH NO\n";
        return 1;
    }
    std::cout << "yay paged and naive is de match\n";
    return 0;
}
