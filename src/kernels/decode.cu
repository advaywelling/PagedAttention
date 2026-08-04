#include <iostream>
#include <cmath>

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
            dot += q[j] * K[paged_index(block_table, i, j, block_sizde, d_k)];
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
    const int seq_len = 3, d_k = 4;

    // host values
    float h_q[d_k] = {1, 0, 1, 0};
    float h_K[seq_len * d_k] = { 1,0,0,0,  0,1,0,0,  1,1,0,0 };
    float h_V[seq_len * d_k] = { 1,2,3,4,   5,6,7,8,   9,10,11,12 };
    float h_out[d_k] = {};

    // gpu pointers
    float *d_q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_q, d_k * sizeof(float));
    cudaMalloc(&d_K, d_k * seq_len * sizeof(float));
    cudaMalloc(&d_V, d_k * seq_len * sizeof(float));
    cudaMalloc(&d_out, d_k * sizeof(float));

    // copy mem to gpu
    cudaMemcpy(d_q, h_q, d_k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, d_k * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, d_k * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, d_k * sizeof(float), cudaMemcpyHostToDevice);

    // just doing serially first lol
    decode_attention<<<1,1>>>(d_q, d_K, d_V, d_out, seq_len, d_k);

    // copy mem to cpu
    cudaMemcpy(h_out, d_out, d_k * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: ";
    for (int i{}; i < d_k; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << '\n';

    cudaFree(d_q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    return 0;
}
