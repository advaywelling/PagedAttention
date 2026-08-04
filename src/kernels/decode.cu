#include <iostream>
#include <cmath>

__global__ void decode_attention(const float* q, const float* K, const float* V, float* out, int seq_len, int d_k) {
    
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
    decode_attention<<1,1>>(d_q, d_K, d_V, d_out, seq_len, d_k);

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
