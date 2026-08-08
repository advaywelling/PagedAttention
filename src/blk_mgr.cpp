#include "blk_mgr.h"
#include <cstdlib>

int main() {
    const int num_seq= 100;
    const int max_seq_len = 128;
    const int block_size = 16;

    // generate varied workload
    srand(42);
    std::vector<int> lengths;
    int total_used_tokens {};
    for(int i {}; i < num_seq; i ++) {
        int len = rand() % max_seq_len + 1;
        lengths.push_back(len);
        total_used_tokens += len;
    }
    int naive = num_seq * max_seq_len;

    Block_Manager mgr;
    mgr.init(100000, block_size); // big pool so we don't run out during benchmark
    for(int i{}; i < num_seq; i++) {
        for(int j{}; j < lengths[i]; j++) {
            mgr.append_token(i);
        }
    }
    auto stats = mgr.get_stats();
    int paged = stats.first;
    int paged_used = stats.second;
    std::cout << "Total real tokens: " << total_used_tokens << "\n\n";

    std::cout << "NAIVE allocated slots: " << naive << "\n";
    std::cout << "NAIVE utilization: " << (double)total_used_tokens / naive * 100 << "%\n\n";

    std::cout << "PAGED allocated slots: " << paged << "\n";
    std::cout << "PAGED utilization: " << (double)paged_used / paged * 100 << "%\n\n";

    std::cout << "Paged uses " << (double)paged / naive * 100 << "% of the memory naive does\n";
    return 0;
}
