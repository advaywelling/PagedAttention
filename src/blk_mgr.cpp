#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>

class Block_Manager {
    public: 
        void init(int num_blocks) {
            for(int i{}; i < num_blocks; i++) {
                free_list.push_back(i);
            }
        }
        bool append_token(int seq_id) {
            if (length[seq_id] % block_size == 0) { // allocate new block if full
                if (free_list.empty()) {
                    return false;
                }
                block_list[seq_id].push_back(free_list.back()); // choose free block
                free_list.pop_back();
            } 
            length[seq_id]++;
            return true;
        }
        void free_sequence(int seq_id) {
            while(!block_list[seq_id].empty()) {
                free_list.push_back(block_list[seq_id].back());
                block_list[seq_id].pop_back();
            }
            length[seq_id] = 0;
        }
        std::pair<int, int> get_physical_location(int position, int seq_id) {
            int block = position / block_size; // gets floored due to "int", logical block
            int slot = position % block_size;
            return {block_list[seq_id][block], slot};
        }
        void print_state(int seq_id) {
            std::cout << "Free blocks for sequence " << seq_id << ": ";
            for(int i : free_list) {
                std::cout << i << " ";
            }
            std::cout << '\n';
            std::cout << "Used blocks: ";
            for(int i : block_list[seq_id]) {
                std::cout << i << " ";
            }
            std::cout << '\n';
        }
        void memory_stats() {
            int token_count {};
            size_t block_count {};
            for(auto& entry : block_list) {
                token_count += length[entry.first]; // entry.first = seq_id
                block_count += entry.second.size(); // entry.second = blocks
            }
            if (block_count == 0) {
                std::cout << "No memory allocated";
                return;
            }
            std::cout << "Total blocks used: " << block_count << '\n';
            std::cout << "Wasted slots: " << (block_count * block_size) - token_count << '\n';
            std::cout << "Utilization: " << (double)token_count / (block_count * block_size) * 100 <<  "%" << '\n';
        }
        std::pair<int,int> get_stats() {
            int tokens {};
            int blocks {};
            for (auto& entry : block_list) {
                tokens += length[entry.first];
                blocks += entry.second.size();
            }
            return {blocks * block_size, tokens};
        }
    private:
        int block_size = 16;
        std::vector<int> free_list {}; // list of available blocks
        std::unordered_map<int, std::vector<int>> block_list {}; // seq_id -> list of blocks used by seq
        std::unordered_map<int, int> length {}; // seq_id -> length of seq
};

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
    mgr.init(100000); // big pool so we don't run out during benchmark
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