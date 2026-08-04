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
    private:
        int block_size = 4;
        std::vector<int> free_list {}; // list of available blocks
        std::unordered_map<int, std::vector<int>> block_list {}; // seq_id -> list of blocks used by seq
        std::unordered_map<int, int> length {}; // seq_id -> length of seq
};

int main() {
    Block_Manager blk_mgr;
    blk_mgr.init(10);

    // sequence 0 appends 5 tokens -> should grab 2 boxes
    for (int i = 0; i < 5; i++) {
        blk_mgr.append_token(0);
    }
    // sequence 1 appends 3 tokens -> should grab 1 box
    for (int i = 0; i < 3; i++) {
        blk_mgr.append_token(1);
    }

    std::cout << "--- after appends ---\n";
    blk_mgr.print_state(0);
    blk_mgr.print_state(1);

    // free sequence 0, its boxes should return to the pool
    blk_mgr.free_sequence(0);
    std::cout << "--- after freeing seq 0 ---\n";
    blk_mgr.print_state(1);   // seq 1 untouched, but free list grew
}