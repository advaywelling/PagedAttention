#include <vector>
#include <utility>
#include <iostream>

class Block_Manager {
    public: 
        void init(int num_blocks) {
            for(int i{}; i < num_blocks; i++) {
                free_list.push_back(i);
            }
        }
        bool append_token() {
            if (length % block_size == 0) { // allocate new block if full
                if (free_list.empty()) {
                    return false;
                }
                block_list.push_back(free_list.back()); // choose free block
                free_list.pop_back();
            } 
            length++;
            return true;
        }
        void free() {
            while(!block_list.empty()) {
                free_list.push_back(block_list.back());
                block_list.pop_back();
            }
            length = 0;
        }
        std::pair<int, int> get_physical_location(int position) {
            int block = position / block_size; // gets floored due to "int", logical block
            int slot = position % block_size;
            return {block_list[block], slot};
        }
        void print_state() {
            std::cout << "Free blocks: ";
            for(int i : free_list) {
                std::cout << i << " ";
            }
            std::cout << '\n';
            std::cout << "Used blocks: ";
            for(int i : block_list) {
                std::cout << i << " ";
            }
            std::cout << '\n';
        }
    private:
        int block_size = 4;
        std::vector<int> free_list; // list of available blocks
        std::vector<int> block_list; // list of blocks used by ONE sequence
        int length = 0; // length of current sequence
};

int main() {
    Block_Manager blk_mgr;
    blk_mgr.init(10);
    for(int i{}; i < 5; i++) {
        blk_mgr.append_token();
    }
    blk_mgr.print_state();
}