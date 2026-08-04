#include <vector>
#include <utility>

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
    private:
        int block_size = 4;
        std::vector<int> free_list; // list of available blocks
        std::vector<int> block_list; // list of blocks used by ONE sequence
        int length = 0; // length of current sequence
};

