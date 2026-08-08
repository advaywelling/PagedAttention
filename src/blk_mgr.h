#pragma once

#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>

class Block_Manager {
    public:
        void init(int num_blocks, int blk_size = 16) {
            block_size = blk_size;
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
        // the block table the kernel actually indexes through: logical block -> physical block
        const std::vector<int>& get_block_table(int seq_id) {
            return block_list[seq_id];
        }
        int get_block_size() const {
            return block_size;
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
