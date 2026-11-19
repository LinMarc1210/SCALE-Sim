// SRAM buffer models (mirrors scalesim/memory/{read_buffer,read_buffer_estimate_bw,write_buffer}.py
// with read_port/write_port in their non-Ramulator configuration).
#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

#include "common.hpp"
#include "csv_writer.hpp"

namespace scalesim {

// Maps an address to the (sorted, de-duplicated) list of hashed-buffer lines that contain it.
// Most operands contain every address once, so a single uint32 per address is enough; addresses
// that appear in several lines (overlapping convolution windows) spill into a side table.
class LineIndex {
public:
    void init(int64_t base, int64_t range);
    void add(int64_t addr, uint32_t line);  // lines must be added in non-decreasing order

    template <class InWindow>
    int contains(int64_t addr, InWindow&& in_window) const {  // -1: unknown address
        const int64_t idx = addr - base_;
        if (idx < 0 || idx >= range_) return -1;
        const uint32_t v = first_[static_cast<size_t>(idx)];
        if (v == kEmpty) return -1;
        if (!(v & kMulti)) return in_window(v) ? 1 : 0;
        for (uint32_t l : multi_[v & ~kMulti])
            if (in_window(l)) return 1;
        return 0;
    }

private:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
    static constexpr uint32_t kMulti = 0x80000000u;
    int64_t base_ = 0, range_ = 0;
    std::vector<uint32_t> first_;
    std::vector<std::vector<uint32_t>> multi_;
};

// read_buffer.py in USER bandwidth mode (no custom layout, no Ramulator).
class UserReadBuffer {
public:
    void set_params(int64_t total_size_bytes, int64_t backing_bw, int64_t num_bank);

    // Feed the prefetch sequence (set_fetch_matrix + prepare_hashed_buffer).
    void begin_prefetch_index(int64_t addr_base, int64_t addr_range);
    void push_prefetch_elem(int64_t elem) {
        ++num_fetch_elems_;
        if (elem == -1) return;
        index_.add(elem, static_cast<uint32_t>(num_valid_ / elems_per_set_));
        ++num_valid_;
    }
    void end_prefetch_index();

    // service_reads() for one demand row; returns the serviced cycle.
    int64_t service(int64_t cycle, const int64_t* addrs, size_t n);

    int64_t num_access() const { return num_access_; }

private:
    bool in_window(uint32_t l) const {
        const int64_t L = l;
        return (win_start_ < win_end_) ? (L >= win_start_ && L < win_end_) : (L >= win_start_ || L < win_end_);
    }
    void prefetch_active_buffer(int64_t start_cycle);
    void new_prefetch();

    int64_t total_elems_ = 0, active_size_ = 0, prefetch_size_ = 0, bw_ = 1;
    int64_t elems_per_set_ = 1;
    LineIndex index_;
    int64_t num_fetch_elems_ = 0, num_valid_ = 0;
    int64_t num_lines_ = 0, num_active_lines_ = 0, num_prefetch_lines_ = 0, fetch_rows_ = 0;
    bool active_full_ = false;
    int64_t win_start_ = 0, win_end_ = 0;
    int64_t last_prefetch_cycle_ = -1;
    int64_t num_access_ = 0;
};

// read_buffer_estimate_bw.py (CALC bandwidth mode).
class EstReadBuffer {
public:
    ~EstReadBuffer();
    void set_params(int64_t total_size_bytes, int64_t default_bw, int64_t addr_base, int64_t addr_range,
                    const std::string& trace_tmp_path /* empty: no trace */);
    int64_t service(int64_t cycle, const int64_t* addrs, size_t n);
    void complete_all_prefetches();
    void write_trace(const std::string& path);  // converts the temp trace into the final CSV

    int64_t num_access() const { return num_access_; }
    bool has_rows() const { return has_rows_; }
    int64_t first_row_cycle() const { return first_row_cycle_; }
    int64_t last_row_cycle() const { return last_row_cycle_; }

private:
    struct SetRec {
        int64_t size = 0;
        std::vector<int64_t> elems;  // insertion order (only kept when tracing)
    };
    void manage_prefetches(int64_t cycle, int64_t addr);
    void prefetch(int64_t start_set, int64_t end_set);

    int64_t items_per_set_ = 1, n_active_sets_ = 50, n_prefetch_sets_ = 50, default_bw_ = 1;
    int64_t prefetch_bw_ = 1;
    int64_t base_ = 0, range_ = 0;
    std::vector<int32_t> latest_set_;  // most recent set id each address was inserted into

    int64_t cur_set_id_ = 0, elems_cur_set_ = 0;
    std::vector<int64_t> cur_elems_;
    std::deque<SetRec> pending_;  // completed sets not yet prefetched
    int64_t pending_base_ = 0;    // set id of pending_.front()

    int64_t read_start_ = 0, read_end_ = 0, pf_start_ = -1, pf_end_ = -1;
    int64_t last_pf_start_ = -2, last_pf_end_ = -1, first_req_cycle_ = 0;
    bool first_seen_ = false, active_done_ = false;

    int64_t num_access_ = 0;
    bool has_rows_ = false;
    int64_t first_row_cycle_ = 0, last_row_cycle_ = 0;

    bool tracing_ = false;
    std::string tmp_path_;
    std::FILE* tmp_ = nullptr;
    int64_t max_width_ = 0;
};

// write_buffer.py (OFMAP SRAM).
class WriteBuffer {
public:
    void set_params(int64_t total_size_bytes, int64_t backing_bw, CsvWriter* dram_trace /* nullable */);
    int64_t service(int64_t cycle, const int64_t* addrs, size_t n);
    void empty_all_buffers(int64_t cycle);

    int64_t num_access() const { return num_access_; }
    bool trace_valid() const { return trace_valid_; }
    int64_t min_cycle() const { return min_cycle_; }
    int64_t max_cycle() const { return max_cycle_; }

private:
    void store(int64_t elem) {
        cur_line_[static_cast<size_t>(line_idx_++)] = elem;
        --free_space_;
        if (line_idx_ >= bw_) push_line();
    }
    void push_line();
    void force_flush() {
        if (line_idx_ != 0) push_line();
    }
    int64_t empty_drain_buf(int64_t start_cycle);
    size_t undrained_lines() const { return (lines_.size() - head_) / static_cast<size_t>(bw_); }

    int64_t total_elems_ = 0, active_size_ = 0, drain_size_ = 0, bw_ = 1, lines_per_drain_ = 1;
    int64_t free_space_ = 0, drain_end_cycle_ = 0;
    std::vector<int64_t> cur_line_;
    int64_t line_idx_ = 0;
    std::vector<int64_t> lines_;  // undrained lines, flattened, starting at head_
    size_t head_ = 0;
    bool any_lines_ = false;
    int64_t num_access_ = 0;
    bool trace_valid_ = false;
    int64_t min_cycle_ = 0, max_cycle_ = 0;
    CsvWriter* trace_ = nullptr;
};

}  // namespace scalesim
