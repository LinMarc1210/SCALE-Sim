#include "memory.hpp"

#include <algorithm>

#include "pyset.hpp"

namespace scalesim {

// ---------------------------------------------------------------------------------------------
// LineIndex
// ---------------------------------------------------------------------------------------------
void LineIndex::init(int64_t base, int64_t range) {
    base_ = base;
    range_ = range;
    first_.assign(static_cast<size_t>(range), kEmpty);
    multi_.clear();
}

void LineIndex::add(int64_t addr, uint32_t line) {
    const int64_t idx = addr - base_;
    if (idx < 0 || idx >= range_) throw SimError("prefetch address out of the operand range");
    if (line >= kMulti) throw SimError("too many hashed-buffer lines");
    uint32_t& v = first_[static_cast<size_t>(idx)];
    if (v == kEmpty) {
        v = line;
    } else if (!(v & kMulti)) {
        if (v != line) {
            if (multi_.size() >= kMulti - 1) throw SimError("line index overflow");
            multi_.push_back({v, line});
            v = kMulti | static_cast<uint32_t>(multi_.size() - 1);
        }
    } else {
        auto& lst = multi_[v & ~kMulti];
        if (lst.back() != line) lst.push_back(line);
    }
}

// ---------------------------------------------------------------------------------------------
// UserReadBuffer  (read_buffer.py, enable_layout_evaluation=False, use_ramulator_trace=False)
// ---------------------------------------------------------------------------------------------
void UserReadBuffer::set_params(int64_t total_size_bytes, int64_t backing_bw, int64_t num_bank) {
    total_elems_ = total_size_bytes;  // word_size = 1
    active_size_ = py_ceil_mul(total_elems_, 0.5);
    prefetch_size_ = total_elems_ - active_size_;
    bw_ = backing_bw;
    if (num_bank == 0) throw SimError("ZeroDivisionError: number of SRAM banks is 0");
    // bw_per_bank = bw // num_bank;  assert bw_per_bank * num_bank == bw
    int64_t q = bw_ / num_bank;
    if ((bw_ % num_bank != 0) && ((bw_ < 0) != (num_bank < 0))) --q;  // Python floor division
    if (q * num_bank != bw_)
        throw SimError("overall bandwidth must be divisible by total number of banks, number of banks = " +
                       std::to_string(num_bank) + ", bandwidth of each as " + std::to_string(q) +
                       ", total bandwidth = " + std::to_string(bw_));
    elems_per_set_ = py_ceil_div(total_elems_, 100);
    if (elems_per_set_ <= 0) throw SimError("SRAM too small");
}

void UserReadBuffer::begin_prefetch_index(int64_t addr_base, int64_t addr_range) {
    index_.init(addr_base, addr_range);
    num_fetch_elems_ = 0;
    num_valid_ = 0;
}

void UserReadBuffer::end_prefetch_index() {
    num_lines_ = num_valid_ / elems_per_set_ + 1;
    fetch_rows_ = py_ceil_div(num_fetch_elems_, bw_);
    const int64_t max_active = py_ceil_div(active_size_, elems_per_set_);
    const int64_t max_prefetch = py_ceil_div(prefetch_size_, elems_per_set_);
    num_active_lines_ = (num_lines_ > max_active) ? max_active : num_lines_;
    const int64_t remaining = num_lines_ - num_active_lines_;
    num_prefetch_lines_ = (remaining > max_prefetch) ? max_prefetch : remaining;
}

void UserReadBuffer::prefetch_active_buffer(int64_t start_cycle) {
    int64_t n = py_ceil_div(active_size_, bw_);
    if (!(n < fetch_rows_)) n = fetch_rows_;
    if (n <= 0) throw SimError("empty prefetch matrix");
    num_access_ += n * bw_;
    // cycles[i] = -(n - start - (i - lat)), responses = cycles + lat (lat = 1): max = start - 1
    last_prefetch_cycle_ = start_cycle - 1;
    win_start_ = 0;
    win_end_ = num_active_lines_;
    active_full_ = true;
}

void UserReadBuffer::new_prefetch() {
    win_start_ = (win_start_ + num_prefetch_lines_) % num_lines_;
    win_end_ = (win_start_ + num_active_lines_) % num_lines_;
    const int64_t n = py_ceil_div(prefetch_size_, bw_);
    if (n <= 0) throw SimError("empty prefetch request");
    num_access_ += n * bw_;
    // cycles[i] = last + i + 1, responses = cycles + 1: max = last + n + 1
    last_prefetch_cycle_ += n + 1;
}

int64_t UserReadBuffer::service(int64_t cycle, const int64_t* addrs, size_t n) {
    if (!active_full_) prefetch_active_buffer(cycle);
    int64_t offset = 1;  // hit latency
    auto inw = [this](uint32_t l) { return in_window(l); };
    for (size_t k = 0; k < n; ++k) {
        const int64_t addr = addrs[k];
        int hit = index_.contains(addr, inw);
        if (hit == 1) continue;
        if (hit < 0)
            throw SimError("address " + std::to_string(addr) +
                           " is not in the prefetch matrix (the Python model would loop forever)");
        int64_t guard = 0;
        while (hit == 0) {
            if (num_prefetch_lines_ == 0 || ++guard > num_lines_ + 1)
                throw SimError("read buffer can never hit address " + std::to_string(addr) +
                               " (the Python model would loop forever)");
            new_prefetch();
            const int64_t potential_stall = last_prefetch_cycle_ - (cycle + offset);
            if (potential_stall > 0) offset += potential_stall;
            hit = index_.contains(addr, inw);
        }
    }
    return cycle + offset;
}

// ---------------------------------------------------------------------------------------------
// EstReadBuffer  (read_buffer_estimate_bw.py)
// ---------------------------------------------------------------------------------------------
EstReadBuffer::~EstReadBuffer() {
    if (tmp_) {
        std::fclose(tmp_);
        std::remove(tmp_path_.c_str());
    }
}

void EstReadBuffer::set_params(int64_t total_size_bytes, int64_t default_bw, int64_t addr_base,
                               int64_t addr_range, const std::string& trace_tmp_path) {
    const int64_t total_elems = total_size_bytes;  // word_size = 1
    items_per_set_ = py_floor_div(total_elems, 100);
    n_active_sets_ = static_cast<int64_t>(0.5 * 100);  // int(round(0.5, 2) * 100)
    n_prefetch_sets_ = 100 - n_active_sets_;
    default_bw_ = default_bw;
    prefetch_bw_ = default_bw;
    read_start_ = 0;
    read_end_ = n_active_sets_ - 1;
    last_pf_start_ = -2;
    last_pf_end_ = -1;
    base_ = addr_base;
    range_ = addr_range;
    latest_set_.assign(static_cast<size_t>(addr_range), -1);
    tracing_ = !trace_tmp_path.empty();
    if (tracing_) {
        tmp_path_ = trace_tmp_path;
        tmp_ = std::fopen(tmp_path_.c_str(), "w+b");
        if (!tmp_) throw SimError("cannot create temp file " + tmp_path_);
    }
}

int64_t EstReadBuffer::service(int64_t cycle, const int64_t* addrs, size_t n) {
    if (!first_seen_) {
        for (size_t k = 0; k < n; ++k)
            if (addrs[k] > -1) {
                first_req_cycle_ = cycle;
                first_seen_ = true;
                break;
            }
    }
    for (size_t k = 0; k < n; ++k) manage_prefetches(cycle, addrs[k]);
    return cycle + 1;  // stall free: always a hit
}

void EstReadBuffer::manage_prefetches(int64_t cycle, int64_t addr) {
    const int64_t idx = addr - base_;
    if (idx < 0 || idx >= range_) throw SimError("demand address out of the operand range");
    int32_t& latest = latest_set_[static_cast<size_t>(idx)];
    // check_hit(): addr is in a completed set inside [read_start, min(cur, read_end + 1)).
    // Since cur <= read_end + 1 always holds, the newest set containing addr decides.
    if (latest >= read_start_ && latest < cur_set_id_) return;
    if (latest == cur_set_id_) return;  // already in the set being filled
    latest = static_cast<int32_t>(cur_set_id_);
    ++elems_cur_set_;
    if (tracing_) cur_elems_.push_back(addr);
    if (elems_cur_set_ != items_per_set_) return;

    SetRec rec;
    rec.size = elems_cur_set_;
    rec.elems.swap(cur_elems_);
    pending_.push_back(std::move(rec));
    elems_cur_set_ = 0;
    ++cur_set_id_;
    if (cur_set_id_ > INT32_MAX - 2) throw SimError("too many buffer sets");

    if (cur_set_id_ != read_end_ + 1) return;
    if (!active_done_) {
        prefetch_bw_ = default_bw_;
        last_pf_end_ = first_req_cycle_ - 1 - 1;  // backing latency = 1
        const int64_t cycles_needed = py_ceil_div(n_prefetch_sets_ * items_per_set_, prefetch_bw_);
        last_pf_start_ = last_pf_end_ - cycles_needed + 1;
        prefetch(0, n_active_sets_ - 1);
        pf_start_ = read_end_ + 1;
        pf_end_ = pf_start_ + n_prefetch_sets_ - 1;
        active_done_ = true;
    } else {
        const int64_t elems = n_prefetch_sets_ * items_per_set_;
        const int64_t cycles_needed = last_pf_end_ - last_pf_start_ + 1;
        prefetch_bw_ = py_ceil_div(elems, cycles_needed);
        prefetch(pf_start_, pf_end_);
        pf_start_ += n_prefetch_sets_;
        pf_end_ += n_prefetch_sets_;
    }
    read_start_ += n_prefetch_sets_;
    read_end_ += n_prefetch_sets_;
    last_pf_start_ = last_pf_end_ + 1;
    last_pf_end_ = cycle;
}

void EstReadBuffer::complete_all_prefetches() {
    if (elems_cur_set_ > 0) {
        SetRec rec;
        rec.size = elems_cur_set_;
        rec.elems.swap(cur_elems_);
        pending_.push_back(std::move(rec));
    } else {
        --cur_set_id_;
    }
    if (!active_done_) {
        prefetch_bw_ = default_bw_;
        last_pf_end_ = -1 - 1;
        const int64_t n_sets = cur_set_id_ + 1;
        n_active_sets_ = n_sets;
        const int64_t cycles_needed = py_ceil_div(n_sets * items_per_set_, prefetch_bw_);
        last_pf_start_ = last_pf_end_ - cycles_needed + 1;
        prefetch(0, n_sets - 1);
        active_done_ = true;
    } else {
        const int64_t n_sets = cur_set_id_ - pf_start_ + 1;
        pf_end_ = cur_set_id_;
        const int64_t elems = n_sets * items_per_set_;
        const int64_t cycles_needed = last_pf_end_ - last_pf_start_ + 1;
        prefetch_bw_ = py_ceil_div(elems, cycles_needed);
        prefetch(pf_start_, pf_end_);
    }
}

void EstReadBuffer::prefetch(int64_t start_set, int64_t end_set) {
    const int64_t nsets = end_set - start_set + 1;
    if (nsets > 0 && start_set != pending_base_)
        throw SimError("internal error: unexpected prefetch set range");
    if (nsets > static_cast<int64_t>(pending_.size()))
        throw SimError("internal error: prefetching sets that do not exist");

    int64_t count = 0;
    std::vector<int64_t> addrs;
    for (int64_t s = 0; s < nsets; ++s) {
        const SetRec& rec = pending_[static_cast<size_t>(s)];
        count += rec.size;
        if (tracing_) {
            PySetOrder order;  // all_addresses += list(this_set)
            for (int64_t a : rec.elems) order.add(a);
            order.for_each([&](int64_t a) { addrs.push_back(a); });
        }
    }
    num_access_ += count;

    const int64_t cycles_needed = last_pf_end_ - last_pf_start_ + 1;
    if (cycles_needed < 0) throw SimError("negative prefetch window (Python would fail here)");
    const int64_t capacity = cycles_needed * prefetch_bw_;
    if (count > capacity) throw SimError("prefetch does not fit the estimated bandwidth (Python reshape error)");

    if (cycles_needed > 0) {
        if (!has_rows_) {
            has_rows_ = true;
            first_row_cycle_ = last_pf_start_ + 1;
        }
        last_row_cycle_ = last_pf_start_ + cycles_needed - 1 + 1;
    }
    if (tracing_) {
        max_width_ = std::max(max_width_, prefetch_bw_ + 1);
        const int64_t hdr[2] = {cycles_needed, prefetch_bw_};
        std::fwrite(hdr, sizeof(int64_t), 2, tmp_);
        addrs.resize(static_cast<size_t>(capacity), -1);
        for (int64_t j = 0; j < cycles_needed; ++j) {
            const int64_t cyc = last_pf_start_ + j + 1;
            std::fwrite(&cyc, sizeof(int64_t), 1, tmp_);
            if (prefetch_bw_ > 0)
                std::fwrite(addrs.data() + j * prefetch_bw_, sizeof(int64_t), static_cast<size_t>(prefetch_bw_), tmp_);
        }
    }
    for (int64_t s = 0; s < nsets; ++s) pending_.pop_front();
    pending_base_ += nsets;
}

void EstReadBuffer::write_trace(const std::string& path) {
    if (!tracing_ || !tmp_) return;
    std::fflush(tmp_);
    std::rewind(tmp_);
    CsvWriter out;
    out.open(path);
    std::vector<int64_t> row;
    int64_t hdr[2];
    while (std::fread(hdr, sizeof(int64_t), 2, tmp_) == 2) {
        const int64_t rows = hdr[0], width = hdr[1];
        row.resize(static_cast<size_t>(width + 1));
        for (int64_t j = 0; j < rows; ++j) {
            if (std::fread(row.data(), sizeof(int64_t), static_cast<size_t>(width + 1), tmp_) !=
                static_cast<size_t>(width + 1))
                throw SimError("corrupt temp trace file");
            for (int64_t k = 0; k <= width; ++k) {
                if (k) out.put_char(',');
                out.put_intfloat(row[static_cast<size_t>(k)]);
            }
            for (int64_t k = width + 1; k < max_width_; ++k) out.put_raw(",1.0", 4);  // np.ones padding
            out.put_char('\n');
        }
    }
    out.close();
    std::fclose(tmp_);
    tmp_ = nullptr;
    std::remove(tmp_path_.c_str());
}

// ---------------------------------------------------------------------------------------------
// WriteBuffer  (write_buffer.py with write_port latency 0)
// ---------------------------------------------------------------------------------------------
void WriteBuffer::set_params(int64_t total_size_bytes, int64_t backing_bw, CsvWriter* dram_trace) {
    total_elems_ = total_size_bytes;  // word_size = 1
    active_size_ = py_ceil_mul(total_elems_, 0.5);
    drain_size_ = total_elems_ - active_size_;
    bw_ = backing_bw;
    if (bw_ <= 0) throw SimError("invalid OFMAP backing bandwidth");
    lines_per_drain_ = py_ceil_div(drain_size_, bw_);
    free_space_ = total_elems_;
    cur_line_.assign(static_cast<size_t>(bw_), -1);
    trace_ = dram_trace;
}

void WriteBuffer::push_line() {
    lines_.insert(lines_.end(), cur_line_.begin(), cur_line_.end());
    std::fill(cur_line_.begin(), cur_line_.end(), -1);
    line_idx_ = 0;
    any_lines_ = true;
}

int64_t WriteBuffer::service(int64_t cycle, const int64_t* addrs, size_t n) {
    int64_t current = cycle;
    for (size_t k = 0; k < n; ++k) {
        store(addrs[k]);
        if (current < drain_end_cycle_) {
            if (!(free_space_ > 0)) current = drain_end_cycle_;
        } else if (free_space_ < active_size_) {
            force_flush();
            drain_end_cycle_ = empty_drain_buf(current);
        }
    }
    return current;
}

int64_t WriteBuffer::empty_drain_buf(int64_t start_cycle) {
    const int64_t avail = static_cast<int64_t>(undrained_lines());
    const int64_t n = std::min(lines_per_drain_, avail);
    if (n <= 0) throw SimError("IndexError: draining an empty OFMAP buffer");
    const size_t bw = static_cast<size_t>(bw_);
    const int64_t* base = lines_.data() + head_;
    int64_t data = n * bw_;
    const int64_t* last = base + static_cast<size_t>(n - 1) * bw;
    for (size_t k = 0; k < bw; ++k)
        if (last[k] == -1) --data;
    num_access_ += data;

    for (int64_t j = 0; j < n; ++j) {
        const int64_t cyc = start_cycle + j;
        if (!trace_valid_) {
            min_cycle_ = max_cycle_ = cyc;
            trace_valid_ = true;
        } else {
            min_cycle_ = std::min(min_cycle_, cyc);
            max_cycle_ = std::max(max_cycle_, cyc);
        }
        if (trace_) {
            trace_->put_intfloat(cyc);
            const int64_t* line = base + static_cast<size_t>(j) * bw;
            for (size_t k = 0; k < bw; ++k) {
                trace_->put_char(',');
                trace_->put_intfloat(line[k]);
            }
            trace_->put_char('\n');
        }
    }
    head_ += static_cast<size_t>(n) * bw;
    if (head_ > (1u << 16) && head_ * 2 > lines_.size()) {
        lines_.erase(lines_.begin(), lines_.begin() + static_cast<std::ptrdiff_t>(head_));
        head_ = 0;
    }
    free_space_ += data;
    return start_cycle + n - 1;
}

void WriteBuffer::empty_all_buffers(int64_t cycle) {
    force_flush();
    if (!any_lines_) return;
    while (undrained_lines() > 0) {
        drain_end_cycle_ = empty_drain_buf(cycle);
        cycle = drain_end_cycle_ + 1;
    }
}

}  // namespace scalesim
