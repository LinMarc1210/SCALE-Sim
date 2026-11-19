// SCALE-Sim .cfg parsing (mirrors scalesim/scale_config.py).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

namespace scalesim {

struct Config {
    std::string run_name = "scale_run";
    bool use_user_bandwidth = false;

    int64_t array_rows = 4;
    int64_t array_cols = 4;
    int64_t ifmap_sz_kb = 256;
    int64_t filter_sz_kb = 256;
    int64_t ofmap_sz_kb = 128;
    std::string df = "ws";
    Dataflow dataflow = Dataflow::WS;
    int64_t ifmap_offset = 0;
    int64_t filter_offset = 10000000;
    int64_t ofmap_offset = 20000000;
    int64_t req_buf_sz_rd = 60;
    int64_t req_buf_sz_wr = 60;
    std::vector<int64_t> bandwidths;

    bool using_ifmap_custom_layout = false;
    int64_t ifmap_sram_bank_bandwidth = 10;
    int64_t ifmap_sram_bank_num = 10;
    int64_t ifmap_sram_bank_port = 2;
    bool using_filter_custom_layout = false;
    int64_t filter_sram_bank_bandwidth = 10;
    int64_t filter_sram_bank_num = 10;
    int64_t filter_sram_bank_port = 2;

    bool sparsity_support = false;
    bool use_ramulator_trace = false;
    std::string time_linear_model = "None";
};

// Parses a SCALE-Sim config file. Throws SimError on malformed input or on features that the
// C++ port does not implement yet (sparsity, custom layouts, Ramulator traces).
Config read_config(const std::string& path);

}  // namespace scalesim
