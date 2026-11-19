// Single layer simulation (mirrors scalesim/single_layer_sim.py + double_buffered_scratchpad_mem.py).
#pragma once

#include <cstdint>
#include <string>

#include "config.hpp"
#include "topology.hpp"

namespace scalesim {

struct LayerResult {
    // COMPUTE_REPORT
    int64_t overall_cycles = 0, total_cycles = 0, stall_cycles = 0;
    double overall_util = 0, mapping_eff = 0, compute_util = 0;
    // BANDWIDTH_REPORT
    double avg_ifmap_sram_bw = 0, avg_filter_sram_bw = 0, avg_ofmap_sram_bw = 0;
    double avg_ifmap_dram_bw = 0, avg_filter_dram_bw = 0, avg_ofmap_dram_bw = 0;
    // DETAILED_ACCESS_REPORT (SRAM start/stop are python ints 0 when the trace has no request)
    bool sram_found[3] = {false, false, false};
    int64_t sram_start[3] = {0, 0, 0}, sram_stop[3] = {0, 0, 0};
    int64_t ifmap_sram_reads = 0, filter_sram_reads = 0, ofmap_sram_writes = 0;
    int64_t dram_start[3] = {0, 0, 0}, dram_stop[3] = {0, 0, 0}, dram_access[3] = {0, 0, 0};

    double seconds = 0;
};

// Simulates one layer. When save_trace is set, the six trace CSVs are written to
// <run_dir>/layer<layer_id>/ while the simulation runs (streaming, O(1) memory).
LayerResult simulate_layer(const Config& cfg, const Layer& layer, int layer_id, const std::string& run_dir,
                           bool save_trace);

}  // namespace scalesim
