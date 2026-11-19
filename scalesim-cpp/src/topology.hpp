// Topology CSV parsing (mirrors scalesim/topology_utils.py).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

namespace scalesim {

struct Layer {
    std::string name;
    int64_t ifmap_h = 1, ifmap_w = 1;
    int64_t filt_h = 1, filt_w = 1;
    int64_t num_ch = 1, num_filt = 1;
    int64_t stride_h = 1, stride_w = 1;
    int64_t sparsity_n = 1, sparsity_m = 1;

    // Derived as in topologies.topo_calc_hyperparams()
    int64_t ofmap_h = 1, ofmap_w = 1;
    int64_t window = 1;  // filt_h * filt_w * num_ch

    int64_t ofmap_px() const { return ofmap_h * ofmap_w; }           // ofmap pixels per filter
    int64_t num_ofmap_px() const { return ofmap_px() * num_filt; }  // get_layer_num_ofmap_px()

    // Spatio-temporal dims (S_r, S_c, T) for the given dataflow.
    void spatiotemporal(Dataflow df, int64_t& s_row, int64_t& s_col, int64_t& t_time) const;

    // Two layers with the same key produce identical simulation results.
    std::string shape_key() const;
};

// gemm=true reads "Layer, M, N, K," rows; otherwise convolution rows.
std::vector<Layer> load_topology(const std::string& path, bool gemm);

}  // namespace scalesim
