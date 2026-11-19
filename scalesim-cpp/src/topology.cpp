#include "topology.hpp"

#include <cmath>
#include <fstream>

#include "textutil.hpp"

namespace scalesim {
namespace {

// topologies.append_topo_arrays(): elems[1:] are converted with int(str(e).strip()).
Layer make_layer(const std::string& name, const std::vector<std::string>& elems) {
    if (elems.size() < 11) throw SimError("topology: too few fields for layer " + name);
    Layer l;
    l.name = name;
    int64_t v[10];
    for (int i = 0; i < 10; ++i) v[i] = py_int(elems[static_cast<size_t>(i) + 1], "topology field of " + name);
    l.ifmap_h = v[0];
    l.ifmap_w = v[1];
    l.filt_h = v[2];
    l.filt_w = v[3];
    l.num_ch = v[4];
    l.num_filt = v[5];
    l.stride_h = v[6];
    l.stride_w = v[7];
    l.sparsity_n = v[8];
    l.sparsity_m = v[9];
    if (l.filt_h > l.ifmap_h) throw SimError("Filter height cannot be larger than IFMAP height");
    if (l.filt_w > l.ifmap_w) throw SimError("Filter width cannot be larger than IFMAP width");

    l.ofmap_h = static_cast<int64_t>(std::ceil(static_cast<double>(l.ifmap_h - l.filt_h + l.stride_h) /
                                               static_cast<double>(l.stride_h)));
    l.ofmap_w = static_cast<int64_t>(std::ceil(static_cast<double>(l.ifmap_w - l.filt_w + l.stride_w) /
                                               static_cast<double>(l.stride_w)));
    l.window = l.filt_h * l.filt_w * l.num_ch;
    return l;
}

std::vector<std::string> split_sparsity(const std::string& s) {
    auto parts = py_split(py_strip(s), ':');
    if (parts.size() < 2) throw SimError("topology: invalid sparsity ratio '" + s + "'");
    return parts;
}

}  // namespace

void Layer::spatiotemporal(Dataflow df, int64_t& s_row, int64_t& s_col, int64_t& t_time) const {
    // calc_spatio_temporal_params(): num_ofmap = int(num_ofmap_px / num_filt)
    const int64_t num_ofmap =
        static_cast<int64_t>(static_cast<double>(num_ofmap_px()) / static_cast<double>(num_filt));
    switch (df) {
        case Dataflow::OS: s_row = num_ofmap; s_col = num_filt; t_time = window; break;
        case Dataflow::WS: s_row = window; s_col = num_filt; t_time = num_ofmap; break;
        case Dataflow::IS: s_row = window; s_col = num_ofmap; t_time = num_filt; break;
    }
}

std::string Layer::shape_key() const {
    return std::to_string(ifmap_h) + "," + std::to_string(ifmap_w) + "," + std::to_string(filt_h) + "," +
           std::to_string(filt_w) + "," + std::to_string(num_ch) + "," + std::to_string(num_filt) + "," +
           std::to_string(stride_h) + "," + std::to_string(stride_w);
}

std::vector<Layer> load_topology(const std::string& path, bool gemm) {
    std::ifstream in(path);
    if (!in) throw SimError("cannot open topology file: " + path);
    std::vector<Layer> layers;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        const std::string row = py_strip(line);
        if (first) {
            first = false;
            continue;
        }
        if (row.empty()) continue;
        std::vector<std::string> elems = py_split(row, ',');
        elems.pop_back();  // row.split(',')[:-1]

        if (gemm) {
            if (elems.size() <= 3) throw SimError("There should be at least 4 entries per row");
            const std::string name = py_strip(elems[0]);
            if (elems.size() < 5) elems.push_back("1:1");
            const auto sp = split_sparsity(elems[4]);
            const std::string m = py_strip(elems[1]), n = py_strip(elems[2]), k = py_strip(elems[3]);
            // [name, m, k, 1, k, 1, n, 1, 1, N, M]
            layers.push_back(make_layer(name, {name, m, k, "1", k, "1", n, "1", "1", sp[0], sp[1]}));
        } else {
            if (elems.size() < 8) throw SimError("topology: too few fields in row: " + row);
            // elems = elems[0:8] + [elems[7]] + elems[8:]
            elems.insert(elems.begin() + 8, elems[7]);
            if (elems.size() < 10) elems.push_back("1:1");
            const std::string sp_field = elems.back();
            elems.pop_back();
            const auto sp = split_sparsity(sp_field);
            elems.push_back(sp[0]);
            elems.push_back(sp[1]);

            const std::string base_name = py_strip(elems[0]);
            if (base_name.find("DP") != std::string::npos) {
                const int64_t nch = py_int(elems[5], "channels");
                for (int64_t d = 0; d < nch; ++d) {
                    elems[5] = "1";
                    layers.push_back(make_layer(base_name + "Channel_" + std::to_string(d), elems));
                }
            } else {
                layers.push_back(make_layer(base_name, elems));
            }
        }
    }
    return layers;
}

}  // namespace scalesim
