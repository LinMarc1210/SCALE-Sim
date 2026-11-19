#include "layer_sim.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

#include "csv_writer.hpp"
#include "geometry.hpp"
#include "memory.hpp"

namespace scalesim {
namespace {

struct ComputeStats {
    double avg_mapping_eff = 0, avg_compute_util = 0;
    int64_t ifmap_reads = 0, filter_reads = 0, ofmap_writes = 0;
};

// Per-fold bookkeeping of systolic_compute_{os,ws,is}.create_*_demand_mat(): mapping efficiency,
// compute utilisation (averaged with Python's sequential float sum, folds in fc-major order) and
// SRAM request counters.
ComputeStats compute_stats(const Geometry& g, const Operands& ops) {
    ComputeStats s;
    const int64_t R = g.R(), C = g.C(), T = g.T();
    const int64_t RC = R * C;
    double map_sum = 0, util_sum = 0;
    int64_t n = 0;
    for (int64_t fc = 0; fc < g.col_fold(); ++fc) {
        for (int64_t fr = 0; fr < g.row_fold(); ++fr) {
            const Fold f = g.fold(fc, fr);
            const int64_t mac_used = f.nr * f.nc;
            double map_eff, util;
            switch (g.df()) {
                case Dataflow::OS:
                    s.ifmap_reads += T * f.nr;
                    s.filter_reads += T * f.nc;
                    s.ofmap_writes += f.nr * f.nc + R + C;  // "+= shape[0] + shape[1]" in the reference
                    map_eff = py_truediv(mac_used, RC);
                    util = py_truediv(mac_used * T, RC * (T + R + C - 2));
                    break;
                case Dataflow::WS: {
                    s.ifmap_reads += T * f.nr;
                    s.filter_reads += f.nr * f.nc;
                    s.ofmap_writes += T * f.nc;
                    // sum_sparse = number of -1 entries in the padded R x C filter block
                    int64_t valid = mac_used;
                    if (ops.filter_offset() < 0) {
                        valid = 0;
                        for (int64_t a = 0; a < f.nr; ++a)
                            for (int64_t c = 0; c < f.nc; ++c)
                                if (ops.F(f.rs + a, f.cs + c) != -1) ++valid;
                    }
                    const int64_t sum_sparse = RC - valid;
                    map_eff = py_truediv(RC - sum_sparse, RC);
                    util = py_truediv(mac_used * T, RC * (2 * R + 2 * C + T - 3));
                    break;
                }
                case Dataflow::IS:
                default:
                    s.ifmap_reads += f.nr * f.nc;
                    s.filter_reads += T * f.nr;
                    s.ofmap_writes += T * f.nc;
                    map_eff = py_truediv(mac_used, RC);
                    util = py_truediv(mac_used * T, RC * (2 * R + 2 * C + T - 3));
                    break;
            }
            map_sum += map_eff;
            util_sum += util;
            ++n;
        }
    }
    s.avg_mapping_eff = map_sum / static_cast<double>(n);
    s.avg_compute_util = util_sum / static_cast<double>(n);
    return s;
}

void write_sram_row(CsvWriter& w, int64_t cycle, std::vector<int64_t>& row) {
    w.put_int(cycle);
    for (int64_t& v : row) {
        w.put_char(',');
        w.put_int(v);
        v = -1;
    }
    w.put_char('\n');
}

}  // namespace

LayerResult simulate_layer(const Config& cfg, const Layer& layer, int layer_id, const std::string& run_dir,
                           bool save_trace) {
    const auto t0 = std::chrono::steady_clock::now();
    LayerResult res;

    const Operands ops(layer, cfg);
    const Geometry g(cfg.dataflow, cfg.array_rows, cfg.array_cols, ops);
    const ComputeStats cs = compute_stats(g, ops);

    // ---------------- memory system setup (single_layer_sim.run, step 2.1) ----------------
    const int64_t ifmap_bytes = 1024 * cfg.ifmap_sz_kb;
    const int64_t filter_bytes = 1024 * cfg.filter_sz_kb;
    const int64_t ofmap_bytes = 1024 * cfg.ofmap_sz_kb;
    const bool user_bw = cfg.use_user_bandwidth;

    std::string layer_dir;
    CsvWriter sram_w[3], ofmap_dram_w;
    if (save_trace) {
        layer_dir = run_dir + "/layer" + std::to_string(layer_id);
        std::filesystem::create_directories(layer_dir);
        sram_w[kIfmap].open(layer_dir + "/IFMAP_SRAM_TRACE.csv");
        sram_w[kFilter].open(layer_dir + "/FILTER_SRAM_TRACE.csv");
        sram_w[kOfmap].open(layer_dir + "/OFMAP_SRAM_TRACE.csv");
        ofmap_dram_w.open(layer_dir + "/OFMAP_DRAM_TRACE.csv");
    }

    std::unique_ptr<UserReadBuffer> user_rd[2];
    std::unique_ptr<EstReadBuffer> est_rd[2];
    const int64_t addr_base[2] = {ops.ifmap_offset(), ops.filter_offset()};
    const int64_t addr_range[2] = {ops.ifmap_range(), ops.filter_range()};
    if (user_bw) {
        if (cfg.bandwidths.empty()) throw SimError("USER bandwidth mode needs 'Bandwidth'");
        const int64_t bytes[2] = {ifmap_bytes, filter_bytes};
        const int64_t bw[2] = {cfg.ifmap_sram_bank_bandwidth, cfg.filter_sram_bank_bandwidth};
        const int64_t banks[2] = {cfg.ifmap_sram_bank_num, cfg.filter_sram_bank_num};
        for (int m = 0; m < 2; ++m) {
            user_rd[m] = std::make_unique<UserReadBuffer>();
            user_rd[m]->set_params(bytes[m], bw[m], banks[m]);
        }
        // 2.2 Install the prefetch matrices (set_read_buf_prefetch_matrices)
        for (int m = 0; m < 2; ++m) {
            UserReadBuffer& b = *user_rd[m];
            b.begin_prefetch_index(addr_base[m], addr_range[m]);
            g.for_each_prefetch(static_cast<Mat>(m), [&](int64_t e) { b.push_prefetch_elem(e); });
            b.end_prefetch_index();
        }
    } else {
        const int64_t bytes[2] = {ifmap_bytes, filter_bytes};
        const char* names[2] = {"IFMAP", "FILTER"};
        for (int m = 0; m < 2; ++m) {
            est_rd[m] = std::make_unique<EstReadBuffer>();
            std::string tmp;
            if (save_trace) tmp = layer_dir + "/." + names[m] + "_DRAM_TRACE.tmp";
            est_rd[m]->set_params(bytes[m], 10, addr_base[m], addr_range[m], tmp);
        }
    }
    WriteBuffer wr;
    wr.set_params(ofmap_bytes, user_bw ? cfg.bandwidths[0] : cfg.array_cols, save_trace ? &ofmap_dram_w : nullptr);

    // ---------------- service_memory_requests ----------------
    std::vector<int64_t> valid[3];
    std::vector<int64_t> rowbuf[3];
    for (int m = 0; m < 3; ++m) {
        valid[m].reserve(static_cast<size_t>(g.width(static_cast<Mat>(m))));
        if (save_trace) rowbuf[m].assign(static_cast<size_t>(g.width(static_cast<Mat>(m))), -1);
    }

    int64_t stall = 0, row_idx = 0;
    int64_t max_ofmap_out = 0, last_ofmap_out = 0;
    bool any_row = false;
    for (int64_t fc = 0; fc < g.col_fold(); ++fc) {
        for (int64_t fr = 0; fr < g.row_fold(); ++fr) {
            const Fold f = g.fold(fc, fr);
            for (int64_t r = 0; r < g.rows_per_fold(); ++r, ++row_idx) {
                for (int m = 0; m < 3; ++m) {
                    std::vector<int64_t>& v = valid[m];
                    v.clear();
                    if (save_trace) {
                        std::vector<int64_t>& rb = rowbuf[m];
                        g.row(static_cast<Mat>(m), f, r, [&](int64_t c, int64_t a) {
                            rb[static_cast<size_t>(c)] = a;
                            if (a != -1) v.push_back(a);
                        });
                    } else {
                        g.row(static_cast<Mat>(m), f, r, [&](int64_t, int64_t a) {
                            if (a != -1) v.push_back(a);
                        });
                    }
                }

                const int64_t cycle = row_idx + stall;
                int64_t out[3];
                for (int m = 0; m < 2; ++m)
                    out[m] = user_bw ? user_rd[m]->service(cycle, valid[m].data(), valid[m].size())
                                     : est_rd[m]->service(cycle, valid[m].data(), valid[m].size());
                out[kOfmap] = wr.service(cycle, valid[kOfmap].data(), valid[kOfmap].size());

                stall += std::max({out[kIfmap] - cycle - 1, out[kFilter] - cycle - 1, out[kOfmap] - cycle});

                for (int m = 0; m < 3; ++m) {
                    if (!valid[m].empty()) {
                        if (!res.sram_found[m]) {
                            res.sram_found[m] = true;
                            res.sram_start[m] = out[m];
                        }
                        res.sram_stop[m] = out[m];
                    }
                    if (save_trace) write_sram_row(sram_w[m], out[m], rowbuf[m]);
                }
                if (!any_row || out[kOfmap] > max_ofmap_out) max_ofmap_out = out[kOfmap];
                any_row = true;
                last_ofmap_out = out[kOfmap];
            }
        }
    }
    if (!any_row) throw SimError("layer has an empty demand matrix");

    if (!user_bw) {
        est_rd[kIfmap]->complete_all_prefetches();
        est_rd[kFilter]->complete_all_prefetches();
    }
    wr.empty_all_buffers(last_ofmap_out);

    // ---------------- traces ----------------
    if (save_trace) {
        for (auto& w : sram_w) w.close();
        ofmap_dram_w.close();
        const char* names[2] = {"/IFMAP_DRAM_TRACE.csv", "/FILTER_DRAM_TRACE.csv"};
        for (int m = 0; m < 2; ++m) {
            if (user_bw) {
                // read_buffer.trace_matrix is never filled in the reference (np.ones((1, 1)))
                CsvWriter w;
                w.open(layer_dir + names[m]);
                w.put_raw("1.0\n", 4);
                w.close();
            } else {
                est_rd[m]->write_trace(layer_dir + names[m]);
            }
        }
        if (!wr.trace_valid()) std::filesystem::remove(layer_dir + "/OFMAP_DRAM_TRACE.csv");
    }

    // ---------------- report items (single_layer_sim.calc_report_data) ----------------
    res.total_cycles = max_ofmap_out;
    res.stall_cycles = stall;
    const int64_t num_compute = layer.num_ofmap_px() * layer.window;
    const int64_t num_mac_unit = cfg.array_rows * cfg.array_cols;
    res.overall_util = py_truediv(num_compute * 100, res.total_cycles * num_mac_unit);
    res.mapping_eff = cs.avg_mapping_eff * 100;
    res.compute_util = cs.avg_compute_util * 100;

    res.ifmap_sram_reads = cs.ifmap_reads;
    res.filter_sram_reads = cs.filter_reads;
    res.ofmap_sram_writes = cs.ofmap_writes;
    res.avg_ifmap_sram_bw = py_truediv(cs.ifmap_reads, res.total_cycles);
    res.avg_filter_sram_bw = py_truediv(cs.filter_reads, res.total_cycles);
    res.avg_ofmap_sram_bw = py_truediv(cs.ofmap_writes, res.total_cycles);

    for (int m = 0; m < 2; ++m) {
        if (user_bw) {
            res.dram_start[m] = 1;  // np.amin / np.amax over np.ones((1, 1))
            res.dram_stop[m] = 1;
            res.dram_access[m] = user_rd[m]->num_access();
        } else {
            if (!est_rd[m]->has_rows()) throw SimError("IndexError: empty estimate-bandwidth DRAM trace");
            res.dram_start[m] = est_rd[m]->first_row_cycle();
            res.dram_stop[m] = est_rd[m]->last_row_cycle();
            res.dram_access[m] = est_rd[m]->num_access();
        }
    }
    if (!wr.trace_valid()) throw SimError("OFMAP buffer was never drained (Python asserts here)");
    res.dram_start[kOfmap] = wr.min_cycle();
    res.dram_stop[kOfmap] = wr.max_cycle();
    res.dram_access[kOfmap] = wr.num_access();

    res.overall_cycles = res.dram_stop[kOfmap] - std::min(res.dram_start[kIfmap], res.dram_start[kFilter]);
    double* dram_bw[3] = {&res.avg_ifmap_dram_bw, &res.avg_filter_dram_bw, &res.avg_ofmap_dram_bw};
    for (int m = 0; m < 3; ++m)
        *dram_bw[m] = static_cast<double>(res.dram_access[m]) /
                      static_cast<double>(res.dram_stop[m] - res.dram_start[m] + 1);

    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace scalesim
