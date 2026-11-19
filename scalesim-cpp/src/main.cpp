// scalesim_cpp: C++ port of SCALE-Sim (scalesim/scale.py + simulator.py).
//
// Command line is compatible with scalesim/scale.py:
//   scalesim_cpp -c <config> -t <topology> [-l <layout>] [-p <log dir>] [-i conv|gemm] [-s Y|N]
// Extra options:
//   -j <threads>   simulate up to <threads> layers in parallel (default: hardware threads)
//   -q             quiet (no per-layer summary)
//   --no-dedup     simulate every layer even if an identical layer was already simulated
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.hpp"
#include "layer_sim.hpp"
#include "pyfmt.hpp"
#include "topology.hpp"

using namespace scalesim;

namespace {

struct Args {
    std::string topology = "./topologies/conv_nets/test.csv";
    std::string layout = "./layouts/conv_nets/test.csv";
    std::string config = "./configs/scale.cfg";
    std::string logdir = "./results/";
    std::string input_type = "conv";
    std::string save_trace = "Y";
    int threads = 0;
    bool quiet = false;
    bool dedup = true;
};

[[noreturn]] void usage(const char* prog, int code) {
    std::fprintf(code ? stderr : stdout,
                 "usage: %s [-t topology.csv] [-l layout.csv] [-c config.cfg] [-p log_dir]\n"
                 "          [-i conv|gemm] [-s Y|N] [-j threads] [-q] [--no-dedup]\n",
                 prog);
    std::exit(code);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], 2);
            return argv[++i];
        };
        if (k == "-t") a.topology = need();
        else if (k == "-l") a.layout = need();
        else if (k == "-c") a.config = need();
        else if (k == "-p") a.logdir = need();
        else if (k == "-i") a.input_type = need();
        else if (k == "-s") a.save_trace = need();
        else if (k == "-j") a.threads = std::stoi(need());
        else if (k == "-q") a.quiet = true;
        else if (k == "--no-dedup") a.dedup = false;
        else if (k == "-h" || k == "--help") usage(argv[0], 0);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", k.c_str());
            usage(argv[0], 2);
        }
    }
    return a;
}

// scalesim/linear_model/tpu.py
double time_linear_model(const std::string& model, int64_t cycles, int64_t s_row, int64_t s_col, int64_t t_time) {
    const double c = static_cast<double>(cycles);
    const bool small = s_row <= 128 && s_col <= 128 && t_time <= 128;
    const bool mid = s_row <= 1024 && s_col <= 1024 && t_time <= 1024;
    if (model == "TPUv4") {
        if (small) return 0.002762 * c - 0.062665;
        if (mid) return 0.000388 * c + 2.05942;
        return 0.000202 * c + 29.7217;
    }
    if (model == "TPUv5e") {
        if (small) return 0.002133 * c - 0.168796;
        if (mid) return 0.000167 * c + 1.158923;
        return 0.000159 * c - 0.380696;
    }
    // TPUv6e
    if (small) return 0.001389 * c + 0.604798;
    if (mid) return 0.000068 * c + 1.546793;
    return 0.000040 * c + 4.384712;
}

std::string join_items(const std::vector<std::string>& items) {
    std::string s;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) s += ", ";
        s += items[i];
    }
    return s;
}

std::string sram_cycle(const LayerResult& r, int m, bool stop) {
    if (!r.sram_found[m]) return "0";
    return pyfmt::intfloat_str(stop ? r.sram_stop[m] : r.sram_start[m]);
}

void write_reports(const std::string& dir, const Config& cfg, const std::vector<Layer>& layers,
                   const std::vector<LayerResult>& res) {
    using pyfmt::float_repr;
    using pyfmt::int_str;
    using pyfmt::intfloat_str;
    std::ofstream comp(dir + "/COMPUTE_REPORT.csv", std::ios::binary);
    std::ofstream time(dir + "/TIME_REPORT.csv", std::ios::binary);
    std::ofstream bw(dir + "/BANDWIDTH_REPORT.csv", std::ios::binary);
    std::ofstream det(dir + "/DETAILED_ACCESS_REPORT.csv", std::ios::binary);
    if (!comp || !time || !bw || !det) throw SimError("cannot write reports into " + dir);

    comp << "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, "
            "Mapping Efficiency %, Compute Util %,\n";
    time << "LayerID, Time (us),\n";
    bw << "LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, "
          "Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n";
    det << "LayerID, "
           "SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, "
           "SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, "
           "SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, "
           "DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, "
           "DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, "
           "DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n";

    for (size_t lid = 0; lid < res.size(); ++lid) {
        const LayerResult& r = res[lid];
        const std::string id = std::to_string(lid) + ", ";
        comp << id
             << join_items({int_str(r.overall_cycles), int_str(r.total_cycles), int_str(r.stall_cycles),
                            float_repr(r.overall_util), float_repr(r.mapping_eff), float_repr(r.compute_util)})
             << ",\n";

        const std::string& model = cfg.time_linear_model;
        std::string t;
        if (model == "TPUv4" || model == "TPUv5e" || model == "TPUv6e") {
            int64_t sr, sc, tt;
            layers[lid].spatiotemporal(cfg.dataflow, sr, sc, tt);
            t = float_repr(time_linear_model(model, r.total_cycles, sr, sc, tt));
        } else {
            t = int_str(r.total_cycles);
        }
        time << id << t << ",\n";

        bw << id
           << join_items({float_repr(r.avg_ifmap_sram_bw), float_repr(r.avg_filter_sram_bw),
                          float_repr(r.avg_ofmap_sram_bw), float_repr(r.avg_ifmap_dram_bw),
                          float_repr(r.avg_filter_dram_bw), float_repr(r.avg_ofmap_dram_bw)})
           << ",\n";

        std::vector<std::string> d;
        const int64_t sram_n[3] = {r.ifmap_sram_reads, r.filter_sram_reads, r.ofmap_sram_writes};
        for (int m = 0; m < 3; ++m) {
            d.push_back(sram_cycle(r, m, false));
            d.push_back(sram_cycle(r, m, true));
            d.push_back(int_str(sram_n[m]));
        }
        for (int m = 0; m < 3; ++m) {
            d.push_back(intfloat_str(r.dram_start[m]));
            d.push_back(intfloat_str(r.dram_stop[m]));
            d.push_back(int_str(r.dram_access[m]));
        }
        det << id << join_items(d) << ",\n";
    }
}

std::string layer_summary(int lid, const LayerResult& r, bool reused) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "\nRunning Layer %d%s\n"
                  "Total cycles: %lld\n"
                  "Compute cycles: %lld\n"
                  "Stall cycles: %lld\n"
                  "Overall utilization: %.2f%%\n"
                  "Mapping efficiency: %.2f%%\n"
                  "Average IFMAP SRAM BW: %.3f words/cycle\n"
                  "Average Filter SRAM BW: %.3f words/cycle\n"
                  "Average OFMAP SRAM BW: %.3f words/cycle\n"
                  "Average IFMAP DRAM BW: %.3f words/cycle\n"
                  "Average Filter DRAM BW: %.3f words/cycle\n"
                  "Average OFMAP DRAM BW: %.3f words/cycle\n"
                  "Layer time: %.3f s\n",
                  lid, reused ? " (identical to an earlier layer, result reused)" : "",
                  static_cast<long long>(r.overall_cycles), static_cast<long long>(r.total_cycles),
                  static_cast<long long>(r.stall_cycles), r.overall_util, r.mapping_eff, r.avg_ifmap_sram_bw,
                  r.avg_filter_sram_bw, r.avg_ofmap_sram_bw, r.avg_ifmap_dram_bw, r.avg_filter_dram_bw,
                  r.avg_ofmap_dram_bw, r.seconds);
    return buf;
}

void print_banner(const Config& cfg, const Args& a) {
    const char* df = cfg.dataflow == Dataflow::WS ? "Weight Stationary"
                     : cfg.dataflow == Dataflow::IS ? "Input Stationary"
                                                    : "Output Stationary";
    std::printf("====================================================\n");
    std::printf("******************* SCALE SIM (C++) ****************\n");
    std::printf("====================================================\n");
    std::printf("Array Size: \t%lldx%lld\n", static_cast<long long>(cfg.array_rows),
                static_cast<long long>(cfg.array_cols));
    std::printf("SRAM IFMAP (kB): \t%lld\n", static_cast<long long>(cfg.ifmap_sz_kb));
    std::printf("SRAM Filter (kB): \t%lld\n", static_cast<long long>(cfg.filter_sz_kb));
    std::printf("SRAM OFMAP (kB): \t%lld\n", static_cast<long long>(cfg.ofmap_sz_kb));
    std::printf("Dataflow: \t%s\n", df);
    std::printf("topology file path: \t%s\n", a.topology.c_str());
    if (cfg.use_user_bandwidth) {
        std::string bws;
        for (size_t i = 0; i < cfg.bandwidths.size(); ++i) bws += (i ? "," : "") + std::to_string(cfg.bandwidths[i]);
        std::printf("Bandwidth: \t%s\nWorking in USER BANDWIDTH mode.\n", bws.c_str());
    } else {
        std::printf("Working in ESTIMATE BANDWIDTH mode.\n");
    }
    std::printf("====================================================\n");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    try {
        if (!std::filesystem::exists(args.topology)) throw SimError("Topology file not found: " + args.topology);
        if (!std::filesystem::exists(args.config)) throw SimError("Config file not found: " + args.config);
        const Config cfg = read_config(args.config);
        const std::vector<Layer> layers = load_topology(args.topology, args.input_type == "gemm");
        if (layers.empty()) throw SimError("topology has no layers");
        const bool save_trace = args.save_trace == "Y";
        const bool verbose = !args.quiet;
        if (verbose) print_banner(cfg, args);

        const std::string run_dir = args.logdir + "/" + cfg.run_name;
        std::filesystem::create_directories(run_dir);

        // Identical layers produce identical reports; without traces we only simulate one of them.
        std::vector<int> source(layers.size());
        std::vector<int> todo;
        std::map<std::string, int> seen;
        for (size_t i = 0; i < layers.size(); ++i) {
            source[i] = static_cast<int>(i);
            if (args.dedup && !save_trace) {
                auto it = seen.find(layers[i].shape_key());
                if (it != seen.end()) {
                    source[i] = it->second;
                    continue;
                }
                seen.emplace(layers[i].shape_key(), static_cast<int>(i));
            }
            todo.push_back(static_cast<int>(i));
        }

        int nthreads = args.threads > 0 ? args.threads : static_cast<int>(std::thread::hardware_concurrency());
        nthreads = std::max(1, std::min<int>(nthreads, static_cast<int>(todo.size())));

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<LayerResult> results(layers.size());
        std::vector<char> done(layers.size(), 0);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::string error;
        std::mutex mu;
        std::condition_variable cv;

        auto worker = [&]() {
            while (!failed) {
                const size_t k = next.fetch_add(1);
                if (k >= todo.size()) return;
                const int lid = todo[k];
                try {
                    LayerResult r = simulate_layer(cfg, layers[static_cast<size_t>(lid)], lid, run_dir, save_trace);
                    std::lock_guard<std::mutex> lk(mu);
                    results[static_cast<size_t>(lid)] = r;
                    done[static_cast<size_t>(lid)] = 1;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(mu);
                    if (!failed) error = "layer " + std::to_string(lid) + ": " + e.what();
                    failed = true;
                }
                cv.notify_all();
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);

        // Print per-layer summaries in layer order as results become available.
        for (size_t lid = 0; lid < layers.size(); ++lid) {
            const size_t src = static_cast<size_t>(source[lid]);
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return done[src] || failed; });
            if (failed) break;
            if (src != lid) {
                results[lid] = results[src];
                results[lid].seconds = 0;
            }
            if (verbose) {
                std::fputs(layer_summary(static_cast<int>(lid), results[lid], src != lid).c_str(), stdout);
                std::fflush(stdout);
            }
        }
        for (auto& th : pool) th.join();
        if (failed) throw SimError(error);

        write_reports(run_dir, cfg, layers, results);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (verbose) {
            std::printf("\nSimulated %zu layers (%zu unique) with %d thread(s) in %.3f s\n", layers.size(), todo.size(),
                        nthreads, secs);
            std::printf("************ SCALE SIM Run Complete ****************\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
