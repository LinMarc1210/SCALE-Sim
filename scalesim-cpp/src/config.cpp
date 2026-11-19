#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>

#include "textutil.hpp"

namespace scalesim {
namespace {

// Minimal configparser.ConfigParser equivalent: section names are case sensitive, option names
// are lower-cased, "key: value" and "key = value" are both accepted, full-line comments start with
// '#' or ';', and indented lines continue the previous value.
class Ini {
public:
    explicit Ini(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw SimError("cannot open config file: " + path);
        std::string line, section, last_key;
        bool have_key = false;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const std::string s = py_strip(line);
            if (s.empty() || s[0] == '#' || s[0] == ';') continue;
            const bool indented = !line.empty() && (line[0] == ' ' || line[0] == '\t');
            if (indented && have_key && !section.empty()) {
                data_[section][last_key] += "\n" + s;
                continue;
            }
            if (s.front() == '[' && s.back() == ']' && s.size() >= 3) {
                section = s.substr(1, s.size() - 2);
                data_[section];
                have_key = false;
                continue;
            }
            const size_t pos = s.find_first_of(":=");
            if (pos == std::string::npos || section.empty())
                throw SimError("config parse error in " + path + ": " + line);
            last_key = py_lower(py_strip(s.substr(0, pos)));
            data_[section][last_key] = py_strip(s.substr(pos + 1));
            have_key = true;
        }
    }

    bool has_section(const std::string& sec) const { return data_.count(sec) != 0; }

    bool has_option(const std::string& sec, const std::string& key) const {
        auto it = data_.find(sec);
        return it != data_.end() && it->second.count(py_lower(key)) != 0;
    }

    const std::string& get(const std::string& sec, const std::string& key) const {
        auto it = data_.find(sec);
        if (it == data_.end()) throw SimError("config: missing section [" + sec + "]");
        auto jt = it->second.find(py_lower(key));
        if (jt == it->second.end())
            throw SimError("config: missing option '" + key + "' in section [" + sec + "]");
        return jt->second;
    }

    int64_t get_int(const std::string& sec, const std::string& key) const {
        return py_int(get(sec, key), sec + "." + key);
    }

    bool get_bool(const std::string& sec, const std::string& key) const {
        const std::string v = py_lower(get(sec, key));
        if (v == "1" || v == "yes" || v == "true" || v == "on") return true;
        if (v == "0" || v == "no" || v == "false" || v == "off") return false;
        throw SimError("config: not a boolean: " + sec + "." + key + " = " + v);
    }

private:
    std::map<std::string, std::map<std::string, std::string>> data_;
};

}  // namespace

Config read_config(const std::string& path) {
    Ini ini(path);
    Config c;

    c.run_name = ini.get("general", "run_name");

    const std::string bw_mode = ini.get("run_presets", "InterfaceBandwidth");
    if (bw_mode == "USER")
        c.use_user_bandwidth = true;
    else if (bw_mode == "CALC")
        c.use_user_bandwidth = false;
    else
        throw SimError("Use either USER or CALC in InterfaceBandwidth field");

    if (ini.has_option("run_presets", "UseRamulatorTrace"))
        c.use_ramulator_trace = (ini.get("run_presets", "UseRamulatorTrace") == "True");
    if (ini.has_option("run_presets", "TimeLinearModel")) {
        c.time_linear_model = ini.get("run_presets", "TimeLinearModel");
        const auto& m = c.time_linear_model;
        if (m != "None" && m != "TPUv4" && m != "TPUv5e" && m != "TPUv6e")
            throw SimError("Invalid time linear model '" + m + "'");
    }

    const std::string arch = "architecture_presets";
    c.array_rows = ini.get_int(arch, "ArrayHeight");
    c.array_cols = ini.get_int(arch, "ArrayWidth");
    c.ifmap_sz_kb = ini.get_int(arch, "ifmapsramszkB");
    c.filter_sz_kb = ini.get_int(arch, "filtersramszkB");
    c.ofmap_sz_kb = ini.get_int(arch, "ofmapsramszkB");
    c.ifmap_offset = ini.get_int(arch, "IfmapOffset");
    c.filter_offset = ini.get_int(arch, "FilterOffset");
    c.ofmap_offset = ini.get_int(arch, "OfmapOffset");
    c.df = ini.get(arch, "Dataflow");
    if (ini.has_option(arch, "ReadRequestBuffer")) c.req_buf_sz_rd = ini.get_int(arch, "ReadRequestBuffer");
    if (ini.has_option(arch, "WriteRequestBuffer")) c.req_buf_sz_wr = ini.get_int(arch, "WriteRequestBuffer");

    const std::string lay = "layout";
    c.using_ifmap_custom_layout = ini.get_bool(lay, "IfmapCustomLayout");
    c.using_filter_custom_layout = ini.get_bool(lay, "FilterCustomLayout");
    c.ifmap_sram_bank_bandwidth = ini.get_int(lay, "IfmapSRAMBankBandwidth");
    c.ifmap_sram_bank_num = ini.get_int(lay, "IfmapSRAMBankNum");
    c.ifmap_sram_bank_port = ini.get_int(lay, "IfmapSRAMBankPort");
    c.filter_sram_bank_bandwidth = ini.get_int(lay, "FilterSRAMBankBandwidth");
    c.filter_sram_bank_num = ini.get_int(lay, "FilterSRAMBankNum");
    c.filter_sram_bank_port = ini.get_int(lay, "FilterSRAMBankPort");

    if (c.use_user_bandwidth) {
        std::stringstream ss(py_strip(ini.get(arch, "Bandwidth")));
        std::string tok;
        while (std::getline(ss, tok, ',')) c.bandwidths.push_back(py_int(tok, "Bandwidth"));
    }

    if (c.df == "os")
        c.dataflow = Dataflow::OS;
    else if (c.df == "ws")
        c.dataflow = Dataflow::WS;
    else if (c.df == "is")
        c.dataflow = Dataflow::IS;
    else
        throw SimError("Invalid dataflow '" + c.df + "'");

    if (ini.has_section("sparsity"))
        c.sparsity_support = (py_lower(ini.get("sparsity", "SparsitySupport")) == "true");

    // Features of the Python simulator that this port does not model yet.
    if (c.sparsity_support)
        throw SimError("sparsity support is not implemented in the C++ port yet (use the Python simulator)");
    if (c.using_ifmap_custom_layout || c.using_filter_custom_layout)
        throw SimError("custom layouts are not implemented in the C++ port yet (use the Python simulator)");
    if (c.use_ramulator_trace)
        throw SimError("Ramulator traces are not implemented in the C++ port yet (use the Python simulator)");

    return c;
}

}  // namespace scalesim
