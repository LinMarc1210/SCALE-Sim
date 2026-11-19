// Formatting that matches Python's str()/repr() output byte for byte.
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace scalesim::pyfmt {

// repr(float) in CPython 3 (also what str(np.float64) prints): shortest round-trip digits,
// positional notation for 1e-4 <= |v| < 1e16, otherwise "d.ddde+XX".
inline std::string float_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";

    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string s(buf, res.ptr);

    bool neg = false;
    size_t p = 0;
    if (s[0] == '-') {
        neg = true;
        p = 1;
    }
    const size_t epos = s.find('e');
    std::string digits;
    for (size_t i = p; i < epos; ++i)
        if (s[i] != '.') digits += s[i];
    const int exp10 = std::atoi(s.c_str() + epos + 1);
    const int nd = static_cast<int>(digits.size());
    const int decpt = exp10 + 1;  // value = 0.DIGITS * 10**decpt

    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (nd > 1) {
            out += '.';
            out.append(digits, 1, std::string::npos);
        }
        const int e = decpt - 1;
        out += 'e';
        out += (e < 0 ? '-' : '+');
        const int ae = std::abs(e);
        if (ae < 10) out += '0';
        out += std::to_string(ae);
    } else if (decpt <= 0) {
        out += "0.";
        out.append(static_cast<size_t>(-decpt), '0');
        out += digits;
    } else if (decpt >= nd) {
        out += digits;
        out.append(static_cast<size_t>(decpt - nd), '0');
        out += ".0";
    } else {
        out.append(digits, 0, static_cast<size_t>(decpt));
        out += '.';
        out.append(digits, static_cast<size_t>(decpt), std::string::npos);
    }
    return out;
}

// str(int)
inline std::string int_str(int64_t v) { return std::to_string(v); }

// str(float(v)) for an integer-valued float, e.g. 33 -> "33.0".
inline std::string intfloat_str(int64_t v) { return float_repr(static_cast<double>(v)); }

}  // namespace scalesim::pyfmt
