// Shared helpers for the C++ port of SCALE-Sim.
//
// Every arithmetic helper here reproduces the exact Python semantics used by the reference
// implementation (true division into an IEEE double, then math.ceil / math.floor), so the C++
// simulator produces bit-identical numbers.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace scalesim {

enum class Dataflow { OS, WS, IS };

struct SimError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// math.ceil(a / b) as evaluated by CPython for ints that fit in a double.
inline int64_t py_ceil_div(int64_t a, int64_t b) {
    if (b == 0) throw SimError("ZeroDivisionError in ceil division");
    return static_cast<int64_t>(std::ceil(static_cast<double>(a) / static_cast<double>(b)));
}

// math.floor(a / b) as evaluated by CPython for ints that fit in a double.
inline int64_t py_floor_div(int64_t a, int64_t b) {
    if (b == 0) throw SimError("ZeroDivisionError in floor division");
    return static_cast<int64_t>(std::floor(static_cast<double>(a) / static_cast<double>(b)));
}

// Python int / int true division (correctly rounded for |a|, |b| < 2**53).
inline double py_truediv(int64_t a, int64_t b) {
    if (b == 0) throw SimError("ZeroDivisionError in true division");
    return static_cast<double>(a) / static_cast<double>(b);
}

// int(math.ceil(n * frac)) used for the active-buffer split.
inline int64_t py_ceil_mul(int64_t n, double frac) {
    return static_cast<int64_t>(std::ceil(static_cast<double>(n) * frac));
}

}  // namespace scalesim
