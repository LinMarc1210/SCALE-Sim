// Reads commands from stdin and prints results, so a Python script can compare them with CPython:
//   F <hex-double>          -> float_repr(value)
//   S <n> <k1> ... <kn>     -> CPython set iteration order after inserting k1..kn
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "pyfmt.hpp"
#include "pyset.hpp"

int main() {
    std::string cmd;
    while (std::cin >> cmd) {
        if (cmd == "F") {
            std::string hex;
            std::cin >> hex;
            const uint64_t bits = std::stoull(hex, nullptr, 16);
            double v;
            std::memcpy(&v, &bits, sizeof v);
            std::cout << scalesim::pyfmt::float_repr(v) << '\n';
        } else if (cmd == "S") {
            size_t n;
            std::cin >> n;
            scalesim::PySetOrder s;
            for (size_t i = 0; i < n; ++i) {
                long long k;
                std::cin >> k;
                s.add(k);
            }
            bool first = true;
            s.for_each([&](int64_t k) {
                if (!first) std::cout << ' ';
                std::cout << k;
                first = false;
            });
            std::cout << '\n';
        }
    }
    return 0;
}
