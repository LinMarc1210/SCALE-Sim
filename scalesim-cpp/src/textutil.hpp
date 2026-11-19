// Small string helpers with Python semantics (str.strip, int()).
#pragma once

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.hpp"

namespace scalesim {

inline bool py_isspace(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\v' || ch == '\f';
}

inline std::string py_strip(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && py_isspace(s[b])) ++b;
    while (e > b && py_isspace(s[e - 1])) --e;
    return s.substr(b, e - b);
}

inline std::string py_lower(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// int(s) for a decimal string (surrounding whitespace allowed).
inline int64_t py_int(const std::string& raw, const std::string& what) {
    std::string s = py_strip(raw);
    std::string cleaned;
    for (char ch : s)
        if (ch != '_') cleaned += ch;
    if (cleaned.empty()) throw SimError("invalid integer for " + what + ": '" + raw + "'");
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(cleaned.c_str(), &end, 10);
    if (errno != 0 || end == cleaned.c_str() || *end != '\0')
        throw SimError("invalid integer for " + what + ": '" + raw + "'");
    return static_cast<int64_t>(v);
}

// str.split(sep) (no maxsplit), keeps empty fields like Python.
inline std::vector<std::string> py_split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

}  // namespace scalesim
