// Buffered CSV writer producing np.savetxt-compatible text.
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "common.hpp"
#include "pyfmt.hpp"

namespace scalesim {

class CsvWriter {
public:
    CsvWriter() = default;
    CsvWriter(const CsvWriter&) = delete;
    CsvWriter& operator=(const CsvWriter&) = delete;
    ~CsvWriter() { close(); }

    void open(const std::string& path) {
        close();
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) throw SimError("cannot open for writing: " + path);
        buf_.resize(kBufSize);
        pos_ = 0;
    }

    bool is_open() const { return f_ != nullptr; }

    void close() {
        if (!f_) return;
        flush();
        std::fclose(f_);
        f_ = nullptr;
    }

    // '%i' formatting of an integer value.
    void put_int(int64_t v) {
        reserve(24);
        auto r = std::to_chars(buf_.data() + pos_, buf_.data() + pos_ + 24, v);
        pos_ = static_cast<size_t>(r.ptr - buf_.data());
    }

    // '%s' formatting of an integer-valued np.float64 (e.g. "-1.0", "10000384.0").
    void put_intfloat(int64_t v) {
        if (v > -10000000000000000LL && v < 10000000000000000LL) {
            put_int(v);
            put_raw(".0", 2);
        } else {
            const std::string s = pyfmt::float_repr(static_cast<double>(v));
            put_raw(s.data(), s.size());
        }
    }

    void put_char(char ch) {
        reserve(1);
        buf_[pos_++] = ch;
    }

    void put_raw(const char* s, size_t n) {
        reserve(n);
        std::copy(s, s + n, buf_.data() + pos_);
        pos_ += n;
    }

private:
    static constexpr size_t kBufSize = size_t{1} << 22;

    void reserve(size_t n) {
        if (pos_ + n > buf_.size()) {
            flush();
            if (n > buf_.size()) buf_.resize(n);
        }
    }

    void flush() {
        if (f_ && pos_ > 0) {
            if (std::fwrite(buf_.data(), 1, pos_, f_) != pos_) throw SimError("write failed");
        }
        pos_ = 0;
    }

    std::FILE* f_ = nullptr;
    std::vector<char> buf_;
    size_t pos_ = 0;
};

}  // namespace scalesim
