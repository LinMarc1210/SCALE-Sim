// Operand address functions and per-dataflow fold geometry.
//
// The Python reference materialises the operand matrices, the prefetch matrices and the full
// demand matrices (all folds, with skew and null padding) as float64 numpy arrays. Here every
// element is computed on demand from closed-form expressions, so a layer needs O(rows + cols)
// memory instead of O(total_cycles * array_width).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "config.hpp"
#include "topology.hpp"

namespace scalesim {

// operand_matrix.calc_{ifmap,filter,ofmap}_elem_addr()
class Operands {
public:
    Operands(const Layer& l, const Config& c)
        : ifmap_offset_(c.ifmap_offset), filter_offset_(c.filter_offset), ofmap_offset_(c.ofmap_offset),
          ifmap_rows_(l.ifmap_h), ifmap_cols_(l.ifmap_w), px_(l.ofmap_px()), window_(l.window),
          num_filt_(l.num_filt) {
        const int64_t ew = l.ofmap_w;
        const int64_t ch = l.num_ch;
        i_row_.resize(static_cast<size_t>(px_));
        i_col_.resize(static_cast<size_t>(px_));
        i_base_.resize(static_cast<size_t>(px_));
        for (int64_t i = 0; i < px_; ++i) {
            const int64_t r = (i / ew) * l.stride_h;
            const int64_t cc = (i % ew) * l.stride_w;
            i_row_[i] = r;
            i_col_[i] = cc;
            i_base_[i] = (r * ifmap_cols_ + cc) * ch;
        }
        j_row_.resize(static_cast<size_t>(window_));
        j_col_.resize(static_cast<size_t>(window_));
        j_inner_.resize(static_cast<size_t>(window_));
        const int64_t row_span = l.filt_w * ch;
        for (int64_t j = 0; j < window_; ++j) {
            const int64_t cr = j / row_span;
            const int64_t k = j % row_span;
            const int64_t ccol = k / ch;
            const int64_t cch = k % ch;
            j_row_[j] = cr;
            j_col_[j] = ccol;
            j_inner_[j] = (cr * ifmap_cols_ + ccol) * ch + cch;
        }
        ifmap_range_ = l.ifmap_h * l.ifmap_w * l.num_ch;
        filter_range_ = window_ * num_filt_;
    }

    // IFMAP operand matrix element [i][j], i < ofmap_px, j < window. -1 if outside the input.
    int64_t I(int64_t i, int64_t j) const {
        if (j_row_[j] + i_row_[i] < ifmap_rows_ && j_col_[j] + i_col_[i] < ifmap_cols_)
            return j_inner_[j] + i_base_[i] + ifmap_offset_;
        return -1;
    }
    // Filter operand matrix element [i][j], i < window, j < num_filt.
    int64_t F(int64_t i, int64_t j) const { return j * window_ + i + filter_offset_; }
    // OFMAP operand matrix element [i][j], i < ofmap_px, j < num_filt.
    int64_t O(int64_t i, int64_t j) const { return num_filt_ * i + j + ofmap_offset_; }

    int64_t px() const { return px_; }
    int64_t window() const { return window_; }
    int64_t num_filt() const { return num_filt_; }
    int64_t ifmap_offset() const { return ifmap_offset_; }
    int64_t filter_offset() const { return filter_offset_; }
    int64_t ifmap_range() const { return ifmap_range_; }
    int64_t filter_range() const { return filter_range_; }

private:
    int64_t ifmap_offset_, filter_offset_, ofmap_offset_;
    int64_t ifmap_rows_, ifmap_cols_;
    int64_t px_, window_, num_filt_;
    int64_t ifmap_range_ = 0, filter_range_ = 0;
    std::vector<int64_t> i_row_, i_col_, i_base_;
    std::vector<int64_t> j_row_, j_col_, j_inner_;
};

enum Mat { kIfmap = 0, kFilter = 1, kOfmap = 2 };

struct Fold {
    int64_t rs, nr;  // first row index / number of used array rows   (over S_r)
    int64_t cs, nc;  // first col index / number of used array cols   (over S_c)
};

// Fold geometry of systolic_compute_{os,ws,is}.py.
class Geometry {
public:
    Geometry(Dataflow df, int64_t arr_r, int64_t arr_c, const Operands& ops)
        : df_(df), R_(arr_r), C_(arr_c), ops_(ops) {
        switch (df_) {
            case Dataflow::OS: Sr_ = ops.px(); Sc_ = ops.num_filt(); T_ = ops.window(); break;
            case Dataflow::WS: Sr_ = ops.window(); Sc_ = ops.num_filt(); T_ = ops.px(); break;
            case Dataflow::IS: Sr_ = ops.window(); Sc_ = ops.px(); T_ = ops.num_filt(); break;
        }
        row_fold_ = py_ceil_div(Sr_, R_);
        col_fold_ = py_ceil_div(Sc_, C_);
        rows_per_fold_ = (df_ == Dataflow::OS) ? T_ + R_ + C_ - 2 : T_ + 2 * R_ + C_ - 2;
    }

    Dataflow df() const { return df_; }
    int64_t R() const { return R_; }
    int64_t C() const { return C_; }
    int64_t Sr() const { return Sr_; }
    int64_t Sc() const { return Sc_; }
    int64_t T() const { return T_; }
    int64_t row_fold() const { return row_fold_; }
    int64_t col_fold() const { return col_fold_; }
    int64_t rows_per_fold() const { return rows_per_fold_; }
    int64_t num_folds() const { return row_fold_ * col_fold_; }

    // Width (number of columns) of each demand matrix.
    int64_t width(Mat m) const {
        if (m == kOfmap) return C_;
        if (m == kIfmap) return df_ == Dataflow::IS ? C_ : R_;
        return df_ == Dataflow::IS ? R_ : C_;
    }

    Fold fold(int64_t fc, int64_t fr) const {
        Fold f;
        f.rs = fr * R_;
        f.nr = std::min(R_, Sr_ - f.rs);
        f.cs = fc * C_;
        f.nc = std::min(C_, Sc_ - f.cs);
        return f;
    }

    // Emits (column, address) for every entry of row r (0 <= r < rows_per_fold) of the given
    // fold's demand matrix that is not part of the null padding, in increasing column order.
    // The address itself can still be -1 (out-of-bounds IFMAP pixel), exactly as in Python.
    template <class Emit>
    void row(Mat m, const Fold& f, int64_t r, Emit&& emit) const {
        const Operands& o = ops_;
        switch (df_) {
            case Dataflow::OS:
                if (m == kIfmap)
                    skew(r, 0, T_, f.nr, emit, [&](int64_t x, int64_t c) { return o.I(f.rs + c, x); });
                else if (m == kFilter)
                    skew(r, 0, T_, f.nc, emit, [&](int64_t x, int64_t c) { return o.F(x, f.cs + c); });
                else
                    skew(r, T_ - 1 + R_ - f.nr, f.nr, f.nc, emit,
                         [&](int64_t x, int64_t c) { return o.O(f.rs + f.nr - 1 - x, f.cs + c); });
                break;
            case Dataflow::WS:
                if (m == kIfmap)
                    skew(r, R_, T_, f.nr, emit, [&](int64_t x, int64_t c) { return o.I(x, f.rs + c); });
                else if (m == kFilter)
                    flat(r, f, emit, [&](int64_t b, int64_t c) { return o.F(f.rs + b, f.cs + c); });
                else
                    skew(r, 2 * R_ - 1, T_, f.nc, emit, [&](int64_t x, int64_t c) { return o.O(x, f.cs + c); });
                break;
            case Dataflow::IS:
                if (m == kIfmap)
                    flat(r, f, emit, [&](int64_t b, int64_t c) { return o.I(f.cs + c, f.rs + b); });
                else if (m == kFilter)
                    skew(r, R_, T_, f.nr, emit, [&](int64_t x, int64_t c) { return o.F(f.rs + c, x); });
                else
                    skew(r, 2 * R_ - 1, T_, f.nc, emit, [&](int64_t x, int64_t c) { return o.O(f.cs + c, x); });
                break;
        }
    }

    // ---- prefetch matrices (only needed in USER bandwidth mode) ----
    // Shape of the prefetch matrix before the anti-diagonal roll-out.
    int64_t prefetch_rows(Mat m) const {
        const bool fold_rows = prefetch_folds_rows(m);
        return (fold_rows ? row_fold_ : col_fold_) * prefetch_block(m);
    }
    int64_t prefetch_cols(Mat m) const { return prefetch_folds_rows(m) ? R_ : C_; }
    // Whether the Python code rolls the prefetch matrix out along anti-diagonals (ISSUE #15/#16).
    bool prefetch_rollout(Mat m) const {
        if (df_ == Dataflow::WS && m == kFilter) return false;
        if (df_ == Dataflow::IS && m == kIfmap) return false;
        return true;
    }

    // Visits the prefetch sequence exactly as read_buffer.set_fetch_matrix() flattens it.
    template <class Visit>
    void for_each_prefetch(Mat m, Visit&& visit) const {
        const int64_t M = prefetch_rows(m), N = prefetch_cols(m);
        if (prefetch_rollout(m)) {
            for (int64_t d = 0; d < M + N; ++d) {
                const int64_t max_r = std::min(d, M - 1);
                const int64_t min_r = std::max<int64_t>(0, d - N + 1);
                for (int64_t row = max_r; row >= min_r; --row) visit(prefetch_elem(m, row, d - row));
            }
        } else {
            for (int64_t row = 0; row < M; ++row)
                for (int64_t col = 0; col < N; ++col) visit(prefetch_elem(m, row, col));
        }
    }

private:
    // Skewed block: row r, column c holds block[x][c] with x = r - c - base, valid for
    // 0 <= x < span and c < nv.
    template <class Emit, class Addr>
    static void skew(int64_t r, int64_t base, int64_t span, int64_t nv, Emit& emit, Addr&& addr) {
        const int64_t c_lo = std::max<int64_t>(0, r - base - span + 1);
        const int64_t c_hi = std::min<int64_t>(nv - 1, r - base);
        for (int64_t c = c_lo; c <= c_hi; ++c) emit(c, addr(r - base - c, c));
    }

    // Un-skewed, vertically flipped R x C block at the top of the fold (WS filter, IS ifmap).
    template <class Emit, class Addr>
    void flat(int64_t r, const Fold& f, Emit& emit, Addr&& addr) const {
        if (r < R_ - f.nr || r >= R_) return;
        const int64_t b = R_ - 1 - r;
        for (int64_t c = 0; c < f.nc; ++c) emit(c, addr(b, c));
    }

    bool prefetch_folds_rows(Mat m) const {
        // ifmap for OS/WS and filter for IS are folded over S_r (array rows), others over S_c.
        if (df_ == Dataflow::IS) return m == kFilter;
        return m == kIfmap;
    }
    int64_t prefetch_block(Mat m) const {
        if (df_ == Dataflow::WS && m == kFilter) return Sr_;
        if (df_ == Dataflow::IS && m == kIfmap) return Sr_;
        return T_;
    }

    int64_t prefetch_elem(Mat m, int64_t row, int64_t col) const {
        const int64_t blk = prefetch_block(m);
        const int64_t f = row / blk;
        const int64_t x = row % blk;
        const Operands& o = ops_;
        if (prefetch_folds_rows(m)) {
            const int64_t start = f * R_;
            if (col >= std::min(R_, Sr_ - start)) return -1;
            if (df_ == Dataflow::OS) return o.I(start + col, x);  // OS ifmap
            if (df_ == Dataflow::WS) return o.I(x, start + col);  // WS ifmap
            return o.F(start + col, x);                           // IS filter
        }
        const int64_t start = f * C_;
        if (col >= std::min(C_, Sc_ - start)) return -1;
        if (df_ == Dataflow::IS) return o.I(start + col, x);  // IS ifmap
        return o.F(x, start + col);                           // OS / WS filter
    }

    Dataflow df_;
    int64_t R_, C_;
    const Operands& ops_;
    int64_t Sr_ = 0, Sc_ = 0, T_ = 0;
    int64_t row_fold_ = 1, col_fold_ = 1, rows_per_fold_ = 0;
};

}  // namespace scalesim
