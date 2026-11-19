// Emulation of CPython 3.10 set insertion / iteration order for integer-valued keys.
//
// The Python reference turns sets of addresses into lists (`list(this_set)`) when it writes
// estimate-bandwidth DRAM traces, so the order of addresses in those traces is the hash-table
// slot order of a CPython set. This class replays CPython's setobject.c probing and resizing
// logic (LINEAR_PROBES = 9, PERTURB_SHIFT = 5, growth to used*4 / used*2) so the C++ port emits
// the same order. Keys are the integer values of np.float64 addresses, whose hash equals
// hash(int(v)) = v mod (2**61 - 1).
#pragma once

#include <cstdint>
#include <vector>

namespace scalesim {

class PySetOrder {
public:
    PySetOrder() { reset(); }

    void reset() {
        mask_ = kMinSize - 1;
        fill_ = 0;
        used_ = 0;
        keys_.assign(kMinSize, 0);
        hashes_.assign(kMinSize, 0);
        active_.assign(kMinSize, 0);
    }

    static uint64_t py_hash(int64_t v) {
        constexpr uint64_t kModulus = (uint64_t{1} << 61) - 1;
        int64_t h;
        if (v >= 0) {
            h = static_cast<int64_t>(static_cast<uint64_t>(v) % kModulus);
        } else {
            const uint64_t mag = static_cast<uint64_t>(-(v + 1)) + 1;  // |v| without overflow
            h = -static_cast<int64_t>(mag % kModulus);
            if (h == -1) h = -2;
        }
        return static_cast<uint64_t>(h);
    }

    void add(int64_t key) {
        const uint64_t h = py_hash(key);
        const uint64_t mask = mask_;
        uint64_t i = h & mask;
        uint64_t perturb = h;
        while (true) {
            uint64_t e = i;
            uint64_t probes = (i + kLinearProbes <= mask) ? kLinearProbes : 0;
            while (true) {
                if (!active_[e]) {
                    insert_at(e, key, h);
                    return;
                }
                if (hashes_[e] == h && keys_[e] == key) return;  // already present
                ++e;
                if (probes-- == 0) break;
            }
            perturb >>= kPerturbShift;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    size_t size() const { return used_; }

    template <class F>
    void for_each(F&& f) const {
        for (size_t s = 0; s <= mask_; ++s)
            if (active_[s]) f(keys_[s]);
    }

private:
    static constexpr uint64_t kMinSize = 8;
    static constexpr uint64_t kLinearProbes = 9;
    static constexpr unsigned kPerturbShift = 5;

    void insert_at(uint64_t slot, int64_t key, uint64_t h) {
        const uint64_t mask = mask_;
        ++fill_;
        ++used_;
        keys_[slot] = key;
        hashes_[slot] = h;
        active_[slot] = 1;
        if (fill_ * 5 < mask * 3) return;
        resize(used_ > 50000 ? used_ * 2 : used_ * 4);
    }

    void resize(uint64_t minused) {
        uint64_t newsize = kMinSize;
        while (newsize <= minused) newsize <<= 1;
        std::vector<int64_t> old_keys;
        std::vector<uint64_t> old_hashes;
        std::vector<uint8_t> old_active;
        old_keys.swap(keys_);
        old_hashes.swap(hashes_);
        old_active.swap(active_);
        keys_.assign(newsize, 0);
        hashes_.assign(newsize, 0);
        active_.assign(newsize, 0);
        mask_ = newsize - 1;
        fill_ = used_;
        for (size_t s = 0; s < old_active.size(); ++s)
            if (old_active[s]) insert_clean(old_keys[s], old_hashes[s]);
    }

    void insert_clean(int64_t key, uint64_t h) {
        const uint64_t mask = mask_;
        uint64_t i = h & mask;
        uint64_t perturb = h;
        while (true) {
            if (!active_[i]) {
                place(i, key, h);
                return;
            }
            if (i + kLinearProbes <= mask) {
                for (uint64_t j = 1; j <= kLinearProbes; ++j) {
                    if (!active_[i + j]) {
                        place(i + j, key, h);
                        return;
                    }
                }
            }
            perturb >>= kPerturbShift;
            i = (i * 5 + 1 + perturb) & mask;
        }
    }

    void place(uint64_t slot, int64_t key, uint64_t h) {
        keys_[slot] = key;
        hashes_[slot] = h;
        active_[slot] = 1;
    }

    uint64_t mask_ = kMinSize - 1;
    uint64_t fill_ = 0;
    uint64_t used_ = 0;
    std::vector<int64_t> keys_;
    std::vector<uint64_t> hashes_;
    std::vector<uint8_t> active_;
};

}  // namespace scalesim
