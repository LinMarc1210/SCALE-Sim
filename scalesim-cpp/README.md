# SCALE-Sim C++

A C++17 version of the Python simulator in `scalesim/`. It writes the same reports and traces as the Python version, runs large GEMM layers about 300x faster, and uses far less memory.

## Build

```bash
cmake -S scalesim-cpp -B scalesim-cpp/build
cmake --build scalesim-cpp/build -j
```

Do not remove `-ffp-contract=off` from `CMakeLists.txt`. Without it the last digits of some report values change.

## Run

The flags are the same as `scalesim/scale.py`:

```bash
scalesim-cpp/build/scalesim_cpp -c configs/scale.cfg -t topologies/conv_nets/alexnet.csv -i conv -p results/
```

Extra flags:

| flag | meaning |
|------|---------|
| `-j N` | number of layers simulated in parallel (default: all cores) |
| `-q` | do not print per-layer results |
| `--no-dedup` | with `-s N`, simulate repeated layer shapes again instead of reusing the result |

Use `-s N` for large models. Their traces can take hundreds of GB.

## Verify

```bash
python3 scalesim-cpp/verify/check_pyfmt_pyset.py scalesim-cpp/build/scalesim_selftest
python3 scalesim-cpp/verify/run_parity.py --jobs 16
```

`run_parity.py` runs 85 configs through both versions and compares every output file. Run it after changing either version.

## Not supported

Sparsity, custom layouts and Ramulator traces. The program exits with an error if the config enables them.

## Notes

- In USER bandwidth mode the IFMAP and FILTER DRAM traces contain only `1.0`, so the DRAM bandwidth columns are not meaningful. The Python version behaves the same way.
- Float sums follow Python 3.10. Python 3.12 can differ in the last digit.
