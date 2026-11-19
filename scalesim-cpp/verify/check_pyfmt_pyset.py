"""Cross-checks the C++ float formatting and CPython-set emulation against the running CPython.

usage: python3 check_pyfmt_pyset.py <path/to/scalesim_selftest>
"""
import random
import struct
import subprocess
import sys

import numpy as np


def main():
    exe = sys.argv[1]
    rng = random.Random(1234)

    floats = [0.0, -0.0, 1.0, -1.0, 1e16, 1e-4, 1e-5, 9999999999999998.0, 123456789.0, 0.1, 1 / 3,
              2.5e-5, 5e-324, 1.7976931348623157e308, float('inf'), float('-inf')]
    for _ in range(200000):
        kind = rng.random()
        if kind < 0.3:
            floats.append(rng.uniform(0, 100))
        elif kind < 0.5:
            floats.append(rng.randint(1, 10**9) / rng.randint(1, 10**7))
        elif kind < 0.7:
            floats.append(struct.unpack('<d', struct.pack('<Q', rng.getrandbits(64)))[0])
        else:
            floats.append(rng.randint(-10**15, 10**15) * 1.0)
    floats = [f for f in floats if f == f]  # drop NaNs from random bit patterns

    sets = []
    for _ in range(400):
        n = rng.choice([1, 5, 9, 50, 655, 1000, 5000, 60000])
        base = rng.choice([0, 10000000, 20000000, rng.randint(0, 2**40)])
        stride = rng.choice([1, 2, 3, 7, 128, 4096, 14336])
        keys = [base + stride * rng.randint(0, n * 4) for _ in range(n)]
        sets.append(keys)

    lines = []
    for f in floats:
        lines.append('F %016x' % struct.unpack('<Q', struct.pack('<d', f))[0])
    for keys in sets:
        lines.append('S %d %s' % (len(keys), ' '.join(map(str, keys))))
    out = subprocess.run([exe], input='\n'.join(lines) + '\n', capture_output=True, text=True,
                         check=True).stdout.split('\n')

    bad = 0
    for i, f in enumerate(floats):
        exp = str(np.float64(f))
        if out[i] != exp or repr(f) != exp:
            bad += 1
            if bad < 10:
                print('float mismatch:', repr(f), 'expected', exp, 'got', out[i])
    off = len(floats)
    for j, keys in enumerate(sets):
        s = set()
        for k in keys:
            s.add(np.float64(k))
        exp = ' '.join(str(int(v)) for v in s)
        if out[off + j] != exp:
            bad += 1
            if bad < 20:
                print('set order mismatch for set #%d (n=%d)' % (j, len(keys)))
    print('checked %d floats and %d sets: %s' % (len(floats), len(sets), 'OK' if bad == 0 else '%d FAILURES' % bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
