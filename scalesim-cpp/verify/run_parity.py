"""Byte-for-byte parity check between the Python SCALE-Sim and the C++ port.

Every case runs scalesim/scale.py and scalesim_cpp on the same config + topology and compares
every output file (reports and traces) byte for byte.

usage: python3 scalesim-cpp/verify/run_parity.py [--exe scalesim-cpp/build/scalesim_cpp] [--work DIR] [--jobs N]
                                        [--filter SUBSTRING] [--quick]
"""
import argparse
import filecmp
import itertools
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CFG_TEMPLATE = """[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight:    {rows}
ArrayWidth:     {cols}
IfmapSramSzkB:   {ifmap_kb}
FilterSramSzkB:  {filter_kb}
OfmapSramSzkB:   {ofmap_kb}
IfmapOffset:    {ifmap_off}
FilterOffset:   {filter_off}
OfmapOffset:    {ofmap_off}
Bandwidth : {bw}
Dataflow : {df}
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: {bank_bw}
IfmapSRAMBankNum: {bank_num}
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: {bank_bw}
FilterSRAMBankNum: {bank_num}
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: {mode}
UseRamulatorTrace: False
{extra}
"""

TOPOLOGIES = {
    # conv: name, ifmap h, ifmap w, filter h, filter w, channels, num filter, stride
    'conv_small.csv': ('conv', """Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
C1, 15, 15, 3, 3, 4, 8, 1,
C2, 17, 13, 5, 3, 3, 16, 2,
C3, 9, 9, 1, 1, 40, 20, 1,
DP_conv, 10, 10, 3, 3, 3, 1, 1,
C1_again, 15, 15, 3, 3, 4, 8, 1,
"""),
    'conv_stride.csv': ('conv', """Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,
S1, 31, 29, 7, 5, 3, 33, 3,
S2, 20, 20, 4, 4, 8, 70, 4,
"""),
    'alexnet_part.csv': ('conv', None),  # topologies/conv_nets/alexnet_part.csv
    'gemm_small.csv': ('gemm', """Layer, M, N, K,
G1, 37, 50, 70,
G2, 1, 300, 17,
G3, 130, 7, 129,
G4, 64, 64, 64,
"""),
    'gemm_mnk.csv': ('gemm', None),  # topologies/GEMM_mnk/test_mnk_input.csv
    'gemm_llm.csv': ('gemm', """Layer, M, N, K,
Q, 32, 512, 512,
FFN, 16, 1024, 384,
Q_dup, 32, 512, 512,
"""),
}

ARRAYS = {'32x32': (32, 32), '8x16': (8, 16), '128x128': (128, 128)}


def make_cases(quick):
    cases = []
    def add(name, topo, **kw):
        cases.append((name, topo, kw))

    for df in ('ws', 'os', 'is'):
        for mode in ('CALC', 'USER'):
            for arr in ('32x32', '8x16'):
                for sram in ((64, 64, 64), (2, 3, 1)):
                    for topo in ('conv_small.csv', 'gemm_small.csv', 'conv_stride.csv'):
                        add(f'{df}_{mode}_{arr}_{sram[0]}k_{topo[:-4]}', topo, df=df, mode=mode, arr=arr,
                            sram=sram)
    # golden-trace style runs (the CI configs) and a TPUv4-like setup
    for df in ('ws', 'os', 'is'):
        add(f'{df}_CALC_32x32_alexnet', 'alexnet_part.csv', df=df, mode='CALC', arr='32x32', sram=(64, 64, 64))
        add(f'{df}_USER_32x32_mnk', 'gemm_mnk.csv', df=df, mode='USER', arr='32x32', sram=(64, 64, 64))
        add(f'{df}_USER_128_tpuv4_llm', 'gemm_llm.csv', df=df, mode='USER', arr='128x128', sram=(4096, 4096, 8192),
            bw=1024, bank_bw=1024, bank_num=32, extra='TimeLinearModel: TPUv4')
        add(f'{df}_CALC_128_llm_v5e', 'gemm_llm.csv', df=df, mode='CALC', arr='128x128', sram=(256, 256, 128),
            extra='TimeLinearModel: TPUv5e')
    add('ws_USER_8x16_offsets_v6e', 'conv_small.csv', df='ws', mode='USER', arr='8x16', sram=(4, 4, 2),
        offsets=(5, 1000, 3000000), bw=20, bank_bw=20, bank_num=4, extra='TimeLinearModel: TPUv6e')
    if quick:
        cases = [c for c in cases if '_2k_' not in c[0] and 'alexnet' not in c[0]]
    return cases


def write_inputs(work):
    tdir = os.path.join(work, 'topologies')
    os.makedirs(tdir, exist_ok=True)
    for fname, (_, text) in TOPOLOGIES.items():
        dst = os.path.join(tdir, fname)
        if text is None:
            src = {'alexnet_part.csv': 'topologies/conv_nets/alexnet_part.csv',
                   'gemm_mnk.csv': 'topologies/GEMM_mnk/test_mnk_input.csv'}[fname]
            shutil.copy(os.path.join(REPO, src), dst)
        else:
            with open(dst, 'w') as f:
                f.write(text)
    return tdir


def run_case(case, exe, work, tdir, save_trace):
    name, topo, kw = case
    rows, cols = ARRAYS[kw['arr']]
    ifmap_kb, filter_kb, ofmap_kb = kw['sram']
    offs = kw.get('offsets', (0, 10000000, 20000000))
    cfg_text = CFG_TEMPLATE.format(run_name=name, rows=rows, cols=cols, ifmap_kb=ifmap_kb, filter_kb=filter_kb,
                                   ofmap_kb=ofmap_kb, ifmap_off=offs[0], filter_off=offs[1], ofmap_off=offs[2],
                                   bw=kw.get('bw', 10), df=kw['df'], bank_bw=kw.get('bank_bw', 10),
                                   bank_num=kw.get('bank_num', 10), mode=kw['mode'], extra=kw.get('extra', ''))
    cdir = os.path.join(work, 'cases', name)
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir)
    cfg = os.path.join(cdir, 'config.cfg')
    with open(cfg, 'w') as f:
        f.write(cfg_text)
    topo_path = os.path.join(tdir, topo)
    itype = TOPOLOGIES[topo][0]
    s = 'Y' if save_trace else 'N'

    t0 = time.time()
    py = subprocess.run([sys.executable, 'scalesim/scale.py', '-c', cfg, '-t', topo_path, '-i', itype,
                         '-p', os.path.join(cdir, 'py'), '-s', s], cwd=REPO, capture_output=True, text=True,
                        env=dict(os.environ, PYTHONPATH=REPO, PYTHONWARNINGS='ignore'))
    t_py = time.time() - t0
    t0 = time.time()
    cc = subprocess.run([exe, '-c', cfg, '-t', topo_path, '-i', itype, '-p', os.path.join(cdir, 'cpp'), '-s', s,
                         '-q', '-j', '4'], cwd=REPO, capture_output=True, text=True)
    t_cc = time.time() - t0

    if py.returncode != 0:
        tail = py.stderr.strip().splitlines()[-1:] if py.stderr.strip() else ['?']
        if cc.returncode != 0:
            return name, 'both-failed', t_py, t_cc, ['python: ' + tail[0], 'cpp: ' + cc.stderr.strip()]
        return name, 'python-failed', t_py, t_cc, tail
    if cc.returncode != 0:
        return name, 'FAIL', t_py, t_cc, ['cpp failed: ' + cc.stderr.strip()]

    pdir = os.path.join(cdir, 'py', name)
    qdir = os.path.join(cdir, 'cpp', name)
    problems = []
    pfiles = sorted(os.path.relpath(os.path.join(r, f), pdir) for r, _, fs in os.walk(pdir) for f in fs)
    qfiles = sorted(os.path.relpath(os.path.join(r, f), qdir) for r, _, fs in os.walk(qdir) for f in fs)
    if pfiles != qfiles:
        problems.append('file sets differ: only-py=%s only-cpp=%s' % (sorted(set(pfiles) - set(qfiles)),
                                                                     sorted(set(qfiles) - set(pfiles))))
    nbytes = 0
    for rel in sorted(set(pfiles) & set(qfiles)):
        a, b = os.path.join(pdir, rel), os.path.join(qdir, rel)
        nbytes += os.path.getsize(a)
        if not filecmp.cmp(a, b, shallow=False):
            problems.append('DIFF ' + rel)
    status = 'OK' if not problems else 'FAIL'
    if status == 'OK':
        shutil.rmtree(cdir, ignore_errors=True)
    return name, status, t_py, t_cc, problems + ['%d files, %.1f MB compared' % (len(pfiles), nbytes / 2**20)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=os.path.join(REPO, 'scalesim-cpp', 'build', 'scalesim_cpp'))
    ap.add_argument('--work', default=os.path.join(REPO, 'scalesim-cpp', 'build', 'parity_work'))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--filter', default='')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-trace', action='store_true', help='compare reports only (-s N)')
    a = ap.parse_args()

    tdir = write_inputs(a.work)
    cases = [c for c in make_cases(a.quick) if a.filter in c[0]]
    print('running %d cases with %d jobs' % (len(cases), a.jobs))
    failures = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(run_case, c, a.exe, a.work, tdir, not a.no_trace) for c in cases]
        for fut in futs:
            name, status, t_py, t_cc, info = fut.result()
            if status in ('FAIL',):
                failures += 1
            print('%-6s %-45s py %7.2fs  cpp %6.2fs  %s' % (status, name, t_py, t_cc, '; '.join(info[-1:])))
            if status != 'OK':
                for line in info[:-1][:12]:
                    print('         ', line)
    print('\n%d / %d cases failed' % (failures, len(cases)))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
