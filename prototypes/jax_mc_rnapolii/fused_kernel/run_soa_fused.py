"""run_soa.py with the fused Pallas sweep (pallas_sweep.TutSamplerFused) in
place of the XLA sweep. Same options and outputs; the per-frame rescore,
records and snapshots stay XLA. The kernel needs dense EV against the
bodies and rotating EM covariances (--ev dense, --cov rotate, set here).
--no-em-cache turns off the kernel's EM cache.

    mpirun -np 4 --bind-to none python run_soa_fused.py --chains 16384 \\
        --frames 20000 --thin 200 --score-every 10 --rescore recorded \\
        --out runs/x
"""
import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import sampler_soa                              # noqa: E402
from pallas_sweep import TutSamplerFused        # noqa: E402
import run_soa                                  # noqa: E402

if __name__ == '__main__':
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--ev', default='dense')
    pre.add_argument('--cov', default='rotate')
    pre.add_argument('--no-em-cache', action='store_true')
    k, rest = pre.parse_known_args()
    if k.ev != 'dense' or k.cov != 'rotate':
        raise SystemExit('the fused kernel needs --ev dense and --cov rotate')
    sys.argv = [sys.argv[0]] + rest + ['--ev', 'dense', '--cov', 'rotate']
    # run_soa.main() imports TutSamplerSoA from sampler_soa when it runs
    sampler_soa.TutSamplerSoA = lambda s: TutSamplerFused(
        s, bc=16, num_warps=2, layout='reg128', em_cache=not k.no_em_cache)
    run_soa.main()
