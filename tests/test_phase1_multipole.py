"""
test_phase1_optimized.py
~~~~~~~~~~~~~~~~~~~~~~~~
Tests MultipolePotentialGPU (optimized, _multipole_potential_kernel.cu) against:
  1. Agama CPU reference   — physics correctness (Phi, F, rho, Hessian)
  2. Baseline GPU (_baseline/gpu_potential.py) — bit-exact regression
  3. Speedup benchmark     — optimized vs baseline vs CPU at varying N and lmax

Accuracy models: Spheroid and Dehnen potentials exported as Multipole coefficients
  via agama.Potential.export(), tested at lmax = 2, 4, 8.

Speedup models: same models at lmax = 0, 2, 4, 8 over N = 1e3 … 1e6.
  If FIRE snapshot coefficient files are present they are used instead for
  more realistic (higher lmax) benchmarks.
"""

import sys, os, time, tempfile
import numpy as np

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpu_potential import MultipolePotentialGPU        # optimized (parent dir)
# Baseline: load without clobbering the parent import
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_baseline_gpu", "../_baseline/gpu_potential.py")
_baseline_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_mod)
MultipolePotentialGPU_Baseline = _baseline_mod.MultipolePotentialGPU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(N, seed=42, lo=-50, hi=50):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lo, hi, (N, 3)).astype(np.float64)
    xyz[:, :2] += 0.1   # avoid exact z-axis pole
    return xyz


def _rel_phi(gpu, cpu):
    diff = np.abs(np.asarray(gpu) - cpu)
    return float(np.max(diff / (np.abs(cpu) + 1e-30)))


def _rel_force(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1))) + 1e-30
    return float(np.max(np.abs(np.asarray(gpu_f) - cpu_f)) / fmag)


def _rel_scalar(gpu, cpu, eps=1e-20):
    g, c = np.asarray(gpu), np.asarray(cpu)
    mask = np.abs(c) > eps
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(g[mask] - c[mask]) / (np.abs(c[mask]) + 1e-30)))


def _time_fn(fn, warmup=5, reps=20, is_gpu=True):
    for _ in range(warmup):
        fn()
    if is_gpu:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if is_gpu:
        cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / reps


# ---------------------------------------------------------------------------
# Precomputed coefficient files  (optional — skipped if absent)
# ---------------------------------------------------------------------------

_PRECOMPUTED_FILES = {
    "LMC":  "100.LMC.none_8.coef_mult",
    "FIRE": "600.dark.none_8.coef_mul_DR",
}

_LMAX_LIST = [0, 2, 4, 6, 8]

def _make_from_coef_file(coef_file, lmax):
    """Load coef file, cap at lmax, return (agama_pot, opt_gpu, base_gpu)."""
    from nbody_streams.agama_helper import read_coefs, load_agama_potential

    mc   = read_coefs(coef_file)
    keep = [(l, m) for l, m in mc.lm_labels if l <= lmax]
    mc2  = mc.zeroed(keep_lm=keep)

    pot_cpu  = load_agama_potential(coef_file, keep_lm_mult=keep)
    pot_opt  = MultipolePotentialGPU(mc2)
    pot_base = MultipolePotentialGPU_Baseline(mc2)
    return pot_cpu, pot_opt, pot_base


# ---------------------------------------------------------------------------
# Build (agama_pot, opt_gpu, base_gpu) for a given analytic potential + lmax
# ---------------------------------------------------------------------------

def _make_from_agama(agama_kw, lmax):
    """Export agama_pot as Multipole BFE, load into optimized and baseline GPU."""
    from nbody_streams.agama_helper import read_coefs

    pot_cpu = agama.Potential(**agama_kw)

    # Export to a temp file, re-read as MultipoleCoefs object
    with tempfile.NamedTemporaryFile(suffix=".coef_mul", delete=False) as f:
        tmp = f.name
    try:
        agama.Potential(type='Multipole', potential=pot_cpu, lmax=lmax,
                        gridSizeR=50).export(tmp)
        mc = read_coefs(tmp)
    finally:
        os.unlink(tmp)

    # Truncate to requested lmax
    keep = [(l, m) for l, m in mc.lm_labels if l <= lmax]
    mc2  = mc.zeroed(keep_lm=keep)

    pot_opt  = MultipolePotentialGPU(mc2)
    pot_base = MultipolePotentialGPU_Baseline(mc2)
    return pot_cpu, pot_opt, pot_base


# ---------------------------------------------------------------------------
# Accuracy test  (optimized vs Agama CPU + vs Baseline GPU)
# ---------------------------------------------------------------------------

_MODELS = [
    # (display name, agama_kw, lmax)
    ("Spheroid (sph)  lmax=2",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 2),
    ("Spheroid (sph)  lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 4),
    ("Spheroid (sph)  lmax=8",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 8),
    ("Spheroid (tri)  lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
          axisRatioY=0.9, axisRatioZ=0.8), 4),
    ("Spheroid (tri)  lmax=8",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
          axisRatioY=0.9, axisRatioZ=0.8), 8),
    ("Dehnen g=1      lmax=4",
     dict(type='Dehnen', mass=1e12, scaleRadius=5., gamma=1.), 4),
    ("Dehnen g=1.5    lmax=8",
     dict(type='Dehnen', mass=1e12, scaleRadius=5., gamma=1.5), 8),
    ("Spheroid cutoff lmax=4",
     dict(type='Spheroid', mass=1e12, scaleRadius=20., gamma=1.,
          outerCutoffRadius=200., cutoffStrength=2.), 4),
]

_TOL_VS_CPU  = (1e-3, 1e-3, 1e-2, 5e-2)   # (phi, force, rho, hess) vs Agama
_TOL_VS_BASE = (1e-10, 1e-10, 1e-10, 1e-10) # vs baseline GPU (bitwise agreement)


def test_accuracy():
    N = 2000
    xyz_np = _pts(N)
    xyz_cp = cp.asarray(xyz_np)

    W = 118
    print("=" * W)
    print(f"{'PHASE 1 ACCURACY:  optimized GPU  vs  Agama CPU  and  vs  Baseline GPU (N=2000)':^{W}}")
    print("=" * W)
    print(f"  {'Model':<30} | {'Phi/CPU':>9} | {'F/CPU':>9} | {'Rho/CPU':>9} | {'Hess/CPU':>9}"
          f" | {'Phi/Base':>9} | {'F/Base':>9} | {'Rho/Base':>9} | Status")
    print("-" * W)

    for name, agama_kw, lmax in _MODELS:
        try:
            pot_cpu, pot_opt, pot_base = _make_from_agama(agama_kw, lmax)

            phi_cpu  = pot_cpu.potential(xyz_np)
            f_cpu    = pot_cpu.force(xyz_np)
            rho_cpu  = pot_cpu.density(xyz_np)
            dF_cpu   = pot_cpu.forceDeriv(xyz_np)[1]   # (N,6)

            phi_opt  = cp.asnumpy(pot_opt.potential(xyz_cp))
            f_opt    = cp.asnumpy(pot_opt.force(xyz_cp))
            rho_opt  = cp.asnumpy(pot_opt.density(xyz_cp))
            _, dF_cp = pot_opt.forceDeriv(xyz_cp)
            dF_opt   = cp.asnumpy(dF_cp)

            phi_base = cp.asnumpy(pot_base.potential(xyz_cp))
            f_base   = cp.asnumpy(pot_base.force(xyz_cp))
            rho_base = cp.asnumpy(pot_base.density(xyz_cp))

            phi_vs_cpu  = _rel_phi(phi_opt, phi_cpu)
            f_vs_cpu    = _rel_force(f_opt, f_cpu)
            rho_vs_cpu  = _rel_scalar(rho_opt, rho_cpu)
            dF_scale    = np.max(np.abs(dF_cpu)) + 1e-30
            hess_vs_cpu = float(np.max(np.abs(dF_opt - dF_cpu)) / dF_scale)

            phi_vs_base = _rel_phi(phi_opt, phi_base)
            f_vs_base   = _rel_force(f_opt, f_base)
            rho_vs_base = _rel_scalar(rho_opt, rho_base)

            pt, ft, rt, ht = _TOL_VS_CPU
            is_ok = (phi_vs_cpu < pt and f_vs_cpu < ft and
                     rho_vs_cpu < rt and hess_vs_cpu < ht)
            status = "PASS" if is_ok else "FAIL"
            print(f"  {name:<30} | {phi_vs_cpu:>9.2e} | {f_vs_cpu:>9.2e} | {rho_vs_cpu:>9.2e}"
                  f" | {hess_vs_cpu:>9.2e} | {phi_vs_base:>9.2e} | {f_vs_base:>9.2e}"
                  f" | {rho_vs_base:>9.2e} | {status}")

        except Exception as e:
            print(f"  {name:<30} | CRITICAL ERROR: {e}")

    print("=" * W)


# ---------------------------------------------------------------------------
# Speedup benchmark
# ---------------------------------------------------------------------------

def run_speedup_benchmark():
    W = 90
    print("\n" + "=" * W)
    print(f"{'PHASE 1 SPEEDUP:  optimized GPU  vs  baseline GPU  vs  Agama CPU  (.force)':^{W}}")
    print("=" * W)

    # Benchmark configs: (label, agama_kw, lmax)
    configs = [
        ("Spheroid lmax=0",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 0),
        ("Spheroid lmax=4",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 4),
        ("Spheroid (tri) lmax=4",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.,
              axisRatioY=0.9, axisRatioZ=0.8), 4),
        ("Spheroid lmax=8",
         dict(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4., gamma=1.), 8),
    ]

    N_list = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    for label, agama_kw, lmax in configs:
        try:
            pot_cpu, pot_opt, pot_base = _make_from_agama(agama_kw, lmax)
        except Exception as e:
            print(f"\n  {label}: SKIP ({e})")
            continue

        n_lm = (lmax + 1)**2
        print(f"\n  {label}  (n_lm={n_lm})")
        hdr = (f"    {'N':>9}  {'CPU (ms)':>10}  {'Base (ms)':>10}  {'Opt (ms)':>10}"
               f"  {'Base/CPU':>9}  {'Opt/CPU':>9}  {'Opt/Base':>9}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))

        for N in N_list:
            xyz_np = _pts(N, seed=7)
            xyz_cp = cp.asarray(xyz_np)

            t_cpu  = _time_fn(lambda: pot_cpu.force(xyz_np),  is_gpu=False) * 1e3
            t_base = _time_fn(lambda: pot_base.force(xyz_cp), is_gpu=True)  * 1e3
            t_opt  = _time_fn(lambda: pot_opt.force(xyz_cp),  is_gpu=True)  * 1e3

            print(f"    {N:>9,}  {t_cpu:>10.2f}  {t_base:>10.2f}  {t_opt:>10.2f}"
                  f"  {t_cpu/t_base:>9.1f}x  {t_cpu/t_opt:>9.1f}x  {t_base/t_opt:>9.1f}x")

    print("=" * W)


# ---------------------------------------------------------------------------
# Precomputed-file accuracy  (lmax = 0,2,4,6,8 for each file if present)
# ---------------------------------------------------------------------------

def test_precomputed_accuracy():
    present = {k: p for k, p in _PRECOMPUTED_FILES.items() if os.path.exists(p)}
    if not present:
        print("\n[Precomputed file accuracy] no files found — skipped")
        return

    N = 2000
    xyz_np = _pts(N)
    xyz_cp = cp.asarray(xyz_np)

    W = 118
    print("\n" + "=" * W)
    print(f"{'PRECOMPUTED FILE ACCURACY:  optimized GPU  vs  Agama CPU  and  vs  Baseline GPU':^{W}}")
    print("=" * W)
    print(f"  {'Model':<30} | {'Phi/CPU':>9} | {'F/CPU':>9} | {'Rho/CPU':>9} | {'Hess/CPU':>9}"
          f" | {'Phi/Base':>9} | {'F/Base':>9} | {'Rho/Base':>9} | Status")
    print("-" * W)

    for label, path in present.items():
        for lmax in _LMAX_LIST:
            name = f"{label} lmax={lmax}"
            try:
                pot_cpu, pot_opt, pot_base = _make_from_coef_file(path, lmax)

                phi_cpu = pot_cpu.potential(xyz_np)
                f_cpu   = pot_cpu.force(xyz_np)
                rho_cpu = pot_cpu.density(xyz_np)
                dF_cpu  = pot_cpu.forceDeriv(xyz_np)[1]

                phi_opt  = cp.asnumpy(pot_opt.potential(xyz_cp))
                f_opt    = cp.asnumpy(pot_opt.force(xyz_cp))
                rho_opt  = cp.asnumpy(pot_opt.density(xyz_cp))
                _, dF_cp = pot_opt.forceDeriv(xyz_cp)
                dF_opt   = cp.asnumpy(dF_cp)

                phi_base = cp.asnumpy(pot_base.potential(xyz_cp))
                f_base   = cp.asnumpy(pot_base.force(xyz_cp))
                rho_base = cp.asnumpy(pot_base.density(xyz_cp))

                phi_vs_cpu  = _rel_phi(phi_opt, phi_cpu)
                f_vs_cpu    = _rel_force(f_opt, f_cpu)
                rho_vs_cpu  = _rel_scalar(rho_opt, rho_cpu)
                dF_scale    = np.max(np.abs(dF_cpu)) + 1e-30
                hess_vs_cpu = float(np.max(np.abs(dF_opt - dF_cpu)) / dF_scale)

                phi_vs_base = _rel_phi(phi_opt, phi_base)
                f_vs_base   = _rel_force(f_opt, f_base)
                rho_vs_base = _rel_scalar(rho_opt, rho_base)

                pt, ft, rt, ht = _TOL_VS_CPU
                is_ok  = phi_vs_cpu < pt and f_vs_cpu < ft and rho_vs_cpu < rt and hess_vs_cpu < ht
                status = "PASS" if is_ok else "FAIL"
                print(f"  {name:<30} | {phi_vs_cpu:>9.2e} | {f_vs_cpu:>9.2e} | {rho_vs_cpu:>9.2e}"
                      f" | {hess_vs_cpu:>9.2e} | {phi_vs_base:>9.2e} | {f_vs_base:>9.2e}"
                      f" | {rho_vs_base:>9.2e} | {status}")

            except Exception as e:
                print(f"  {name:<30} | CRITICAL ERROR: {e}")

        print("-" * W)

    print("=" * W)


# ---------------------------------------------------------------------------
# Precomputed-file speedup  (lmax = 0,2,4,6,8 for each file if present)
# ---------------------------------------------------------------------------

def run_precomputed_benchmark():
    present = {k: p for k, p in _PRECOMPUTED_FILES.items() if os.path.exists(p)}
    if not present:
        print("\n[Precomputed file benchmark] no files found — skipped")
        return

    W = 90
    print("\n" + "=" * W)
    print(f"{'PRECOMPUTED FILE SPEEDUP:  optimized GPU  vs  baseline GPU  vs  Agama CPU':^{W}}")
    print("=" * W)

    N_list = [1_000, 10_000, 100_000, 1_000_000]

    for label, path in present.items():
        for lmax in _LMAX_LIST:
            n_lm = (lmax + 1)**2
            try:
                pot_cpu, pot_opt, pot_base = _make_from_coef_file(path, lmax)
            except Exception as e:
                print(f"\n  {label} lmax={lmax}: SKIP ({e})")
                continue

            print(f"\n  {label} lmax={lmax}  (n_lm={n_lm})")
            hdr = (f"    {'N':>9}  {'CPU (ms)':>10}  {'Base (ms)':>10}  {'Opt (ms)':>10}"
                   f"  {'Base/CPU':>9}  {'Opt/CPU':>9}  {'Opt/Base':>9}")
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            for N in N_list:
                xyz_np = _pts(N, seed=7)
                xyz_cp = cp.asarray(xyz_np)

                t_cpu  = _time_fn(lambda: pot_cpu.force(xyz_np),  is_gpu=False) * 1e3
                t_base = _time_fn(lambda: pot_base.force(xyz_cp), is_gpu=True)  * 1e3
                t_opt  = _time_fn(lambda: pot_opt.force(xyz_cp),  is_gpu=True)  * 1e3

                print(f"    {N:>9,}  {t_cpu:>10.2f}  {t_base:>10.2f}  {t_opt:>10.2f}"
                      f"  {t_cpu/t_base:>9.1f}x  {t_cpu/t_opt:>9.1f}x  {t_base/t_opt:>9.1f}x")

        print("=" * W)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_accuracy()
    run_speedup_benchmark()
    test_precomputed_accuracy()
    run_precomputed_benchmark()
