"""
test_gpu_potential.py
~~~~~~~~~~~~~~~~~~~~~
Correctness + timing tests for MultipolePotentialGPU vs agama.Potential (CPU).

Tests every available lmax: 0, 1, 2, 4, 6, 8, 10
Compares .potential(), .force(), .density(), .forceDeriv()
Timing sweep: N = 1, 10, 100, 1000, 10k, 100k, 1M
"""

import sys, os, time
import numpy as np
import cupy as cp
import agama

agama.setUnits(mass=1, length=1, velocity=1)

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpu_potential import MultipolePotentialGPU, PotentialGPU

FIRE_DIR = "/mnt/d/Research/firesims_metaldiff/m12i_res7100/potential/10kpc"

COEF_FILES = {
    0:  os.path.join(FIRE_DIR, "600.dark.none_0.coef_mul_DR"),
    1:  os.path.join(FIRE_DIR, "600.dark.none_1.coef_mul_DR"),
    2:  os.path.join(FIRE_DIR, "600.dark.none_2.coef_mul_DR"),
    4:  os.path.join(FIRE_DIR, "600.dark.none_4.coef_mul_DR"),
    6:  os.path.join(FIRE_DIR, "600.dark.none_6.coef_mul_DR"),
    8:  os.path.join(FIRE_DIR, "600.dark.none_8.coef_mul_DR"),
    10: os.path.join(FIRE_DIR, "600.dark.none_10.coef_mul_DR"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rel_err(a, b, eps=1e-30):
    """Max relative error across array."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))

def abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

def check(label, rel, threshold=1e-6):
    status = PASS if rel < threshold else (WARN if rel < 1e-4 else FAIL)
    print(f"    {label:<30s}  rel={rel:.2e}   {status}")
    return rel < threshold

# ---------------------------------------------------------------------------
# GPU timer (events)
# ---------------------------------------------------------------------------

def gpu_time_ms(fn, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    starts = [cp.cuda.Event() for _ in range(repeats)]
    ends   = [cp.cuda.Event() for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()
    cp.cuda.Stream.null.synchronize()
    times = [cp.cuda.get_elapsed_time(starts[i], ends[i]) for i in range(repeats)]
    return np.median(times)

def cpu_time_ms(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return np.median(times)

# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_correctness(lmax, coef_file, N=2000, seed=42):
    print(f"\n{'='*60}")
    print(f"  lmax={lmax}   file: {os.path.basename(coef_file)}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    # Positions in kpc: mix of near-centre, typical halo, and outer
    xyz_np = np.concatenate([
        rng.uniform(-5,   5,   (N//3, 3)),
        rng.uniform(-50,  50,  (N//3, 3)),
        rng.uniform(-200, 200, (N - 2*(N//3), 3)),
    ], axis=0).astype(np.float64)

    # --- CPU reference ---
    pot_cpu = agama.Potential(coef_file)

    # --- GPU ---
    from nbody_streams.agama_helper import read_coefs
    mc = read_coefs(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)

    xyz_cp = cp.asarray(xyz_np)

    all_pass = True

    # ---- .potential() ----
    phi_cpu = pot_cpu.potential(xyz_np)
    phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
    all_pass &= check("potential()", rel_err(phi_gpu, phi_cpu))

    # ---- .force() ----
    # agama returns force as (N,3) array
    f_cpu = pot_cpu.force(xyz_np)   # (N,3)
    f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))
    for i, comp in enumerate("xyz"):
        all_pass &= check(f"force()[{comp}]", rel_err(f_gpu[:,i], f_cpu[:,i]))

    # ---- .density() ----
    rho_cpu = pot_cpu.density(xyz_np)
    rho_gpu = cp.asnumpy(pot_gpu.density(xyz_cp))
    # density can pass through zero — use a floor of |rho_cpu|.mean() for rel err
    floor = max(np.abs(rho_cpu).mean() * 1e-4, 1e-30)
    rho_rel = float(np.max(np.abs(rho_gpu - rho_cpu) / (np.abs(rho_cpu) + floor)))
    all_pass &= check("density()", rho_rel, threshold=1e-5)

    # ---- .forceDeriv() ----
    # agama: forceDeriv returns (force, deriv) where deriv is (N,6)
    # deriv[i] = [dFx/dx, dFy/dx, dFz/dx, dFy/dy, dFz/dy, dFz/dz] = -Hessian
    f_cpu2, d_cpu = pot_cpu.forceDeriv(xyz_np)
    f_gpu2, d_gpu = pot_gpu.forceDeriv(xyz_cp)
    f_gpu2 = cp.asnumpy(f_gpu2)
    d_gpu  = cp.asnumpy(d_gpu)
    for i, comp in enumerate("xyz"):
        all_pass &= check(f"forceDeriv() force[{comp}]", rel_err(f_gpu2[:,i], f_cpu2[:,i]))
    hess_names = ["xx","xy","xz","yy","yz","zz"]
    for i, nm in enumerate(hess_names):
        floor_h = max(np.abs(d_cpu[:,i]).mean() * 1e-4, 1e-30)
        h_rel = float(np.max(np.abs(d_gpu[:,i] - d_cpu[:,i]) / (np.abs(d_cpu[:,i]) + floor_h)))
        all_pass &= check(f"forceDeriv() deriv[{nm}]", h_rel, threshold=1e-5)

    # ---- single-point test ----
    xyz_1 = cp.asarray(xyz_np[0])   # shape (3,)
    phi_1  = pot_gpu.potential(xyz_1)
    f_1    = pot_gpu.force(xyz_1)
    assert phi_1.shape == (), f"single-point phi shape: {phi_1.shape}"
    assert f_1.shape  == (3,), f"single-point force shape: {f_1.shape}"
    print(f"    {'single-point shapes':<30s}  phi:{phi_1.shape} force:{f_1.shape}  {PASS}")

    summary = PASS if all_pass else FAIL
    print(f"  → Overall: {summary}")
    return all_pass

# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def bench_timing(lmax, coef_file):
    print(f"\n{'='*60}")
    print(f"  TIMING  lmax={lmax}")
    print(f"{'='*60}")

    from nbody_streams.agama_helper import read_coefs
    mc      = read_coefs(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)
    pot_cpu = agama.Potential(coef_file)

    rng = np.random.default_rng(0)

    Ns = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]

    print(f"  {'N':>10}  {'CPU force ms':>14}  {'GPU force ms':>14}  {'Speedup':>9}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*9}")

    for N in Ns:
        xyz_np = rng.standard_normal((N, 3)).astype(np.float64) * 50.0
        xyz_cp = cp.asarray(xyz_np)

        t_cpu = cpu_time_ms(lambda: pot_cpu.force(xyz_np))
        t_gpu = gpu_time_ms(lambda: (pot_gpu.force(xyz_cp), cp.cuda.Stream.null.synchronize()))

        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        print(f"  {N:>10d}  {t_cpu:>14.3f}  {t_gpu:>14.3f}  {speedup:>8.1f}x")
        del xyz_cp
        cp.get_default_memory_pool().free_all_blocks()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("\n" + "="*60)
    print("  GPU Multipole Potential — Correctness Tests")
    print("="*60)

    results = {}
    for lmax, fpath in sorted(COEF_FILES.items()):
        if os.path.exists(fpath):
            results[lmax] = test_correctness(lmax, fpath)
        else:
            print(f"\nSKIP lmax={lmax}: file not found ({fpath})")

    print("\n\n" + "="*60)
    print("  GPU Multipole Potential — Timing Benchmarks")
    print("="*60)

    for lmax in [4, 8]:
        fpath = COEF_FILES.get(lmax)
        if fpath and os.path.exists(fpath):
            bench_timing(lmax, fpath)

    print("\n\n  Summary")
    print("  " + "-"*30)
    for lmax, passed in sorted(results.items()):
        tag = PASS if passed else FAIL
        print(f"  lmax={lmax:<4d}  {tag}")
