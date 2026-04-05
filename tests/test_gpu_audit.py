"""
test_gpu_audit.py
~~~~~~~~~~~~~~~~~
Absolute-error and edge-case audit of GPU multipole potential vs Agama CPU.

Tests:
  1. Absolute & relative errors (potential, force, Hessian) at random points
  2. Grid-point vs midpoint comparison
  3. Z-axis singularity: (0, 0, z) — Fy must not vanish if Agama gives nonzero
  4. Near-origin behavior
  5. Error trends with lmax and radius
  6. --use_fast_math impact assessment (FMA)
"""

import sys, os, time
import numpy as np

# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/aarora/py_scripts/Agama")

import agama
agama.setUnits(mass=1, length=1, velocity=1)

import cupy as cp
from gpu_potential import MultipolePotentialGPU

FIRE_DIR = "/mnt/d/Research/firesims_metaldiff/m12i_res7100/potential/10kpc"

COEF_FILES = {
    0:  os.path.join(FIRE_DIR, "600.dark.none_0.coef_mul_DR"),
    2:  os.path.join(FIRE_DIR, "600.dark.none_2.coef_mul_DR"),
    4:  os.path.join(FIRE_DIR, "600.dark.none_4.coef_mul_DR"),
    6:  os.path.join(FIRE_DIR, "600.dark.none_6.coef_mul_DR"),
    8:  os.path.join(FIRE_DIR, "600.dark.none_8.coef_mul_DR"),
    10: os.path.join(FIRE_DIR, "600.dark.none_10.coef_mul_DR"),
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gpu(coef_file):
    from nbody_streams.agama_helper import read_coefs
    mc = read_coefs(coef_file)
    return MultipolePotentialGPU(mc)

def get_grid(coef_file):
    """Return the radial grid from a coef file."""
    from nbody_streams.agama_helper import read_coefs
    mc = read_coefs(coef_file)
    return np.asarray(mc.R_grid, dtype=np.float64)

# ---------------------------------------------------------------------------
# TEST 1: Absolute errors — are forces/potentials physically small where
#          relative errors look large?
# ---------------------------------------------------------------------------

def test_absolute_errors(lmax, coef_file, N=5000, seed=42):
    print(f"\n{'='*70}")
    print(f"  TEST 1: Absolute errors — lmax={lmax}")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    xyz_np = np.concatenate([
        rng.uniform(-5,   5,   (N//3, 3)),
        rng.uniform(-50,  50,  (N//3, 3)),
        rng.uniform(-200, 200, (N - 2*(N//3), 3)),
    ], axis=0).astype(np.float64)

    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)
    xyz_cp  = cp.asarray(xyz_np)

    # Potential
    phi_cpu = pot_cpu.potential(xyz_np)
    phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
    abs_phi = np.abs(phi_gpu - phi_cpu)
    rel_phi = abs_phi / (np.abs(phi_cpu) + 1e-30)

    print(f"  Potential [units: (km/s)^2]")
    print(f"    max |Φ_cpu|           = {np.max(np.abs(phi_cpu)):.6e}")
    print(f"    median |Φ_cpu|        = {np.median(np.abs(phi_cpu)):.6e}")
    print(f"    max |ΔΦ| (absolute)   = {np.max(abs_phi):.6e}")
    print(f"    median |ΔΦ| (abs)     = {np.median(abs_phi):.6e}")
    print(f"    max relative error    = {np.max(rel_phi):.6e}")
    print(f"    median relative error = {np.median(rel_phi):.6e}")

    # Force
    f_cpu = pot_cpu.force(xyz_np)
    f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))
    abs_f = np.abs(f_gpu - f_cpu)
    fmag_cpu = np.sqrt(np.sum(f_cpu**2, axis=1))

    print(f"\n  Force [units: (km/s)^2 / kpc]")
    print(f"    max |F_cpu|           = {np.max(fmag_cpu):.6e}")
    print(f"    median |F_cpu|        = {np.median(fmag_cpu):.6e}")
    for i, c in enumerate("xyz"):
        abs_err = np.max(abs_f[:, i])
        rel_err = np.max(abs_f[:, i] / (np.abs(f_cpu[:, i]) + 1e-30))
        print(f"    F{c}: max|Δ|={abs_err:.6e}  max_rel={rel_err:.6e}")

    # Hessian
    f_cpu2, d_cpu = pot_cpu.forceDeriv(xyz_np)
    f_gpu2, d_gpu = pot_gpu.forceDeriv(xyz_cp)
    d_gpu = cp.asnumpy(d_gpu)
    abs_d = np.abs(d_gpu - d_cpu)

    hess_names = ["dFx/dx","dFy/dy","dFz/dz","dFx/dy","dFy/dz","dFx/dz"]
    print(f"\n  Hessian [units: (km/s)^2 / kpc^2]")
    for i, nm in enumerate(hess_names):
        ae = np.max(abs_d[:, i])
        scale = np.max(np.abs(d_cpu[:, i]))
        re = ae / (scale + 1e-30)
        print(f"    {nm}: max|Δ|={ae:.6e}  |scale|={scale:.6e}  rel={re:.6e}")

# ---------------------------------------------------------------------------
# TEST 2: Grid-point vs midpoint errors
# ---------------------------------------------------------------------------

def test_grid_vs_midpoint(lmax, coef_file, N_per_shell=200, seed=99):
    print(f"\n{'='*70}")
    print(f"  TEST 2: Grid-point vs midpoint errors — lmax={lmax}")
    print(f"{'='*70}")

    r_grid = get_grid(coef_file)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)
    rng = np.random.default_rng(seed)

    def random_on_shells(radii, n_per):
        pts = []
        for r in radii:
            # random directions
            z = rng.uniform(-1, 1, n_per)
            phi = rng.uniform(0, 2*np.pi, n_per)
            ct = z
            st = np.sqrt(1 - ct**2)
            pts.append(np.column_stack([r*st*np.cos(phi), r*st*np.sin(phi), r*ct]))
        return np.vstack(pts).astype(np.float64)

    # At grid points
    grid_r = r_grid[1:-1:3]  # every 3rd interior grid point
    xyz_grid = random_on_shells(grid_r, N_per_shell)

    # At midpoints (geometric mean between consecutive grid points)
    mid_r = np.sqrt(r_grid[:-1] * r_grid[1:])[1:-1:3]
    xyz_mid = random_on_shells(mid_r, N_per_shell)

    for label, xyz_np in [("GRID POINTS", xyz_grid), ("MIDPOINTS", xyz_mid)]:
        xyz_cp = cp.asarray(xyz_np)
        phi_cpu = pot_cpu.potential(xyz_np)
        phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
        f_cpu = pot_cpu.force(xyz_np)
        f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))

        phi_rel = np.max(np.abs(phi_gpu - phi_cpu) / (np.abs(phi_cpu) + 1e-30))
        f_rel = np.max(np.abs(f_gpu - f_cpu) / (np.abs(f_cpu) + 1e-30))
        phi_abs = np.max(np.abs(phi_gpu - phi_cpu))
        f_abs = np.max(np.abs(f_gpu - f_cpu))

        status_phi = PASS if phi_rel < 1e-8 else (WARN if phi_rel < 1e-5 else FAIL)
        status_f   = PASS if f_rel < 1e-6 else (WARN if f_rel < 1e-4 else FAIL)
        print(f"  {label}  (N={len(xyz_np)})")
        print(f"    Φ: max_rel={phi_rel:.2e}  max_abs={phi_abs:.2e}  {status_phi}")
        print(f"    F: max_rel={f_rel:.2e}  max_abs={f_abs:.2e}  {status_f}")

# ---------------------------------------------------------------------------
# TEST 3: Z-axis — (0, 0, z)  for various z values
#          Key question: does GPU produce nonzero Fy when Agama does?
# ---------------------------------------------------------------------------

def test_z_axis(lmax, coef_file):
    print(f"\n{'='*70}")
    print(f"  TEST 3: Z-axis singularity — lmax={lmax}")
    print(f"{'='*70}")

    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)

    z_vals = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    # Also test points very close to z-axis but not exactly on it
    # eps_vals: tiny x,y offset from z-axis
    eps_vals = [0.0, 1e-15, 1e-10, 1e-6, 1e-3]

    print(f"  {'z':>6s}  {'eps':>10s}  {'CPU_Fx':>12s}  {'GPU_Fx':>12s}  "
          f"{'CPU_Fy':>12s}  {'GPU_Fy':>12s}  {'CPU_Fz':>12s}  {'GPU_Fz':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

    for z in z_vals:
        for eps in eps_vals:
            xyz_np = np.array([[eps, eps, z]], dtype=np.float64)
            xyz_cp = cp.asarray(xyz_np)

            f_cpu = pot_cpu.force(xyz_np)[0]
            f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))[0]

            print(f"  {z:6.1f}  {eps:10.1e}  {f_cpu[0]:12.4e}  {f_gpu[0]:12.4e}  "
                  f"{f_cpu[1]:12.4e}  {f_gpu[1]:12.4e}  {f_cpu[2]:12.4e}  {f_gpu[2]:12.4e}")

    # Detailed: for exact z-axis, show potential and all Hessian components
    print(f"\n  Exact z-axis detail: potential + Hessian")
    for z in [1.0, 10.0, 50.0]:
        xyz_np = np.array([[0.0, 0.0, z]], dtype=np.float64)
        xyz_cp = cp.asarray(xyz_np)

        phi_cpu = pot_cpu.potential(xyz_np)[0]
        phi_gpu = float(cp.asnumpy(pot_gpu.potential(xyz_cp))[0])

        f_cpu2, d_cpu = pot_cpu.forceDeriv(xyz_np)
        f_gpu2, d_gpu = pot_gpu.forceDeriv(xyz_cp)
        d_cpu = d_cpu[0]
        d_gpu = cp.asnumpy(d_gpu)[0]

        hess_names = ["dFx/dx","dFy/dy","dFz/dz","dFx/dy","dFy/dz","dFx/dz"]
        print(f"\n  z={z}  Φ_cpu={phi_cpu:.8e}  Φ_gpu={phi_gpu:.8e}  "
              f"ΔΦ={abs(phi_gpu-phi_cpu):.2e}")
        for i, nm in enumerate(hess_names):
            print(f"    {nm}: cpu={d_cpu[i]:12.4e}  gpu={d_gpu[i]:12.4e}  "
                  f"Δ={abs(d_gpu[i]-d_cpu[i]):.2e}")

# ---------------------------------------------------------------------------
# TEST 4: Near-origin behavior
# ---------------------------------------------------------------------------

def test_near_origin(lmax, coef_file):
    print(f"\n{'='*70}")
    print(f"  TEST 4: Near-origin behavior — lmax={lmax}")
    print(f"{'='*70}")

    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)
    r_grid = get_grid(coef_file)

    r_inner = r_grid[0]
    print(f"  Innermost grid point: r_min = {r_inner:.6e} kpc")

    # Points inside, at, and outside the inner grid point
    test_r = [r_inner * 0.01, r_inner * 0.1, r_inner * 0.5,
              r_inner, r_inner * 2.0, r_inner * 10.0]

    print(f"\n  {'r':>12s}  {'r/r_min':>8s}  {'Φ_cpu':>14s}  {'Φ_gpu':>14s}  "
          f"{'|ΔΦ|':>10s}  {'|F_cpu|':>12s}  {'|F_gpu|':>12s}  {'|ΔF|':>10s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*10}")

    for r in test_r:
        # Direction: (1,1,1)/sqrt(3)
        s3 = r / np.sqrt(3)
        xyz_np = np.array([[s3, s3, s3]], dtype=np.float64)
        xyz_cp = cp.asarray(xyz_np)

        phi_cpu = pot_cpu.potential(xyz_np)[0]
        phi_gpu = float(cp.asnumpy(pot_gpu.potential(xyz_cp))[0])
        f_cpu = pot_cpu.force(xyz_np)[0]
        f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))[0]

        fmag_cpu = np.sqrt(np.sum(f_cpu**2))
        fmag_gpu = np.sqrt(np.sum(f_gpu**2))
        delta_f  = np.sqrt(np.sum((f_gpu - f_cpu)**2))

        print(f"  {r:12.4e}  {r/r_inner:8.3f}  {phi_cpu:14.6e}  {phi_gpu:14.6e}  "
              f"{abs(phi_gpu-phi_cpu):10.2e}  {fmag_cpu:12.4e}  {fmag_gpu:12.4e}  {delta_f:10.2e}")

# ---------------------------------------------------------------------------
# TEST 5: Error trend with radius (binned by radial distance)
# ---------------------------------------------------------------------------

def test_radial_trend(lmax, coef_file, N=10000, seed=77):
    print(f"\n{'='*70}")
    print(f"  TEST 5: Error vs radius — lmax={lmax}")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)
    r_grid = get_grid(coef_file)

    # Log-uniform radii across the grid range
    log_r = np.log10(r_grid)
    r_samples = 10**rng.uniform(log_r[0], log_r[-1], N)
    # Random directions
    cos_theta = rng.uniform(-1, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi_az = rng.uniform(0, 2*np.pi, N)
    xyz_np = np.column_stack([
        r_samples * sin_theta * np.cos(phi_az),
        r_samples * sin_theta * np.sin(phi_az),
        r_samples * cos_theta,
    ]).astype(np.float64)
    xyz_cp = cp.asarray(xyz_np)

    phi_cpu = pot_cpu.potential(xyz_np)
    phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
    f_cpu = pot_cpu.force(xyz_np)
    f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))

    abs_phi = np.abs(phi_gpu - phi_cpu)
    rel_phi = abs_phi / (np.abs(phi_cpu) + 1e-30)
    abs_f = np.sqrt(np.sum((f_gpu - f_cpu)**2, axis=1))
    fmag = np.sqrt(np.sum(f_cpu**2, axis=1))
    rel_f = abs_f / (fmag + 1e-30)

    # Bin by log(r)
    r_vals = np.sqrt(np.sum(xyz_np**2, axis=1))
    bins = np.logspace(np.log10(r_grid[0]), np.log10(r_grid[-1]), 12)

    print(f"  {'r_bin':>12s}  {'N':>5s}  {'Φ max_rel':>10s}  {'Φ max_abs':>10s}  "
          f"{'F max_rel':>10s}  {'F max_abs':>10s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for i in range(len(bins)-1):
        mask = (r_vals >= bins[i]) & (r_vals < bins[i+1])
        n = mask.sum()
        if n == 0:
            continue
        mr_phi = np.max(rel_phi[mask])
        ma_phi = np.max(abs_phi[mask])
        mr_f   = np.max(rel_f[mask])
        ma_f   = np.max(abs_f[mask])
        rmid = np.sqrt(bins[i] * bins[i+1])
        print(f"  {rmid:12.2f}  {n:5d}  {mr_phi:10.2e}  {ma_phi:10.2e}  "
              f"{mr_f:10.2e}  {ma_f:10.2e}")

# ---------------------------------------------------------------------------
# TEST 6: lmax sweep — worst-case errors at a few representative radii
# ---------------------------------------------------------------------------

def test_lmax_sweep(coef_files, N_per_r=500, seed=88):
    print(f"\n{'='*70}")
    print(f"  TEST 6: Error trend with lmax")
    print(f"{'='*70}")

    rng = np.random.default_rng(seed)
    results = []

    for lmax, fpath in sorted(coef_files.items()):
        if not os.path.exists(fpath):
            continue
        pot_cpu = agama.Potential(fpath)
        pot_gpu = make_gpu(fpath)
        r_grid = get_grid(fpath)

        # Three radial shells: inner, middle, outer
        r_shells = [r_grid[2], np.sqrt(r_grid[0]*r_grid[-1]), r_grid[-3]]
        labels = ["inner", "mid", "outer"]

        for r_val, lab in zip(r_shells, labels):
            cos_t = rng.uniform(-1, 1, N_per_r)
            sin_t = np.sqrt(1 - cos_t**2)
            phi_a = rng.uniform(0, 2*np.pi, N_per_r)
            xyz_np = np.column_stack([
                r_val * sin_t * np.cos(phi_a),
                r_val * sin_t * np.sin(phi_a),
                r_val * cos_t,
            ]).astype(np.float64)
            xyz_cp = cp.asarray(xyz_np)

            phi_cpu = pot_cpu.potential(xyz_np)
            phi_gpu = cp.asnumpy(pot_gpu.potential(xyz_cp))
            f_cpu = pot_cpu.force(xyz_np)
            f_gpu = cp.asnumpy(pot_gpu.force(xyz_cp))

            rel_phi = float(np.max(np.abs(phi_gpu - phi_cpu) / (np.abs(phi_cpu) + 1e-30)))
            abs_phi = float(np.max(np.abs(phi_gpu - phi_cpu)))
            abs_f   = float(np.max(np.abs(f_gpu - f_cpu)))
            fmag    = float(np.max(np.sqrt(np.sum(f_cpu**2, axis=1))))
            rel_f   = abs_f / (fmag + 1e-30)

            results.append((lmax, lab, r_val, rel_phi, abs_phi, rel_f, abs_f))

    print(f"  {'lmax':>4s}  {'shell':>6s}  {'r':>10s}  {'Φ_rel':>10s}  {'Φ_abs':>10s}  "
          f"{'F_rel':>10s}  {'F_abs':>10s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for lmax, lab, r_val, rp, ap, rf, af in results:
        print(f"  {lmax:4d}  {lab:>6s}  {r_val:10.2f}  {rp:10.2e}  {ap:10.2e}  "
              f"{rf:10.2e}  {af:10.2e}")

# ---------------------------------------------------------------------------
# TEST 7: Hessian ordering verification — does our [xx,yy,zz,xy,yz,xz]
#          match Agama's exact ordering?
# ---------------------------------------------------------------------------

def test_hessian_ordering(coef_file):
    """Verify by finite differencing that our Hessian components are in the
    correct slots matching Agama's convention."""
    lmax = 4  # use a non-trivial lmax
    print(f"\n{'='*70}")
    print(f"  TEST 7: Hessian ordering check (finite difference)")
    print(f"{'='*70}")

    pot_cpu = agama.Potential(coef_file)
    pot_gpu = make_gpu(coef_file)

    # Test at an asymmetric point
    xyz0 = np.array([[3.0, 5.0, 7.0]], dtype=np.float64)
    xyz0_cp = cp.asarray(xyz0)

    # Get Agama Hessian
    _, d_cpu = pot_cpu.forceDeriv(xyz0)
    # Get GPU Hessian
    _, d_gpu = pot_gpu.forceDeriv(xyz0_cp)
    d_gpu = cp.asnumpy(d_gpu)

    # Finite difference the force to get Hessian numerically
    h = 1e-5
    fd_hess = np.zeros(6)  # [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFx/dz]

    # dFi/dxj by central difference
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
    # pair (i, j) means dF_i / dx_j
    for slot, (fi, xj) in enumerate(pairs):
        xyz_p = xyz0.copy(); xyz_p[0, xj] += h
        xyz_m = xyz0.copy(); xyz_m[0, xj] -= h
        fp = pot_cpu.force(xyz_p)[0, fi]
        fm = pot_cpu.force(xyz_m)[0, fi]
        fd_hess[slot] = (fp - fm) / (2*h)

    hess_names = ["dFx/dx","dFy/dy","dFz/dz","dFx/dy","dFy/dz","dFx/dz"]
    print(f"  {'component':>10s}  {'Agama':>14s}  {'GPU':>14s}  {'FD':>14s}  "
          f"{'Agama-FD':>10s}  {'GPU-FD':>10s}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}")
    for i, nm in enumerate(hess_names):
        print(f"  {nm:>10s}  {d_cpu[0,i]:14.6e}  {d_gpu[0,i]:14.6e}  {fd_hess[i]:14.6e}  "
              f"{abs(d_cpu[0,i]-fd_hess[i]):10.2e}  {abs(d_gpu[0,i]-fd_hess[i]):10.2e}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Pick a representative coef file for single-lmax tests
    lmax_single = 4
    fpath_single = COEF_FILES.get(lmax_single)

    available = {l: f for l, f in COEF_FILES.items() if os.path.exists(f)}

    if not available:
        print("No coefficient files found. Exiting.")
        sys.exit(1)

    # Use first available if lmax_single not present
    if lmax_single not in available:
        lmax_single = min(available.keys())
        fpath_single = available[lmax_single]
    else:
        fpath_single = available[lmax_single]

    print(f"\nUsing lmax={lmax_single} for single-lmax tests: {os.path.basename(fpath_single)}")

    test_absolute_errors(lmax_single, fpath_single)
    test_grid_vs_midpoint(lmax_single, fpath_single)
    test_z_axis(lmax_single, fpath_single)
    test_near_origin(lmax_single, fpath_single)
    test_radial_trend(lmax_single, fpath_single)
    test_lmax_sweep(available)
    test_hessian_ordering(fpath_single)

    # Also run z-axis test for lmax=0 (should be perfect)
    if 0 in available:
        test_z_axis(0, available[0])

    print(f"\n{'='*70}")
    print("  AUDIT COMPLETE")
    print(f"{'='*70}")
