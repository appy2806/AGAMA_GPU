"""
test_sanitizers.py
~~~~~~~~~~~~~~~~~~
Minimal single-shot test for compute-sanitizer (memcheck + racecheck).

Design constraints
------------------
* Each GPU kernel is called exactly ONCE — no warmup, no repeat loops.
* N is small (256 particles) so racecheck finishes in seconds per kernel.
* No timing code (no perf_counter loops, no _time_fn helpers).
* Covers all three phases:
    Phase 1 — MultipolePotentialGPU  (from analytic Agama export + precomputed file)
    Phase 2 — Analytic GPU potentials (NFW, Plummer, MiyamotoNagai, LogHalo,
               Dehnen, Isochrone, DiskAnsatz, UniformAcceleration, PotentialGPU)
    Phase 3 — CylSplinePotentialGPU  (from precomputed coef_cylsp file)

Pass/fail criteria
------------------
* All output arrays are finite (no NaN or Inf).
* Max relative error vs Agama CPU is within the stated tolerance.

Usage
-----
    # memcheck
    compute-sanitizer --tool memcheck python test_sanitizers.py

    # racecheck (slow on large N; small N here makes it feasible)
    compute-sanitizer --tool racecheck python test_sanitizers.py
"""

import sys, os, tempfile
import numpy as np
import cupy as cp
import agama

agama.setUnits(mass=1, length=1, velocity=1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpu_potential import MultipolePotentialGPU, CylSplinePotentialGPU, PotentialGPU
from _analytic_potentials import (
    NFWPotentialGPU, PlummerPotentialGPU, HernquistPotentialGPU,
    IsochronePotentialGPU, MiyamotoNagaiPotentialGPU,
    LogHaloPotentialGPU, DehnenSphericalPotentialGPU,
    DiskAnsatzPotentialGPU, UniformAccelerationGPU,
)

_HERE     = os.path.dirname(__file__)
_COEF_MUL = os.path.join(_HERE, "600.dark.none_8.coef_mul_DR")
_COEF_CYL = os.path.join(_HERE, "600.bar.none_8.coef_cylsp_DR")

N = 1024000  # small — keeps racecheck fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pts(N=N, seed=7, lo=-40.0, hi=40.0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(lo, hi, (N, 3)).astype(np.float64)
    xyz[:, :2] += 0.1   # stay off z-axis
    return xyz


def _pts_sphere(N=N, seed=7, rmin=1.0, rmax=80.0):
    """Spherical shell, avoiding z-axis."""
    rng  = np.random.default_rng(seed)
    r    = rng.uniform(rmin, rmax, N)
    cost = rng.uniform(-1, 1, N)
    phi  = rng.uniform(0, 2 * np.pi, N)
    sint = np.sqrt(1 - cost**2)
    xyz  = np.column_stack([r * sint * np.cos(phi),
                            r * sint * np.sin(phi),
                            r * cost]).astype(np.float64)
    xyz[:, :2] += 0.1
    return xyz


def _check_finite(arr, label):
    a = np.asarray(cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr)
    if not np.isfinite(a).all():
        raise AssertionError(f"{label}: non-finite values detected "
                             f"(nan={np.isnan(a).sum()}, inf={np.isinf(a).sum()})")


def _rel_phi(gpu, cpu):
    g, c = np.asarray(cp.asnumpy(gpu)), np.asarray(cpu)
    return float(np.max(np.abs(g - c) / (np.abs(c) + 1e-30)))


def _rel_force(gpu_f, cpu_f):
    fmag = np.max(np.sqrt(np.sum(cpu_f**2, axis=1))) + 1e-30
    return float(np.max(np.sqrt(np.sum((cp.asnumpy(gpu_f) - cpu_f)**2, axis=1))) / fmag)


# ---------------------------------------------------------------------------
# Registry entry: (tag, gpu_pot, cpu_pot_or_None, xyz_np, phi_tol, f_tol)
# ---------------------------------------------------------------------------

def _build_registry():
    entries = []
    xyz = _pts()
    xyz_cyl = _pts_sphere()

    # ------------------------------------------------------------------
    # Phase 2 — analytic kernels (Category A)
    # ------------------------------------------------------------------
    cat_a = [
        ("NFW",
         NFWPotentialGPU(mass=1e12, scaleRadius=20.0),
         agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0),
         xyz, 1e-4, 1e-4),
        ("Plummer",
         PlummerPotentialGPU(mass=1e11, scaleRadius=5.0),
         agama.Potential(type='Plummer', mass=1e11, scaleRadius=5.0),
         xyz, 1e-4, 1e-4),
        ("Hernquist",
         HernquistPotentialGPU(mass=5e11, scaleRadius=10.0),
         agama.Potential(type='Dehnen', mass=5e11, scaleRadius=10.0, gamma=1.0),
         xyz, 1e-4, 1e-4),
        ("Isochrone",
         IsochronePotentialGPU(mass=1e11, scaleRadius=2.0),
         agama.Potential(type='Isochrone', mass=1e11, scaleRadius=2.0),
         xyz, 1e-4, 1e-4),
        ("MiyamotoNagai",
         MiyamotoNagaiPotentialGPU(mass=5e10, scaleRadius=3.0, scaleHeight=0.3),
         agama.Potential(type='MiyamotoNagai', mass=5e10, scaleRadius=3.0, scaleHeight=0.3),
         xyz, 1e-4, 1e-4),
        ("LogHalo",
         LogHaloPotentialGPU(velocity=200.0, coreRadius=0.1),
         agama.Potential(type='Logarithmic', v0=200.0, scaleRadius=0.1),
         xyz, 1e-4, 1e-4),
        ("LogHalo (triaxial)",
         LogHaloPotentialGPU(velocity=200.0, coreRadius=0.1,
                             axisRatioY=0.9, axisRatioZ=0.8),
         agama.Potential(type='Logarithmic', v0=200.0, scaleRadius=0.1,
                         axisRatioY=0.9, axisRatioZ=0.8),
         xyz, 1e-4, 1e-4),
        ("Dehnen g=0",
         DehnenSphericalPotentialGPU(mass=1e12, scaleRadius=5.0, gamma=0.0),
         agama.Potential(type='Dehnen', mass=1e12, scaleRadius=5.0, gamma=0.0),
         xyz, 1e-4, 1e-4),
        ("Dehnen g=1.5",
         DehnenSphericalPotentialGPU(mass=1e12, scaleRadius=5.0, gamma=1.5),
         agama.Potential(type='Dehnen', mass=1e12, scaleRadius=5.0, gamma=1.5),
         xyz, 1e-4, 1e-4),
        # GPU-only: just check finite outputs
        ("DiskAnsatz (exp)",
         DiskAnsatzPotentialGPU(surfaceDensity=1e8, scaleRadius=3.0, scaleHeight=0.3),
         None, xyz, None, None),
        ("DiskAnsatz (sech2)",
         DiskAnsatzPotentialGPU(surfaceDensity=1e8, scaleRadius=3.0, scaleHeight=-0.3),
         None, xyz, None, None),
        ("UniformAccel",
         UniformAccelerationGPU(ax=0.01, ay=-0.02, az=0.005),
         None, xyz, None, None),
    ]

    # Category B — Agama-exported Multipole route
    cat_b = [
        ("Spheroid (sph)",
         PotentialGPU(type='Spheroid', mass=1e12, scaleRadius=20.,
                      alpha=1., beta=4., gamma=1.),
         agama.Potential(type='Spheroid', mass=1e12, scaleRadius=20.,
                         alpha=1., beta=4., gamma=1.),
         xyz, 1e-3, 1e-3),
        ("Spheroid (triaxial)",
         PotentialGPU(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4.,
                      gamma=1., axisRatioY=0.9, axisRatioZ=0.8),
         agama.Potential(type='Spheroid', mass=1e12, scaleRadius=20., alpha=1., beta=4.,
                         gamma=1., axisRatioY=0.9, axisRatioZ=0.8),
         xyz, 1e-3, 1e-3),
        ("Dehnen g=2 (Agama route)",
         PotentialGPU(type='Dehnen', mass=1e12, scaleRadius=5., gamma=2.),
         agama.Potential(type='Dehnen', mass=1e12, scaleRadius=5., gamma=2.),
         xyz, 1e-3, 1e-3),
    ]

    entries.extend(cat_a)
    entries.extend(cat_b)

    # ------------------------------------------------------------------
    # Phase 1 — MultipolePotentialGPU from precomputed file
    # ------------------------------------------------------------------
    if os.path.exists(_COEF_MUL):
        from nbody_streams.agama_helper import read_coefs, load_agama_potential
        mc       = read_coefs(_COEF_MUL)
        pot_cpu  = load_agama_potential(_COEF_MUL)
        pot_gpu  = MultipolePotentialGPU(mc)
        entries.append(("Phase1/Multipole (FIRE dark)", pot_gpu, pot_cpu,
                        xyz, 1e-3, 1e-3))
    else:
        print(f"  [Phase 1] SKIP: {_COEF_MUL} not found")

    # ------------------------------------------------------------------
    # Phase 3 — CylSplinePotentialGPU from precomputed file
    # ------------------------------------------------------------------
    if os.path.exists(_COEF_CYL):
        from nbody_streams.agama_helper import load_agama_potential
        pot_cpu = load_agama_potential(_COEF_CYL)
        pot_gpu = CylSplinePotentialGPU.from_file(_COEF_CYL)
        entries.append(("Phase3/CylSpline (FIRE bar)", pot_gpu, pot_cpu,
                        xyz_cyl, 5e-3, 5e-3))
    else:
        print(f"  [Phase 3] SKIP: {_COEF_CYL} not found")

    return entries


# ---------------------------------------------------------------------------
# Single-shot evaluation check
# ---------------------------------------------------------------------------

def _run_one(tag, gpu_pot, cpu_pot, xyz_np, phi_tol, f_tol):
    """
    Call potential/force/forceDeriv/density each exactly once.
    Check finiteness; check accuracy vs CPU if cpu_pot is provided.
    Returns True on pass, False on fail (prints reason).
    """
    xyz_cp = cp.asarray(xyz_np)
    errors = []

    # --- potential ---
    phi_gpu = gpu_pot.potential(xyz_cp)
    _check_finite(phi_gpu, f"{tag} potential")

    # --- force ---
    f_gpu = gpu_pot.force(xyz_cp)
    _check_finite(f_gpu, f"{tag} force")

    # --- forceDeriv ---
    phi2_gpu, dF_gpu = gpu_pot.forceDeriv(xyz_cp)
    _check_finite(phi2_gpu, f"{tag} forceDeriv/phi")
    _check_finite(dF_gpu,   f"{tag} forceDeriv/hess")

    # --- density (not all potentials implement it; skip gracefully) ---
    try:
        rho_gpu = gpu_pot.density(xyz_cp)
        _check_finite(rho_gpu, f"{tag} density")
    except (AttributeError, NotImplementedError):
        pass

    cp.cuda.Stream.null.synchronize()

    # --- accuracy vs CPU ---
    if cpu_pot is not None and phi_tol is not None:
        phi_cpu = cpu_pot.potential(xyz_np)
        f_cpu   = cpu_pot.force(xyz_np)

        e_phi = _rel_phi(phi_gpu, phi_cpu)
        e_f   = _rel_force(f_gpu, f_cpu)

        if e_phi > phi_tol:
            errors.append(f"phi err {e_phi:.2e} > tol {phi_tol:.2e}")
        if e_f > f_tol:
            errors.append(f"force err {e_f:.2e} > tol {f_tol:.2e}")

    # free pooled allocations immediately
    del xyz_cp, phi_gpu, f_gpu, phi2_gpu, dF_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    W = 72
    print()
    print("=" * W)
    print(f"{'SANITIZER TEST  —  all phases, single-shot, N=256':^{W}}")
    print("=" * W)
    print(f"  {'Model':<42}  {'Status':>8}  {'Notes'}")
    print("-" * W)

    registry = _build_registry()

    n_pass = n_fail = 0
    for tag, gpu_pot, cpu_pot, xyz_np, phi_tol, f_tol in registry:
        try:
            errs = _run_one(tag, gpu_pot, cpu_pot, xyz_np, phi_tol, f_tol)
            if errs:
                status = "FAIL"
                note   = "; ".join(errs)
                n_fail += 1
            else:
                status = "PASS"
                note   = ""
                n_pass += 1
        except Exception as exc:
            status = "ERROR"
            note   = str(exc)
            n_fail += 1

        print(f"  {tag:<42}  {status:>8}  {note}")

    print("=" * W)
    print(f"  Result: {n_pass} passed, {n_fail} failed")
    print()

    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
