"""
Focused bug diagnosis:
  1. Z-axis asymmetry: GPU gives Fx!=0, Fy=0 at (0,0,z)
  2. Hessian: FD of GPU's OWN force vs GPU Hessian kernel
  3. Component-by-component harmonic testing
"""

import sys, os
import numpy as np

import agama
agama.setUnits(mass=1, length=1, velocity=1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cupy as cp
from gpu_potential import MultipolePotentialGPU

FIRE_DIR = "/mnt/d/Research/firesims_metaldiff/m12i_res7100/potential/10kpc"

def make_gpu(coef_file):
    from nbody_streams.agama_helper import read_coefs
    return MultipolePotentialGPU(read_coefs(coef_file))

def make_gpu_selective(coef_file, keep_lm):
    """Build GPU pot keeping only specified (l,m) pairs."""
    from nbody_streams.agama_helper import read_coefs
    mc = read_coefs(coef_file)
    mc2 = mc.zeroed(keep_lm=keep_lm)
    return MultipolePotentialGPU(mc2)

# =========================================================================
# BUG 1: Z-axis asymmetry
# =========================================================================

def test_zaxis_bug():
    print("="*70)
    print("  BUG 1: Z-axis force asymmetry at (0, 0, z)")
    print("="*70)

    f4 = os.path.join(FIRE_DIR, "600.dark.none_4.coef_mul_DR")

    # ----- First: axisymmetric only (l=0,2,4 m=0) -----
    lm_axi = [(0,0), (2,0), (4,0)]
    from nbody_streams.agama_helper import load_agama_potential
    pot_cpu_axi = load_agama_potential(f4, keep_lm_mult=lm_axi)
    pot_gpu_axi = make_gpu_selective(f4, lm_axi)

    print("\n  Axisymmetric (l=0,2,4 m=0 only):")
    for z in [1.0, 10.0, 50.0]:
        xyz_np = np.array([[0.0, 0.0, z]])
        xyz_cp = cp.asarray(xyz_np)
        f_cpu = pot_cpu_axi.force(xyz_np)[0]
        f_gpu = cp.asnumpy(pot_gpu_axi.force(xyz_cp))[0]
        print(f"    z={z:5.1f}  CPU=({f_cpu[0]:+.6e}, {f_cpu[1]:+.6e}, {f_cpu[2]:+.6e})")
        print(f"           GPU=({f_gpu[0]:+.6e}, {f_gpu[1]:+.6e}, {f_gpu[2]:+.6e})")

    # ----- Full lmax=4 potential -----
    pot_cpu = agama.Potential(f4)
    pot_gpu = make_gpu(f4)

    print("\n  Full lmax=4 (all m terms):")
    print("  Testing approach direction dependence at (eps, eps, z):")
    z = 10.0
    for eps in [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        # Approach along (1,1,0) direction
        xyz_11 = np.array([[eps, eps, z]])
        f_cpu_11 = pot_cpu.force(xyz_11)[0]
        f_gpu_11 = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_11)))[0]

        # Approach along (1,0,0) direction
        xyz_10 = np.array([[eps, 0.0, z]])
        f_cpu_10 = pot_cpu.force(xyz_10)[0]
        f_gpu_10 = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_10)))[0]

        # Approach along (0,1,0) direction
        xyz_01 = np.array([[0.0, eps, z]])
        f_cpu_01 = pot_cpu.force(xyz_01)[0]
        f_gpu_01 = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_01)))[0]

        print(f"\n    eps={eps:.0e}:")
        print(f"      (eps,eps,z) CPU: Fx={f_cpu_11[0]:+12.6e} Fy={f_cpu_11[1]:+12.6e} Fz={f_cpu_11[2]:+12.6e}")
        print(f"      (eps,eps,z) GPU: Fx={f_gpu_11[0]:+12.6e} Fy={f_gpu_11[1]:+12.6e} Fz={f_gpu_11[2]:+12.6e}")
        print(f"      (eps, 0, z) CPU: Fx={f_cpu_10[0]:+12.6e} Fy={f_cpu_10[1]:+12.6e} Fz={f_cpu_10[2]:+12.6e}")
        print(f"      (eps, 0, z) GPU: Fx={f_gpu_10[0]:+12.6e} Fy={f_gpu_10[1]:+12.6e} Fz={f_gpu_10[2]:+12.6e}")
        print(f"      ( 0,eps, z) CPU: Fx={f_cpu_01[0]:+12.6e} Fy={f_cpu_01[1]:+12.6e} Fz={f_cpu_01[2]:+12.6e}")
        print(f"      ( 0,eps, z) GPU: Fx={f_gpu_01[0]:+12.6e} Fy={f_gpu_01[1]:+12.6e} Fz={f_gpu_01[2]:+12.6e}")

    # Exact z-axis
    print(f"\n    eps=0 (exact z-axis):")
    xyz_0 = np.array([[0.0, 0.0, z]])
    f_cpu_0 = pot_cpu.force(xyz_0)[0]
    f_gpu_0 = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_0)))[0]
    print(f"      (0, 0, z)   CPU: Fx={f_cpu_0[0]:+12.6e} Fy={f_cpu_0[1]:+12.6e} Fz={f_cpu_0[2]:+12.6e}")
    print(f"      (0, 0, z)   GPU: Fx={f_gpu_0[0]:+12.6e} Fy={f_gpu_0[1]:+12.6e} Fz={f_gpu_0[2]:+12.6e}")

# =========================================================================
# BUG 2: Hessian dFz/dz — FD of GPU's own force
# =========================================================================

def test_hessian_self_consistency():
    print("\n" + "="*70)
    print("  BUG 2: Hessian self-consistency (FD of GPU force vs GPU Hessian)")
    print("="*70)

    f4 = os.path.join(FIRE_DIR, "600.dark.none_4.coef_mul_DR")
    pot_gpu = make_gpu(f4)
    pot_cpu = agama.Potential(f4)

    test_pts = [
        (3.0, 5.0, 7.0),
        (8.0, 3.0, 5.0),
        (1.0, 1.0, 1.0),
        (20.0, 10.0, 15.0),
        (0.5, 0.5, 0.5),
        (50.0, 30.0, 40.0),
    ]

    h = 1e-5
    hess_names = ["dFx/dx","dFy/dy","dFz/dz","dFx/dy","dFy/dz","dFx/dz"]
    # pairs: (force_component, perturbed_axis)
    pairs = [(0,0), (1,1), (2,2), (0,1), (1,2), (0,2)]

    for pt in test_pts:
        xyz0 = np.array([list(pt)], dtype=np.float64)
        xyz0_cp = cp.asarray(xyz0)

        # GPU Hessian
        _, d_gpu = pot_gpu.forceDeriv(xyz0_cp)
        d_gpu = cp.asnumpy(d_gpu)[0]

        # CPU Hessian
        _, d_cpu = pot_cpu.forceDeriv(xyz0)
        d_cpu = d_cpu[0]

        # FD of GPU force
        fd_gpu = np.zeros(6)
        for slot, (fi, xj) in enumerate(pairs):
            xyz_p = xyz0.copy(); xyz_p[0, xj] += h
            xyz_m = xyz0.copy(); xyz_m[0, xj] -= h
            fp = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_p)))[0, fi]
            fm = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_m)))[0, fi]
            fd_gpu[slot] = (fp - fm) / (2*h)

        # FD of CPU force
        fd_cpu = np.zeros(6)
        for slot, (fi, xj) in enumerate(pairs):
            xyz_p = xyz0.copy(); xyz_p[0, xj] += h
            xyz_m = xyz0.copy(); xyz_m[0, xj] -= h
            fp = pot_cpu.force(xyz_p)[0, fi]
            fm = pot_cpu.force(xyz_m)[0, fi]
            fd_cpu[slot] = (fp - fm) / (2*h)

        print(f"\n  Point: {pt}")
        print(f"    {'comp':>10s}  {'GPU_Hess':>12s}  {'GPU_FD':>12s}  {'CPU_Hess':>12s}  "
              f"{'CPU_FD':>12s}  {'GPU:H-FD':>10s}  {'CPU:H-FD':>10s}")
        print(f"    {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")
        for i, nm in enumerate(hess_names):
            gpu_hf = abs(d_gpu[i] - fd_gpu[i])
            cpu_hf = abs(d_cpu[i] - fd_cpu[i])
            flag = " <<<" if gpu_hf > 0.1 * (abs(d_gpu[i]) + 1e-30) else ""
            print(f"    {nm:>10s}  {d_gpu[i]:12.4e}  {fd_gpu[i]:12.4e}  {d_cpu[i]:12.4e}  "
                  f"{fd_cpu[i]:12.4e}  {gpu_hf:10.2e}  {cpu_hf:10.2e}{flag}")

# =========================================================================
# TEST 3: Component-by-component harmonic accuracy
# =========================================================================

def test_per_harmonic():
    print("\n" + "="*70)
    print("  TEST 3: Per-harmonic accuracy")
    print("="*70)

    f4 = os.path.join(FIRE_DIR, "600.dark.none_4.coef_mul_DR")
    from nbody_streams.agama_helper import read_coefs, load_agama_potential
    mc = read_coefs(f4)

    rng = np.random.default_rng(42)
    N = 1000
    xyz_np = rng.uniform(-50, 50, (N, 3)).astype(np.float64)
    xyz_cp = cp.asarray(xyz_np)

    print(f"\n  Individual (l,m) components at N={N} random points:")
    print(f"    {'(l,m)':>8s}  {'Φ max_rel':>10s}  {'Φ max_abs':>10s}  "
          f"{'F max_rel':>10s}  {'F max_abs':>10s}")
    print(f"    {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for l, m in mc.lm_labels:
        lm = [(l, m)]
        # Also include (0,0) as it provides the radial structure
        if (l, m) != (0, 0):
            lm_with_base = [(0, 0), (l, m)]
        else:
            lm_with_base = [(0, 0)]

        try:
            pot_cpu_lm = load_agama_potential(f4, keep_lm_mult=lm_with_base)
            mc_lm = mc.zeroed(keep_lm=lm_with_base)
            pot_gpu_lm = MultipolePotentialGPU(mc_lm)
        except Exception as e:
            print(f"    ({l:2d},{m:+2d})  SKIP: {e}")
            continue

        phi_cpu = pot_cpu_lm.potential(xyz_np)
        phi_gpu = cp.asnumpy(pot_gpu_lm.potential(xyz_cp))
        f_cpu = pot_cpu_lm.force(xyz_np)
        f_gpu = cp.asnumpy(pot_gpu_lm.force(xyz_cp))

        phi_rel = np.max(np.abs(phi_gpu - phi_cpu) / (np.abs(phi_cpu) + 1e-30))
        phi_abs = np.max(np.abs(phi_gpu - phi_cpu))
        f_abs = np.max(np.abs(f_gpu - f_cpu))
        fmag = np.max(np.sqrt(np.sum(f_cpu**2, axis=1)))
        f_rel = f_abs / (fmag + 1e-30)

        print(f"    ({l:2d},{m:+2d})  {phi_rel:10.2e}  {phi_abs:10.2e}  "
              f"{f_rel:10.2e}  {f_abs:10.2e}")

    # Cumulative: add harmonics one at a time
    print(f"\n  Cumulative (adding harmonics to l=0 base):")
    print(f"    {'keep_lm':>30s}  {'Φ max_rel':>10s}  {'F max_rel':>10s}")
    print(f"    {'-'*30}  {'-'*10}  {'-'*10}")

    cumul_sets = [
        [(0,0)],
        [(l,m) for l,m in mc.lm_labels if m == 0],  # axisymmetric
        [(l,m) for l,m in mc.lm_labels if abs(m) <= 1],  # up to m=1
        [(l,m) for l,m in mc.lm_labels if abs(m) <= 2],  # up to m=2
        list(mc.lm_labels),  # all
    ]
    cumul_names = ["l=0 only", "m=0 (axisym)", "|m|<=1", "|m|<=2", "all"]

    for name, lm_set in zip(cumul_names, cumul_sets):
        try:
            pot_cpu_c = load_agama_potential(f4, keep_lm_mult=lm_set)
            mc_c = mc.zeroed(keep_lm=lm_set)
            pot_gpu_c = MultipolePotentialGPU(mc_c)
        except Exception as e:
            print(f"    {name:>30s}  SKIP: {e}")
            continue

        phi_cpu = pot_cpu_c.potential(xyz_np)
        phi_gpu = cp.asnumpy(pot_gpu_c.potential(xyz_cp))
        f_cpu = pot_cpu_c.force(xyz_np)
        f_gpu = cp.asnumpy(pot_gpu_c.force(xyz_cp))

        phi_rel = np.max(np.abs(phi_gpu - phi_cpu) / (np.abs(phi_cpu) + 1e-30))
        f_abs = np.max(np.abs(f_gpu - f_cpu))
        fmag = np.max(np.sqrt(np.sum(f_cpu**2, axis=1)))
        f_rel = f_abs / (fmag + 1e-30)

        lm_str = name
        print(f"    {lm_str:>30s}  {phi_rel:10.2e}  {f_rel:10.2e}")


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_zaxis_bug()
    test_hessian_self_consistency()
    test_per_harmonic()
