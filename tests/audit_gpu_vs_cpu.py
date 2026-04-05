import numpy as np
import cupy as cp
import agama
from nbody_streams.agama_helper import read_coefs
from gpu_potential import MultipolePotentialGPU

agama.setUnits(mass=1, length=1, velocity=1)
FIRE_DIR = '/mnt/d/Research/firesims_metaldiff/m12i_res7100/potential/10kpc'

print("="*60)
print("  AUDIT: Coordinate transforms, Hessian, extrapolation")
print("="*60)

# -----------------------------------------------------------------------
# 1. Verify gradient (force) direction and magnitude
# -----------------------------------------------------------------------
print("\n--- 1. Force = -grad(Phi) verification ---")
# Use finite differences to verify the GPU force is the gradient of the GPU potential

for lmax in [0, 2, 4]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    mc = read_coefs(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)
    pot_cpu = agama.Potential(coef_file)

    eps = 1e-5  # finite difference step
    xyz0 = np.array([[8.0, 3.0, 5.0]])
    phi0 = float(cp.asnumpy(pot_gpu.potential(cp.asarray(xyz0)))[0])
    f_gpu = cp.asnumpy(pot_gpu.force(cp.asarray(xyz0)))[0]
    f_cpu = pot_cpu.force(xyz0)[0]

    fd_force = np.zeros(3)
    for i in range(3):
        xyz_p = xyz0.copy(); xyz_p[0, i] += eps
        xyz_m = xyz0.copy(); xyz_m[0, i] -= eps
        phi_p = float(cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_p)))[0])
        phi_m = float(cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_m)))[0])
        fd_force[i] = -(phi_p - phi_m) / (2*eps)

    print(f"\n  lmax={lmax} at (8,3,5):")
    print(f"    GPU force:   {f_gpu}")
    print(f"    CPU force:   {f_cpu}")
    print(f"    FD force:    {fd_force}")
    for i, c in enumerate("xyz"):
        rel_fd = abs(f_gpu[i] - fd_force[i]) / (abs(f_gpu[i]) + 1e-30)
        rel_cpu = abs(f_gpu[i] - f_cpu[i]) / (abs(f_cpu[i]) + 1e-30)
        print(f"    force[{c}]: GPU vs FD rel={rel_fd:.2e}, GPU vs CPU rel={rel_cpu:.2e}")

# -----------------------------------------------------------------------
# 2. Verify Hessian via finite differences of force
# -----------------------------------------------------------------------
print("\n--- 2. Hessian (forceDeriv) verification ---")

for lmax in [0, 2, 4]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    mc = read_coefs(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)
    pot_cpu = agama.Potential(coef_file)

    xyz0 = np.array([[8.0, 3.0, 5.0]])
    f_gpu_raw, d_gpu_raw = pot_gpu.forceDeriv(cp.asarray(xyz0))
    d_gpu = cp.asnumpy(d_gpu_raw)[0]
    f_cpu_raw, d_cpu = pot_cpu.forceDeriv(xyz0)
    d_cpu = d_cpu[0]  # (6,)

    # Finite difference Hessian of force from GPU
    eps = 1e-5
    fd_hess = np.zeros(6)  # [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
    # dF_i/dx_j indices: [0,0], [1,1], [2,2], [0,1], [1,2], [0,2]
    ij_pairs = [(0,0), (1,1), (2,2), (0,1), (1,2), (0,2)]
    for idx, (fi, xj) in enumerate(ij_pairs):
        xyz_p = xyz0.copy(); xyz_p[0, xj] += eps
        xyz_m = xyz0.copy(); xyz_m[0, xj] -= eps
        fp = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_p)))[0]
        fm = cp.asnumpy(pot_gpu.force(cp.asarray(xyz_m)))[0]
        fd_hess[idx] = (fp[fi] - fm[fi]) / (2*eps)

    labels = ["dFx/dx", "dFy/dy", "dFz/dz", "dFx/dy", "dFy/dz", "dFz/dx"]
    print(f"\n  lmax={lmax} at (8,3,5):")
    for idx in range(6):
        rel_fd = abs(d_gpu[idx] - fd_hess[idx]) / (abs(d_gpu[idx]) + 1e-30)
        rel_cpu = abs(d_gpu[idx] - d_cpu[idx]) / (abs(d_cpu[idx]) + 1e-30)
        print(f"    {labels[idx]:>7s}: GPU={d_gpu[idx]:+.6e}, CPU={d_cpu[idx]:+.6e}, FD={fd_hess[idx]:+.6e}, GPU-CPU={rel_cpu:.2e}, GPU-FD={rel_fd:.2e}")

# -----------------------------------------------------------------------
# 3. Verify Agama forceDeriv ordering
# -----------------------------------------------------------------------
print("\n--- 3. Agama forceDeriv ordering check ---")
# At (r, 0, 0) for a spherical potential, only dFx/dx, dFy/dy, dFz/dz should be nonzero
xyz_x = np.array([[10.0, 0.0, 0.0]])
pot_sph = agama.Potential(f'{FIRE_DIR}/600.dark.none_0.coef_mul_DR')
f_x, d_x = pot_sph.forceDeriv(xyz_x)
print(f"  lmax=0 at (10,0,0):")
print(f"    force: {f_x[0]}")
print(f"    deriv: {d_x[0]}")
print(f"    Expected: [dFx/dx, dFy/dy, dFz/dz, 0, 0, 0]")
print(f"    dFy/dy==dFz/dz? {abs(d_x[0,1]-d_x[0,2])/(abs(d_x[0,1])+1e-30):.2e}")

# -----------------------------------------------------------------------
# 4. Grid-point vs between-grid accuracy
# -----------------------------------------------------------------------
print("\n--- 4. Accuracy at grid points vs between grid points ---")

for lmax in [0, 2, 4, 8]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    mc = read_coefs(coef_file)
    r_grid = np.array(mc.R_grid)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)

    # At grid points (along x-axis)
    xyz_grid = np.column_stack([r_grid, np.zeros(len(r_grid)), np.zeros(len(r_grid))])
    phi_c = pot_cpu.potential(xyz_grid)
    phi_g = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_grid)))
    rel_grid = np.abs(phi_g - phi_c) / (np.abs(phi_c) + 1e-30)

    # Between grid points (midpoints in log)
    logr = np.log(r_grid)
    r_mid = np.exp(0.5*(logr[:-1] + logr[1:]))
    xyz_mid = np.column_stack([r_mid, np.zeros(len(r_mid)), np.zeros(len(r_mid))])
    phi_c2 = pot_cpu.potential(xyz_mid)
    phi_g2 = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_mid)))
    rel_mid = np.abs(phi_g2 - phi_c2) / (np.abs(phi_c2) + 1e-30)

    # At quarter-grid points
    r_qtr = np.exp(0.25*logr[:-1] + 0.75*logr[1:])
    xyz_qtr = np.column_stack([r_qtr, np.zeros(len(r_qtr)), np.zeros(len(r_qtr))])
    phi_c3 = pot_cpu.potential(xyz_qtr)
    phi_g3 = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_qtr)))
    rel_qtr = np.abs(phi_g3 - phi_c3) / (np.abs(phi_c3) + 1e-30)

    print(f"\n  lmax={lmax}:")
    print(f"    At grid pts:    max={rel_grid.max():.2e}, med={np.median(rel_grid):.2e}")
    print(f"    At midpoints:   max={rel_mid.max():.2e}, med={np.median(rel_mid):.2e}")
    print(f"    At 3/4 points:  max={rel_qtr.max():.2e}, med={np.median(rel_qtr):.2e}")

# -----------------------------------------------------------------------
# 5. Extrapolation: what happens outside the radial grid?
# -----------------------------------------------------------------------
print("\n--- 5. Behavior outside radial grid ---")
for lmax in [0, 2]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    mc = read_coefs(coef_file)
    r_grid = np.array(mc.R_grid)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)

    r_inner = r_grid[0] * np.array([0.01, 0.1, 0.5, 1.0])
    r_outer = r_grid[-1] * np.array([1.0, 2.0, 5.0, 10.0])
    r_test = np.concatenate([r_inner, r_outer])

    xyz_test = np.column_stack([r_test, np.zeros(len(r_test)), np.zeros(len(r_test))])
    phi_c = pot_cpu.potential(xyz_test)
    phi_g = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz_test)))

    print(f"\n  lmax={lmax}: R_grid=[{r_grid[0]:.3f}, {r_grid[-1]:.3f}]")
    for i, r in enumerate(r_test):
        rel = abs(phi_g[i] - phi_c[i]) / (abs(phi_c[i]) + 1e-30)
        where = "INNER" if r < r_grid[0] else ("OUTER" if r > r_grid[-1] else "IN")
        print(f"    r={r:10.4f}: CPU={phi_c[i]:+.6e}, GPU={phi_g[i]:+.6e}, rel={rel:.2e} [{where}]")

# -----------------------------------------------------------------------
# 6. Error trend with lmax
# -----------------------------------------------------------------------
print("\n--- 6. Error trend with lmax ---")
rng = np.random.default_rng(42)
N = 500
for lmax in [0, 1, 2, 4, 6, 8, 10]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    if not __import__('os').path.exists(coef_file): continue
    mc = read_coefs(coef_file)
    r_grid = np.array(mc.R_grid)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)

    xyz = rng.uniform(-50, 50, (N, 3)).astype(np.float64)
    phi_c = pot_cpu.potential(xyz)
    phi_g = cp.asnumpy(pot_gpu.potential(cp.asarray(xyz)))
    f_c = pot_cpu.force(xyz)
    f_g = cp.asnumpy(pot_gpu.force(cp.asarray(xyz)))

    rel_phi = np.abs(phi_g - phi_c) / (np.abs(phi_c) + 1e-30)
    rel_f = np.max(np.abs(f_g - f_c) / (np.abs(f_c) + 1e-30), axis=1)

    print(f"  lmax={lmax:2d}: phi max={rel_phi.max():.2e} med={np.median(rel_phi):.2e}, force max={rel_f.max():.2e} med={np.median(rel_f):.2e}")

# -----------------------------------------------------------------------
# 7. Density audit
# -----------------------------------------------------------------------
print("\n--- 7. Density comparison ---")
for lmax in [0, 2, 4]:
    coef_file = f'{FIRE_DIR}/600.dark.none_{lmax}.coef_mul_DR'
    mc = read_coefs(coef_file)
    pot_cpu = agama.Potential(coef_file)
    pot_gpu = MultipolePotentialGPU(mc)

    xyz = rng.uniform(-30, 30, (300, 3)).astype(np.float64)
    rho_c = pot_cpu.density(xyz)
    rho_g = cp.asnumpy(pot_gpu.density(cp.asarray(xyz)))

    floor = max(np.abs(rho_c).mean() * 1e-4, 1e-30)
    rel_rho = np.abs(rho_g - rho_c) / (np.abs(rho_c) + floor)

    print(f"  lmax={lmax}: density max_rel={rel_rho.max():.2e}, med_rel={np.median(rel_rho):.2e}")

print("\n\nDONE: Coordinate transform + Hessian + extrapolation audit complete.")
