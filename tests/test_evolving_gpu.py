"""
Benchmark EvolvingPotentialGPU vs Agama CPU load_agama_evolving_potential.

HDF5: mult_halo_none_4_spl.h5  (301 snapshots, 6.6–13.8 Gyr)
We restrict to [8.0, 11.0] Gyr (~118 snapshots, dt ~ 0.025 Gyr).
"""

import sys, os
import time
import h5py
import numpy as np
import cupy as cp

H5_PATH   = "/mnt/d/Research/firesims_metaldiff/m12i_res7100/potential/10kpc/mult_halo_none_4_spl.h5"
TIME_MIN  = 8.0
TIME_MAX  = 11.0

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------------------------------------------------------
# Load snapshot times and filter to [TIME_MIN, TIME_MAX]
# ---------------------------------------------------------------------------
print("Reading HDF5 metadata …")
with h5py.File(H5_PATH, "r") as f:
    all_times  = np.asarray(f["times"])
    all_groups = sorted([k for k in f.keys() if k != "times"],
                        key=lambda s: int(s.split("_")[1]))

mask       = (all_times >= TIME_MIN) & (all_times <= TIME_MAX)
sel_times  = all_times[mask]
sel_groups = [g for g, m in zip(all_groups, mask) if m]
print(f"  Total snapshots : {len(all_times)}  ({all_times[0]:.3f}–{all_times[-1]:.3f} Gyr)")
print(f"  Filtered [8,11] : {len(sel_times)} snapshots  "
      f"({sel_times[0]:.3f}–{sel_times[-1]:.3f} Gyr)")

# ---------------------------------------------------------------------------
# Build CPU evolving potential via load_agama_evolving_potential
# ---------------------------------------------------------------------------
print("\nBuilding Agama CPU evolving potential …")
from nbody_streams.agama_helper import load_agama_evolving_potential

t0 = time.perf_counter()
cpu_pot = load_agama_evolving_potential(
    H5_PATH,
    times=sel_times.tolist(),
    group_names=sel_groups,
)
t_cpu_build = time.perf_counter() - t0
print(f"  CPU build time : {t_cpu_build:.2f} s")

# ---------------------------------------------------------------------------
# Build GPU evolving potential
# ---------------------------------------------------------------------------
print("\nBuilding GPU evolving potential …")
from gpu_potential import MultipolePotentialGPU, EvolvingPotentialGPU
from nbody_streams.agama_helper._io import _resolve_coef_string
from nbody_streams.agama_helper import read_coefs

t0 = time.perf_counter()
gpu_snaps = []
for grp in sel_groups:
    coef_str = _resolve_coef_string(H5_PATH, grp, "coefs")
    coefs    = read_coefs(coef_str)          # MultipoleCoefs from raw string
    gpu_snaps.append(MultipolePotentialGPU(coefs))
gpu_pot = EvolvingPotentialGPU(gpu_snaps, sel_times)
t_gpu_build = time.perf_counter() - t0
print(f"  GPU build time : {t_gpu_build:.2f} s  ({len(gpu_snaps)} snapshots)")

# ---------------------------------------------------------------------------
# Benchmark at several evaluation times
# ---------------------------------------------------------------------------
EVAL_TIMES = [8.5, 9.0, 9.5, 10.0, 10.5]
N_SIZES    = [1_000, 10_000, 100_000, 1_000_000]
N_WARMUP   = 3
N_REPS     = 10

rng = np.random.default_rng(42)

print("\n" + "=" * 70)
print(f"{'N':>10}  {'t_eval':>6}  {'CPU ms':>8}  {'GPU ms':>8}  {'speedup':>8}")
print("=" * 70)

for N in N_SIZES:
    xyz_np = rng.standard_normal((N, 3)) * 30.0   # kpc-scale positions

    for t_eval in EVAL_TIMES:
        xyz_cp = cp.asarray(xyz_np, dtype=cp.float64)

        # -- CPU warmup + timing --
        for _ in range(N_WARMUP):
            _ = cpu_pot.force(xyz_np, t=t_eval)
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            res_cpu = cpu_pot.force(xyz_np, t=t_eval)
        cpu_ms = (time.perf_counter() - t0) / N_REPS * 1e3

        # -- GPU warmup + timing --
        for _ in range(N_WARMUP):
            _ = gpu_pot.force(xyz_cp, t_eval)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            _ = gpu_pot.force(xyz_cp, t_eval)
        cp.cuda.Stream.null.synchronize()
        gpu_ms = (time.perf_counter() - t0) / N_REPS * 1e3

        speedup = cpu_ms / gpu_ms
        print(f"{N:>10,}  {t_eval:>6.1f}  {cpu_ms:>8.2f}  {gpu_ms:>8.2f}  {speedup:>7.1f}x")

# ---------------------------------------------------------------------------
# Accuracy check at t=9.0, N=10k
# ---------------------------------------------------------------------------
print("\n--- Accuracy check (t=9.0, N=10k, force) ---")
N_acc    = 10_000
xyz_acc  = rng.standard_normal((N_acc, 3)) * 30.0
xyz_acc_cp = cp.asarray(xyz_acc)

f_cpu = cpu_pot.force(xyz_acc, t=9.0)                     # (N, 3) numpy
f_gpu = cp.asnumpy(gpu_pot.force(xyz_acc_cp, 9.0))        # (N, 3) numpy

rel_err = np.abs(f_cpu - f_gpu) / (np.abs(f_cpu) + 1e-30)
print(f"  Force rel err  : median={np.median(rel_err):.2e}  max={np.max(rel_err):.2e}")

phi_cpu = cpu_pot.potential(xyz_acc, t=9.0)
phi_gpu = cp.asnumpy(gpu_pot.potential(xyz_acc_cp, 9.0))
rel_phi = np.abs(phi_cpu - phi_gpu) / (np.abs(phi_cpu) + 1e-30)
print(f"  Phi   rel err  : median={np.median(rel_phi):.2e}  max={np.max(rel_phi):.2e}")

print("\nDone.")
