# AGAMA_GPU

GPU-accelerated drop-in for [Agama](https://github.com/GalacticDynamics-Oxford/Agama) potential evaluation. Targets N >= 50k particles where GPU throughput gives a 5–10× speedup over Agama CPU.

## Overview

Three files comprise the package:

| File | Purpose |
|------|---------|
| `gpu_potential.py` | Python wrapper, quintic spline builder, unified factory `PotentialGPU` |
| `_multipole_potential_kernel.cu` | CUDA kernels: potential, force, density, hessian |
| `_analytic_potentials.py` | Fused CuPy `ElementwiseKernel` analytic types |

The CUDA module is compiled once at first import via `cp.RawModule` + nvcc.

---

## Supported potential types

### Multipole BFE — `MultipolePotentialGPU`

Quintic C2 splines with Agama log-scaling (replicates `MultipoleInterp1d` from Agama's `potential_multipole.cpp`). Requires coefficient files with dPhi/dr data (`_DR` suffix).

- lmax up to 32
- Inner power-law extrapolation, outer Keplerian extrapolation in density kernel
- Non-uniform radial grid: auto-resampled to log-uniform via `CubicHermiteSpline` when max spacing error > 0.1%

### Analytic — `_analytic_potentials.py`

| Class | Agama type |
|-------|-----------|
| `NFWPotentialGPU` | NFW |
| `PlummerPotentialGPU` | Plummer |
| `HernquistPotentialGPU` | Hernquist |
| `IsochronePotentialGPU` | Isochrone |
| `MiyamotoNagaiPotentialGPU` | MiyamotoNagai |
| `LogHaloPotentialGPU` | Logarithmic (triaxial) |
| `DehnenSphericalPotentialGPU` | Dehnen spherical (gamma in [0,2)) |
| `DiskAnsatzPotentialGPU` | DiskAnsatz |
| `UniformAccelerationGPU` | UniformAcceleration |

### Modifiers

- `ShiftedPotentialGPU`: static offset `(3,)`, cubic-spline center trajectory `(T,4)`, or Hermite-spline `(T,7)`. Linear extrapolation outside time range.
- `ScaledPotentialGPU`: static float, or time-dependent `(T,2)` / `(T,3)` tables.

### Composite types

- `CompositePotentialGPU`: sum of arbitrary GPU components, built automatically when multiple components are passed.
- `EvolvingPotentialGPU`: linear lerp between BFE snapshots at fixed timestamps (matches Agama `Evolving` with `interpLinear=True`).

---

## Factory — `PotentialGPU`

Mirrors `agama.Potential` constructor:

```python
from gpu_potential import PotentialGPU

# From INI file (same as agama.Potential('file.ini'))
pot = PotentialGPU('McMillan17.ini')

# From coefficient file
pot = PotentialGPU('halo.coef_mul')

# Analytic by keyword
pot = PotentialGPU(type='NFW', mass=1e12, scaleRadius=20.0)

# Component dicts (Agama-style)
pot = PotentialGPU(dict(type='NFW', mass=1e12, scaleRadius=20),
                   dict(type='MiyamotoNagai', mass=5e10, scaleRadius=3, scaleHeight=0.3))

# Combine existing GPU objects
pot = pot_halo + pot_disk   # via __add__ on _GPUPotBase

# Time-dependent center
pot = PotentialGPU('halo.ini', center=center_traj)   # (T,4): [t, x, y, z]
```

---

## API

All methods accept CuPy or NumPy arrays, shape `(N,3)` or `(3,)` (scalar squeezed):

```python
pot.potential(xyz, t=0.)     # -> (N,)    [km/s]^2
pot.force(xyz, t=0.)         # -> (N,3)   [km/s]^2/kpc  (= -grad Phi)
pot.density(xyz, t=0.)       # -> (N,)    [Msol/kpc^3]
pot.forceDeriv(xyz, t=0.)    # -> (force (N,3), deriv (N,6))
pot.evalDeriv(xyz, t=0.)     # -> (phi (N,), force (N,3), deriv (N,6))
```

`forceDeriv` returns `deriv = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]`, matching `agama.Potential.forceDeriv` exactly.

Units follow Agama convention: mass = Msol, length = kpc, velocity = km/s.

---

## Accuracy

| Component | phi rel err | force rel err |
|-----------|-------------|---------------|
| Multipole l=0 (monopole) | ~1e-12 | ~1e-12 |
| Multipole l>0 harmonics | ~1e-7 | ~1e-5 |
| Disk composite (DiskAnsatz + Multipole) | ~5e-5 | ~3e-3 |
| Analytic (NFW, Hernquist, etc.) | ~1e-12 | ~1e-12 |

l>0 errors are a numerical floor from log-scaling derivative cancellation — both GPU and Agama CPU hit the same floor. BFE fitting error for N-body data is typically >1%, so this is not a practical issue.

---

## Requirements

- CUDA GPU (tested on NVIDIA L40)
- `cupy >= 10.0` (matching CUDA version)
- `nvcc` accessible on PATH
- `scipy` (quintic spline construction; falls back gracefully if missing)
- `agama` (for `Disk`/`Spheroid`/`King`/`Dehnen` types that use CPU export to build BFE)

---

## File layout

```
AGAMA_GPU/
  gpu_potential.py                  <- main wrapper: PotentialGPU factory + GPU classes
  _multipole_potential_kernel.cu    <- CUDA kernels (potential, force, density, hessian)
  _analytic_potentials.py           <- analytic GPU potentials
  _baseline/
    gpu_potential.py                <- unmodified baseline (regression reference)
  tests/
    test_gpu_bugs.py                <- z-axis, hessian self-consistency, per-harmonic accuracy
    test_gpu_potential.py           <- correctness + timing vs Agama CPU
    test_phase2_analytic.py         <- analytic potential tests
  tech_err.md                       <- architecture decisions and precision notes
```

---

## Known gotchas

- **`Disk` type requires Agama**: `PotentialGPU(type='Disk', ...)` calls `agama.Potential` internally to export a Multipole coefficient file, then wraps it as `DiskAnsatz + MultipolePotentialGPU`.
- **`from_agama()` raises for pure analytic types**: Agama does not export NFW/Plummer/etc. parameters programmatically; these must be constructed directly by keyword.
- **EvolvingPotential interpolation**: GPU `interpolate=True` is linear lerp. Agama default (`interpLinear=False`) is nearest-neighbor. The INI parser maps `interpLinear=True` → GPU linear lerp.
- **lmax limit**: Kernel supports lmax <= 32. Python raises `ValueError` if exceeded.
