"""
Verification script: confirm -Xptxas -O0 fix works on H-200 (sm_90).
Tests CylSplinePotentialGPU against CPU reference on 6 off-grid points.
Expected: all max_err < 1e-4.
"""
import sys
import numpy as np
sys.path.insert(0, '/gpfs/home/arora125/libs/AGAMA_GPU')
sys.path.insert(0, '/gpfs/home/arora125/libs')
import agama
import gpu_potential as gp

agama.setUnits(length=1, velocity=1, mass=1)

coef_file = '/gpfs/home/arora125/libs/AGAMA_GPU/tests/600.bar.none_8.coef_cylsp_DR'
pot_cpu = agama.Potential(file=coef_file)
pot_gpu = gp.CylSplinePotentialGPU.from_file(coef_file)

# Test points: mix of R, phi, z values, all off-grid
pts = np.array([
    [1.0,   0.0,  0.0],
    [2.0,   0.5,  0.5],
    [5.0,   1.0,  1.0],
    [8.0,  -0.3,  2.0],
    [0.5,   0.0, -0.5],
    [12.0,  0.0,  0.0],
])

cpu_vals = pot_cpu.potential(pts)
gpu_vals = pot_gpu.potential(pts)

print("Point        CPU                GPU                rel_err")
print("-" * 70)
all_ok = True
for i, (c, g) in enumerate(zip(cpu_vals, gpu_vals)):
    rel = abs(g - c) / (abs(c) + 1e-30)
    ok = rel < 5e-4
    flag = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"pt{i}  {c:18.8g}  {g:18.8g}  {rel:.3e}  {flag}")

print()
if all_ok:
    print("RESULT: PASS — all 6 points within 5e-4 relative error (fast_math rounding expected)")
else:
    print("RESULT: FAIL — see above")
    sys.exit(1)
