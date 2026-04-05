// =============================================================================
//  _multipole_potential_kernel.cu
//  GPU kernels for evaluating Agama Multipole BFE potentials.
//
//  Phase 1 optimizations over baseline potential_kernels.cu:
//    - __ldg() on poly reads  → 128-byte read-only (texture) cache path
//    - fma() throughout Horner evaluation in quintic_eval
//    - Shared memory for lm_l[], lm_m[] arrays (loaded cooperatively per block)
//    - Designed to pair with Python-side radius sorting for L1 cache reuse
//
//  Replicates Agama's MultipoleInterp1d (potential_multipole.cpp) exactly:
//    - Quintic C2 splines with log-scaling of radial coefficients
//    - Legendre recurrence from src/math_sphharm.cpp (PREFACT/COEF arrays)
//    - Angular assembly:  mul = 2*sqrt(pi) for m=0, 2*sqrt(2*pi) for m!=0
//    - Flat harmonic index: c = l*(l+1)+m,  m in [-l, l]
//      cos-modes: m >= 0,  T_m = cos(m*phi)
//      sin-modes: m <  0,  T_m = sin(|m|*phi)
//
//  Log-scaling (Agama convention, when all Phi[l=0] < 0):
//    c=0:  stored = log(invPhi0 - 1/Phi_0)
//          eval:   expX = exp(stored), Phi_0 = 1/(invPhi0 - expX)
//    c>0:  stored = Phi_c / Phi_0
//          eval:   Phi_c = stored * Phi_0   (with full chain rule for derivs)
//
//  Spline polynomial layout (6 coefficients per interval per harmonic):
//    C(s) = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))   s in [0,1]
//    poly[(c * n_intervals + k) * 6 + {0..5}] = {a0..a5}
//    dC/ds      = a1 + s*(2a2 + s*(3a3 + s*(4a4 + 5a5*s)))
//    d2C/ds2    = 2a2 + s*(6a3 + s*(12a4 + 20a5*s))
//
//  Hessian output layout (matches agama.Potential.forceDeriv):
//    hess_out per particle: [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
//    → force derivatives = -hess = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
//
//  Performance notes:
//    - NORM_LM constant table: eliminates sqrt() from Legendre recurrence
//    - Chebyshev trig recurrence: one sincos() per particle, ~4 FMAs per m
//    - Template DO_GRAD: compiler eliminates dead branches
//    - __launch_bounds__(256,2): guides register allocation
//    - Separate x,y,z inputs: coalesced warp reads
//
//  Supported lmax <= 16.
// =============================================================================

#include <math.h>

// ---------------------------------------------------------------------------
// Normalization constants from math_sphharm.cpp lines 26-35.
// PREFACT[m] = |COEF[m]|  where  P_m^m = COEF[m] * sin^m(theta)
// ---------------------------------------------------------------------------

__constant__ double PREFACT[17] = {
    0.2820947917738782,   0.3454941494713355,   0.1287580673410632,
    0.02781492157551894,  0.004214597070904597, 0.0004911451888263050,
    4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
    1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11,
    1.790656309174350e-12, 7.299068453727266e-14, 2.751209457796109e-15,
    9.643748535232993e-17, 3.159120301003413e-18
};

__constant__ double COEF[17] = {
     0.2820947917738782, -0.3454941494713355,  0.3862742020231896,
    -0.4172238236327841,  0.4425326924449826,  -0.4641322034408582,
     0.4830841135800662, -0.5000395635705506,   0.5154289843972843,
    -0.5295529414924496,  0.5426302919442215,  -0.5548257538066191,
     0.5662666637421912, -0.5770536647012670,   0.5872677968601020,
    -0.5969753602424046,  0.6062313441538353
};

// ---------------------------------------------------------------------------
// NORM_LM[c] = fully accumulated normalization factor at flat index c=l*(l+1)+m.
// NORM_LM[l*(l+1)+m] = PREFACT[m] * prod_{l'=m+1}^{l} sqrt((2l'+1)/(2l'-1)*(l'-m)/(l'+m))
// ---------------------------------------------------------------------------

__constant__ double NORM_LM[289] = {
    2.8209479177387820e-01, 0.0000000000000000e+00, 4.8860251190291998e-01, 3.4549414947133550e-01,
    0.0000000000000000e+00, 0.0000000000000000e+00, 6.3078313050504009e-01, 2.5751613468212642e-01,
    1.2875806734106321e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    7.4635266518023091e-01, 2.1545345607610047e-01, 6.8132365095552164e-02, 2.7814921575518941e-02,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    8.4628437532163459e-01, 1.8923493915151207e-01, 4.4603102903819289e-02, 1.1920680675222404e-02,
    4.2145970709045969e-03, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 9.3560257962738902e-01, 1.7081687924064815e-01,
    3.2281355871636185e-02, 6.5894041742255291e-03, 1.5531374585246046e-03, 4.9114518882630498e-04,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 1.0171072362820550e+00, 1.5694305382900609e-01,
    2.4814875652103462e-02, 4.1358126086839097e-03, 7.5509261979682136e-04, 1.6098628745551689e-04,
    4.6472738199140567e-05, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.0925484305920792e+00, 1.4599792520475469e-01, 1.9867801125370677e-02, 2.8097313806030645e-03,
    4.2358294323398304e-04, 7.0597157205663839e-05, 1.3845241622917590e-05, 3.7002964707185449e-06,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.1631066229203197e+00, 1.3707343005165717e-01, 1.6383408517733743e-02, 2.0166581817713460e-03,
    2.6034945176644505e-04, 3.6103972995494481e-05, 5.5709639801456997e-06, 1.0171142129915205e-06,
    2.5427855324788019e-07, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 1.2296226898414842e+00, 1.2961361208406261e-01,
    1.3816857472880173e-02, 1.5075427437116580e-03, 1.7069560266960422e-04, 2.0402026779828733e-05,
    2.6338903315718134e-06, 3.8016932298723471e-07, 6.5198501006896228e-08, 1.5367434061724761e-08,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 1.2927207364566031e+00, 1.2325608605533819e-01,
    1.1860322410551530e-02, 1.1630002963250777e-03, 1.1748077086477522e-04, 1.2383560573501302e-05,
    1.3845241622917595e-06, 1.6789821653966854e-07, 2.2848053291417064e-08, 3.7064436750050733e-09,
    8.2878600120854774e-10, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.3528790949515030e+00, 1.1775301082031181e-01, 1.0327622243750185e-02, 9.2005771562235072e-04,
    8.3989394175505792e-05, 7.9362517769773305e-06, 7.8580601974183673e-07, 8.2831227381848063e-08,
    9.5013934086030256e-09, 1.2266246145761483e-09, 1.8927228717505940e-10, 4.0352987211987470e-11,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.4104739588693913e+00, 1.1292829551195402e-01, 9.1000213811776830e-03, 7.4301363441007791e-04,
    6.1917802867506483e-05, 5.3094077934550720e-06, 4.7299964023276903e-07, 4.4300475192525714e-08,
    4.4300475192525726e-09, 4.8335781164824416e-10, 5.9497233712289827e-11, 8.7723885243451015e-12,
    1.7906563091743501e-12, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 1.4658075357087605e+00, 1.0865288342008114e-01,
    8.0985077759553742e-03, 6.1044799215044456e-04, 4.6819223749156532e-05, 3.6784656225465392e-06,
    2.9836296043507733e-07, 2.5216272546510160e-08, 2.2464441057274933e-09, 2.1419004136424348e-10,
    2.2330855485962905e-11, 2.6317165573035307e-12, 3.7218092476604861e-13, 7.2990684537272655e-14,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 1.5191269449366254e+00, 1.0482971849697341e-01,
    7.2686331775633339e-03, 5.0890611383235819e-04, 3.6166382675602854e-05, 2.6237851683540062e-06,
    1.9556539982650959e-07, 1.5088198164953164e-08, 1.2158416567084866e-09, 1.0349931506416358e-10,
    9.4481515911627671e-12, 9.4481515911627658e-13, 1.0697925061790326e-13, 1.4558032059954717e-14,
    2.7512094577961089e-15, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.5706373285785551e+00, 1.0138420360860752e-01, 6.5717618288470159e-03, 4.2960951030737357e-04,
    2.8451584865249228e-05, 1.9182054603012299e-06, 1.3236875238963260e-07, 9.4070376108560291e-09,
    6.9349601348365155e-10, 5.3504379032941761e-11, 4.3686142545058030e-12, 3.8315281651733123e-13,
    3.6868896959507437e-14, 4.0227264549159520e-15, 5.2820986116545115e-16, 9.6437485352329924e-17,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
    1.6205112036071443e+00, 9.8257924411309761e-02, 5.9797868504124288e-03, 3.6664425597227604e-04,
    2.2738311489089525e-05, 1.4323789865155638e-06, 9.2076808599479035e-08, 6.0713649643097745e-09,
    4.1310406124360974e-10, 2.9210868304105939e-11, 2.1652536169495931e-12, 1.7011839061486178e-13,
    1.4377628163571825e-14, 1.3349292620091874e-15, 1.4071389943855713e-16, 1.7870683099388800e-17,
    3.1591203010034129e-18,
};

// 2*sqrt(pi) and 2*sqrt(2*pi)
#define MUL0  3.5449077018110318
#define MUL1  5.0132565706694072

// ---------------------------------------------------------------------------
//  quintic_eval — evaluate a degree-5 polynomial and its 1st/2nd derivatives.
//
//  Layout: poly[(c * n_intervals + k) * 6 + {0..5}] = {a0, a1, a2, a3, a4, a5}
//    C(s)    = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))
//    C'(s)   = a1 + s*(2a2 + s*(3a3 + s*(4a4 + 5a5*s)))
//    C''(s)  = 2a2 + s*(6a3 + s*(12a4 + 20a5*s))
//  Pass NULL to skip d/d2 outputs.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
quintic_eval(const double* __restrict__ poly,
             int c, int k, int n_intervals, double s,
             double* val, double* dval_ds, double* d2val_ds2)
{
    // __ldg: route through 128-byte read-only (texture) cache.
    // After radius sorting, consecutive threads in a warp land in the same
    // radial interval k, so all 6 doubles for a given (c,k) entry are shared
    // across the warp → high L1 hit rate.
    int base = (c * n_intervals + k) * 6;
    double a0=__ldg(poly+base  ), a1=__ldg(poly+base+1), a2=__ldg(poly+base+2);
    double a3=__ldg(poly+base+3), a4=__ldg(poly+base+4), a5=__ldg(poly+base+5);
    // fma() Horner form: fewer rounding operations + compiler can pipeline MAs.
    *val = fma(s, fma(s, fma(s, fma(s, fma(s, a5, a4), a3), a2), a1), a0);
    if (dval_ds)
        *dval_ds = fma(s, fma(s, fma(s, fma(s, 5.0*a5, 4.0*a4), 3.0*a3), 2.0*a2), a1);
    if (d2val_ds2)
        *d2val_ds2 = fma(s, fma(s, fma(s, 20.0*a5, 12.0*a4), 6.0*a3), 2.0*a2);
}


// ---------------------------------------------------------------------------
//  compute_Plm — fill normalized associated Legendre P_l^|m|(cos theta)
//  for all (l,m) with 0 <= l <= lmax, 0 <= m <= l.
//  Flat index: c = l*(l+1)+m  (positive m only)
// ---------------------------------------------------------------------------

__device__ void
compute_Plm(double cos_theta, double sin_theta, int lmax,
            double* __restrict__ Plm,
            double* __restrict__ dPlm,
            double* __restrict__ d2Plm)
{
    const double POLE_EPS = 1.0e-10;
    bool near_pole = (fabs(sin_theta) < POLE_EPS);

    int n_lm = (lmax + 1) * (lmax + 1);
    for (int i = 0; i < n_lm; i++) {
        Plm[i] = 0.0;
        if (dPlm)  dPlm[i]  = 0.0;
        if (d2Plm) d2Plm[i] = 0.0;
    }

    double sin_pow = 1.0;   // accumulates sin^m

    for (int m = 0; m <= lmax; m++) {
        // Diagonal: P_m^m = COEF[m] * sin^m
        double Pmm, dPmm;
        if (m == 0) {
            Pmm = PREFACT[0]; dPmm = 0.0;
        } else if (m == 1) {
            Pmm  = -sin_theta * PREFACT[1];
            dPmm = -cos_theta * PREFACT[1];
        } else {
            double c_m = COEF[m];
            Pmm  = c_m * sin_pow;
            dPmm = near_pole ? 0.0
                 : (double)m * c_m * sin_pow / sin_theta * cos_theta;
        }

        int cidx = m*(m+1)+m;
        Plm[cidx] = Pmm;
        if (dPlm)  dPlm[cidx]  = dPmm;
        if (d2Plm) {
            double d2;
            if      (m == 0)    d2 = 0.0;
            else if (m == 1)    d2 = sin_theta * PREFACT[1];
            else if (near_pole) d2 = 0.0;
            else {
                double c_m = COEF[m];
                double sp2 = sin_pow / (sin_theta * sin_theta);
                d2 = (double)(m*(m-1)) * c_m * sp2 * cos_theta*cos_theta
                   - (double)m * c_m * sin_pow;
            }
            d2Plm[cidx] = d2;
        }

        if (m == lmax) break;

        // Upward recurrence l = m+1, ..., lmax
        double pf       = PREFACT[m];
        double raw_prev = (pf != 0.0) ? Pmm / pf : 0.0;     // P_m^m / PREFACT[m]
        double raw_cur  = raw_prev * cos_theta * (double)(2*m+1);

        double der_prev = dPlm ? (pf != 0.0 ? dPmm / pf : 0.0) : 0.0;
        double der_cur  = dPlm ? (der_prev*cos_theta*(double)(2*m+1)
                                  - raw_prev*sin_theta*(double)(2*m+1)) : 0.0;

        for (int l = m+1; l <= lmax; l++) {
            if (l > m+1) {
                double inv_lm = 1.0 / (double)(l - m);
                double new_raw = ((double)(2*l-1)*cos_theta*raw_cur
                                  - (double)(l+m-1)*raw_prev) * inv_lm;
                double new_der = dPlm
                    ? (((double)(2*l-1)*(cos_theta*der_cur - sin_theta*raw_cur)
                        - (double)(l+m-1)*der_prev) * inv_lm)
                    : 0.0;
                raw_prev = raw_cur;
                der_prev = der_cur;
                raw_cur  = new_raw;
                der_cur  = new_der;
            }
            int c2 = l*(l+1)+m;
            Plm[c2]  = raw_cur * NORM_LM[c2];
            if (dPlm) dPlm[c2] = der_cur * NORM_LM[c2];

            if (d2Plm) {
                double d2;
                if (near_pole) {
                    d2 = 0.0;
                } else {
                    double P_val = Plm[c2];
                    double P_der = dPlm ? dPlm[c2] : 0.0;
                    double m2_s2 = (double)(m*m) / (sin_theta * sin_theta);
                    d2 = (cos_theta / (-sin_theta)) * P_der
                       - ((double)(l*(l+1)) - m2_s2) * P_val;
                }
                d2Plm[c2] = d2;
            }
        }

        sin_pow *= sin_theta;
    }
}


// ---------------------------------------------------------------------------
//  multipole_eval_device<DO_GRAD>
//  One thread per particle: potential + (optionally) gradient.
//  Handles quintic splines and Agama log-scaling.
// ---------------------------------------------------------------------------

template<bool DO_GRAD>
__device__ void
multipole_eval_device(
    int tid,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out,
    double* __restrict__ grad_out,
    int N)
{
    if (tid >= N) return;

    double px = x[tid], py = y[tid], pz = z[tid];
    double r2   = px*px + py*py + pz*pz;
    double r    = sqrt(r2);
    double Rcyl = sqrt(px*px + py*py);

    if (r < 1.0e-300) {
        // At exact origin: Phi = W (finite), force = 0
        phi_out[tid] = MUL0 * inner_W * NORM_LM[0];
        if (DO_GRAD) { grad_out[3*tid]=0.; grad_out[3*tid+1]=0.; grad_out[3*tid+2]=0.; }
        return;
    }

    double logr = log(r);

    // Inner extrapolation: for r < r_min, use power-law model
    // Phi_0(r) = U * (r/r0)^s + W; higher harmonics scale as (r/r0)^l
    if (logr < logr_min) {
        double dlr = logr - logr_min;  // negative
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        double Phi0 = inner_U * r_ratio_s + inner_W;
        phi_out[tid] = MUL0 * Phi0 * NORM_LM[0];  // l=0 only (higher l vanish as r^l → 0)
        if (DO_GRAD) {
            // dPhi/d(logr) = U * s * (r/r0)^s
            double dPhi0_dlr = inner_U * inner_s * r_ratio_s;
            double inv_r2 = 1.0 / r2;
            // gradient = dPhi/dlogr * (x/r²) for each component
            double g = MUL0 * dPhi0_dlr * NORM_LM[0];
            grad_out[3*tid+0] = g * (px * inv_r2);
            grad_out[3*tid+1] = g * (py * inv_r2);
            grad_out[3*tid+2] = g * (pz * inv_r2);
        }
        return;
    }

    int k = (int)((logr - logr_min) * inv_dlogr);
    if (k < 0) k = 0;
    if (k >= n_intervals) {
        // Outer extrapolation (l=0 only): Phi(r) = W*(r/r_max)^(-1) + U*(r/r_max)^s
        double logr_max = logr_min + n_intervals * dlogr;
        double dlr      = logr - logr_max;   // > 0
        double exp_neg  = exp(-dlr);
        double exp_s    = exp(outer_s * dlr);
        double Phi0     = outer_W * exp_neg + outer_U * exp_s;
        phi_out[tid]    = MUL0 * Phi0 * NORM_LM[0];
        if (DO_GRAD) {
            double dPhi0_dlr = -outer_W * exp_neg + outer_s * outer_U * exp_s;
            double g = MUL0 * dPhi0_dlr * NORM_LM[0];
            double inv_r2 = 1.0 / r2;
            grad_out[3*tid+0] = g * px * inv_r2;
            grad_out[3*tid+1] = g * py * inv_r2;
            grad_out[3*tid+2] = g * pz * inv_r2;
        }
        return;
    }
    double s = (logr - (logr_min + k * dlogr)) * inv_dlogr;
    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;

    double cos_theta = pz   / r;
    double sin_theta = Rcyl / r;
    double phi_az    = atan2(py, px);

    // Chebyshev trig recurrence
    double cos_mf[17], sin_mf[17];
    cos_mf[0] = 1.0; sin_mf[0] = 0.0;
    {
        double cph, sph;
        sincos(phi_az, &sph, &cph);
        if (lmax >= 1) { cos_mf[1] = cph; sin_mf[1] = sph; }
        for (int m = 2; m <= lmax; m++) {
            cos_mf[m] = cph*cos_mf[m-1] - sph*sin_mf[m-1];
            sin_mf[m] = sph*cos_mf[m-1] + cph*sin_mf[m-1];
        }
    }

    const int NLM_MAX = (16+1)*(16+1);
    double Plm_arr[NLM_MAX];
    double dPlm_arr[NLM_MAX];
    compute_Plm(cos_theta, sin_theta, lmax,
                Plm_arr,
                DO_GRAD ? dPlm_arr : (double*)0,
                (double*)0);

    double Phi            = 0.0;
    double dPhi_dlogr     = 0.0;
    double dPhi_dtheta    = 0.0;
    double dPhi_dphi_os   = 0.0;  // dPhi/dphi / sin_theta (pole-safe)

    // -----------------------------------------------------------------------
    // c=0 (l=0, m=0): evaluate and apply log-scaling inverse transform
    // -----------------------------------------------------------------------
    double C0_sc, dC0_sc_ds = 0.0;
    quintic_eval(poly, 0, k, n_intervals, s,
                 &C0_sc,
                 DO_GRAD ? &dC0_sc_ds : (double*)0,
                 (double*)0);

    double C0_val     = C0_sc;           // actual Phi_0 after unscaling
    double dC0_dlr    = 0.0;             // actual dPhi_0/d(logr) after unscaling

    if (log_scaling) {
        double expX    = exp(C0_sc);
        double Phi0    = 1.0 / (invPhi0 - expX);
        double dPhidX  = Phi0 * Phi0 * expX;
        if (DO_GRAD) {
            dC0_dlr = dPhidX * (dC0_sc_ds * inv_dlogr);
        }
        C0_val = Phi0;
    } else {
        if (DO_GRAD) dC0_dlr = dC0_sc_ds * inv_dlogr;
    }

    // c=0 angular contribution: l=0, m=0, Ylm=NORM_LM[0], Tlm=1
    Phi += MUL0 * C0_val * Plm_arr[0];
    if (DO_GRAD) {
        dPhi_dlogr  += MUL0 * dC0_dlr    * Plm_arr[0];
        dPhi_dtheta += MUL0 * C0_val     * dPlm_arr[0];
        // dPhi_dphi: m=0 → dTlm/dphi = 0
    }

    // -----------------------------------------------------------------------
    // c>0: evaluate spline and apply log-scaling if needed
    // -----------------------------------------------------------------------
    for (int c = 1; c < n_lm; c++) {
        int l    = lm_l[c];
        int m    = lm_m[c];
        int absm = m < 0 ? -m : m;
        int legidx = l*(l+1)+absm;

        double Cc_sc, dCc_sc_ds = 0.0;
        quintic_eval(poly, c, k, n_intervals, s,
                     &Cc_sc,
                     DO_GRAD ? &dCc_sc_ds : (double*)0,
                     (double*)0);

        double Cc_val  = Cc_sc;
        double dCc_dlr = 0.0;

        if (log_scaling) {
            // Cc_sc = Phi_c / Phi_0 → actual = Cc_sc * Phi_0
            double dCc_sc_dlr = dCc_sc_ds * inv_dlogr;
            if (DO_GRAD) {
                dCc_dlr = dCc_sc_dlr * C0_val + Cc_sc * dC0_dlr;
            }
            Cc_val = Cc_sc * C0_val;
        } else {
            if (DO_GRAD) dCc_dlr = dCc_sc_ds * inv_dlogr;
        }

        double mul  = (absm == 0) ? MUL0 : MUL1;
        double Ylm  = Plm_arr[legidx];
        double Tlm  = (m >= 0) ? cos_mf[absm] : sin_mf[absm];

        Phi += mul * Cc_val * Ylm * Tlm;

        if (DO_GRAD) {
            double dTlm = (m >= 0) ? -(double)absm * sin_mf[absm]
                                    :  (double)absm * cos_mf[absm];
            dPhi_dlogr  += mul * dCc_dlr        * Ylm * Tlm;
            dPhi_dtheta += mul * Cc_val          * dPlm_arr[legidx] * Tlm;

            // Accumulate dPhi/dphi / sin_theta (pole-safe).
            // P_l^m / sin_theta: finite for m=1 (limit = dP/dtheta), 0 for m>=2, N/A for m=0.
            double Plm_os;
            if (absm == 0) {
                Plm_os = 0.0;
            } else if (sin_theta > 1.0e-10) {
                Plm_os = Ylm / sin_theta;
            } else {
                Plm_os = (absm == 1) ? dPlm_arr[legidx] : 0.0;
            }
            dPhi_dphi_os += mul * Cc_val * Plm_os * dTlm;
        }
    }

    phi_out[tid] = Phi;

    if (DO_GRAD) {
        double inv_r  = 1.0 / r;
        double inv_r2 = inv_r * inv_r;
        double cos_phi = (Rcyl > 1.0e-300) ? px / Rcyl : 1.0;
        double sin_phi = (Rcyl > 1.0e-300) ? py / Rcyl : 0.0;

        // Gradient in Cartesian using pole-safe phi contribution:
        //   dPhi/dphi * dphi/dx_i = (dPhi/dphi / sin_theta) * (-sin_phi / r)  for x
        //                         = (dPhi/dphi / sin_theta) * ( cos_phi / r)  for y
        // Since R = r*sin_theta, this is equivalent to the standard formula
        // away from the pole, but finite at sin_theta=0.
        grad_out[3*tid+0] = dPhi_dlogr*(px*inv_r2)
                          + dPhi_dtheta*(cos_theta*cos_phi*inv_r)
                          + dPhi_dphi_os*(-sin_phi*inv_r);
        grad_out[3*tid+1] = dPhi_dlogr*(py*inv_r2)
                          + dPhi_dtheta*(cos_theta*sin_phi*inv_r)
                          + dPhi_dphi_os*(cos_phi*inv_r);
        grad_out[3*tid+2] = dPhi_dlogr*(pz*inv_r2)
                          + dPhi_dtheta*(-sin_theta*inv_r);
    }
}


// ---- C-linkage __global__ wrappers ----

extern "C" __global__ __launch_bounds__(256,2) void
multipole_potential_kernel(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l, const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out, double* __restrict__ grad_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    multipole_eval_device<false>(tid, x,y,z,poly,logr_min,dlogr,inv_dlogr,
                                 n_intervals,n_lm,lmax,lm_l,lm_m,
                                 log_scaling,invPhi0,inner_s,inner_U,inner_W,
                                 outer_s,outer_U,outer_W,
                                 phi_out,grad_out,N);
}

extern "C" __global__ __launch_bounds__(256,2) void
multipole_force_kernel(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l, const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out, double* __restrict__ grad_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    multipole_eval_device<true>(tid, x,y,z,poly,logr_min,dlogr,inv_dlogr,
                                n_intervals,n_lm,lmax,lm_l,lm_m,
                                log_scaling,invPhi0,inner_s,inner_U,inner_W,
                                outer_s,outer_U,outer_W,
                                phi_out,grad_out,N);
}


// ---------------------------------------------------------------------------
//  multipole_hess_kernel — potential + gradient + Hessian (6 components)
//
//  Output layout matches agama.Potential.forceDeriv:
//    hess_out[6*i + {0..5}] = [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
//  → force derivatives = −H = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256, 2) void
multipole_hess_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out,
    double* __restrict__ grad_out,
    double* __restrict__ hess_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double px = x[tid], py = y[tid], pz = z[tid];
    double r2   = px*px + py*py + pz*pz;
    double r    = sqrt(r2);
    double Rcyl = sqrt(px*px + py*py);

    if (r < 1.0e-300) {
        phi_out[tid] = MUL0 * inner_W * NORM_LM[0];
        for (int i=0;i<3;i++) grad_out[3*tid+i]=0.;
        for (int i=0;i<6;i++) hess_out[6*tid+i]=0.;
        return;
    }

    double logr = log(r);

    // Inner extrapolation (same as force kernel, plus Hessian from radial-only l=0)
    if (logr < logr_min) {
        double dlr = logr - logr_min;
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        double Phi0 = inner_U * r_ratio_s + inner_W;
        phi_out[tid] = MUL0 * Phi0 * NORM_LM[0];
        double dPhi0_dlr = inner_U * inner_s * r_ratio_s;
        double d2Phi0_dlr2 = inner_U * inner_s * inner_s * r_ratio_s;
        double g = MUL0 * dPhi0_dlr * NORM_LM[0];
        double inv_r2 = 1.0 / r2;
        grad_out[3*tid+0] = g * px * inv_r2;
        grad_out[3*tid+1] = g * py * inv_r2;
        grad_out[3*tid+2] = g * pz * inv_r2;
        // Hessian: d2Phi/dxi dxj = (d2Phi/dlogr^2 - dPhi/dlogr)*xi*xj/r^4
        //                         + dPhi/dlogr * delta_ij / r^2
        double A = MUL0 * NORM_LM[0] * (d2Phi0_dlr2 - dPhi0_dlr) * inv_r2 * inv_r2;
        double B = MUL0 * NORM_LM[0] * dPhi0_dlr * inv_r2;
        double coords[3] = {px, py, pz};
        int ii[6]={0,1,2,0,1,0}, jj[6]={0,1,2,1,2,2};
        for (int p=0;p<6;p++)
            hess_out[6*tid+p] = A*coords[ii[p]]*coords[jj[p]] + (ii[p]==jj[p] ? B : 0.0);
        return;
    }

    int k = (int)((logr - logr_min) * inv_dlogr);
    if (k < 0) k = 0;
    if (k >= n_intervals) {
        // Outer extrapolation (l=0 only): Phi(r) = W*(r/r_max)^(-1) + U*(r/r_max)^s
        double logr_max  = logr_min + n_intervals * dlogr;
        double dlr       = logr - logr_max;   // > 0
        double exp_neg   = exp(-dlr);
        double exp_s     = exp(outer_s * dlr);
        double Phi0      = outer_W * exp_neg + outer_U * exp_s;
        double dPhi0     = -outer_W * exp_neg + outer_s * outer_U * exp_s;
        double d2Phi0    = outer_W * exp_neg + outer_s * outer_s * outer_U * exp_s;
        phi_out[tid] = MUL0 * Phi0 * NORM_LM[0];
        double inv_r2 = 1.0 / r2;
        double g = MUL0 * dPhi0 * NORM_LM[0];
        grad_out[3*tid+0] = g * px * inv_r2;
        grad_out[3*tid+1] = g * py * inv_r2;
        grad_out[3*tid+2] = g * pz * inv_r2;
        double A = MUL0 * NORM_LM[0] * (d2Phi0 - dPhi0) * inv_r2 * inv_r2;
        double B = MUL0 * NORM_LM[0] * dPhi0 * inv_r2;
        double coords[3] = {px, py, pz};
        int ii[6]={0,1,2,0,1,0}, jj[6]={0,1,2,1,2,2};
        for (int p=0;p<6;p++)
            hess_out[6*tid+p] = A*coords[ii[p]]*coords[jj[p]] + (ii[p]==jj[p] ? B : 0.0);
        return;
    }
    double s = (logr - (logr_min + k * dlogr)) * inv_dlogr;
    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;

    double cos_theta = pz   / r;
    double sin_theta = Rcyl / r;
    double phi_az    = atan2(py, px);

    double cos_mf[17], sin_mf[17];
    cos_mf[0] = 1.0; sin_mf[0] = 0.0;
    {
        double cph, sph;
        sincos(phi_az, &sph, &cph);
        if (lmax >= 1) { cos_mf[1] = cph; sin_mf[1] = sph; }
        for (int m = 2; m <= lmax; m++) {
            cos_mf[m] = cph*cos_mf[m-1] - sph*sin_mf[m-1];
            sin_mf[m] = sph*cos_mf[m-1] + cph*sin_mf[m-1];
        }
    }

    const int NLM_MAX = (16+1)*(16+1);
    double Plm_arr[NLM_MAX], dPlm_arr[NLM_MAX], d2Plm_arr[NLM_MAX];
    compute_Plm(cos_theta, sin_theta, lmax, Plm_arr, dPlm_arr, d2Plm_arr);

    double Phi=0., dPhi_dlr=0., d2Phi_dlr2=0.;
    double dPhi_dth=0., d2Phi_dth2=0., d2Phi_dlr_dth=0.;
    double dPhi_dph=0., d2Phi_dph2=0., d2Phi_dlr_dph=0., d2Phi_dth_dph=0.;
    double dPhi_dph_os=0.;  // dPhi/dphi / sin_theta (pole-safe, for gradient)

    // -----------------------------------------------------------------------
    // c=0: evaluate + log-scaling inverse transform (with 2nd derivatives)
    // -----------------------------------------------------------------------
    double C0_sc, dC0_sc_ds, d2C0_sc_ds2;
    quintic_eval(poly, 0, k, n_intervals, s, &C0_sc, &dC0_sc_ds, &d2C0_sc_ds2);

    double C0_val      = C0_sc;
    double dC0_dlr     = dC0_sc_ds * inv_dlogr;
    double d2C0_dlr2   = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;

    if (log_scaling) {
        double expX   = exp(C0_sc);
        double Phi0   = 1.0 / (invPhi0 - expX);
        double dPhidX = Phi0 * Phi0 * expX;
        // d2Phi/dX2 = dPhidX * (1 + 2*Phi0*expX) ... from Agama:
        // d2Phi0 = dPhidX*(d2C0_dlr2 + dC0_dlr^2 * Phi0*(invPhi0+expX))
        double d2Phi0_dlr2 = dPhidX * (d2C0_dlr2
                             + dC0_dlr * dC0_dlr * Phi0 * (invPhi0 + expX));
        double dPhi0_dlr   = dPhidX * dC0_dlr;
        C0_val    = Phi0;
        dC0_dlr   = dPhi0_dlr;
        d2C0_dlr2 = d2Phi0_dlr2;
    }

    // c=0 angular contribution (l=0, m=0)
    {
        double Ylm   = Plm_arr[0];
        double dYlm  = dPlm_arr[0];
        double d2Ylm = d2Plm_arr[0];
        // Tlm=1, dTlm=0, d2Tlm=0 for m=0
        Phi              += MUL0 * C0_val    * Ylm;
        dPhi_dlr         += MUL0 * dC0_dlr   * Ylm;
        d2Phi_dlr2       += MUL0 * d2C0_dlr2 * Ylm;
        dPhi_dth         += MUL0 * C0_val    * dYlm;
        d2Phi_dth2       += MUL0 * C0_val    * d2Ylm;
        d2Phi_dlr_dth    += MUL0 * dC0_dlr   * dYlm;
        // dPhi_dph, d2Phi_dph2, d2Phi_dlr_dph, d2Phi_dth_dph all 0 for m=0
    }

    // -----------------------------------------------------------------------
    // c>0 loop
    // -----------------------------------------------------------------------
    for (int c = 1; c < n_lm; c++) {
        int l    = lm_l[c];
        int m    = lm_m[c];
        int absm = m < 0 ? -m : m;
        int legidx = l*(l+1)+absm;

        double Cc_sc, dCc_sc_ds, d2Cc_sc_ds2;
        quintic_eval(poly, c, k, n_intervals, s, &Cc_sc, &dCc_sc_ds, &d2Cc_sc_ds2);
        double dCc_sc_dlr  = dCc_sc_ds  * inv_dlogr;
        double d2Cc_sc_dlr2= d2Cc_sc_ds2* inv_dlogr * inv_dlogr;

        double Cc_val    = Cc_sc;
        double dCc_dlr   = dCc_sc_dlr;
        double d2Cc_dlr2 = d2Cc_sc_dlr2;

        if (log_scaling) {
            // chain rule: actual = scaled * Phi_0
            d2Cc_dlr2 = d2Cc_sc_dlr2*C0_val + 2.0*dCc_sc_dlr*dC0_dlr + Cc_sc*d2C0_dlr2;
            dCc_dlr   = dCc_sc_dlr*C0_val + Cc_sc*dC0_dlr;
            Cc_val    = Cc_sc * C0_val;
        }

        double mul   = (absm==0) ? MUL0 : MUL1;
        double Ylm   = Plm_arr[legidx];
        double dYlm  = dPlm_arr[legidx];
        double d2Ylm = d2Plm_arr[legidx];
        double Tlm   = (m>=0) ? cos_mf[absm] : sin_mf[absm];
        double dTlm  = (m>=0) ? -(double)absm*sin_mf[absm] : (double)absm*cos_mf[absm];
        double d2Tlm = -(double)(absm*absm) * Tlm;

        double YT  = Ylm*Tlm,  dYT = dYlm*Tlm, d2YT = d2Ylm*Tlm;
        double YdT = Ylm*dTlm, dYdT= dYlm*dTlm, Yd2T = Ylm*d2Tlm;

        Phi              += mul*Cc_val    *YT;
        dPhi_dlr         += mul*dCc_dlr  *YT;
        d2Phi_dlr2       += mul*d2Cc_dlr2*YT;
        dPhi_dth         += mul*Cc_val   *dYT;
        d2Phi_dth2       += mul*Cc_val   *d2YT;
        d2Phi_dlr_dth    += mul*dCc_dlr  *dYT;
        dPhi_dph         += mul*Cc_val   *YdT;
        d2Phi_dph2       += mul*Cc_val   *Yd2T;
        d2Phi_dlr_dph    += mul*dCc_dlr  *YdT;
        d2Phi_dth_dph    += mul*Cc_val   *dYdT;

        // Pole-safe: P_l^m / sin_theta for gradient phi contribution
        double Plm_os;
        if (absm == 0) {
            Plm_os = 0.0;
        } else if (sin_theta > 1.0e-10) {
            Plm_os = Ylm / sin_theta;
        } else {
            Plm_os = (absm == 1) ? dPlm_arr[legidx] : 0.0;
        }
        dPhi_dph_os += mul*Cc_val * Plm_os * dTlm;
    }

    phi_out[tid] = Phi;

    // Cartesian gradient (pole-safe: uses dPhi_dph_os = dPhi/dphi / sin_theta)
    double inv_r  = 1.0/r, inv_r2 = inv_r*inv_r;
    double inv_R  = (Rcyl>1.e-300) ? 1./Rcyl : 0.;
    double cos_phi= (Rcyl>1.e-300) ? px/Rcyl : 1.;
    double sin_phi= (Rcyl>1.e-300) ? py/Rcyl : 0.;

    double dlr_dx=px*inv_r2, dlr_dy=py*inv_r2, dlr_dz=pz*inv_r2;
    double dth_dx= cos_theta*cos_phi*inv_r;
    double dth_dy= cos_theta*sin_phi*inv_r;
    double dth_dz=-sin_theta*inv_r;
    // Pole-safe gradient: dPhi/dphi * dphi/dx = (dPhi_dph/sin_theta) * (-sin_phi/r)
    grad_out[3*tid+0]=dPhi_dlr*dlr_dx+dPhi_dth*dth_dx+dPhi_dph_os*(-sin_phi*inv_r);
    grad_out[3*tid+1]=dPhi_dlr*dlr_dy+dPhi_dth*dth_dy+dPhi_dph_os*(cos_phi*inv_r);
    grad_out[3*tid+2]=dPhi_dlr*dlr_dz+dPhi_dth*dth_dz;

    // Cartesian Hessian — full chain rule
    // (uses standard inv_R formulation; at exact pole, phi-dependent terms → 0)
    double dph_dx=-sin_phi*inv_R, dph_dy=cos_phi*inv_R;
    double inv_r3=inv_r2*inv_r, inv_r4=inv_r2*inv_r2, inv_r5=inv_r4*inv_r;
    double inv_R2=inv_R*inv_R;

    double d2lr_xx=inv_r2-2.*px*px*inv_r4, d2lr_xy=-2.*px*py*inv_r4,
           d2lr_xz=-2.*px*pz*inv_r4,       d2lr_yy=inv_r2-2.*py*py*inv_r4,
           d2lr_yz=-2.*py*pz*inv_r4,        d2lr_zz=inv_r2-2.*pz*pz*inv_r4;

    double df_dx=-pz*px*inv_r3, df_dy=-pz*py*inv_r3, df_dz=inv_r-pz*pz*inv_r3;
    double p3z=3.*pz;
    double d2f_xx=-pz*inv_r3+p3z*px*px*inv_r5, d2f_xy=p3z*px*py*inv_r5,
           d2f_xz=-px*inv_r3+p3z*px*pz*inv_r5,  d2f_yy=-pz*inv_r3+p3z*py*py*inv_r5,
           d2f_yz=-py*inv_r3+p3z*py*pz*inv_r5,   d2f_zz=-3.0*pz*inv_r3+p3z*pz*pz*inv_r5;

    double inv_sth=(sin_theta>1.e-14)?1./sin_theta:0.;
    double c_o_s3=cos_theta*inv_sth*inv_sth*inv_sth;

    double d2th_xx=-d2f_xx*inv_sth-df_dx*df_dx*c_o_s3;
    double d2th_xy=-d2f_xy*inv_sth-df_dx*df_dy*c_o_s3;
    double d2th_xz=-d2f_xz*inv_sth-df_dx*df_dz*c_o_s3;
    double d2th_yy=-d2f_yy*inv_sth-df_dy*df_dy*c_o_s3;
    double d2th_yz=-d2f_yz*inv_sth-df_dy*df_dz*c_o_s3;
    double d2th_zz=-d2f_zz*inv_sth-df_dz*df_dz*c_o_s3;

    double d2ph_xx=0.,d2ph_xy=0.,d2ph_yy=0.;
    if (Rcyl>1.e-14) {
        double inv_R4=inv_R2*inv_R2;
        d2ph_xx= 2.*px*py*inv_R4;
        d2ph_xy=(py*py-px*px)*inv_R4;
        d2ph_yy=-2.*px*py*inv_R4;
    }

    double dlr[3]={dlr_dx,dlr_dy,dlr_dz};
    double dth[3]={dth_dx,dth_dy,dth_dz};
    double dph[3]={dph_dx,dph_dy,0.};
    double d2lr[6]={d2lr_xx,d2lr_xy,d2lr_xz,d2lr_yy,d2lr_yz,d2lr_zz};
    double d2th[6]={d2th_xx,d2th_xy,d2th_xz,d2th_yy,d2th_yz,d2th_zz};
    double d2ph[6]={d2ph_xx,d2ph_xy,0.,d2ph_yy,0.,0.};

    // Agama forceDeriv layout: [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
    // Mapping: p=0→(i=0,j=0), p=1→(i=1,j=1), p=2→(i=2,j=2)
    //          p=3→(i=0,j=1), p=4→(i=1,j=2), p=5→(i=0,j=2)
    int ii[6]={0,1,2,0,1,0};
    int jj[6]={0,1,2,1,2,2};
    // d2lr/d2th/d2ph indexed by canonical {xx,xy,xz,yy,yz,zz} order → map ii,jj:
    // canonical slot for (i,j): 0=(0,0)=xx, 1=(0,1)=xy, 2=(0,2)=xz, 3=(1,1)=yy, 4=(1,2)=yz, 5=(2,2)=zz
    // For output p, need canonical slot of (ii[p], jj[p]):
    int can[6];
    for (int p=0;p<6;p++) {
        int i=ii[p], j=jj[p];
        // canonical index: encode upper triangle in xx,xy,xz,yy,yz,zz order
        if      (i==0&&j==0) can[p]=0;
        else if (i==0&&j==1) can[p]=1;
        else if (i==0&&j==2) can[p]=2;
        else if (i==1&&j==1) can[p]=3;
        else if (i==1&&j==2) can[p]=4;
        else                  can[p]=5;
    }

    for (int p=0;p<6;p++) {
        int i=ii[p], j=jj[p], q=can[p];
        hess_out[6*tid+p] =
            d2Phi_dlr2    *dlr[i]*dlr[j]
          + d2Phi_dlr_dth *(dlr[i]*dth[j]+dlr[j]*dth[i])
          + d2Phi_dlr_dph *(dlr[i]*dph[j]+dlr[j]*dph[i])
          + d2Phi_dth2    *dth[i]*dth[j]
          + d2Phi_dth_dph *(dth[i]*dph[j]+dth[j]*dph[i])
          + d2Phi_dph2    *dph[i]*dph[j]
          + dPhi_dlr*d2lr[q] + dPhi_dth*d2th[q] + dPhi_dph*d2ph[q];
    }
}


// ---------------------------------------------------------------------------
//  multipole_density_kernel — rho = nabla^2 Phi / (4 pi G)
//  After unscaling via log-scaling inverse, uses standard Laplacian formula:
//    nabla^2 Phi = sum_c (d2C/dlogr^2 + dC/dlogr - l(l+1)*C) / r^2 * Y * T * mul
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256, 2) void
multipole_density_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double inv_4piG,
    double* __restrict__ rho_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double px=x[tid], py=y[tid], pz=z[tid];
    double r2=px*px+py*py+pz*pz, r=sqrt(r2), Rcyl=sqrt(px*px+py*py);

    if (r < 1.0e-300) { rho_out[tid]=0.; return; }

    double logr=log(r);

    // Inner extrapolation: density from l=0 power-law
    // nabla^2 (U*r^s + W) = U * s*(s+1) * r^(s-2) / (4*pi)
    if (logr < logr_min) {
        double dlr = logr - logr_min;
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        // For l=0: Laplacian = (d2C/dlogr^2 + dC/dlogr) / r^2
        double dC_dlr = inner_U * inner_s * r_ratio_s;
        double d2C_dlr2 = inner_U * inner_s * inner_s * r_ratio_s;
        double laplacian = (d2C_dlr2 + dC_dlr) / (r * r);
        rho_out[tid] = MUL0 * NORM_LM[0] * laplacian * inv_4piG;
        return;
    }

    int k=(int)((logr-logr_min)*inv_dlogr);
    if (k<0) k=0;
    if (k>=n_intervals) { rho_out[tid]=0.; return; }  // beyond grid: density→0 (Keplerian tail)
    double s=(logr-(logr_min+k*dlogr))*inv_dlogr;
    if (s<0.) s=0.; if (s>1.) s=1.;

    double cos_theta=pz/r, sin_theta=Rcyl/r, phi_az=atan2(py,px);

    double cos_mf[17], sin_mf[17];
    cos_mf[0]=1.; sin_mf[0]=0.;
    { double cph,sph; sincos(phi_az,&sph,&cph);
      if (lmax>=1) {cos_mf[1]=cph; sin_mf[1]=sph;}
      for(int m=2;m<=lmax;m++){
          cos_mf[m]=cph*cos_mf[m-1]-sph*sin_mf[m-1];
          sin_mf[m]=sph*cos_mf[m-1]+cph*sin_mf[m-1];
      }
    }

    const int NLM_MAX=(16+1)*(16+1);
    double Plm_arr[NLM_MAX];
    compute_Plm(cos_theta,sin_theta,lmax,Plm_arr,(double*)0,(double*)0);

    // -----------------------------------------------------------------------
    // c=0: evaluate + log-scaling inverse
    // -----------------------------------------------------------------------
    double C0_sc, dC0_sc_ds, d2C0_sc_ds2;
    quintic_eval(poly, 0, k, n_intervals, s, &C0_sc, &dC0_sc_ds, &d2C0_sc_ds2);
    double C0_val    = C0_sc;
    double dC0_dlr   = dC0_sc_ds * inv_dlogr;
    double d2C0_dlr2 = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;

    if (log_scaling) {
        double expX   = exp(C0_sc);
        double Phi0   = 1.0 / (invPhi0 - expX);
        double dPhidX = Phi0 * Phi0 * expX;
        double dC0_sc_dlr = dC0_sc_ds * inv_dlogr;
        double d2C0_sc_dlr2 = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;
        double d2Phi0 = dPhidX * (d2C0_sc_dlr2
                        + dC0_sc_dlr * dC0_sc_dlr * Phi0 * (invPhi0 + expX));
        double dPhi0  = dPhidX * dC0_sc_dlr;
        C0_val    = Phi0;
        dC0_dlr   = dPhi0;
        d2C0_dlr2 = d2Phi0;
    }

    // c=0 Laplacian contribution (l=0)
    double inv_r2=1./r2;
    double lap_sum = (d2C0_dlr2 + dC0_dlr - 0.) * inv_r2   // l(l+1)=0
                   * Plm_arr[0] * MUL0;

    // -----------------------------------------------------------------------
    // c>0 loop
    // -----------------------------------------------------------------------
    for (int c=1; c<n_lm; c++) {
        int l=lm_l[c], m=lm_m[c], absm=m<0?-m:m;

        double Cc_sc, dCc_sc_ds, d2Cc_sc_ds2;
        quintic_eval(poly, c, k, n_intervals, s, &Cc_sc, &dCc_sc_ds, &d2Cc_sc_ds2);
        double dCc_sc_dlr   = dCc_sc_ds  * inv_dlogr;
        double d2Cc_sc_dlr2 = d2Cc_sc_ds2* inv_dlogr * inv_dlogr;

        double Cc_val    = Cc_sc;
        double dCc_dlr   = dCc_sc_dlr;
        double d2Cc_dlr2 = d2Cc_sc_dlr2;

        if (log_scaling) {
            d2Cc_dlr2 = d2Cc_sc_dlr2*C0_val + 2.0*dCc_sc_dlr*dC0_dlr + Cc_sc*d2C0_dlr2;
            dCc_dlr   = dCc_sc_dlr*C0_val + Cc_sc*dC0_dlr;
            Cc_val    = Cc_sc * C0_val;
        }

        double lap_c = (d2Cc_dlr2 + dCc_dlr - (double)(l*(l+1))*Cc_val) * inv_r2;
        double mul   = (absm==0) ? MUL0 : MUL1;
        double Tlm   = (m>=0) ? cos_mf[absm] : sin_mf[absm];
        lap_sum += mul * lap_c * Plm_arr[l*(l+1)+absm] * Tlm;
    }

    rho_out[tid] = inv_4piG * lap_sum;
}
