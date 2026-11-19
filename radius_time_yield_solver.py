"""
Nuclear Yield Estimator from Optical Bhangmeter Light Curves
=============================================================
Simplified version:
  - Each (time, power) pair is processed independently
  - Uses Brent’s method to recover yield at every point
  - No averaging and no least-squares fitting
"""

import numpy as np
from scipy.optimize import root_scalar

# ==============================================================================
# Constants
# ==============================================================================

S_GAMMA        = 1.032
GAMMA          = 1.4
R_SPECIFIC_AIR = 287.05                 # J kg⁻¹ K⁻¹
SIGMA          = 5.670374419e-8         # Stefan-Boltzmann [W m⁻² K⁻⁴]
KT_TO_J        = 4.184e12
G_GAMMA        = (25.0 / 4.0) * S_GAMMA**(-5)   # ≈ 5.33927
BLAST_COUPLING = 0.53                   # Y_hydro / Y_total
K_EILERS       = 1.2e-8
K_IDEAL        = 6.4e-9

# ==============================================================================
# Physics models
# ==============================================================================

def post_shock_temperature(v):
    """Strong-shock Rankine-Hugoniot temperature (ideal gas)."""

    prefactor = (2.0 * GAMMA * (GAMMA - 1.0)) / (GAMMA + 1.0)**2

    T = prefactor * (v**2) / R_SPECIFIC_AIR

    return np.maximum(T, 300.0)

def calculated_power(R, v):
    """Black-body power radiated by the opaque shock front [W]."""

    T = post_shock_temperature(v)

    return 4.0 * np.pi * R**2 * SIGMA * T**4

def self_similar_velocity(R, t):

    return 0.4 * R / t

def total_yield_kt(R, v, rho):
    """Convert hydrodynamic energy to total nuclear yield [kt]."""

    Y_hydro = G_GAMMA * rho * R**3 * v**2

    return (Y_hydro / BLAST_COUPLING) / KT_TO_J

# ==============================================================================
# Core solver – Brent’s method (one point at a time)
# ==============================================================================

def recover_yield(t, P_measured, rho=1.225, R_bracket=(1.0, 10000.0)):
    """
    Recover total nuclear yield from a single (time, absolute power) pair
    using Brent’s method.

    How Brent’s method is used here
    -------------------------------
    We do **not** optimize yield directly. Instead we solve a root-finding
    problem for the shock radius R.

    At fixed time t we have a measured power P_measured.

    We define the residual function:
    f(R) = P_calculated(R) - P_measured

    where P_calculated is obtained by:
      1. Assuming self-similar flow → v = (2/5) * R / t
      2. Computing post-shock temperature from Rankine-Hugoniot: T = T(v)
      3. Computing black-body power: P = 4π R² σ T⁴

    Brent’s method (scipy.optimize.root_scalar with method='brentq')
    finds the value of R that makes f(R) = 0, i.e. the radius at which
    the predicted optical power exactly matches the measured power.

    Once that R is known, velocity, temperature, and finally yield
    are computed from the Sedov-Taylor relations + coupling factor.
    """

    def objective(R):

        """
        Residual function whose root we seek:
        f(R) = calculated_power(R) - measured_power
        """

        if R <= 0.0:
            return 1e30

        v = self_similar_velocity(R, t)

        return calculated_power(R, v) - P_measured

    # Brent’s method finds R such that objective(R) ≈ 0
    sol = root_scalar(objective, bracket=R_bracket, method='brentq', xtol=1e-10)
    R = sol.root

    # Once R is known, the remaining quantities follow directly
    v = self_similar_velocity(R, t)
    T = post_shock_temperature(v)
    Y_kt = total_yield_kt(R, v, rho)

    return Y_kt, R, v, T

# ==============================================================================
# Main analysis – process every point independently
# ==============================================================================

def analyze_points(t_arr, P_arr, rho=1.225, verbose=True):
    """
    Calculate yield independently for every (t, P) pair.
    Returns a list of results (one dictionary per point).
    """

    t_arr = np.asarray(t_arr, dtype=float)
    P_arr = np.asarray(P_arr, dtype=float)

    results = []

    if verbose:
        print("\n" + "="*78)
        print("INDIVIDUAL YIELD RESULTS (Brent method)")
        print("="*78)
        print(f"{'t (ms)':>8}  {'Power (W)':>12}  {'Yield (kt)':>10}  {'R (m)':>8}  {'v (m/s)':>9}  {'T (K)':>8}")
        print("-"*78)

    for t, P in zip(t_arr, P_arr):
        Y_kt, R, v, T = recover_yield(t, P, rho)
        results.append({
            't_s'   : t,
            't_ms'  : t * 1000.0,
            'P_W'   : P,
            'Y_kt'  : Y_kt,
            'R_m'   : R,
            'v_mps' : v,
            'T_K'   : T
        })

        if verbose:
            print(f"{t*1000:8.2f}  {P:12.3e}  {Y_kt:10.4f}  {R:8.1f}  {v:9.0f}  {T:8.0f}")

    if verbose:
        print("="*78)

    return results

# ==============================================================================
# Synthetic data generator (for testing)
# ==============================================================================

def generate_synthetic_lightcurve(Y_true_kt=1.0, rho=1.225,
                                  t_start=0.004, t_end=0.009, n_points=8,
                                  noise_level=0.0, seed=None):
    """Generate synthetic (t, P) data with optional relative noise."""

    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(t_start, t_end, n_points)
    Y_hydro_J = Y_true_kt * KT_TO_J * BLAST_COUPLING

    P = []

    for ti in t:
        R = S_GAMMA * Y_hydro_J**0.2 * ti**0.4 * rho**(-0.2)
        v = self_similar_velocity(R, ti)
        P_clean = calculated_power(R, v)
        if noise_level > 0.0:
            P_clean *= (1.0 + np.random.normal(0.0, noise_level))
        P.append(P_clean)

    return t, np.array(P)
 
# ==============================================================================
# Demo
# ==============================================================================

if __name__ == "__main__":
    print("Nuclear Yield Estimator – Individual Point Version")
    print(f"G(γ) = {G_GAMMA:.5f} | Coupling = {BLAST_COUPLING}")

    # Example 1: clean data
    print("\n>>> Clean synthetic data (1 kt, zero noise)")
    t_clean, P_clean = generate_synthetic_lightcurve(Y_true_kt=1.0, noise_level=0.0, n_points=8)
    analyze_points(t_clean, P_clean)

    # Example 2: noisy data
    print("\n>>> Noisy synthetic data (1 kt, 5% noise)")
    t_noisy, P_noisy = generate_synthetic_lightcurve(Y_true_kt=1.0, noise_level=0.05, n_points=8, seed=42)
    analyze_points(t_noisy, P_noisy)