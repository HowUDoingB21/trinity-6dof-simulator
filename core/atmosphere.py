"""
core/atmosphere.py — ICAO International Standard Atmosphere (ISA)
Trinity 6-DOF Simulator | Orbital Dynamics

Implements the full ISA model from sea level to 86 km across all 7 atmospheric
layers (troposphere, stratosphere, mesosphere). Above 86 km the model returns
extrapolated values; above ~120 km aerodynamic forces become negligible.

Reference: ICAO Doc 7488-CD, 3rd edition; ISO 2533:1975
"""

from __future__ import annotations
import numpy as np

# ─── Physical constants ───────────────────────────────────────────────────────
G0    = 9.80665    # Standard gravity [m/s²]
R_AIR = 287.05287  # Specific gas constant for dry air [J/(kg·K)]
GAMMA = 1.40       # Ratio of specific heats (air)
P0    = 101325.0   # Sea-level pressure [Pa]
T0    = 288.15     # Sea-level temperature [K]
RHO0  = 1.225      # Sea-level density [kg/m³]

# ─── ISA layer table ──────────────────────────────────────────────────────────
# Each entry: (base_altitude_m, base_temperature_K, lapse_rate_K/m)
# Lapse rate < 0 → temperature decreases with altitude
# Lapse rate = 0 → isothermal layer
_LAYERS = np.array([
    (      0.0, 288.15, -6.5e-3),   # Layer 0: Troposphere
    ( 11_000.0, 216.65,  0.0   ),   # Layer 1: Tropopause (isothermal)
    ( 20_000.0, 216.65, +1.0e-3),   # Layer 2: Stratosphere lower
    ( 32_000.0, 228.65, +2.8e-3),   # Layer 3: Stratosphere upper
    ( 47_000.0, 270.65,  0.0   ),   # Layer 4: Stratopause (isothermal)
    ( 51_000.0, 270.65, -2.8e-3),   # Layer 5: Mesosphere lower
    ( 71_000.0, 214.65, -2.0e-3),   # Layer 6: Mesosphere upper
    ( 86_000.0, 186.87,  0.0   ),   # Layer 7: Boundary (stop here)
])

# Precompute base pressures for each layer
_P_BASE = np.zeros(len(_LAYERS))
_P_BASE[0] = P0

for _i in range(1, len(_LAYERS)):
    h0, T0_layer, L0_layer = _LAYERS[_i - 1]
    h1, T1_layer, _        = _LAYERS[_i]
    dh = h1 - h0
    if abs(L0_layer) < 1e-10:
        # Isothermal: P = P_base * exp(-g * dh / (R * T))
        _P_BASE[_i] = _P_BASE[_i - 1] * np.exp(-G0 * dh / (R_AIR * T0_layer))
    else:
        # Gradient: P = P_base * (T/T_base)^(-g/(R*L))
        T_top = T0_layer + L0_layer * dh
        _P_BASE[_i] = _P_BASE[_i - 1] * (T_top / T0_layer) ** (-G0 / (R_AIR * L0_layer))


def isa(altitude_m: float) -> tuple[float, float, float]:
    """
    ICAO ISA atmosphere at geometric altitude.

    Parameters
    ----------
    altitude_m : float
        Geometric altitude [m], referenced to MSL. May be negative (below MSL).

    Returns
    -------
    rho : float
        Air density [kg/m³]
    P : float
        Static pressure [Pa]
    T : float
        Temperature [K]
    """
    h = float(altitude_m)

    # Clamp: below sea level → surface values; above 86 km → extrapolate
    if h < 0.0:
        h = 0.0

    # Find which layer we're in
    layer_idx = 0
    for i in range(len(_LAYERS) - 1):
        if h >= _LAYERS[i, 0]:
            layer_idx = i
        else:
            break

    h_base = _LAYERS[layer_idx, 0]
    T_base = _LAYERS[layer_idx, 1]
    L      = _LAYERS[layer_idx, 2]
    P_base = _P_BASE[layer_idx]

    dh = h - h_base
    T  = T_base + L * dh

    if abs(L) < 1e-10:
        # Isothermal layer
        P = P_base * np.exp(-G0 * dh / (R_AIR * T_base))
    else:
        P = P_base * (T / T_base) ** (-G0 / (R_AIR * L))

    rho = P / (R_AIR * T)
    return float(rho), float(P), float(T)


def speed_of_sound(T_K: float) -> float:
    """Speed of sound at temperature T [K] → [m/s]."""
    return float(np.sqrt(GAMMA * R_AIR * T_K))


def altitude_to_mach(altitude_m: float, speed_ms: float) -> float:
    """Compute Mach number from altitude and airspeed."""
    _, _, T = isa(altitude_m)
    a = speed_of_sound(T)
    return speed_ms / a if a > 0 else 0.0


def dynamic_pressure(altitude_m: float, speed_ms: float) -> float:
    """q_dyn = ½ρV² [Pa]."""
    rho, _, _ = isa(altitude_m)
    return 0.5 * rho * speed_ms ** 2


# ─── Convenience: density → approximate altitude (for CSV import) ─────────────
def density_to_altitude(rho: float) -> float:
    """
    Inverse ISA: given density [kg/m³], return approximate altitude [m].
    Uses bisection on the troposphere layer (good for rho in [0.32, 1.225]).
    Falls back to linear extrapolation for rarefied air.
    """
    if rho >= RHO0:
        return 0.0
    if rho <= 1e-5:
        return 86_000.0

    lo, hi = 0.0, 86_000.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        rho_mid, _, _ = isa(mid)
        if rho_mid > rho:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ─── Vectorised versions ──────────────────────────────────────────────────────
def isa_vec(altitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ISA for arrays of altitudes."""
    results = [isa(h) for h in np.asarray(altitudes, dtype=float)]
    rho = np.array([r[0] for r in results])
    P   = np.array([r[1] for r in results])
    T   = np.array([r[2] for r in results])
    return rho, P, T
