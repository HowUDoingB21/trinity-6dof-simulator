"""
core/mass_model.py — Variable Mass Model
Trinity 6-DOF Simulator | Orbital Dynamics

═══════════════════════════════════════════════════════════════════════════════
REFERENCE FRAME — ALL POSITIONS FROM THE NOSE OF STAGE 2
═══════════════════════════════════════════════════════════════════════════════

Every CG and CP value is measured from the NOSE TIP OF STAGE 2, which is:
  • The nose of the complete rocket during Stage 1 flight
  • The nose of the sustainer during Stage 2 flight

This matches OpenRocket's convention directly:
  • OpenRocket "CG: 193 cm" → enter 1.93 m for CG during S1 flight  ✓
  • OpenRocket S2 "CG: 121 cm" → enter 1.21 m in Stage 2 tab        ✓
  • OpenRocket "CP: 212 cm"   → enter 2.12 m in Aero S1 tab         ✓

To convert S1-local values to this frame: add S2_length
  cg_s1_dry_from_s2_nose = L_s2 + cg_s1_dry_from_s1_nose
  e.g. 1.80 + 0.613 = 2.413 m

Classes
───────
  StageGeometry     — geometry dataclass (all CG from S2 nose)
  StageInertia      — dry inertia tensor (referenced to dry CG)
  MassModel         — Stage 2 flying solo
  TwoStageMassModel — Stage 1 flight with S2 as payload
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


# ─── PropellantModel protocol ─────────────────────────────────────────────────

class IPropulsion(Protocol):
    def get_propellant_remaining(self, t: float) -> float: ...
    propellant_mass_kg: float


# ─── Inertia dataclass ───────────────────────────────────────────────────────

@dataclass
class StageInertia:
    """Dry inertia tensor [kg·m²] at the stage's own dry CG."""
    Ixx: float = 0.005
    Iyy: float = 0.005
    Izz: float = 0.001
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0

    def as_matrix(self) -> np.ndarray:
        return np.array([
            [self.Ixx, self.Ixy, self.Ixz],
            [self.Ixy, self.Iyy, self.Iyz],
            [self.Ixz, self.Iyz, self.Izz],
        ])


# ─── Geometry dataclass ───────────────────────────────────────────────────────

@dataclass
class StageGeometry:
    """
    Physical geometry of one rocket stage.

    cg_dry_from_nose_m and cg_prop_from_nose_m are ALWAYS measured from
    Stage 2's nose tip (nose of the complete rocket). This is OpenRocket's
    natural output convention.

    For Stage 2: these values equal OpenRocket's S2 CG readout.
    For Stage 1: these values = L_s2 + (CG measured from S1's own nose).
    """
    diameter_m:           float
    length_m:             float           # this stage's own structural length
    cg_dry_from_nose_m:   float           # dry CG from S2 nose [m]
    cg_prop_from_nose_m:  float           # propellant CG from S2 nose [m]
    grain_length_m:       float
    grain_od_m:           float
    grain_id_m:           float = 0.0
    inertia_dry:          StageInertia = field(default_factory=StageInertia)


# ─── Math helpers ─────────────────────────────────────────────────────────────

def _cylinder_inertia(mass: float, r_outer: float, r_inner: float,
                      length: float) -> tuple[float, float]:
    """Hollow cylinder: (I_lateral, I_axial) about own CG."""
    if mass <= 0.0:
        return 0.0, 0.0
    ro2, ri2 = r_outer ** 2, r_inner ** 2
    I_axial   = 0.5  * mass * (ro2 + ri2)
    I_lateral = mass / 12.0 * (3.0 * (ro2 + ri2) + length ** 2)
    return float(I_lateral), float(I_axial)


def _parallel_axis(I: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """I_new = I + m * (|d|² E − d dᵀ)"""
    if mass <= 0.0:
        return I.copy()
    return I + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


# ─── Stage 2 mass model (S2 alone) ────────────────────────────────────────────

class MassModel:
    """
    Mass model for Stage 2 flying solo after separation.

    Uses two CG states that OpenRocket provides directly — same convention
    as Stage 1:

      cg_dry_from_nose_m   → CG of S2 at ignition (S2 propellant full).
                             From OpenRocket's S2-alone view (with motor).
                             e.g. 1.21 m from S2 nose.

      cg_prop_from_nose_m  → CG of S2 at burnout (S2 propellant empty).
                             From OpenRocket simulation or estimate.
                             e.g. 1.13 m from S2 nose.

    CG at any time t is linearly interpolated between these two states
    proportional to S2 propellant consumed. At ignition CG = cg_dry value;
    at burnout CG = cg_prop value.
    """

    def __init__(self, geometry: StageGeometry, propulsion: IPropulsion,
                 dry_mass_override: float | None = None):
        self.geom  = geometry
        self.prop  = propulsion
        self._dry  = dry_mass_override
        self._cg_full    = geometry.cg_dry_from_nose_m   # CG at ignition (full)
        self._cg_burnout = geometry.cg_prop_from_nose_m  # CG at burnout (empty)
        self._m_prop_total = propulsion.propellant_mass_kg

    @property
    def dry_mass(self) -> float:
        if self._dry is not None:
            return self._dry
        if hasattr(self.prop, 'total_mass_kg'):
            return self.prop.total_mass_kg - self.prop.propellant_mass_kg
        return 0.0

    def get_properties(self, t: float) -> tuple[float, float, np.ndarray]:
        """
        Returns (mass [kg], CG_from_S2_nose [m], I [3×3 kg·m²]).
        CG at ignition = cg_dry (full)   — matches OpenRocket S2 view.
        CG at burnout  = cg_prop (empty) — matches OpenRocket simulation.
        """
        m_prop  = self.prop.get_propellant_remaining(t)
        m_dry   = self.dry_mass
        m_total = m_dry + m_prop

        frac = (1.0 - m_prop / self._m_prop_total
                if self._m_prop_total > 1e-9 else 1.0)
        cg = self._cg_full + (self._cg_burnout - self._cg_full) * frac

        I = self._inertia(m_dry, m_prop, cg)
        return float(m_total), float(cg), I

    def _inertia(self, m_dry: float, m_prop: float, cg_total: float) -> np.ndarray:
        def shift(I_own, mass, cg_comp):
            return _parallel_axis(I_own, mass,
                                  np.array([0., 0., cg_total - cg_comp]))
        I_dry  = shift(self.geom.inertia_dry.as_matrix(), m_dry,
                       self._cg_burnout)   # at burnout, structure dominates
        ro, ri = self.geom.grain_od_m / 2, self.geom.grain_id_m / 2
        Il, Ia = _cylinder_inertia(m_prop, ro, ri, self.geom.grain_length_m)
        I_prop = shift(np.diag([Il, Il, Ia]), m_prop, self._cg_full)
        I = 0.5 * ((I_dry + I_prop) + (I_dry + I_prop).T)
        for i in range(3): I[i, i] = max(I[i, i], 1e-6)
        return I

    def get_static_margin(self, cg: float, cp: float) -> float:
        """Static margin in calibers. Positive = stable. Both from S2 nose."""
        return (cp - cg) / self.geom.diameter_m


# ─── Two-stage mass model (S1+S2 flying together) ─────────────────────────────

class TwoStageMassModel:
    """
    Mass model for Stage 1 flight with Stage 2 aboard.

    Uses two CG states that OpenRocket provides directly — no component
    decomposition needed:

      s1_geom.cg_dry_from_nose_m   → CG of COMPLETE ROCKET at pad (S1 full)
                                      Read from OpenRocket static CG display.
                                      e.g. 1.93 m

      s1_geom.cg_prop_from_nose_m  → CG of COMPLETE ROCKET at S1 burnout
                                      (S1 propellant = 0, S2 still full).
                                      Read from OpenRocket simulation CG plot
                                      at the end of Stage 1 burn.
                                      e.g. 2.15 m (CG shifts aft as S1 burns)

    CG at any time t is linearly interpolated between these two states
    proportional to S1 propellant fraction consumed. This is physically
    exact for a uniform propellant grain and a very good approximation
    otherwise.

    Mass is computed correctly from all four components (S1 dry, S1 prop,
    S2 dry, S2 prop) so that mass evolution is independent and accurate.
    """

    def __init__(self,
                 s1_geom:  StageGeometry,
                 s1_motor: IPropulsion,
                 s1_dry:   float,
                 s2_geom:  StageGeometry,
                 s2_motor: IPropulsion,
                 s2_dry:   float):
        self.s1_geom  = s1_geom
        self.s1_motor = s1_motor
        self.s1_dry   = s1_dry
        self.s2_geom  = s2_geom
        self.s2_motor = s2_motor
        self.s2_dry   = s2_dry

        # The two CG anchor points (both from S2 nose, complete rocket)
        self._cg_pad     = s1_geom.cg_dry_from_nose_m   # at liftoff (S1 full)
        self._cg_burnout = s1_geom.cg_prop_from_nose_m  # at S1 burnout (S1 empty)
        self._m_s1_prop  = s1_motor.propellant_mass_kg

        # S2 CG (constant)
        m_s2d = s2_dry
        m_s2p = s2_motor.propellant_mass_kg
        self._m_s2 = m_s2d + m_s2p

    def get_properties(self, t: float) -> tuple[float, float, np.ndarray]:
        """
        Returns (mass [kg], CG_from_S2_nose [m], I [3×3 kg·m²]).

        CG at t=0  = cg_pad     (the OpenRocket liftoff CG you entered)
        CG at t=t_burnout = cg_burnout (the OpenRocket burnout CG you entered)
        """
        m_s1p = self.s1_motor.get_propellant_remaining(t)
        m_s1d = self.s1_dry
        m_tot = m_s1d + m_s1p + self._m_s2

        # Fraction of S1 propellant consumed (0 = full, 1 = empty)
        if self._m_s1_prop > 1e-9:
            frac_consumed = 1.0 - m_s1p / self._m_s1_prop
        else:
            frac_consumed = 1.0

        # Linear interpolation between the two known CG states
        cg = self._cg_pad + (self._cg_burnout - self._cg_pad) * frac_consumed

        I = self._inertia(m_s1d, m_s1p, cg)
        return float(m_tot), float(cg), I

    def _inertia(self, m_s1d, m_s1p, cg_total):
        """
        Compute inertia tensor using propellant grain geometry for S1 and
        full component geometry for S2.
        """
        def shift(I_own, mass, cg_comp):
            return _parallel_axis(I_own, mass,
                                  np.array([0., 0., cg_total - cg_comp]))

        # S1 dry — use inertia tensor shifted from dry CG
        # Approximate S1 dry CG: interpolate using mass fractions
        cg_s1d_approx = self._cg_pad - (self._cg_burnout - self._cg_pad) * (
            self._m_s1_prop / max(m_s1d, 1e-9))
        I_s1d = shift(self.s1_geom.inertia_dry.as_matrix(), m_s1d, cg_s1d_approx)

        # S1 propellant grain (hollow cylinder about its geometric centre)
        ro, ri = self.s1_geom.grain_od_m / 2, self.s1_geom.grain_id_m / 2
        Il, Ia = _cylinder_inertia(m_s1p, ro, ri, self.s1_geom.grain_length_m)
        # Approximate grain position: midpoint between pad CG and burnout CG
        cg_grain_approx = (self._cg_pad + self._cg_burnout) / 2
        I_s1p = shift(np.diag([Il, Il, Ia]), m_s1p, cg_grain_approx)

        # S2 dry
        m_s2d = self.s2_dry
        I_s2d = shift(self.s2_geom.inertia_dry.as_matrix(),
                      m_s2d, self.s2_geom.cg_dry_from_nose_m)

        # S2 propellant
        m_s2p = self.s2_motor.propellant_mass_kg
        ro2, ri2 = self.s2_geom.grain_od_m / 2, self.s2_geom.grain_id_m / 2
        Il2, Ia2 = _cylinder_inertia(m_s2p, ro2, ri2, self.s2_geom.grain_length_m)
        I_s2p = shift(np.diag([Il2, Il2, Ia2]),
                      m_s2p, self.s2_geom.cg_prop_from_nose_m)

        I = I_s1d + I_s1p + I_s2d + I_s2p
        I = 0.5 * (I + I.T)
        for i in range(3):
            I[i, i] = max(I[i, i], 1e-6)
        return I
