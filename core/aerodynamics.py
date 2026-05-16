"""
core/aerodynamics.py — Aerodynamic Forces and Moments
Trinity 6-DOF Simulator | Orbital Dynamics

Computes ALL aerodynamic contributions in the body frame:

 1. BODY DRAG         Fd  — axial force opposing the velocity vector
 2. BODY NORMAL FORCE Fn  — lateral force from body angle-of-attack
                            acts through the Center of Pressure (CP)
 3. FIN CONTROL       Fc  — four independently deflected fins
                            produce lateral forces and moments
 4. AERO DAMPING      τd  — angular velocity-proportional moment resisting rotation

SIGN / DIRECTION CONVENTIONS (body frame)
──────────────────────────────────────────
Body Z = rocket nose direction (positive)
Body X = fin-1 / fin-3 axis
Body Y = fin-2 / fin-4 axis

All forces returned in body frame [N].
All moments returned as torque about body-frame axes [N·m].

FIN LAYOUT (viewed from base):
        Fin 1 (+X)
           |
  Fin 4   ─┼─   Fin 2 (+Y)
  (-Y)     |
        Fin 3 (-X)

Mixing matrix (from Caronte V1 / ServoController.cpp):
  δ1 = +pitch + roll   (Fin 1)
  δ2 = +yaw  - roll    (Fin 2)
  δ3 = -pitch + roll   (Fin 3)
  δ4 = -yaw  - roll    (Fin 4)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.atmosphere import G0


# ─── Configuration (passed from GUI / SimConfig) ──────────────────────────────

@dataclass
class AeroConfig:
    """
    All aerodynamic parameters for one stage.
    Populated from the GUI and from the parsed aero data files.
    """
    # Reference geometry
    body_diameter_m: float = 0.16        # rocket body outer diameter [m]
    body_aref_m2:    float = 0.0201      # reference area = π(d/2)²  [m²]

    # Center of Pressure (from nose, constant for MVP)
    cp_from_nose_m: float  = 1.20        # [m]

    # Fin geometry
    fin_count:      int    = 4
    fin_sref_m2:    float  = 0.0120      # single-fin reference area [m²]
    fin_lever_arm_m: float = 0.80        # CG → fin lift centroid [m] (LEVER_ARM_D)
    fin_radius_m:   float  = 0.12        # radial distance from axis to fin centroid

    # Aerodynamic damping coefficient  [N·m / (rad/s)]
    # Scales pitch/yaw angular velocity → resisting torque
    # Tune with CFD or flight data; typical 0.01–0.5 for high-power rockets
    aero_damping_lat: float = 0.05       # lateral axes (pitch, yaw)
    aero_damping_roll: float = 0.01      # roll axis

    # Body aero model (from SimSweep CSV)  — may be None (uses fallback)
    body_aero: object = None             # BodyAeroModel instance

    # Fin aero model (from aero_table.h)   — may be None (uses linear approx)
    fin_aero:  object = None             # AeroTableFin instance

    # Fallback Cd(Mach) polynomial if body_aero is not loaded
    # Coefficients for  Cd = c0 + c1*M + c2*M² + c3*M³ + c4*M⁴
    cd_fallback_coeffs: tuple = (0.45, -0.10, 0.50, -0.30, 0.05)

    def aref(self) -> float:
        return self.body_aref_m2 or (np.pi * (self.body_diameter_m / 2) ** 2)


# ─── AerodynamicsModel ───────────────────────────────────────────────────────

class AerodynamicsModel:
    """
    Computes aerodynamic forces F [N] and torques τ [N·m] in body frame.
    """

    def __init__(self, config: AeroConfig):
        self.cfg = config

    # ── Public interface ──────────────────────────────────────────────────────

    def compute(
        self,
        vel_body:    np.ndarray,   # velocity vector in body frame [m/s]
        omega_body:  np.ndarray,   # angular velocity in body frame [rad/s]
        rho:         float,        # air density [kg/m³]
        mach:        float,        # Mach number
        cg_from_nose:float,        # current CG position from nose [m]
        servo_angles: np.ndarray,  # [δ1, δ2, δ3, δ4] fin deflections [rad]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        F_body : np.ndarray (3,)   total aero force in body frame [N]
        tau_body : np.ndarray (3,) total aero torque about body CG [N·m]
        """
        v_mag = float(np.linalg.norm(vel_body))
        if v_mag < 0.5:
            return np.zeros(3), np.zeros(3)

        q_dyn = 0.5 * rho * v_mag ** 2
        Aref  = self.cfg.aref()

        # Decompose body-frame velocity
        vx, vy, vz = vel_body   # vz = axial (along rocket nose)

        # Total angle of attack
        v_lat = np.hypot(vx, vy)       # lateral speed magnitude
        aoa_total = np.arctan2(v_lat, abs(vz))   # [rad], always ≥ 0
        aoa_deg   = np.degrees(aoa_total)

        # ── 1. DRAG ───────────────────────────────────────────────────────────
        Cd = self._Cd(mach, aoa_deg)
        F_drag_mag = Cd * q_dyn * Aref   # [N]
        # Drag acts along -velocity direction
        F_drag = -F_drag_mag * (vel_body / v_mag)

        # ── 2. BODY NORMAL FORCE ──────────────────────────────────────────────
        Cn = self._Cn(mach, aoa_deg)
        F_normal_mag = Cn * q_dyn * Aref   # [N]
        F_normal = np.zeros(3)
        if v_lat > 1e-6 and F_normal_mag > 0:
            # Normal force direction: perpendicular to rocket axis, in plane of AoA
            # Components: opposite to lateral velocity (restoring when CP behind CG)
            # The lateral velocity unit vector
            lat_unit = np.array([vx, vy, 0.0]) / v_lat
            # Normal force acts to oppose the lateral component
            # (i.e., toward the nose direction in the AoA plane)
            F_normal[:2] = -F_normal_mag * lat_unit[:2]

        # ── Moment from body normal force (about CG) ──────────────────────────
        # Moment arm = distance from CG to CP along body Z
        # cg_from_nose, cp_from_nose measured from nose; base is positive direction
        # Moment arm sign: CP aft of CG → stabilizing
        cp_from_nose = self.cfg.cp_from_nose_m
        moment_arm = cp_from_nose - cg_from_nose   # positive = CP aft of CG
        # The normal force at CP creates a torque about CG
        # Torque = r_CP × F_normal, where r_CP = -moment_arm * ẑ (body)
        # τ = (-moment_arm * ẑ) × F_normal
        r_cp = np.array([0.0, 0.0, -moment_arm])  # CP position rel to CG in body
        tau_normal = np.cross(r_cp, F_normal)

        # ── 3. FIN CONTROL FORCES AND MOMENTS ────────────────────────────────
        F_fins, tau_fins = self._fin_forces_and_moments(
            servo_angles, q_dyn, vel_body, aoa_deg
        )

        # ── 4. AERODYNAMIC DAMPING ────────────────────────────────────────────
        tau_damp = -np.array([
            self.cfg.aero_damping_lat  * q_dyn * Aref * omega_body[0],
            self.cfg.aero_damping_lat  * q_dyn * Aref * omega_body[1],
            self.cfg.aero_damping_roll * q_dyn * Aref * omega_body[2],
        ])

        # ── Sum all contributions ─────────────────────────────────────────────
        F_total   = F_drag + F_normal + F_fins
        tau_total = tau_normal + tau_fins + tau_damp

        return F_total, tau_total

    # ── Private helpers ────────────────────────────────────────────────────────

    def _Cd(self, mach: float, aoa_deg: float) -> float:
        """Drag coefficient from body aero model or fallback polynomial."""
        if self.cfg.body_aero is not None:
            return self.cfg.body_aero.Cd(mach, aoa_deg)
        # Fallback: polynomial in Mach (calibrated for M2000-class rocket at AoA=0)
        c = self.cfg.cd_fallback_coeffs
        M = np.clip(mach, 0.0, 5.0)
        Cd0 = c[0] + c[1]*M + c[2]*M**2 + c[3]*M**3 + c[4]*M**4
        # AoA correction (rough): induced drag ≈ k * AoA²
        Cd_aoa = 0.012 * (np.radians(aoa_deg)) ** 2
        return max(0.0, float(Cd0 + Cd_aoa))

    def _Cn(self, mach: float, aoa_deg: float) -> float:
        """Body normal-force coefficient from model or simple approximation."""
        if self.cfg.body_aero is not None:
            return self.cfg.body_aero.Cn(mach, aoa_deg)
        # Simple Barrowman approximation: Cn ≈ 2 * sin(AoA) * cos(AoA)
        # For small AoA: Cn ≈ 2 * AoA [rad]
        alpha = np.radians(aoa_deg)
        return float(2.0 * np.sin(alpha) * np.cos(alpha))

    def _fin_force_single(self, delta_rad: float, q_dyn: float) -> float:
        """
        Aerodynamic force [N] produced by one fin at deflection delta_rad.
        Uses AeroTableFin if available, otherwise linear CL approximation.
        """
        if self.cfg.fin_aero is not None:
            return self.cfg.fin_aero.fin_force(delta_rad, q_dyn)
        # Fallback: simple linear CL model
        # CL ≈ 2π * α (thin airfoil theory)
        CL_slope = 2.5   # per radian (typical flat-plate fin)
        cl = CL_slope * delta_rad
        return float(cl * q_dyn * self.cfg.fin_sref_m2)

    def _fin_forces_and_moments(
        self,
        servo_angles: np.ndarray,   # [δ1, δ2, δ3, δ4] [rad]
        q_dyn:        float,
        vel_body:     np.ndarray,
        aoa_deg:      float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute combined fin force vector and torque vector in body frame.

        Fin layout:
          Fin 1 (+X span direction) → force in +Y body
          Fin 2 (+Y span direction) → force in -X body
          Fin 3 (-X span direction) → force in -Y body
          Fin 4 (-Y span direction) → force in +X body

        Torques about CG (using lever arm D):
          τ_pitch (about X) = (F1_y - F3_y) * D
          τ_yaw   (about Y) = (F2_x - F4_x) * D   (signs included)
          τ_roll  (about Z) = r_fin * (F1 - F2 + F3 - F4) * sign
        """
        if len(servo_angles) < 4:
            servo_angles = np.zeros(4)

        d1, d2, d3, d4 = servo_angles
        D   = self.cfg.fin_lever_arm_m
        r   = self.cfg.fin_radius_m

        # Force magnitudes (with sign from deflection direction)
        F1 = self._fin_force_single(d1, q_dyn)   # Fin 1: force in +Y
        F2 = self._fin_force_single(d2, q_dyn)   # Fin 2: force in -X
        F3 = self._fin_force_single(d3, q_dyn)   # Fin 3: force in -Y
        F4 = self._fin_force_single(d4, q_dyn)   # Fin 4: force in +X

        # Net translational forces (body frame)
        F_x = -F2 + F4    # X: Fin2 pushes -X, Fin4 pushes +X
        F_y =  F1 - F3    # Y: Fin1 pushes +Y, Fin3 pushes -Y
        F_z =  0.0        # No net axial component from fin deflection
        F_fins = np.array([F_x, F_y, F_z])

        # Torques about CG
        tau_pitch = (F1 - F3) * D          # about body X
        tau_yaw   = (F2 - F4) * D          # about body Y  (F2=-X→ yaw, F4=+X→-yaw)
        tau_roll  = r * (F1 - F2 + F3 - F4)  # about body Z (differential)

        tau_fins = np.array([tau_pitch, tau_yaw, tau_roll])
        return F_fins, tau_fins
