"""
control/fsm.py — Flight State Machine + Servo Controller
Trinity 6-DOF Simulator | Orbital Dynamics

Direct port of FlightFSM.h / ServoController.cpp from Caronte V1.

FLIGHT PHASES
─────────────
  PAD          → On the pad, awaiting launch detect
  POWERED_S1   → Stage 1 motor burning
  COAST_S1     → Stage 1 coasting to staging altitude/condition
  STAGING      → Separation event (instantaneous)
  POWERED_S2   → Stage 2 motor burning
  COAST_UP     → Coasting to apogee
  APOGEE       → Apogee detected (vz sign change)
  DESCENT      → Falling under gravity / drogue
  LANDED       → Below threshold altitude & near-zero speed

The FSM outputs the active FlightPhase and whether active control is enabled.
The ServoController converts PID lift commands to servo deflection angles
using the CL(α) inverse from aero_table.h (or a linear approximation).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


# ─── Flight Phases ────────────────────────────────────────────────────────────

class FlightPhase(IntEnum):
    PAD        = 0
    POWERED_S1 = 1
    COAST_S1   = 2
    STAGING    = 3
    POWERED_S2 = 4
    COAST_UP   = 5
    APOGEE     = 6
    DESCENT    = 7
    LANDED     = 8


PHASE_NAMES = {
    FlightPhase.PAD:        "PAD",
    FlightPhase.POWERED_S1: "S1 BURN",
    FlightPhase.COAST_S1:   "S1 COAST",
    FlightPhase.STAGING:    "STAGING",
    FlightPhase.POWERED_S2: "S2 BURN",
    FlightPhase.COAST_UP:   "COAST UP",
    FlightPhase.APOGEE:     "APOGEE",
    FlightPhase.DESCENT:    "DESCENT",
    FlightPhase.LANDED:     "LANDED",
}


# ─── FSM configuration ────────────────────────────────────────────────────────

@dataclass
class FSMConfig:
    """Thresholds that determine phase transitions."""
    # Launch detect
    launch_accel_threshold_ms2: float = 20.0   # accel > this → powered
    launch_speed_threshold_ms:  float =  2.0   # or speed > this

    # Stage 1 burnout detect
    s1_burnout_thrust_n: float = 5.0            # thrust below this → burnout
    # Alternatively: use the motor burn time from the .eng file

    # Staging: separation condition
    staging_mode: str  = 'altitude'             # 'altitude' | 'time' | 'speed'
    staging_altitude_m: float = 15_000.0        # AGL
    staging_time_s:     float = 30.0            # from launch
    staging_delay_s:    float =  0.5            # delay after S1 burnout

    # S2 ignition: delay after staging event
    s2_ignition_delay_s: float = 0.1

    # Apogee detect
    apogee_vz_threshold_ms: float = 0.0         # vz < this → apogee

    # Landing detect
    landing_alt_m:    float = 50.0              # AGL below this
    landing_speed_ms: float =  5.0             # AND speed below this

    # Control enable window
    control_phase_active: tuple = (
        FlightPhase.POWERED_S1,
        FlightPhase.COAST_S1,
        FlightPhase.POWERED_S2,
        FlightPhase.COAST_UP,
    )


# ─── Flight State Machine ─────────────────────────────────────────────────────

class FlightFSM:
    """
    Mirrors Caronte V1 FlightFSM.
    Driven by the simulation runner at each 100 Hz control tick.
    """

    def __init__(self, config: FSMConfig | None = None,
                 s1_burn_time_s: float = 10.0,
                 s2_burn_time_s: float = 10.0):
        self.cfg = config or FSMConfig()
        self.s1_burn_time = s1_burn_time_s
        self.s2_burn_time = s2_burn_time_s

        self.phase         = FlightPhase.PAD
        self.t_launch      = None
        self.t_s1_burnout  = None
        self.t_staging     = None
        self.t_s2_ignition = None
        self.t_apogee      = None

        self._prev_vz      = 0.0
        self._staging_done = False
        self._apogee_done  = False

    def update(self, t: float, state: dict) -> FlightPhase:
        """
        Evaluate current state and advance FSM if transition criteria are met.

        Parameters
        ----------
        t     : float  — simulation time [s]
        state : dict   — keys: alt_agl, speed, thrust, vz (vertical speed)
        """
        alt    = state.get('alt_agl', 0.0)
        speed  = state.get('speed',   0.0)
        thrust = state.get('thrust',  0.0)
        vz     = state.get('vz',      0.0)

        phase = self.phase

        # ── PAD → POWERED_S1 ──────────────────────────────────────────────
        if phase == FlightPhase.PAD:
            if (thrust > self.cfg.launch_accel_threshold_ms2 or
                    speed  > self.cfg.launch_speed_threshold_ms):
                self.phase    = FlightPhase.POWERED_S1
                self.t_launch = t

        # ── POWERED_S1 → COAST_S1 (burnout detect) ────────────────────────
        elif phase == FlightPhase.POWERED_S1:
            t_from_launch = t - (self.t_launch or t)
            if thrust < self.cfg.s1_burnout_thrust_n or \
                    t_from_launch >= self.s1_burn_time:
                self.phase        = FlightPhase.COAST_S1
                self.t_s1_burnout = t

        # ── COAST_S1 → STAGING ────────────────────────────────────────────
        elif phase == FlightPhase.COAST_S1:
            t_since_burnout = t - (self.t_s1_burnout or t)
            stage_cond = False
            mode = self.cfg.staging_mode
            if mode == 'altitude':
                stage_cond = alt >= self.cfg.staging_altitude_m
            elif mode == 'time':
                stage_cond = t_since_burnout >= self.cfg.staging_delay_s
            elif mode == 'speed':
                stage_cond = speed <= self.cfg.staging_altitude_m  # reuse field

            if stage_cond and not self._staging_done:
                self.phase        = FlightPhase.STAGING
                self.t_staging    = t
                self._staging_done = True

        # ── STAGING → POWERED_S2 ──────────────────────────────────────────
        elif phase == FlightPhase.STAGING:
            t_since_staging = t - (self.t_staging or t)
            if t_since_staging >= self.cfg.s2_ignition_delay_s:
                self.phase          = FlightPhase.POWERED_S2
                self.t_s2_ignition  = t

        # ── POWERED_S2 → COAST_UP ─────────────────────────────────────────
        elif phase == FlightPhase.POWERED_S2:
            t_from_s2 = t - (self.t_s2_ignition or t)
            if thrust < self.cfg.s1_burnout_thrust_n or \
                    t_from_s2 >= self.s2_burn_time:
                self.phase = FlightPhase.COAST_UP

        # ── COAST_UP → APOGEE ─────────────────────────────────────────────
        elif phase == FlightPhase.COAST_UP:
            if vz <= self.cfg.apogee_vz_threshold_ms and self._prev_vz > 0:
                if not self._apogee_done:
                    self.phase      = FlightPhase.APOGEE
                    self.t_apogee   = t
                    self._apogee_done = True

        # ── APOGEE → DESCENT ──────────────────────────────────────────────
        elif phase == FlightPhase.APOGEE:
            self.phase = FlightPhase.DESCENT

        # ── DESCENT → LANDED ─────────────────────────────────────────────
        elif phase == FlightPhase.DESCENT:
            if alt < self.cfg.landing_alt_m and speed < self.cfg.landing_speed_ms:
                self.phase = FlightPhase.LANDED

        self._prev_vz = vz
        return self.phase

    @property
    def control_active(self) -> bool:
        return self.phase in self.cfg.control_phase_active

    @property
    def stage_2_active(self) -> bool:
        return self.phase in (FlightPhase.POWERED_S2,
                              FlightPhase.COAST_UP,
                              FlightPhase.APOGEE,
                              FlightPhase.DESCENT,
                              FlightPhase.LANDED)

    @property
    def name(self) -> str:
        return PHASE_NAMES.get(self.phase, "?")


# ─── Servo Controller ─────────────────────────────────────────────────────────

@dataclass
class ServoConfig:
    """Servo actuator parameters (matches Caronte V1 config.h)."""
    delta_max_deg:  float = 15.0     # maximum deflection [°]
    slew_rate_deg_s:float = 300.0    # max angular velocity [°/s]
    latency_s:      float = 0.010    # transport delay [s]
    lever_arm_d_m:  float = 0.80     # [m]


class ServoController:
    """
    Converts PIDOutput (lift forces) to servo deflection angles.
    Applies:
      1. Aerodynamic inversion (force → AoA via aero_table.h)
      2. Mixing matrix
      3. Slew rate limiting
      4. Hard saturation at ±delta_max
    Matches Caronte V1 ServoController.cpp.
    """

    def __init__(self, config: ServoConfig | None = None,
                 aero_fin=None):
        self.cfg      = config or ServoConfig()
        self.aero_fin = aero_fin   # AeroTableFin or None

        # Current servo positions [rad]
        self._pos = np.zeros(4)
        # Latency buffer: [(t_cmd, angles), ...]
        self._cmd_buffer: list = []

    def compute(self, pid_out, q_dyn: float, t: float, dt: float) -> np.ndarray:
        """
        Parameters
        ----------
        pid_out : PIDOutput   — torques / lift forces from PID
        q_dyn   : float       — dynamic pressure [Pa]
        t       : float       — current sim time [s]
        dt      : float       — control timestep [s]

        Returns
        -------
        servo_angles : np.ndarray (4,) in radians, [δ1, δ2, δ3, δ4]
        """
        # ── Aerodynamic inversion (lift force → AoA) ──────────────────────
        d_pitch = self._aero_inverse(pid_out.lift_pitch, q_dyn)
        d_yaw   = self._aero_inverse(pid_out.lift_yaw,   q_dyn)
        d_roll  = self._aero_inverse(pid_out.lift_roll,  q_dyn)

        # ── Mixing matrix (Caronte V1 sign convention) ────────────────────
        raw = np.array([
            +d_pitch + d_roll,   # δ1 — top / fin 1
            +d_yaw   - d_roll,   # δ2 — right / fin 2
            -d_pitch + d_roll,   # δ3 — bottom / fin 3
            -d_yaw   - d_roll,   # δ4 — left / fin 4
        ])

        # ── Hard saturation ───────────────────────────────────────────────
        d_max_rad = np.deg2rad(self.cfg.delta_max_deg)
        raw_sat   = np.clip(raw, -d_max_rad, d_max_rad)

        # ── Latency buffer ────────────────────────────────────────────────
        self._cmd_buffer.append((t + self.cfg.latency_s, raw_sat.copy()))
        # Release commands whose delay has expired
        delayed = raw_sat  # default: no old command ready yet
        while self._cmd_buffer and self._cmd_buffer[0][0] <= t:
            _, delayed = self._cmd_buffer.pop(0)

        # ── Slew rate limiting ────────────────────────────────────────────
        max_step = np.deg2rad(self.cfg.slew_rate_deg_s) * dt
        delta    = np.clip(delayed - self._pos, -max_step, max_step)
        self._pos = np.clip(self._pos + delta, -d_max_rad, d_max_rad)

        return self._pos.copy()

    def _aero_inverse(self, force_n: float, q_dyn: float) -> float:
        """Returns deflection angle [rad] that produces force_n [N]."""
        if q_dyn < 0.5:
            return 0.0
        if self.aero_fin is not None:
            aoa_deg = self.aero_fin.aoa_for_force(force_n, q_dyn)
            return np.deg2rad(aoa_deg) * np.sign(force_n)
        # Fallback: linear CL model CL ≈ CL_slope * α
        CL_slope = 2.5
        sref     = 0.012
        cl_cmd   = abs(force_n) / (q_dyn * sref)
        aoa_rad  = cl_cmd / CL_slope
        return float(np.clip(aoa_rad, 0, np.deg2rad(self.cfg.delta_max_deg)) *
                     np.sign(force_n))

    def reset(self):
        self._pos[:] = 0.0
        self._cmd_buffer.clear()
