"""
core/physics.py — 6-DOF Equations of Motion
Trinity 6-DOF Simulator | Orbital Dynamics

The physics engine solves the full 6-DOF rigid-body equations of motion
using scipy.integrate.solve_ivp with the RK45 adaptive solver.

STATE VECTOR (13 elements)
───────────────────────────
  y[0:3]   position     [px, py, pz]    ENU [m]
  y[3:6]   velocity     [vx, vy, vz]    ENU [m/s]
  y[6:10]  quaternion   [qw, qx, qy, qz] body→ENU
  y[10:13] ang. velocity [ωx, ωy, ωz]   body frame [rad/s]

EQUATIONS OF MOTION
────────────────────
Translational (inertial ENU frame):
  ṗ = v
  v̇ = (1/m) [ F_grav_ENU + R·(F_thrust_body + F_aero_body) ]

Rotational (body frame — Euler's equations):
  q̇ = ½ · q ⊗ [0, ω]            (quaternion kinematics)
  Iω̇ = τ_aero + τ_control − ω × (I·ω)   (Euler's moment equation)

Variable mass:
  m(t), CG(t), I(t) updated from mass model at every evaluation.

CONTROL INTEGRATION
────────────────────
The control system runs at a fixed 100 Hz discrete rate.
Between control updates the servo deflections are held constant (ZOH).
The ODE function is wrapped by PhysicsContext which maintains the control
timer and servo state. solve_ivp calls this wrapper; the wrapper
triggers a control update when the simulation time crosses a control tick.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from core.state import (
    unpack_state, quat_normalize, quat_to_rotmat, quat_derivative,
    quat_to_euler_zyx, STATE_DIM
)
from core.atmosphere import isa, speed_of_sound, G0
from core.mass_model import MassModel
from core.aerodynamics import AerodynamicsModel, AeroConfig


# ─── Gravity ──────────────────────────────────────────────────────────────────
_G_ENU = np.array([0.0, 0.0, -G0])   # gravity in ENU frame [m/s²]


# ─── PhysicsContext  (holds all stateful objects during integration) ───────────

class PhysicsContext:
    """
    Bundles all sub-models and integration state.
    The `__call__` method is passed to scipy.solve_ivp as the ODE function.

    Parameters
    ----------
    mass_model      : MassModel     — mass, CG, inertia vs. time
    aero_model      : AerodynamicsModel
    propulsion      : object        — must implement get_thrust(t_motor) → N
    t_motor_offset  : float         — offset from sim time to motor time
                                     (e.g. staging delay shifts t_motor)
    launch_alt_m    : float         — launch site altitude MSL [m]
    control_hz      : float         — control loop frequency [Hz]
    control_cb      : callable      — f(t, state_dict) → servo_angles [rad] × 4
                                     called at each control tick
    """

    def __init__(
        self,
        mass_model:     MassModel,
        aero_model:     AerodynamicsModel,
        propulsion,
        t_motor_offset: float = 0.0,
        launch_alt_m:   float = 1580.0,
        control_hz:     float = 100.0,
        control_cb      = None,
    ):
        self.mass_model      = mass_model
        self.aero_model      = aero_model
        self.propulsion      = propulsion
        self.t_motor_offset  = t_motor_offset
        self.launch_alt_m    = launch_alt_m
        self.dt_control      = 1.0 / control_hz
        self.control_cb      = control_cb

        # Control state
        self._servo_angles   = np.zeros(4)
        self._next_ctrl_time = 0.0

        # Telemetry buffer (filled during integration)
        self.telemetry: list[dict] = []
        self._last_tel_t  = -1.0
        self._tel_dt      = 0.02   # record at ~50 Hz to keep buffer size reasonable

    def reset_control(self, t_start: float = 0.0):
        """Reset control timer (call before each solve_ivp invocation)."""
        self._servo_angles   = np.zeros(4)
        self._next_ctrl_time = t_start

    # ── ODE function (called by scipy) ────────────────────────────────────────

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        """dy/dt — the 6-DOF equations of motion."""

        # ── Unpack state ──────────────────────────────────────────────────
        pos, vel, quat, omega = unpack_state(y)
        R = quat_to_rotmat(quat)   # body → ENU

        # ── Altitude and atmosphere ───────────────────────────────────────
        alt_msl = pos[2]
        alt_agl = alt_msl - self.launch_alt_m
        rho, P, T = isa(alt_msl)
        a_sound   = speed_of_sound(T)

        # ── Velocity in body frame ────────────────────────────────────────
        vel_body = R.T @ vel
        v_mag    = float(np.linalg.norm(vel))
        mach     = v_mag / a_sound if a_sound > 0 else 0.0

        # ── Mass properties ───────────────────────────────────────────────
        t_motor = t - self.t_motor_offset
        m, cg_from_nose, I = self.mass_model.get_properties(max(t_motor, 0.0))

        # ── 100 Hz control update (ZOH) ───────────────────────────────────
        if self.control_cb is not None and t >= self._next_ctrl_time:
            state_dict = {
                't'          : t,
                'pos_enu'    : pos,
                'vel_enu'    : vel,
                'quat'       : quat,
                'omega_body' : omega,
                'alt_agl'    : alt_agl,
                'speed'      : v_mag,
                'mach'       : mach,
                'rho'        : rho,
                'mass'       : m,
                'cg'         : cg_from_nose,
                'thrust'     : self.propulsion.get_thrust(max(t_motor, 0.0)),
            }
            try:
                self._servo_angles = np.asarray(
                    self.control_cb(t, state_dict), dtype=float
                )
            except Exception:
                self._servo_angles = np.zeros(4)
            self._next_ctrl_time = t + self.dt_control

        servo_angles = self._servo_angles

        # ── Thrust (along body +Z = rocket nose direction) ────────────────
        T_thrust = self.propulsion.get_thrust(max(t_motor, 0.0))
        F_thrust_body = np.array([0.0, 0.0, T_thrust])

        # ── Launch rail constraint ─────────────────────────────────────────
        # The rocket cannot move downward until thrust > weight.
        # Simulates the launch rail holding the rocket until liftoff.
        alt_agl = alt_msl - self.launch_alt_m
        if alt_agl <= 0.0:
            weight = m * G0
            if T_thrust < weight:
                # Hold on rail: zero all derivatives
                dq_dt = np.zeros(4)
                return np.concatenate([np.zeros(3), np.zeros(3), dq_dt, np.zeros(3)])
            else:
                # Constrain to vertical-only motion until AGL > 0
                vel = np.array([0.0, 0.0, max(vel[2], 0.0)])

        # ── Aerodynamic forces and moments (body frame) ────────────────────
        q_dyn = 0.5 * rho * v_mag ** 2
        F_aero_body, tau_aero_body = self.aero_model.compute(
            vel_body    = vel_body,
            omega_body  = omega,
            rho         = rho,
            mach        = mach,
            cg_from_nose= cg_from_nose,
            servo_angles= servo_angles,
        )

        # ── Total force: rotate body forces to ENU, add gravity ────────────
        F_body_total = F_thrust_body + F_aero_body
        F_enu_total  = R @ F_body_total + m * _G_ENU

        # ── Translational dynamics ────────────────────────────────────────
        dp_dt = vel.copy()
        dv_dt = F_enu_total / m

        # ── Quaternion kinematics ─────────────────────────────────────────
        dq_dt = quat_derivative(quat, omega)

        # ── Euler's rotational equations ──────────────────────────────────
        # I·ω̇ = τ − ω × (I·ω)
        I_omega   = I @ omega
        tau_gyro  = np.cross(omega, I_omega)
        try:
            domega_dt = np.linalg.solve(I, tau_aero_body - tau_gyro)
        except np.linalg.LinAlgError:
            domega_dt = np.zeros(3)

        # ── Angular rate clamping (physical safety limit) ─────────────────
        # Prevents numerical runaway during tumbling.
        # 200 rad/s ≈ 32 rev/s — well beyond any real rocket structural limit.
        OMEGA_MAX = 200.0
        omega_next = omega + domega_dt * 0.001  # rough estimate
        if np.any(np.abs(omega_next) > OMEGA_MAX):
            domega_dt = np.clip(domega_dt, -OMEGA_MAX, OMEGA_MAX)
            # Hard-clamp current omega too
            omega = np.clip(omega, -OMEGA_MAX, OMEGA_MAX)

        # ── Telemetry snapshot (throttled) ────────────────────────────────
        if t - self._last_tel_t >= self._tel_dt:
            self._record_telemetry(
                t, pos, vel, quat, omega, m, T_thrust,
                F_aero_body, tau_aero_body, servo_angles,
                mach, q_dyn, rho, cg_from_nose, alt_agl
            )
            self._last_tel_t = t

        return np.concatenate([dp_dt, dv_dt, dq_dt, domega_dt])

    def _record_telemetry(self, t, pos, vel, quat, omega, m, thrust,
                          F_aero, tau_aero, servos, mach, q_dyn, rho,
                          cg, alt_agl):
        roll, pitch, yaw = quat_to_euler_zyx(quat)
        v_mag = float(np.linalg.norm(vel))
        self.telemetry.append({
            't'         : float(t),
            'alt_agl'   : float(alt_agl),
            'alt_msl'   : float(pos[2]),
            'pos_x'     : float(pos[0]),
            'pos_y'     : float(pos[1]),
            'speed'     : v_mag,
            'vz'        : float(vel[2]),
            'mach'      : float(mach),
            'q_dyn'     : float(q_dyn),
            'rho'       : float(rho),
            'mass'      : float(m),
            'thrust'    : float(thrust),
            'drag'      : float(-F_aero[2]),  # axial component
            'roll_deg'  : float(np.degrees(roll)),
            'pitch_deg' : float(np.degrees(pitch)),
            'yaw_deg'   : float(np.degrees(yaw)),
            'roll_rate' : float(omega[0]),
            'pitch_rate': float(omega[1]),
            'yaw_rate'  : float(omega[2]),
            'servo_1_deg': float(np.degrees(servos[0])),
            'servo_2_deg': float(np.degrees(servos[1])),
            'servo_3_deg': float(np.degrees(servos[2])),
            'servo_4_deg': float(np.degrees(servos[3])),
            'tau_pitch' : float(tau_aero[0]),
            'tau_yaw'   : float(tau_aero[1]),
            'tau_roll'  : float(tau_aero[2]),
            'cg'        : float(cg),
            'qw': float(quat[0]), 'qx': float(quat[1]),
            'qy': float(quat[2]), 'qz': float(quat[3]),
        })


# ─── Solve_ivp event functions ────────────────────────────────────────────────

def make_apogee_event(launch_alt: float):
    """Event: vertical velocity crosses zero (apogee)."""
    def apogee(t, y):
        return y[5]   # ENU vz = 0 at apogee
    apogee.terminal  = True
    apogee.direction = -1   # trigger on vz going negative
    return apogee


def make_landing_event(launch_alt: float):
    """Event: rocket returns to launch altitude (AGL = 0)."""
    def landing(t, y):
        return y[2] - launch_alt
    landing.terminal  = True
    landing.direction = -1
    return landing


def make_burnout_event(motor_duration: float, t_offset: float = 0.0):
    """Event: motor burn complete."""
    def burnout(t, y):
        return (t - t_offset) - motor_duration
    burnout.terminal  = False   # don't stop — just note the event time
    burnout.direction = +1
    return burnout
