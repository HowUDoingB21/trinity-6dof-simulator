"""
control/pid.py — PID Controller (Python reimplementation)
Trinity 6-DOF Simulator | Orbital Dynamics

Faithful port of PIDController.cpp / PIDController.h from Caronte V1.
All configurable gains are exposed as instance attributes (GUI-adjustable).

Full dynamic inversion:  τ = I·α_cmd + ω × (I·ω)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PIDGains:
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.1


@dataclass
class PIDOutput:
    torque_roll:   float = 0.0
    torque_pitch:  float = 0.0
    torque_yaw:    float = 0.0
    lift_roll:     float = 0.0
    lift_pitch:    float = 0.0
    lift_yaw:      float = 0.0
    saturated_roll:  bool = False
    saturated_pitch: bool = False
    saturated_yaw:   bool = False


class PIDChannel:
    def __init__(self, Kp: float = 1.0, Ki: float = 0.0, Kd: float = 0.1,
                 integral_limit: float = 10.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.integral_limit = integral_limit
        self.integral    = 0.0
        self.initialized = False

    def compute(self, error: float, error_dot: float, dt: float) -> float:
        if not self.initialized:
            self.initialized = True
        P = self.Kp * error
        self.integral = np.clip(
            self.integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        I = self.Ki * self.integral
        D = self.Kd * error_dot
        return float(P + I + D)

    def reset(self):
        self.integral = 0.0; self.initialized = False

    @property
    def saturated(self) -> bool:
        return abs(self.integral) >= self.integral_limit - 1e-6


class PIDController:
    """
    3-axis PID controller with full dynamic inversion.
    Matches Caronte V1 PIDController exactly.

    Configurable per axis: gains, integral limit, target angle.
    """

    # Default inertia tensor (kg·m²) — overridden by GUI / SimConfig
    INERTIA_DEFAULT = np.array([
        [ 0.005441, -0.011128, -0.085051],
        [-0.011128,  0.005457,  0.041753],
        [-0.085051,  0.041753,  0.048966],
    ])
    LEVER_ARM_D_DEFAULT = 0.80  # [m]

    def __init__(self,
                 gains_pitch: PIDGains = None,
                 gains_yaw:   PIDGains = None,
                 gains_roll:  PIDGains = None,
                 integral_limit: float = 10.0,
                 lever_arm_d: float = None,
                 inertia: np.ndarray = None):
        self.gains_pitch = gains_pitch or PIDGains(1.0, 0.0, 0.1)
        self.gains_yaw   = gains_yaw   or PIDGains(1.0, 0.0, 0.1)
        self.gains_roll  = gains_roll  or PIDGains(0.5, 0.0, 0.05)

        self._roll  = PIDChannel(self.gains_roll.Kp,  self.gains_roll.Ki,
                                 self.gains_roll.Kd,  integral_limit)
        self._pitch = PIDChannel(self.gains_pitch.Kp, self.gains_pitch.Ki,
                                 self.gains_pitch.Kd, integral_limit)
        self._yaw   = PIDChannel(self.gains_yaw.Kp,   self.gains_yaw.Ki,
                                 self.gains_yaw.Kd,   integral_limit)

        self.target_pitch = 0.0
        self.target_yaw   = 0.0
        self.target_roll  = 0.0

        self.lever_arm_d = lever_arm_d or self.LEVER_ARM_D_DEFAULT
        self.inertia     = inertia if inertia is not None else self.INERTIA_DEFAULT.copy()

    def update(self, roll: float, pitch: float, yaw: float,
               roll_rate: float, pitch_rate: float, yaw_rate: float,
               dt: float) -> PIDOutput:
        """
        Compute PID control output.

        Parameters match Caronte V1: angles [rad], rates [rad/s], dt [s].
        """
        # ── Errors ──────────────────────────────────────────────────────
        e_roll  = self._wrap(self.target_roll  - roll)
        e_pitch = self._wrap(self.target_pitch - pitch)
        e_yaw   = self._wrap(self.target_yaw   - yaw)

        # Error derivative = negative of angular rate
        edot_roll  = -roll_rate
        edot_pitch = -pitch_rate
        edot_yaw   = -yaw_rate

        # ── PID → angular acceleration command ───────────────────────────
        alpha_roll  = self._roll.compute(e_roll,  edot_roll,  dt)
        alpha_pitch = self._pitch.compute(e_pitch, edot_pitch, dt)
        alpha_yaw   = self._yaw.compute(e_yaw,   edot_yaw,   dt)

        # ── Dynamic inversion: τ = I·α + ω × (I·ω) ──────────────────────
        alpha   = np.array([alpha_roll, alpha_pitch, alpha_yaw])
        omega   = np.array([roll_rate,  pitch_rate,  yaw_rate])
        I_alpha = self.inertia @ alpha
        I_omega = self.inertia @ omega
        gyro_c  = np.cross(omega, I_omega)
        tau     = I_alpha + gyro_c

        # ── Lift force required (τ / d) ───────────────────────────────────
        lift = tau / self.lever_arm_d

        out = PIDOutput(
            torque_roll  = float(tau[0]),
            torque_pitch = float(tau[1]),
            torque_yaw   = float(tau[2]),
            lift_roll    = float(lift[0]),
            lift_pitch   = float(lift[1]),
            lift_yaw     = float(lift[2]),
            saturated_roll  = self._roll.saturated,
            saturated_pitch = self._pitch.saturated,
            saturated_yaw   = self._yaw.saturated,
        )
        return out

    def reset(self):
        self._roll.reset(); self._pitch.reset(); self._yaw.reset()

    def set_gains(self, axis: str, Kp: float, Ki: float, Kd: float):
        ch = {'roll': self._roll, 'pitch': self._pitch, 'yaw': self._yaw}[axis]
        ch.Kp = Kp; ch.Ki = Ki; ch.Kd = Kd

    @staticmethod
    def _wrap(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)
