"""
core/state.py — State Vector & Quaternion Math
Trinity 6-DOF Simulator | Orbital Dynamics

═══════════════════════════════════════════════════════════════════════════════
COORDINATE SYSTEM CONVENTIONS
═══════════════════════════════════════════════════════════════════════════════

INERTIAL FRAME (ENU — East / North / Up):
  +X = East
  +Y = North
  +Z = Up (altitude)

BODY FRAME (rocket-fixed):
  +Z = longitudinal axis, pointing toward nose (thrust direction)
  +X = fin 1 / fin 3 radial plane
  +Y = fin 2 / fin 4 radial plane  (right-hand: X × Y = Z)

At vertical launch (nose pointing up):
  Body Z  ≡  ENU Z  →  quaternion q = [1, 0, 0, 0]  (identity)

QUATERNION CONVENTION: Hamilton [w, x, y, z]
  q represents the rotation that transforms vectors from BODY → ENU
  v_ENU  = R(q) @ v_body
  v_body = R(q).T @ v_ENU

═══════════════════════════════════════════════════════════════════════════════
STATE VECTOR  y  (13 elements):
  y[0:3]   = position  [px, py, pz]    ENU [m]
  y[3:6]   = velocity  [vx, vy, vz]    ENU [m/s]
  y[6:10]  = quaternion [qw, qx, qy, qz]  body → ENU
  y[10:13] = angular velocity [ωx, ωy, ωz]  body frame [rad/s]
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ─── Index constants ──────────────────────────────────────────────────────────
IDX_POS   = slice(0,  3)   # position
IDX_VEL   = slice(3,  6)   # velocity
IDX_QUAT  = slice(6,  10)  # quaternion [w, x, y, z]
IDX_OMEGA = slice(10, 13)  # angular velocity
STATE_DIM = 13


# ─── Quaternion utilities ─────────────────────────────────────────────────────

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion in-place; return unit quaternion."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both in [w, x, y, z] form."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Rotation matrix R(q) — transforms body → ENU.
    v_ENU = R @ v_body
    """
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build unit quaternion from ZYX Euler angles (rad), convention:
    yaw (ψ) → pitch (θ) → roll (φ).
    """
    cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def quat_to_euler_zyx(q: np.ndarray) -> tuple[float, float, float]:
    """
    Extract ZYX Euler angles (roll, pitch, yaw) in radians from quaternion.
    Returns (roll, pitch, yaw).
    """
    w, x, y, z = quat_normalize(q)
    # Roll (φ)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # Pitch (θ) — clamp to avoid arcsin domain error
    sinp = 2.0 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # Yaw (ψ)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """
    Quaternion kinematic equation:  q̇ = ½ · q ⊗ [0, ωx, ωy, ωz]
    omega_body: angular velocity vector in body frame [rad/s]
    """
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_quat)


# ─── State vector packing / unpacking ────────────────────────────────────────

def pack_state(pos: np.ndarray, vel: np.ndarray,
               quat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Pack individual arrays into the 13-element ODE state vector."""
    return np.concatenate([pos, vel, quat_normalize(quat), omega])


def unpack_state(y: np.ndarray) -> tuple:
    """
    Unpack ODE state vector → (pos, vel, quat_norm, omega).
    Quaternion is always returned normalized.
    """
    pos   = y[IDX_POS].copy()
    vel   = y[IDX_VEL].copy()
    quat  = quat_normalize(y[IDX_QUAT].copy())
    omega = y[IDX_OMEGA].copy()
    return pos, vel, quat, omega


def initial_state(launch_altitude: float = 1580.0,
                  initial_tilt_pitch_deg: float = 0.0,
                  initial_tilt_yaw_deg: float = 0.0) -> np.ndarray:
    """
    Build initial state vector for a vertical launch.

    Parameters
    ----------
    launch_altitude : float
        Altitude MSL of the launch site [m]. Guadalajara ≈ 1580 m.
    initial_tilt_pitch_deg : float
        Small initial pitch misalignment from vertical [°].
    initial_tilt_yaw_deg : float
        Small initial yaw misalignment from vertical [°].
    """
    pos   = np.array([0.0, 0.0, launch_altitude])
    vel   = np.zeros(3)
    quat  = quat_from_euler_zyx(
        roll=0.0,
        pitch=np.deg2rad(initial_tilt_pitch_deg),
        yaw=np.deg2rad(initial_tilt_yaw_deg),
    )
    omega = np.zeros(3)
    return pack_state(pos, vel, quat, omega)


# ─── Rich simulation record (one per time step) ──────────────────────────────

@dataclass
class SimRecord:
    """
    Complete record of one simulation time step.
    Stored in the results list produced by SimulationRunner.
    """
    t: float                          # simulation time [s]
    pos: np.ndarray                   # ENU position [m]
    vel: np.ndarray                   # ENU velocity [m/s]
    quat: np.ndarray                  # quaternion [w,x,y,z]
    omega: np.ndarray                 # angular velocity body [rad/s]
    roll: float = 0.0                 # [rad]
    pitch: float = 0.0                # [rad]
    yaw: float = 0.0                  # [rad]
    altitude_agl: float = 0.0        # above launch altitude [m]
    speed: float = 0.0               # |velocity| [m/s]
    mach: float = 0.0
    aoa_total: float = 0.0           # total angle of attack [rad]
    q_dynamic: float = 0.0           # dynamic pressure [Pa]
    mass: float = 0.0                # total mass [kg]
    thrust: float = 0.0             # [N]
    drag: float = 0.0               # [N]
    normal_force: float = 0.0       # [N]
    servo_1: float = 0.0            # fin deflections [rad]
    servo_2: float = 0.0
    servo_3: float = 0.0
    servo_4: float = 0.0
    tau_pitch: float = 0.0          # control torques [N·m]
    tau_yaw: float = 0.0
    tau_roll: float = 0.0
    phase: int = 0                   # FlightPhase enum value
    air_density: float = 1.225
    static_margin: float = 0.0      # (CP - CG) / diameter [calibers]

    def __post_init__(self):
        # Compute derived scalars from arrays
        roll, pitch, yaw = quat_to_euler_zyx(self.quat)
        self.roll  = roll
        self.pitch = pitch
        self.yaw   = yaw
        self.speed = float(np.linalg.norm(self.vel))
