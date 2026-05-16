"""
control/sensors.py — Sensor Noise Injection
Trinity 6-DOF Simulator | Orbital Dynamics

Converts the "true" physics state into simulated sensor readings by injecting:
  • Gaussian white noise (σ per sensor channel)
  • Constant bias
  • Optional quantization

Matches the Caronte V1 dual-IMU architecture:
  • ICM-45686:  ±20g accel, ±2000 dps gyro — nominal flight
  • ADXL375:    ±200g accel                  — high-g backup
  • MS5611:     barometric altitude
  • NEO-6M:     GPS (position + velocity)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from core.state import quat_to_rotmat, quat_to_euler_zyx
from core.atmosphere import isa, G0


@dataclass
class SensorNoiseConfig:
    """Per-channel noise parameters. All units match the sensor output."""
    # ICM-45686 accelerometer
    accel_noise_sigma: float  = 0.05     # [m/s²] RMS noise
    accel_bias:        np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # [m/s²] constant bias

    # ICM-45686 gyroscope
    gyro_noise_sigma:  float  = 0.003    # [rad/s] RMS noise
    gyro_bias:         np.ndarray = field(
        default_factory=lambda: np.zeros(3))  # [rad/s] bias

    # MS5611 barometer
    baro_noise_sigma:  float  = 0.50     # [m] altitude noise
    baro_bias:         float  = 0.0      # [m] constant offset

    # NEO-6M GPS
    gps_pos_noise:     float  = 5.0      # [m] horizontal position noise
    gps_alt_noise:     float  = 10.0     # [m] vertical noise
    gps_vel_noise:     float  = 0.3      # [m/s] velocity noise
    gps_hdop:          float  = 1.2

    # Enable/disable noise injection
    enabled:           bool   = True


class SensorSimulator:
    """
    Converts true state dict (from PhysicsContext telemetry) to
    simulated noisy sensor readings as would be seen by the Caronte V1 EKF.
    """

    def __init__(self, config: SensorNoiseConfig | None = None, seed: int = 42):
        self.cfg = config or SensorNoiseConfig()
        self.rng = np.random.default_rng(seed)
        self._gps_update_dt = 1.0
        self._last_gps_t    = -1.0

    def read(self, t: float, true_state: dict,
             launch_alt_msl: float) -> dict:
        """
        Convert a true state dict into a simulated sensor data dict.

        Parameters
        ----------
        t              : float  — simulation time [s]
        true_state     : dict   — keys: pos_enu, vel_enu, quat, omega_body, …
        launch_alt_msl : float  — launch site altitude MSL [m]

        Returns
        -------
        sensor_data : dict with keys:
            accel_x/y/z   [m/s²]  body frame (gravity included)
            gyro_x/y/z    [rad/s] body frame
            baro_alt_agl  [m]
            gps_*         [m, m/s] — updated at 1 Hz only
        """
        quat       = np.asarray(true_state['quat'])
        omega_body = np.asarray(true_state['omega_body'])
        vel_enu    = np.asarray(true_state['vel_enu'])
        pos_enu    = np.asarray(true_state['pos_enu'])
        R          = quat_to_rotmat(quat)   # body → ENU

        alt_msl    = float(pos_enu[2])
        rho, P, T  = isa(alt_msl)

        # ── Accelerometer (body frame, includes gravity reaction) ──────────
        # At rest on pad: accel = [0, 0, +g] in body frame (measures reaction)
        vel_body  = R.T @ vel_enu
        # True specific force (non-gravitational acceleration in body frame)
        accel_true = R.T @ np.array([0., 0., G0])  # gravity reaction
        # In flight, add kinematic acceleration contribution
        # (the physics engine has dv/dt; we approximate with the recorded thrust)
        # For sensor simulation: specific force = R^T @ (a_enu + g_enu) = R^T @ F/m
        # Use the recorded accel directly if available
        if 'accel_body' in true_state:
            accel_true = np.asarray(true_state['accel_body'])
        else:
            # Best estimate from gravity reaction (sensor at rest measures gravity)
            gravity_body = R.T @ np.array([0., 0., G0])
            accel_true   = gravity_body  # placeholder for PAD phase

        noise = (self.cfg.accel_noise_sigma * self.rng.standard_normal(3)
                 if self.cfg.enabled else np.zeros(3))
        accel_measured = accel_true + self.cfg.accel_bias + noise

        # ── Gyroscope (body frame) ─────────────────────────────────────────
        noise_g = (self.cfg.gyro_noise_sigma * self.rng.standard_normal(3)
                   if self.cfg.enabled else np.zeros(3))
        gyro_measured = omega_body + self.cfg.gyro_bias + noise_g

        # ── Barometer → altitude AGL ───────────────────────────────────────
        alt_agl_true = alt_msl - launch_alt_msl
        noise_b = (self.cfg.baro_noise_sigma * self.rng.standard_normal()
                   if self.cfg.enabled else 0.0)
        baro_alt_agl = alt_agl_true + self.cfg.baro_bias + noise_b

        sensor_data = {
            'accel_x': float(accel_measured[0]),
            'accel_y': float(accel_measured[1]),
            'accel_z': float(accel_measured[2]),
            'gyro_x':  float(gyro_measured[0]),
            'gyro_y':  float(gyro_measured[1]),
            'gyro_z':  float(gyro_measured[2]),
            'baro_alt_agl': float(baro_alt_agl),
            'temperature': float(T),
        }

        # ── GPS (1 Hz update rate) ─────────────────────────────────────────
        if t - self._last_gps_t >= self._gps_update_dt:
            self._last_gps_t = t
            noise_p = (self.cfg.gps_pos_noise * self.rng.standard_normal(2)
                       if self.cfg.enabled else np.zeros(2))
            noise_z = (self.cfg.gps_alt_noise * self.rng.standard_normal()
                       if self.cfg.enabled else 0.0)
            noise_v = (self.cfg.gps_vel_noise * self.rng.standard_normal(3)
                       if self.cfg.enabled else np.zeros(3))
            sensor_data.update({
                'gps_x'    : float(pos_enu[0] + noise_p[0]),
                'gps_y'    : float(pos_enu[1] + noise_p[1]),
                'gps_alt'  : float(alt_msl    + noise_z),
                'gps_vx'   : float(vel_enu[0] + noise_v[0]),
                'gps_vy'   : float(vel_enu[1] + noise_v[1]),
                'gps_vz'   : float(vel_enu[2] + noise_v[2]),
                'gps_hdop' : self.cfg.gps_hdop,
                'gps_fix'  : True,
            })
        else:
            sensor_data.update({'gps_fix': False})

        return sensor_data
