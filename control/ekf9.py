"""
control/ekf9.py — EKF9 State Estimator (Python reimplementation)
Trinity 6-DOF Simulator | Orbital Dynamics

Direct port of EKF9.cpp / EKF9.h from Caronte V1 firmware.
All algorithm logic, state definitions, and noise parameters are preserved.
Configurable parameters (Q, R matrices) are exposed as instance attributes
so they can be adjusted from the GUI.

STATE VECTOR x[9]:
  x[0]  altitude AGL           [m]
  x[1]  vertical speed Vz      [m/s]
  x[2]  roll  φ                [rad]
  x[3]  pitch θ                [rad]
  x[4]  yaw   ψ                [rad]
  x[5]  gyro bias X            [rad/s]
  x[6]  gyro bias Y            [rad/s]
  x[7]  gyro bias Z            [rad/s]
  x[8]  accel vertical bias    [m/s²]

MEASUREMENTS z[3]:
  z[0]  barometric altitude AGL
  z[1]  roll from accelerometer  = atan2(ay, az)
  z[2]  pitch from accelerometer = atan2(-ax, √(ay²+az²))

Reference: Caronte V1 EKF9.cpp — Orbital Dynamics Rev 9.0
"""

from __future__ import annotations
import numpy as np

EKF_N = 9    # State dimension
EKF_M = 3    # Measurement dimension


class EKF9:
    """
    9-state Extended Kalman Filter for altitude + attitude estimation.
    Matches the Caronte V1 firmware implementation exactly.
    """

    # ── Tunable noise parameters (configurable from GUI) ──────────────────────
    Q_ALT    = 0.005e0    # Process noise: altitude    [m²/s]
    Q_VZ     = 0.10e0     # Process noise: vert speed  [(m/s)²/s]
    Q_ATT    = 1e-5       # Process noise: attitude    [rad²/s]
    Q_GBIAS  = 1e-8       # Process noise: gyro bias   [(rad/s)²/s]
    Q_ABIAS  = 2e-6       # Process noise: accel bias  [(m/s²)²/s]

    R_BARO   = 0.25e0     # Meas. noise: barometer     [m²]
    R_ATT_MIN = 0.01e0    # Meas. noise: attitude min  [rad²]
    R_ATT_MAX = 1e6       # Meas. noise: attitude max  [rad²]
    R_ATT_DEV = 1.0       # Deviation threshold: |a|/g departure

    R_GPS_ALT_BASE = 100.0   # GPS alt noise base [m²] at HDOP=1
    GPS_HDOP_MAX   = 4.0

    G0 = 9.80665

    def __init__(self):
        self.x    = np.zeros(EKF_N)           # state vector
        self.P    = np.zeros((EKF_N, EKF_N))  # covariance matrix
        self._alpha_eq        = 0.9997
        self._high_g_mode     = False
        self._ground_alt_msl  = 0.0

    # ── Initialisation ────────────────────────────────────────────────────────

    def begin(self, roll0: float = 0.0, pitch0: float = 0.0,
              yaw0: float = 0.0, alt0: float = 0.0):
        """Reset state and covariance. Call once before the flight."""
        self._high_g_mode = False
        self.x[:] = 0.0
        self.x[0] = alt0
        self.x[2] = roll0
        self.x[3] = pitch0
        self.x[4] = yaw0

        self.P[:] = 0.0
        self.P[0, 0] = 5.0      # altitude: ±2.2 m
        self.P[1, 1] = 1.0      # vertical speed: ±1 m/s
        self.P[2, 2] = 0.01     # roll:  ±0.1 rad
        self.P[3, 3] = 0.01     # pitch: ±0.1 rad
        self.P[4, 4] = 0.1      # yaw:   ±0.3 rad
        self.P[5, 5] = 1e-4     # gyro bias X
        self.P[6, 6] = 1e-4     # gyro bias Y
        self.P[7, 7] = 1e-4     # gyro bias Z
        self.P[8, 8] = 0.25     # accel bias

        self._alpha_eq = 0.9997

    # ── Prediction step (100 Hz) ───────────────────────────────────────────────

    def predict(self, accel_x: float, accel_y: float, accel_z: float,
                gyro_x: float, gyro_y: float, gyro_z: float, dt: float):
        """
        EKF prediction step — propagate state with IMU data.
        Matches EKF9::predict() in Caronte V1.
        """
        dt2 = dt * dt

        phi   = self.x[2]  # roll
        theta = self.x[3]  # pitch

        sp, cp = np.sin(theta), np.cos(theta)
        sr, cr = np.sin(phi),   np.cos(phi)

        # Bias-corrected angular rates
        wx = gyro_x - self.x[5]
        wy = gyro_y - self.x[6]
        wz = gyro_z - self.x[7]

        ax, ay, az = accel_x, accel_y, accel_z

        # Vertical inertial acceleration
        az_in = (-ax*sp + ay*sr*cp + az*cr*cp) - self.G0 - self.x[8]

        # Partial derivatives of az_in for Jacobian
        daz_dphi   = ay*cr*cp - az*sr*cp
        daz_dtheta = -ax*cp - ay*sr*sp - az*cr*sp

        # ── State propagation ────────────────────────────────────────────
        self.x[0] += self.x[1]*dt + 0.5*az_in*dt2
        self.x[1] += az_in * dt
        self.x[2]  = self._wrap(self.x[2] + wx*dt)
        self.x[3]  = self._wrap(self.x[3] + wy*dt)
        self.x[4]  = self._wrap(self.x[4] + wz*dt)
        # x[5..8]: bias random walk (driven only by Q)

        # ── Jacobian F (identity + sparse deviations) ─────────────────────
        F = np.eye(EKF_N)
        F[0, 1] = dt
        F[0, 2] = daz_dphi   * 0.5 * dt2
        F[0, 3] = daz_dtheta * 0.5 * dt2
        F[0, 8] = -0.5 * dt2
        F[1, 2] = daz_dphi   * dt
        F[1, 3] = daz_dtheta * dt
        F[1, 8] = -dt
        F[2, 5] = -dt
        F[3, 6] = -dt
        F[4, 7] = -dt

        # ── Process noise Q (diagonal) ───────────────────────────────────
        Q_diag = np.array([
            self.Q_ALT, self.Q_VZ,
            self.Q_ATT, self.Q_ATT, self.Q_ATT,
            self.Q_GBIAS, self.Q_GBIAS, self.Q_GBIAS,
            self.Q_ABIAS,
        ])
        if self._high_g_mode:
            Q_diag[1] *= 25.0
            Q_diag[8] *= 5.0

        # ── P = F·P·F^T + Q ──────────────────────────────────────────────
        FP   = F @ self.P
        self.P = FP @ F.T
        for i in range(EKF_N):
            self.P[i, i] += Q_diag[i]
        self._symmetrise_P()

    # ── Update step (barometric + attitude) ────────────────────────────────────

    def update(self, baro_alt_agl: float,
               accel_x: float, accel_y: float, accel_z: float):
        """
        EKF update with barometric altitude and accelerometer-derived attitude.
        Matches EKF9::update() in Caronte V1.
        """
        ax, ay, az = accel_x, accel_y, accel_z

        # Adaptive R for attitude (inflates during high-g)
        a_mag     = np.sqrt(ax*ax + ay*ay + az*az)
        deviation = abs(a_mag - self.G0) / self.G0
        t         = min(deviation / self.R_ATT_DEV, 1.0)
        r_att     = self.R_ATT_MIN + t * (self.R_ATT_MAX - self.R_ATT_MIN)

        self._alpha_eq = 1.0 - (self.R_ATT_MIN / r_att) * (1.0 - 0.95)
        R_diag = np.array([self.R_BARO, r_att, r_att])

        # Measurement model h(x)
        h0 = self.x[0]
        h1 = np.arctan2(ay, az)
        h2 = np.arctan2(-ax, np.sqrt(ay*ay + az*az))

        # Innovation
        y_inn = np.array([
            baro_alt_agl - h0,
            self._wrap(h1 - self.x[2]),
            self._wrap(h2 - self.x[3]),
        ])

        # H sparse: H[0] → x[0], H[1] → x[2], H[2] → x[3]
        idx = [0, 2, 3]

        # S = H·P·H^T + R  (3×3 submatrix)
        S = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                S[i, j] = self.P[idx[i], idx[j]]
            S[i, i] += R_diag[i]

        # S^{-1} analytic 3×3
        Sinv = self._invert3x3(S)
        if Sinv is None:
            return

        # PHt = P · H^T  (9×3)
        PHt = self.P[:, idx]   # columns idx[0], idx[1], idx[2]

        # K = PHt · Sinv  (9×3)
        K = PHt @ Sinv

        # State update: x += K · y
        self.x += K @ y_inn
        self.x[2] = self._wrap(self.x[2])
        self.x[3] = self._wrap(self.x[3])
        self.x[4] = self._wrap(self.x[4])

        # Covariance update: P = (I - K·H)·P
        H = np.zeros((3, EKF_N))
        H[0, 0] = 1.0
        H[1, 2] = 1.0
        H[2, 3] = 1.0
        IKH = np.eye(EKF_N) - K @ H
        Pnew = IKH @ self.P
        np.fill_diagonal(Pnew, np.maximum(np.diag(Pnew), 1e-9))
        self.P = Pnew
        self._symmetrise_P()

    # ── GPS altitude update ────────────────────────────────────────────────────

    def update_gps_alt(self, gps_alt_msl: float, hdop: float):
        """
        Loosely-coupled GPS altitude correction.
        Matches EKF9::updateGPSAlt() in Caronte V1.
        """
        if hdop > self.GPS_HDOP_MAX or self._ground_alt_msl == 0.0:
            return
        gps_agl = gps_alt_msl - self._ground_alt_msl
        if abs(gps_agl - self.x[0]) > 500.0:
            return

        hdop_c = max(hdop, 1.0)
        R_gps  = self.R_GPS_ALT_BASE * hdop_c * hdop_c

        innov  = gps_agl - self.x[0]
        S      = self.P[0, 0] + R_gps
        K      = self.P[:, 0] / S

        self.x    += K * innov
        self.x[2]  = self._wrap(self.x[2])
        self.x[3]  = self._wrap(self.x[3])
        self.x[4]  = self._wrap(self.x[4])

        for i in range(EKF_N):
            self.P[i, :] -= K[i] * self.P[0, :]
        np.fill_diagonal(self.P, np.maximum(np.diag(self.P), 1e-9))
        self._symmetrise_P()

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def altitude(self)      -> float: return float(self.x[0])
    @property
    def vertical_speed(self)-> float: return float(self.x[1])
    @property
    def roll(self)          -> float: return float(self.x[2])
    @property
    def pitch(self)         -> float: return float(self.x[3])
    @property
    def yaw(self)           -> float: return float(self.x[4])
    @property
    def gyro_bias(self)     -> np.ndarray: return self.x[5:8].copy()
    @property
    def var_altitude(self)  -> float: return float(self.P[0, 0])
    @property
    def var_roll(self)      -> float: return float(self.P[2, 2])
    @property
    def var_pitch(self)     -> float: return float(self.P[3, 3])
    @property
    def alpha_equivalent(self) -> float: return self._alpha_eq

    def inject_gyro_bias(self, dbx: float, dby: float, dbz: float):
        self.x[5] += dbx; self.x[6] += dby; self.x[7] += dbz

    def set_high_g_mode(self, enable: bool):
        self._high_g_mode = enable

    def set_ground_alt_msl(self, alt_msl: float):
        self._ground_alt_msl = float(alt_msl)

    def reset_yaw(self, yaw_rad: float = 0.0):
        self.x[4] = yaw_rad
        self.P[4, 4] = 0.1

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def _symmetrise_P(self):
        avg = 0.5 * (self.P + self.P.T)
        self.P[:] = avg

    @staticmethod
    def _invert3x3(A: np.ndarray) -> np.ndarray | None:
        a,b,c = A[0,0],A[0,1],A[0,2]
        d,e,f = A[1,0],A[1,1],A[1,2]
        g,h,ii= A[2,0],A[2,1],A[2,2]
        det = a*(e*ii - f*h) - b*(d*ii - f*g) + c*(d*h - e*g)
        if abs(det) < 1e-12:
            return None
        inv = 1.0 / det
        return np.array([
            [ (e*ii-f*h)*inv, -(b*ii-c*h)*inv,  (b*f-c*e)*inv],
            [-(d*ii-f*g)*inv,  (a*ii-c*g)*inv, -(a*f-d*c)*inv],
            [ (d*h-e*g)*inv,  -(a*h-b*g)*inv,   (a*e-b*d)*inv],
        ])
