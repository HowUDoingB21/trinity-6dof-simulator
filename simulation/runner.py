"""
simulation/runner.py — Simulation Orchestrator
Trinity 6-DOF Simulator | Orbital Dynamics

Coordinates all sub-models into a single simulation run:
  • Stage 1 integration  (PAD → S1 BURN → S1 COAST → STAGING)
  • Staging transition   (instantaneous mass/inertia reconfiguration)
  • Stage 2 integration  (S2 BURN → COAST → APOGEE → DESCENT → LANDED)

The physics integrator (RK45 via scipy.solve_ivp) runs with adaptive step size.
The control system (EKF9 + PID + ServoController) runs at a fixed 100 Hz
discrete rate inside a ZOH wrapper within the ODE function.

OUTPUT
──────
SimResults dataclass containing:
  • pandas DataFrame of telemetry (50 Hz sampled during integration)
  • Apogee altitude, time, max Mach, max q
  • Raw solution objects for post-processing
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from core.state import initial_state, unpack_state, quat_to_euler_zyx
from core.atmosphere import isa, speed_of_sound
from core.mass_model import MassModel, TwoStageMassModel, StageGeometry, StageInertia
from core.aerodynamics import AerodynamicsModel, AeroConfig
from core.physics import (PhysicsContext,
                           make_apogee_event,
                           make_landing_event,
                           make_burnout_event)
from control.sensors import SensorSimulator, SensorNoiseConfig
from control.ekf9 import EKF9
from control.pid import PIDController, PIDGains
from control.fsm import FlightFSM, FSMConfig, ServoController, ServoConfig, FlightPhase
from utils.eng_parser import EngData


# ─── Per-stage configuration (built from GUI values) ──────────────────────────

@dataclass
class StageConfig:
    name: str
    geometry: StageGeometry
    motor: EngData
    aero: AeroConfig
    dry_mass_kg: float


# ─── Full simulation configuration ────────────────────────────────────────────

@dataclass
class SimConfig:
    stage1:           StageConfig
    stage2:           StageConfig

    # Launch conditions
    launch_alt_msl_m: float = 1580.0
    launch_tilt_pitch: float = 0.0
    launch_tilt_yaw:   float = 0.0

    # Staging
    staging_cfg: FSMConfig = field(default_factory=FSMConfig)

    # Control
    control_enabled:  bool   = True
    control_hz:       float  = 100.0
    pid_pitch: PIDGains = field(default_factory=lambda: PIDGains(4.0, 0.05, 0.5))
    pid_yaw:   PIDGains = field(default_factory=lambda: PIDGains(4.0, 0.05, 0.5))
    pid_roll:  PIDGains = field(default_factory=lambda: PIDGains(1.0, 0.0,  0.1))
    servo_cfg: ServoConfig = field(default_factory=ServoConfig)

    # Sensor noise
    noise_cfg: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    noise_seed: int = 42

    # Integrator tolerances
    rtol: float = 1e-7
    atol: float = 1e-9

    # Simulation time limits
    max_time_s: float = 400.0


# ─── Results container ────────────────────────────────────────────────────────

@dataclass
class SimResults:
    success: bool
    message: str = ""
    telemetry_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    apogee_alt_agl_m: float  = 0.0
    apogee_alt_msl_m: float  = 0.0
    apogee_time_s:    float  = 0.0
    max_mach:         float  = 0.0
    max_q_pa:         float  = 0.0
    max_accel_g:      float  = 0.0
    staging_time_s:   float  = 0.0
    landing_time_s:   float  = 0.0
    runtime_s:        float  = 0.0


# ─── SimulationRunner ─────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Main entry point. Call `run(config)` to execute a complete two-stage flight.

    Parameters
    ----------
    progress_cb : callable(float, str) → None
        Optional callback invoked with (fraction_done, message).
        Used to update a progress bar in the GUI.
        IMPORTANT: may be called from a worker thread.
    """

    def __init__(self, progress_cb: Callable[[float, str], None] | None = None):
        self.progress_cb = progress_cb or (lambda f, m: None)
        self._abort      = threading.Event()

    def abort(self):
        """Signal the running simulation to stop cleanly."""
        self._abort.set()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, config: SimConfig) -> SimResults:
        """Execute a full two-stage flight simulation."""
        self._abort.clear()
        t_wall_start = time.perf_counter()

        try:
            results = self._run_internal(config)
        except Exception as exc:
            import traceback
            results = SimResults(
                success=False,
                message=f"Simulation error: {exc}\n{traceback.format_exc()}"
            )

        results.runtime_s = time.perf_counter() - t_wall_start
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_internal(self, cfg: SimConfig) -> SimResults:
        self.progress_cb(0.0, "Initialising…")

        # ── Build models ───────────────────────────────────────────────────
        # All CG positions are from S2's nose (complete rocket nose).
        # No frame conversion needed — the user enters values in this frame.
        s1 = cfg.stage1
        s2 = cfg.stage2

        # Stage 1 flight: S1+S2 fly together as one vehicle
        mass1 = TwoStageMassModel(
            s1_geom  = s1.geometry,
            s1_motor = s1.motor,
            s1_dry   = s1.dry_mass_kg,
            s2_geom  = s2.geometry,
            s2_motor = s2.motor,
            s2_dry   = s2.dry_mass_kg,
        )
        aero1 = AerodynamicsModel(s1.aero)  # CP already in S2-nose frame

        # Stage 2 flight: S2 alone
        mass2 = MassModel(s2.geometry, s2.motor, dry_mass_override=s2.dry_mass_kg)
        aero2 = AerodynamicsModel(s2.aero)  # CP from S2 nose frame

        # ── Sensor simulator + control system ─────────────────────────────
        sensor_sim  = SensorSimulator(cfg.noise_cfg, seed=cfg.noise_seed)
        ekf         = EKF9()
        pid         = PIDController(
            gains_pitch = cfg.pid_pitch,
            gains_yaw   = cfg.pid_yaw,
            gains_roll  = cfg.pid_roll,
            lever_arm_d = cfg.servo_cfg.lever_arm_d_m,
            inertia     = s2.geometry.inertia_dry.as_matrix(),
        )
        servo_ctrl  = ServoController(cfg.servo_cfg, aero_fin=s2.aero.fin_aero)
        fsm         = FlightFSM(
            config       = cfg.staging_cfg,
            s1_burn_time_s = s1.motor.burn_time_s,
            s2_burn_time_s = s2.motor.burn_time_s,
        )

        ekf.begin(alt0=0.0)
        ekf.set_ground_alt_msl(cfg.launch_alt_msl_m)

        # Mutable control state (captured in closure below)
        ctrl_state = {
            'servo_angles': np.zeros(4),
            'phase': FlightPhase.PAD,
        }

        def control_callback(t: float, state_dict: dict) -> np.ndarray:
            return self._control_step(
                t, state_dict, cfg, sensor_sim, ekf, pid,
                servo_ctrl, fsm, ctrl_state
            )

        # ── Initial conditions ─────────────────────────────────────────────
        y0 = initial_state(
            launch_altitude      = cfg.launch_alt_msl_m,
            initial_tilt_pitch_deg = cfg.launch_tilt_pitch,
            initial_tilt_yaw_deg   = cfg.launch_tilt_yaw,
        )

        # ════════════════════════════════════════════════════════════════════
        #  PHASE A: Stage 1 (PAD → BURNOUT → STAGING)
        # ════════════════════════════════════════════════════════════════════
        self.progress_cb(0.05, "Integrating Stage 1…")

        ctx1 = PhysicsContext(
            mass_model    = mass1,
            aero_model    = aero1,
            propulsion    = s1.motor,
            t_motor_offset= 0.0,
            launch_alt_m  = cfg.launch_alt_msl_m,
            control_hz    = cfg.control_hz,
            control_cb    = control_callback if cfg.control_enabled else None,
        )
        ctx1.reset_control(0.0)

        # Events for Stage 1 integration
        ev_burnout1 = make_burnout_event(s1.motor.burn_time_s, t_offset=0.0)
        ev_staging  = self._make_staging_event(cfg.staging_cfg, cfg.launch_alt_msl_m)

        t_end_s1 = min(s1.motor.burn_time_s
                       + max(cfg.staging_cfg.staging_delay_s, 60.0),
                       cfg.max_time_s * 0.4)

        sol1 = solve_ivp(
            fun    = ctx1,
            t_span = (0.0, t_end_s1),
            y0     = y0,
            method = 'RK45',
            rtol   = cfg.rtol,
            atol   = cfg.atol,
            events = [ev_staging],
            dense_output=False,
            max_step = 0.1,
        )

        if self._abort.is_set():
            return SimResults(success=False, message="Aborted by user.")

        self.progress_cb(0.35, "Applying staging…")

        # Staging time & state
        t_staging = float(sol1.t[-1])
        y_staging = sol1.y[:, -1].copy()

        # Record staging time
        staging_time_s = t_staging
        ctrl_state['phase'] = FlightPhase.STAGING

        # ════════════════════════════════════════════════════════════════════
        #  STAGING TRANSITION — reconfigure to Stage 2
        # ════════════════════════════════════════════════════════════════════
        # At staging, Stage 1 hardware is jettisoned.
        # The state vector position/velocity/attitude are PRESERVED.
        # Only mass/inertia/aero model switch to Stage 2.
        # A small separation impulse may be applied (future feature).
        y_s2_init = y_staging.copy()

        # Optional: tiny separation velocity kick along body -Z (pushes stages apart)
        # y_s2_init[5] -= separation_dv   (not implemented in MVP)

        # ════════════════════════════════════════════════════════════════════
        #  PHASE B: Stage 2 (S2 BURN → COAST → APOGEE → DESCENT)
        # ════════════════════════════════════════════════════════════════════
        self.progress_cb(0.40, "Integrating Stage 2…")

        # Stage 2 motor time offset: motor time starts at 0 when S2 ignites
        t_s2_ignition = t_staging + cfg.staging_cfg.s2_ignition_delay_s
        ekf.reset_yaw()
        pid.reset()
        servo_ctrl.reset()

        ctx2 = PhysicsContext(
            mass_model     = mass2,
            aero_model     = aero2,
            propulsion     = s2.motor,
            t_motor_offset = t_s2_ignition,   # maps sim time → motor time
            launch_alt_m   = cfg.launch_alt_msl_m,
            control_hz     = cfg.control_hz,
            control_cb     = control_callback if cfg.control_enabled else None,
        )
        ctx2.reset_control(t_staging)

        # Events for Stage 2 integration
        alt_at_staging = float(y_s2_init[2]) - cfg.launch_alt_msl_m
        ev_apogee  = make_apogee_event(cfg.launch_alt_msl_m)
        ev_landing = make_landing_event(cfg.launch_alt_msl_m)

        t_end_s2 = cfg.max_time_s

        sol2 = solve_ivp(
            fun    = ctx2,
            t_span = (t_staging, t_end_s2),
            y0     = y_s2_init,
            method = 'RK45',
            rtol   = cfg.rtol,
            atol   = cfg.atol,
            events = [ev_apogee, ev_landing],
            dense_output=False,
            max_step = 0.1,
        )

        if self._abort.is_set():
            return SimResults(success=False, message="Aborted by user.")

        self.progress_cb(0.90, "Post-processing results…")

        # ── Combine telemetry from both phases ────────────────────────────
        all_tel = ctx1.telemetry + ctx2.telemetry
        df      = self._build_dataframe(all_tel)

        # ── Compute summary statistics ────────────────────────────────────
        res = self._compute_summary(df, staging_time_s, cfg.launch_alt_msl_m)
        res.telemetry_df = df
        res.staging_time_s = staging_time_s
        res.success = True

        self.progress_cb(1.0, f"Done! Apogee = {res.apogee_alt_agl_m/1000:.1f} km "
                              f"at t = {res.apogee_time_s:.1f} s")
        return res

    # ── 100 Hz control step ───────────────────────────────────────────────────

    def _control_step(self, t: float, state_dict: dict,
                      cfg: SimConfig,
                      sensor_sim: SensorSimulator,
                      ekf: EKF9, pid: PIDController,
                      servo_ctrl: ServoController,
                      fsm: FlightFSM,
                      ctrl_state: dict) -> np.ndarray:
        """
        Called by PhysicsContext at each 100 Hz control tick.
        Returns servo deflection angles [rad] × 4.
        """
        dt = 1.0 / cfg.control_hz

        # Build sensor data
        true_state_for_sensor = {
            'quat'       : state_dict['quat'],
            'omega_body' : state_dict['omega_body'],
            'vel_enu'    : state_dict['vel_enu'],
            'pos_enu'    : state_dict['pos_enu'],
        }
        sensors = sensor_sim.read(t, true_state_for_sensor, cfg.launch_alt_msl_m)

        # EKF predict + update
        ekf.predict(
            accel_x = sensors['accel_x'],
            accel_y = sensors['accel_y'],
            accel_z = sensors['accel_z'],
            gyro_x  = sensors['gyro_x'],
            gyro_y  = sensors['gyro_y'],
            gyro_z  = sensors['gyro_z'],
            dt      = dt,
        )
        ekf.update(
            baro_alt_agl = sensors['baro_alt_agl'],
            accel_x      = sensors['accel_x'],
            accel_y      = sensors['accel_y'],
            accel_z      = sensors['accel_z'],
        )
        if sensors.get('gps_fix'):
            ekf.update_gps_alt(
                gps_alt_msl = sensors.get('gps_alt', cfg.launch_alt_msl_m),
                hdop        = sensors.get('gps_hdop', 2.0),
            )

        # FSM update
        fsm_state = {
            'alt_agl': ekf.altitude,
            'speed'  : state_dict.get('speed', 0.0),
            'thrust' : state_dict.get('thrust', 0.0),
            'vz'     : state_dict.get('vel_enu', np.zeros(3))[2],
        }
        phase = fsm.update(t, fsm_state)
        ctrl_state['phase'] = phase

        # Control: active only during burn and coast phases
        if not fsm.control_active or not cfg.control_enabled:
            servo_ctrl.reset()
            return np.zeros(4)

        # PID update
        pid_out = pid.update(
            # ── MAPPING CORRECTO DE EJES ────────────────────────────────────
            #
            # Separación de canales (sin acoplamiento cruzado):
            #
            #  d_roll  → tau_Z (body-Z spin) via r_fin*(F1-F2+F3-F4)
            #  d_pitch → tau_X (body-X)      via (F1-F3)*D   [tau_roll=0]
            #  d_yaw   → tau_Y (body-Y)      via (F2-F4)*D   [tau_roll=0]
            #
            #  EKF.yaw   = giro sobre body-Z      → canal roll  → d_roll  ✓
            #  EKF.roll  = inclinación hacia +Y    → canal pitch → d_pitch ✓ (negado)
            #  EKF.pitch = inclinación hacia +X    → canal yaw   → d_yaw   ✓
            #
            roll       = ekf.yaw,          # spin control
            pitch      = -ekf.roll,         # tilt-Y correction (body-X torque)
            yaw        = ekf.pitch,         # tilt-X correction (body-Y torque)
            roll_rate  = sensors['gyro_z'] - ekf.gyro_bias[2],  # spin rate → amortigua giro
            pitch_rate = sensors['gyro_x'] - ekf.gyro_bias[0],  # body-X rate
            yaw_rate   = sensors['gyro_y'] - ekf.gyro_bias[1],  # body-Y rate
            dt         = dt,
        )

        # Servo controller
        rho   = state_dict.get('rho', 1.225)
        speed = state_dict.get('speed', 0.0)
        q_dyn = 0.5 * rho * speed ** 2
        servo_angles = servo_ctrl.compute(pid_out, q_dyn, t, dt)

        return servo_angles

    # ── Staging event factory ─────────────────────────────────────────────────

    @staticmethod
    def _make_staging_event(fsm_cfg: FSMConfig, launch_alt_msl: float):
        """Creates a solve_ivp event function for staging."""
        if fsm_cfg.staging_mode == 'altitude':
            target_alt_msl = launch_alt_msl + fsm_cfg.staging_altitude_m
            def ev(t, y):
                return y[2] - target_alt_msl
        elif fsm_cfg.staging_mode == 'time':
            def ev(t, y):
                return t - fsm_cfg.staging_time_s
        else:
            def ev(t, y):
                return y[2] - (launch_alt_msl + 5_000.0)   # fallback

        ev.terminal  = True
        ev.direction = +1
        return ev

    # ── Telemetry assembly ────────────────────────────────────────────────────

    @staticmethod
    def _build_dataframe(telemetry: list[dict]) -> pd.DataFrame:
        if not telemetry:
            return pd.DataFrame()
        df = pd.DataFrame(telemetry)
        df.sort_values('t', inplace=True)
        df.drop_duplicates(subset='t', keep='last', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _compute_summary(df: pd.DataFrame, staging_t: float,
                         launch_alt_msl: float) -> SimResults:
        if df.empty:
            return SimResults(success=True, message="No telemetry data.")

        # Apogee
        idx_apo = df['alt_agl'].idxmax()
        apogee_agl = float(df.loc[idx_apo, 'alt_agl'])
        apogee_msl = float(df.loc[idx_apo, 'alt_msl'])
        apogee_t   = float(df.loc[idx_apo, 't'])

        max_mach = float(df['mach'].max())
        max_q    = float(df['q_dyn'].max())
        # Max accel: need to derive from speed/thrust if not directly recorded
        max_g = max_mach   # placeholder — computed from thrust data

        return SimResults(
            success          = True,
            apogee_alt_agl_m = apogee_agl,
            apogee_alt_msl_m = apogee_msl,
            apogee_time_s    = apogee_t,
            max_mach         = max_mach,
            max_q_pa         = max_q,
            staging_time_s   = staging_t,
        )
