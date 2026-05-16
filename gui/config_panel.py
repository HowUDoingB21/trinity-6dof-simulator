"""
gui/config_panel.py — Simulation Configuration Panel
Trinity 6-DOF Simulator | Orbital Dynamics

Left-side tabbed panel with four tabs:
  • Stages     — mass, geometry, motor (.eng), inertia tensor
  • Aero       — Cd/Cn CSV import, aero_table.h import, CP, fin geometry
  • Control    — PID gains, EKF Q/R, servo parameters
  • Environment — launch site, wind (future)
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QGroupBox, QLabel, QDoubleSpinBox, QSpinBox, QLineEdit,
    QPushButton, QFileDialog, QCheckBox, QComboBox, QSizePolicy,
    QScrollArea, QFrame, QGridLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import numpy as np

from core.mass_model import StageGeometry, StageInertia
from core.aerodynamics import AeroConfig
from control.pid import PIDGains
from control.fsm import FSMConfig, ServoConfig
from control.sensors import SensorNoiseConfig


# ─── Shared widget factories ──────────────────────────────────────────────────

def _dspin(val=0.0, lo=0.0, hi=1e6, step=0.01, dec=4, suffix="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setSingleStep(step)
    w.setDecimals(dec); w.setValue(val)
    w.setSuffix(f"  {suffix}" if suffix else "")
    return w


def _group(title: str, layout) -> QGroupBox:
    g = QGroupBox(title)
    g.setLayout(layout)
    return g


def _file_row(label: str, placeholder: str, parent_layout,
              cb_browse) -> QLineEdit:
    """Add a file-picker row and return the QLineEdit."""
    row = QHBoxLayout()
    le  = QLineEdit(); le.setPlaceholderText(placeholder); le.setReadOnly(True)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(80)
    btn.clicked.connect(cb_browse)
    row.addWidget(le); row.addWidget(btn)
    parent_layout.addRow(label, row)   # type: ignore
    return le


# ─── Stage configuration widget ───────────────────────────────────────────────

class StageWidget(QWidget):
    """Form for one rocket stage (mass, geometry, inertia, motor)."""

    def __init__(self, defaults: dict, parent=None):
        super().__init__(parent)
        d = defaults

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Mass & geometry ─────────────────────────────────────────────
        fm = QFormLayout(); fm.setSpacing(6)
        self.dry_mass   = _dspin(d.get('dry_mass', 8.0),    0, 500,  0.1,  3, "kg")
        self.prop_mass  = _dspin(d.get('prop_mass', 5.0),   0, 500,  0.1,  3, "kg")
        self.diameter   = _dspin(d.get('diameter', 0.16),   0, 2.0,  0.001,4, "m")
        self.length     = _dspin(d.get('length', 2.0),      0, 20.0, 0.01, 3, "m")
        self.cg_dry     = _dspin(d.get('cg_dry', 0.90),     0, 30.0, 0.01, 4, "m")
        self.cg_prop    = _dspin(d.get('cg_prop', 1.20),    0, 30.0, 0.01, 4, "m")
        self.grain_len  = _dspin(d.get('grain_len', 0.60),  0, 5.0,  0.01, 3, "m")
        self.grain_od   = _dspin(d.get('grain_od', 0.075),  0, 0.5,  0.001,4, "m")
        self.grain_id   = _dspin(d.get('grain_id', 0.0),    0, 0.5,  0.001,4, "m")
        fm.addRow("Dry mass",          self.dry_mass)
        fm.addRow("Propellant mass",   self.prop_mass)
        fm.addRow("Body diameter",     self.diameter)
        fm.addRow("Total length",      self.length)
        fm.addRow("CG lleno (desde nariz S2)", self.cg_dry)
        fm.addRow("CG vacío / burnout (desde nariz S2)", self.cg_prop)
        fm.addRow("Grain length",      self.grain_len)
        fm.addRow("Grain OD",          self.grain_od)
        fm.addRow("Grain ID (BATES)",  self.grain_id)
        root.addWidget(_group("Mass & Geometry", fm))

        # ── Motor ────────────────────────────────────────────────────────
        fm2 = QFormLayout(); fm2.setSpacing(6)
        self._eng_path = ""
        self._eng_le   = _file_row("Thrust curve (.eng)", "No file loaded",
                                   fm2, self._browse_eng)
        self.eng_label = QLabel("—")
        fm2.addRow("Motor summary", self.eng_label)
        root.addWidget(_group("Propulsion", fm2))

        # ── Dry inertia tensor ───────────────────────────────────────────
        fi = QFormLayout(); fi.setSpacing(6)
        self.Ixx = _dspin(d.get('Ixx', 0.005), 0, 1e4, 0.0001, 6, "kg·m²")
        self.Iyy = _dspin(d.get('Iyy', 0.005), 0, 1e4, 0.0001, 6, "kg·m²")
        self.Izz = _dspin(d.get('Izz', 0.001), 0, 1e4, 0.0001, 6, "kg·m²")
        self.Ixy = _dspin(d.get('Ixy', 0.0), -1e4, 1e4, 0.0001, 6, "kg·m²")
        self.Ixz = _dspin(d.get('Ixz', 0.0), -1e4, 1e4, 0.0001, 6, "kg·m²")
        self.Iyz = _dspin(d.get('Iyz', 0.0), -1e4, 1e4, 0.0001, 6, "kg·m²")
        fi.addRow("Ixx (lateral X)", self.Ixx)
        fi.addRow("Iyy (lateral Y)", self.Iyy)
        fi.addRow("Izz (axial)",     self.Izz)
        fi.addRow("Ixy",             self.Ixy)
        fi.addRow("Ixz",             self.Ixz)
        fi.addRow("Iyz",             self.Iyz)
        root.addWidget(_group("Dry Inertia Tensor (body CG)", fi))
        root.addStretch()

    def _browse_eng(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select thrust curve", "", "ENG Files (*.eng);;All Files (*)"
        )
        if path:
            self._eng_path = path
            self._eng_le.setText(Path(path).name)
            self._update_eng_label()

    def _update_eng_label(self):
        try:
            from utils.eng_parser import load_eng
            eng = load_eng(self._eng_path)
            self.eng_label.setText(str(eng))
        except Exception as e:
            self.eng_label.setText(f"Parse error: {e}")

    def get_geometry(self) -> StageGeometry:
        length   = self.length.value()
        cg_dry   = self.cg_dry.value()
        cg_prop  = self.cg_prop.value()

        # ── Validate: CG must be positive ─────────────────────────────────
        errors = []
        if cg_dry < 0.0:
            errors.append(f"CG seco = {cg_dry:.4f} m es negativo. "
                          f"Mídelo desde la nariz de S2 (punta del cohete completo).")
        if cg_prop < 0.0:
            errors.append(f"CG prop = {cg_prop:.4f} m es negativo. "
                          f"Mídelo desde la nariz de S2 (punta del cohete completo).")
        if errors:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error — CG inválido", "\n\n".join(errors))
            raise ValueError("Invalid CG: " + "; ".join(errors))

        return StageGeometry(
            diameter_m           = self.diameter.value(),
            length_m             = length,
            cg_dry_from_nose_m   = cg_dry,
            cg_prop_from_nose_m  = cg_prop,
            grain_length_m       = self.grain_len.value(),
            grain_od_m           = self.grain_od.value(),
            grain_id_m           = self.grain_id.value(),
            inertia_dry          = self.get_inertia(),
        )

    def get_inertia(self) -> StageInertia:
        return StageInertia(
            Ixx=self.Ixx.value(), Iyy=self.Iyy.value(), Izz=self.Izz.value(),
            Ixy=self.Ixy.value(), Ixz=self.Ixz.value(), Iyz=self.Iyz.value(),
        )

    def get_dry_mass(self) -> float: return self.dry_mass.value()
    def get_eng_path(self) -> str:   return self._eng_path


# ─── Aerodynamics configuration widget ────────────────────────────────────────

class AeroWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Body aero CSV (SimSweep) ─────────────────────────────────────
        fm = QFormLayout(); fm.setSpacing(6)
        self._s1_csv_path = self._s2_csv_path = ""
        self._s1_csv_le = _file_row("S1 Aero CSV", "SimSweep CSV — fuerzas en Newtons (drag_N, lift_N)", fm,
                                     lambda: self._browse_csv('s1'))
        self._s2_csv_le = _file_row("S2 Aero CSV", "SimSweep CSV — fuerzas en Newtons (drag_N, lift_N)", fm,
                                     lambda: self._browse_csv('s2'))
        root.addWidget(_group("Body Aerodynamics — SimSweep CSV (drag_N / lift_N vs Mach, AoA)", fm))

        # ── Fin aero (aero_table.h) ───────────────────────────────────────
        fm2 = QFormLayout(); fm2.setSpacing(6)
        self._at_s1_path = self._at_s2_path = ""
        self._at_s1_le = _file_row("S1 aero_table.h", "S1 fin CL table", fm2,
                                    lambda: self._browse_at('s1'))
        self._at_s2_le = _file_row("S2 aero_table.h", "S2 fin CL table", fm2,
                                    lambda: self._browse_at('s2'))
        self.at_label  = QLabel("No file loaded")
        fm2.addRow("Fin table info", self.at_label)
        root.addWidget(_group("Fin Aerodynamics (aero_table.h)", fm2))

        # ── Aero geometry ─────────────────────────────────────────────────
        fm3 = QFormLayout(); fm3.setSpacing(6)
        self.cp_s1    = _dspin(1.20, 0, 20, 0.01, 3, "m (from nose)")
        self.cp_s2    = _dspin(1.10, 0, 20, 0.01, 3, "m (from nose)")
        self.fin_sref = _dspin(0.012, 0, 1, 0.0001, 4, "m²")
        self.fin_d    = _dspin(0.80, 0, 5, 0.01, 3, "m")
        self.fin_r    = _dspin(0.12, 0, 1, 0.01, 3, "m")
        self.damp_lat = _dspin(0.05, 0, 10, 0.001, 4)
        self.damp_rol = _dspin(0.01, 0, 10, 0.001, 4)
        fm3.addRow("CP position S1",    self.cp_s1)
        fm3.addRow("CP position S2",    self.cp_s2)
        fm3.addRow("Fin Sref (1 fin)",  self.fin_sref)
        fm3.addRow("Lever arm D",       self.fin_d)
        fm3.addRow("Fin radius r_fin",  self.fin_r)
        fm3.addRow("Damp lat (pitch/yaw)", self.damp_lat)
        fm3.addRow("Damp roll",         self.damp_rol)
        root.addWidget(_group("Geometry & Damping", fm3))
        root.addStretch()

    def _browse_csv(self, stage: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SimSweep CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path: return
        if stage == 's1':
            self._s1_csv_path = path; self._s1_csv_le.setText(Path(path).name)
        else:
            self._s2_csv_path = path; self._s2_csv_le.setText(Path(path).name)

    def _browse_at(self, stage: str):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select aero_table.h", "", "Header Files (*.h);;All Files (*)"
        )
        if not path: return
        if stage == 's1':
            self._at_s1_path = path; self._at_s1_le.setText(Path(path).name)
        else:
            self._at_s2_path = path; self._at_s2_le.setText(Path(path).name)
        self._update_at_label()

    def _update_at_label(self):
        p = self._at_s2_path or self._at_s1_path
        if not p: return
        try:
            from utils.aero_table_parser import parse_aero_table_h
            ft = parse_aero_table_h(p)
            self.at_label.setText(
                f"Sref={ft.sref_m2:.5f} m²  |  N={ft.n_points}  |  "
                f"CL [{ft.cl_min:.3f}…{ft.cl_max:.3f}]"
            )
        except Exception as e:
            self.at_label.setText(f"Error: {e}")

    def get_aero_config(self, stage: int,
                        diameter_m: float) -> AeroConfig:
        """Build AeroConfig for stage 1 or 2."""
        import numpy as np
        csv_path = self._s1_csv_path if stage == 1 else self._s2_csv_path
        at_path  = self._at_s1_path  if stage == 1 else self._at_s2_path
        cp       = self.cp_s1.value() if stage == 1 else self.cp_s2.value()

        aref = np.pi * (diameter_m / 2) ** 2
        body_aero = None
        if csv_path:
            try:
                from utils.aero_table_parser import parse_simsweep_csv
                body_aero = parse_simsweep_csv(csv_path, diameter_m, aref)
            except Exception as e:
                print(f"[AeroWidget] CSV parse error: {e}")

        fin_aero = None
        if at_path:
            try:
                from utils.aero_table_parser import parse_aero_table_h
                fin_aero = parse_aero_table_h(at_path)
            except Exception as e:
                print(f"[AeroWidget] aero_table.h parse error: {e}")

        return AeroConfig(
            body_diameter_m  = diameter_m,
            body_aref_m2     = aref,
            cp_from_nose_m   = cp,
            fin_sref_m2      = self.fin_sref.value(),
            fin_lever_arm_m  = self.fin_d.value(),
            fin_radius_m     = self.fin_r.value(),
            aero_damping_lat  = self.damp_lat.value(),
            aero_damping_roll = self.damp_rol.value(),
            body_aero        = body_aero,
            fin_aero         = fin_aero,
        )


# ─── Control configuration widget ─────────────────────────────────────────────

class ControlWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Control enable ────────────────────────────────────────────────
        self.ctrl_en = QCheckBox("Enable active fin control")
        self.ctrl_en.setChecked(True)
        root.addWidget(self.ctrl_en)

        # ── PID Gains ─────────────────────────────────────────────────────
        for axis, attr in [("Pitch", "pitch"), ("Yaw", "yaw"), ("Roll", "roll")]:
            fm = QFormLayout(); fm.setSpacing(4)
            kp = _dspin(4.0 if axis != "Roll" else 1.0, 0, 1000, 0.1, 3)
            ki = _dspin(0.05 if axis != "Roll" else 0.0, 0, 100, 0.01, 3)
            kd = _dspin(0.5  if axis != "Roll" else 0.1, 0, 100, 0.01, 3)
            fm.addRow("Kp", kp); fm.addRow("Ki", ki); fm.addRow("Kd", kd)
            setattr(self, f'pid_{attr}_kp', kp)
            setattr(self, f'pid_{attr}_ki', ki)
            setattr(self, f'pid_{attr}_kd', kd)
            root.addWidget(_group(f"PID — {axis}", fm))

        # ── Servo ─────────────────────────────────────────────────────────
        fm4 = QFormLayout(); fm4.setSpacing(4)
        self.d_max    = _dspin(15.0, 0, 45, 0.5, 1, "°")
        self.slew     = _dspin(300.0, 10, 2000, 10, 1, "°/s")
        self.latency  = _dspin(10.0,  0, 200,  1,  1, "ms")
        self.lever_d  = _dspin(0.80,  0, 5,    0.01, 3, "m")
        fm4.addRow("Max deflection", self.d_max)
        fm4.addRow("Slew rate",      self.slew)
        fm4.addRow("Latency",        self.latency)
        fm4.addRow("Lever arm D",    self.lever_d)
        root.addWidget(_group("Servo Actuator", fm4))

        # ── EKF noise tuning ──────────────────────────────────────────────
        fm5 = QFormLayout(); fm5.setSpacing(4)
        self.Q_alt   = _dspin(0.005, 0, 10, 0.001, 4)
        self.Q_vz    = _dspin(0.10,  0, 10, 0.01,  3)
        self.Q_att   = _dspin(1e-5,  0, 1,  1e-6,  7)
        self.R_baro  = _dspin(0.25,  0, 100, 0.01, 3)
        fm5.addRow("Q altitude",  self.Q_alt)
        fm5.addRow("Q vert speed",self.Q_vz)
        fm5.addRow("Q attitude",  self.Q_att)
        fm5.addRow("R barometer", self.R_baro)
        root.addWidget(_group("EKF Process / Measurement Noise", fm5))

        # ── Sensor noise ──────────────────────────────────────────────────
        fm6 = QFormLayout(); fm6.setSpacing(4)
        self.sn_en       = QCheckBox("Inject sensor noise"); self.sn_en.setChecked(True)
        self.accel_sigma = _dspin(0.05, 0, 10, 0.001, 4, "m/s²")
        self.gyro_sigma  = _dspin(0.003, 0, 1, 0.0001, 5, "rad/s")
        self.baro_sigma  = _dspin(0.50, 0, 50, 0.1, 2, "m")
        fm6.addRow(self.sn_en)
        fm6.addRow("Accel noise σ", self.accel_sigma)
        fm6.addRow("Gyro noise σ",  self.gyro_sigma)
        fm6.addRow("Baro noise σ",  self.baro_sigma)
        root.addWidget(_group("Sensor Noise", fm6))
        root.addStretch()

    def get_pid_gains(self, axis: str) -> PIDGains:
        return PIDGains(
            Kp=getattr(self, f'pid_{axis}_kp').value(),
            Ki=getattr(self, f'pid_{axis}_ki').value(),
            Kd=getattr(self, f'pid_{axis}_kd').value(),
        )

    def get_servo_config(self) -> ServoConfig:
        return ServoConfig(
            delta_max_deg   = self.d_max.value(),
            slew_rate_deg_s = self.slew.value(),
            latency_s       = self.latency.value() / 1000.0,
            lever_arm_d_m   = self.lever_d.value(),
        )

    def get_sensor_noise_config(self) -> SensorNoiseConfig:
        return SensorNoiseConfig(
            accel_noise_sigma = self.accel_sigma.value(),
            gyro_noise_sigma  = self.gyro_sigma.value(),
            baro_noise_sigma  = self.baro_sigma.value(),
            enabled           = self.sn_en.isChecked(),
        )


# ─── Environment configuration widget ────────────────────────────────────────

class EnvWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        fm = QFormLayout(); fm.setSpacing(6)
        self.launch_alt  = _dspin(1580.0, -500, 5000, 1, 1, "m MSL")
        self.tilt_pitch  = _dspin(0.0, -10, 10, 0.1, 2, "°")
        self.tilt_yaw    = _dspin(0.0, -10, 10, 0.1, 2, "°")
        fm.addRow("Launch altitude", self.launch_alt)
        fm.addRow("Initial pitch tilt", self.tilt_pitch)
        fm.addRow("Initial yaw tilt",   self.tilt_yaw)
        root.addWidget(_group("Launch Conditions", fm))

        fm2 = QFormLayout(); fm2.setSpacing(6)
        self.staging_mode = QComboBox()
        self.staging_mode.addItems(["Altitude (AGL)", "Time from launch"])
        self.staging_alt  = _dspin(15000, 0, 1e5, 100, 1, "m AGL")
        self.staging_time = _dspin(30.0, 0, 300, 1, 1, "s")
        self.s2_delay     = _dspin(0.1, 0, 10, 0.1, 2, "s")
        fm2.addRow("Staging trigger",   self.staging_mode)
        fm2.addRow("Staging altitude",  self.staging_alt)
        fm2.addRow("Staging time",      self.staging_time)
        fm2.addRow("S2 ignition delay", self.s2_delay)
        root.addWidget(_group("Staging Configuration", fm2))

        fm3 = QFormLayout(); fm3.setSpacing(6)
        self.max_time = _dspin(400.0, 10, 3600, 10, 0, "s")
        self.rtol     = _dspin(1e-7,  1e-12, 1e-2, 1e-8, 10)
        self.atol     = _dspin(1e-9,  1e-14, 1e-4, 1e-10, 12)
        fm3.addRow("Max sim time", self.max_time)
        fm3.addRow("RK45 rtol",   self.rtol)
        fm3.addRow("RK45 atol",   self.atol)
        root.addWidget(_group("Integrator", fm3))
        root.addStretch()

    def get_fsm_config(self) -> FSMConfig:
        mode = 'altitude' if self.staging_mode.currentIndex() == 0 else 'time'
        return FSMConfig(
            staging_mode        = mode,
            staging_altitude_m  = self.staging_alt.value(),
            staging_time_s      = self.staging_time.value(),
            s2_ignition_delay_s = self.s2_delay.value(),
        )


# ─── Master config panel ──────────────────────────────────────────────────────

class ConfigPanel(QWidget):
    """Left-side tabbed configuration panel."""

    run_requested    = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(360)
        self.setMaximumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Title ─────────────────────────────────────────────────────────
        title = QLabel("Trinity 6-DOF Simulator")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Orbital Dynamics — Physics Engine v1.0")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(title); root.addWidget(sub)

        # ── Tabs ──────────────────────────────────────────────────────────
        self._tabs = QTabWidget()

        # Stage sub-tabs inside a Stages tab
        stage_tabs = QTabWidget()
        self.s1_widget = StageWidget(self._s1_defaults())
        self.s2_widget = StageWidget(self._s2_defaults())
        sa1 = QScrollArea(); sa1.setWidget(self.s1_widget)
        sa1.setWidgetResizable(True); sa1.setFrameShape(QFrame.Shape.NoFrame)
        sa2 = QScrollArea(); sa2.setWidget(self.s2_widget)
        sa2.setWidgetResizable(True); sa2.setFrameShape(QFrame.Shape.NoFrame)
        stage_tabs.addTab(sa1, "Stage 1")
        stage_tabs.addTab(sa2, "Stage 2")

        self.aero_widget = AeroWidget()
        sa3 = QScrollArea(); sa3.setWidget(self.aero_widget)
        sa3.setWidgetResizable(True); sa3.setFrameShape(QFrame.Shape.NoFrame)

        self.ctrl_widget = ControlWidget()
        sa4 = QScrollArea(); sa4.setWidget(self.ctrl_widget)
        sa4.setWidgetResizable(True); sa4.setFrameShape(QFrame.Shape.NoFrame)

        self.env_widget = EnvWidget()
        sa5 = QScrollArea(); sa5.setWidget(self.env_widget)
        sa5.setWidgetResizable(True); sa5.setFrameShape(QFrame.Shape.NoFrame)

        self._tabs.addTab(stage_tabs, "Stages")
        self._tabs.addTab(sa3, "Aero")
        self._tabs.addTab(sa4, "Control")
        self._tabs.addTab(sa5, "Environment")
        root.addWidget(self._tabs, 1)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run    = QPushButton("▶  Run Simulation")
        self.btn_export = QPushButton("⬇  Export CSV")
        self.btn_run.setFixedHeight(36)
        self.btn_export.setFixedHeight(36)
        self.btn_run.setStyleSheet(
            "QPushButton{background:#2a7aff;color:white;border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#1a5ae0;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.btn_run.clicked.connect(self.run_requested)
        self.btn_export.clicked.connect(self.export_requested)
        btn_row.addWidget(self.btn_run, 2)
        btn_row.addWidget(self.btn_export, 1)
        root.addLayout(btn_row)

        # ── Save / Load config buttons ────────────────────────────────────
        io_row = QHBoxLayout()
        btn_save = QPushButton("💾  Guardar Setup")
        btn_load = QPushButton("📂  Cargar Setup")
        btn_save.setFixedHeight(30)
        btn_load.setFixedHeight(30)
        btn_save.setToolTip("Guarda todos los parámetros del cohete en un archivo .json")
        btn_load.setToolTip("Carga un setup guardado previamente (.json)")
        btn_save.clicked.connect(self._save_config)
        btn_load.clicked.connect(self._load_config)
        io_row.addWidget(btn_save)
        io_row.addWidget(btn_load)
        root.addLayout(io_row)

    # ── Config assembly ───────────────────────────────────────────────────────

    def build_sim_config(self):
        """
        Collect all widget values → SimConfig.
        Raises ValueError with a user-friendly message if required files are missing.
        """
        from simulation.runner import SimConfig, StageConfig
        from utils.eng_parser import load_eng

        errors = []
        # Validate .eng files
        for name, widget in [("Stage 1", self.s1_widget), ("Stage 2", self.s2_widget)]:
            if not widget.get_eng_path():
                errors.append(f"{name}: No .eng thrust curve loaded.")
        if errors:
            raise ValueError("\n".join(errors))

        # Load motors
        s1_motor = load_eng(self.s1_widget.get_eng_path())
        s2_motor = load_eng(self.s2_widget.get_eng_path())

        # Override propellant mass from GUI if eng file differs
        s1_motor.propellant_mass_kg = self.s1_widget.prop_mass.value()
        s2_motor.propellant_mass_kg = self.s2_widget.prop_mass.value()

        s1_geom = self.s1_widget.get_geometry()
        s2_geom = self.s2_widget.get_geometry()

        s1_aero = self.aero_widget.get_aero_config(1, s1_geom.diameter_m)
        s2_aero = self.aero_widget.get_aero_config(2, s2_geom.diameter_m)

        env  = self.env_widget
        ctrl = self.ctrl_widget

        from simulation.runner import StageConfig as SC
        stage1 = SC(
            name="Stage 1", geometry=s1_geom, motor=s1_motor,
            aero=s1_aero, dry_mass_kg=self.s1_widget.get_dry_mass()
        )
        stage2 = SC(
            name="Stage 2", geometry=s2_geom, motor=s2_motor,
            aero=s2_aero, dry_mass_kg=self.s2_widget.get_dry_mass()
        )

        return SimConfig(
            stage1            = stage1,
            stage2            = stage2,
            launch_alt_msl_m  = env.launch_alt.value(),
            launch_tilt_pitch = env.tilt_pitch.value(),
            launch_tilt_yaw   = env.tilt_yaw.value(),
            staging_cfg       = env.get_fsm_config(),
            control_enabled   = ctrl.ctrl_en.isChecked(),
            pid_pitch         = ctrl.get_pid_gains('pitch'),
            pid_yaw           = ctrl.get_pid_gains('yaw'),
            pid_roll          = ctrl.get_pid_gains('roll'),
            servo_cfg         = ctrl.get_servo_config(),
            noise_cfg         = ctrl.get_sensor_noise_config(),
            max_time_s        = env.max_time.value(),
            rtol              = env.rtol.value(),
            atol              = env.atol.value(),
        )

    # ── Save / Load config ────────────────────────────────────────────────────

    def _save_config(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar configuración", "trinity_setup.json",
            "Trinity Setup (*.json);;All Files (*)"
        )
        if path:
            try:
                from utils.config_io import save_config
                save_config(self, path)
                # Remember last saved path for auto-load
                import pathlib
                _last = pathlib.Path(__file__).parent.parent / ".last_config"
                _last.write_text(path)
                QMessageBox.information(self, "Guardado", f"Setup guardado en:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error al guardar", str(e))

    def _load_config(self, path: str = ""):
        from PyQt6.QtWidgets import QFileDialog
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Cargar configuración", "",
                "Trinity Setup (*.json);;All Files (*)"
            )
        if not path:
            return
        try:
            from utils.config_io import load_config
            load_config(self, path)
            # Remember for auto-load next session
            import pathlib
            _last = pathlib.Path(__file__).parent.parent / ".last_config"
            _last.write_text(str(path))
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar", str(e))

    def auto_load_last(self):
        """Called at startup — restores the last saved session if it exists."""
        import pathlib
        _last = pathlib.Path(__file__).parent.parent / ".last_config"
        if _last.exists():
            p = _last.read_text().strip()
            if p and pathlib.Path(p).exists():
                try:
                    from utils.config_io import load_config
                    load_config(self, p)
                except Exception:
                    pass  # silently skip if restore fails

    @staticmethod
    def _s1_defaults():
        return dict(dry_mass=10.0, prop_mass=6.0, diameter=0.16, length=1.0,
                    cg_dry=2.41, cg_prop=2.43, grain_len=0.65, grain_od=0.075)

    @staticmethod
    def _s2_defaults():
        return dict(dry_mass=5.0, prop_mass=0.81, diameter=0.115, length=1.8,
                    cg_dry=1.13, cg_prop=1.21, grain_len=0.16, grain_od=0.062)
