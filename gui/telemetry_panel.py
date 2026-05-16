"""
gui/telemetry_panel.py — Telemetry Dashboard
Trinity 6-DOF Simulator | Orbital Dynamics

Six synchronized PyQtGraph plot panels:
  1. Altitude AGL  +  Mach number
  2. Velocity ENU  +  Dynamic pressure
  3. Attitude (roll / pitch / yaw)
  4. Angular rates  (ωx / ωy / ωz)
  5. Fin deflections  (δ1 / δ2 / δ3 / δ4)
  6. Control torques  +  Mass

All X axes are linked to the same time cursor.
Staging and apogee events are annotated with vertical dashed lines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QSizePolicy, QPushButton
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg


# ─── Colour palette ────────────────────────────────────────────────────────────
_C = {
    'alt':     '#00d2ff',
    'mach':    '#ff9900',
    'vz':      '#44ff88',
    'vxy':     '#bb88ff',
    'q_dyn':   '#ff4488',
    'roll':    '#ff5050',
    'pitch':   '#50ff50',
    'yaw':     '#5088ff',
    'wr':      '#ff5050',
    'wp':      '#50ff50',
    'wy':      '#5088ff',
    's1':      '#ffdd00',
    's2':      '#ff8800',
    's3':      '#00cc44',
    's4':      '#44aaff',
    'tau_p':   '#ff7700',
    'tau_y':   '#00ccff',
    'tau_r':   '#ff44aa',
    'mass':    '#aaaaff',
    'staging': '#ff6600',
    'apogee':  '#00ff99',
}

pg.setConfigOption('background', '#1a1a2e')
pg.setConfigOption('foreground',  '#d0d0d0')


def _pen(color: str, width: float = 1.5) -> pg.mkPen:
    return pg.mkPen(color=color, width=width)


def _vline(plot: pg.PlotWidget, x: float, color: str, label: str):
    """Add a vertical dashed annotation line."""
    line = pg.InfiniteLine(pos=x, angle=90,
                            pen=pg.mkPen(color=color, style=Qt.PenStyle.DashLine, width=1.5))
    plot.addItem(line)
    txt = pg.TextItem(label, color=color, anchor=(0, 1))
    txt.setPos(x, 0)
    plot.addItem(txt)


def _make_plot(title: str, xlabel: str, ylabels: list[str],
               twin_axis: bool = False) -> pg.PlotWidget:
    pw = pg.PlotWidget(title=title)
    pw.setLabel('bottom', xlabel)
    pw.setLabel('left', ylabels[0])
    pw.showGrid(x=True, y=True, alpha=0.3)
    pw.getAxis('bottom').setTextPen('#888')
    pw.addLegend(offset=(5, 5))
    pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return pw


class TelemetryPanel(QWidget):
    """Six-panel telemetry dashboard, populated after a simulation run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ───────────────────────────────────────────────────────
        bar = QHBoxLayout()
        lbl = QLabel("Telemetry")
        lbl.setStyleSheet("font-weight:bold; font-size:13px; color:#ccc;")
        self._status = QLabel("No data — run a simulation first.")
        self._status.setStyleSheet("color:#888; font-size:11px;")
        btn_reset = QPushButton("Reset zoom")
        btn_reset.setFixedWidth(90)
        btn_reset.clicked.connect(self._reset_zoom)
        bar.addWidget(lbl)
        bar.addWidget(self._status, 1)
        bar.addWidget(btn_reset)
        root.addLayout(bar)

        # ── Plots in a scrollable area ─────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        grid  = QVBoxLayout(inner)
        grid.setSpacing(6)

        # ── Row 1: Altitude + Mach ────────────────────────────────────────
        self.pw_alt = _make_plot("Altitude & Mach", "Time [s]", ["Altitude AGL [m]"])
        self.pw_alt.setMinimumHeight(200)
        self.ci_alt  = self.pw_alt.plot([], [], pen=_pen(_C['alt']),   name="Alt AGL [m]")
        self._alt_ax2 = pg.ViewBox()
        self.pw_alt.scene().addItem(self._alt_ax2)
        self.pw_alt.getAxis('right').linkToView(self._alt_ax2)
        self.pw_alt.getAxis('right').setLabel("Mach")
        self.pw_alt.getViewBox().sigResized.connect(self._sync_alt_ax2)
        self._alt_ax2.setXLink(self.pw_alt)
        self.ci_mach = pg.PlotCurveItem(pen=_pen(_C['mach']), name="Mach")
        self._alt_ax2.addItem(self.ci_mach)
        grid.addWidget(self.pw_alt)

        # ── Row 2: Velocity + Dynamic pressure ────────────────────────────
        self.pw_vel = _make_plot("Velocity & Dynamic Pressure", "Time [s]",
                                 ["Speed [m/s]"])
        self.pw_vel.setMinimumHeight(180)
        self.ci_vz   = self.pw_vel.plot([], [], pen=_pen(_C['vz']),   name="Vz [m/s]")
        self.ci_spd  = self.pw_vel.plot([], [], pen=_pen(_C['vxy']),  name="|V| [m/s]")
        self.ci_qdyn = self.pw_vel.plot([], [], pen=_pen(_C['q_dyn'], 1.0), name="q_dyn/1000 [kPa]")
        grid.addWidget(self.pw_vel)

        # ── Row 3: Attitude angles ────────────────────────────────────────
        self.pw_att = _make_plot("Attitude", "Time [s]", ["Angle [°]"])
        self.pw_att.setMinimumHeight(180)
        self.ci_roll  = self.pw_att.plot([], [], pen=_pen(_C['roll']),  name="Roll [°]")
        self.ci_pitch = self.pw_att.plot([], [], pen=_pen(_C['pitch']), name="Pitch [°]")
        self.ci_yaw   = self.pw_att.plot([], [], pen=_pen(_C['yaw']),   name="Yaw [°]")
        grid.addWidget(self.pw_att)

        # ── Row 4: Angular rates ──────────────────────────────────────────
        self.pw_rate = _make_plot("Angular Rates", "Time [s]", ["Rate [rad/s]"])
        self.pw_rate.setMinimumHeight(160)
        self.ci_wr = self.pw_rate.plot([], [], pen=_pen(_C['wr']),  name="ωx roll [rad/s]")
        self.ci_wp = self.pw_rate.plot([], [], pen=_pen(_C['wp']),  name="ωy pitch [rad/s]")
        self.ci_wy = self.pw_rate.plot([], [], pen=_pen(_C['wy']),  name="ωz yaw [rad/s]")
        grid.addWidget(self.pw_rate)

        # ── Row 5: Servo deflections ──────────────────────────────────────
        self.pw_srv = _make_plot("Servo Deflections", "Time [s]", ["Deflection [°]"])
        self.pw_srv.setMinimumHeight(160)
        self.ci_s1 = self.pw_srv.plot([], [], pen=_pen(_C['s1']), name="δ1 [°]")
        self.ci_s2 = self.pw_srv.plot([], [], pen=_pen(_C['s2']), name="δ2 [°]")
        self.ci_s3 = self.pw_srv.plot([], [], pen=_pen(_C['s3']), name="δ3 [°]")
        self.ci_s4 = self.pw_srv.plot([], [], pen=_pen(_C['s4']), name="δ4 [°]")
        grid.addWidget(self.pw_srv)

        # ── Row 6: Mass + Thrust ──────────────────────────────────────────
        self.pw_mass = _make_plot("Mass & Thrust", "Time [s]", ["Mass [kg]"])
        self.pw_mass.setMinimumHeight(160)
        self.ci_mass   = self.pw_mass.plot([], [], pen=_pen(_C['mass']), name="Mass [kg]")
        self.ci_thrust = self.pw_mass.plot([], [], pen=_pen(_C['s1']),   name="Thrust [N]")
        grid.addWidget(self.pw_mass)

        # ── Link X axes ────────────────────────────────────────────────────
        for pw in [self.pw_vel, self.pw_att, self.pw_rate, self.pw_srv, self.pw_mass]:
            pw.setXLink(self.pw_alt)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Store annotation lines so we can remove them on update
        self._event_lines: list = []

    # ── Public API ────────────────────────────────────────────────────────────

    def update_data(self, df: pd.DataFrame,
                    staging_t: float | None = None,
                    apogee_t:  float | None = None):
        """Populate all curves from the telemetry DataFrame."""
        if df is None or df.empty:
            self._status.setText("No data — run a simulation first.")
            return

        t = df['t'].values

        # Clear old event lines
        for item in self._event_lines:
            for pw in [self.pw_alt, self.pw_vel, self.pw_att,
                       self.pw_rate, self.pw_srv, self.pw_mass]:
                try: pw.removeItem(item)
                except Exception: pass
        self._event_lines.clear()

        # ── Row 1 ──────────────────────────────────────────────────────────
        self.ci_alt.setData(t, df['alt_agl'].values)
        self.ci_mach.setData(t, df['mach'].values)
        self._sync_alt_ax2()

        # ── Row 2 ──────────────────────────────────────────────────────────
        self.ci_vz.setData(t,  df['vz'].values)
        self.ci_spd.setData(t, df['speed'].values)
        self.ci_qdyn.setData(t, df['q_dyn'].values / 1000.0)

        # ── Row 3 ──────────────────────────────────────────────────────────
        self.ci_roll.setData(t,  df['roll_deg'].values)
        self.ci_pitch.setData(t, df['pitch_deg'].values)
        self.ci_yaw.setData(t,   df['yaw_deg'].values)

        # ── Row 4 ──────────────────────────────────────────────────────────
        self.ci_wr.setData(t, df['roll_rate'].values)
        self.ci_wp.setData(t, df['pitch_rate'].values)
        self.ci_wy.setData(t, df['yaw_rate'].values)

        # ── Row 5 ──────────────────────────────────────────────────────────
        for ci, col in [(self.ci_s1, 'servo_1_deg'), (self.ci_s2, 'servo_2_deg'),
                        (self.ci_s3, 'servo_3_deg'), (self.ci_s4, 'servo_4_deg')]:
            if col in df.columns:
                ci.setData(t, df[col].values)

        # ── Row 6 ──────────────────────────────────────────────────────────
        self.ci_mass.setData(t,   df['mass'].values)
        self.ci_thrust.setData(t, df['thrust'].values)

        # ── Event lines ────────────────────────────────────────────────────
        if staging_t is not None:
            for pw in [self.pw_alt, self.pw_vel, self.pw_att]:
                line = pg.InfiniteLine(
                    pos=staging_t, angle=90,
                    pen=pg.mkPen(_C['staging'], style=Qt.PenStyle.DashLine, width=2)
                )
                pw.addItem(line)
                self._event_lines.append(line)

        if apogee_t is not None:
            for pw in [self.pw_alt, self.pw_vel]:
                line = pg.InfiniteLine(
                    pos=apogee_t, angle=90,
                    pen=pg.mkPen(_C['apogee'], style=Qt.PenStyle.DashLine, width=2)
                )
                pw.addItem(line)
                self._event_lines.append(line)

        # ── Status ─────────────────────────────────────────────────────────
        apogee_agl = float(df['alt_agl'].max())
        max_mach   = float(df['mach'].max())
        n_pts      = len(t)
        self._status.setText(
            f"Apogee AGL: {apogee_agl/1000:.2f} km  |  "
            f"Max Mach: {max_mach:.3f}  |  "
            f"{n_pts} data points  |  "
            f"t = {t[0]:.1f} … {t[-1]:.1f} s"
        )
        self._reset_zoom()

    def _reset_zoom(self):
        for pw in [self.pw_alt, self.pw_vel, self.pw_att,
                   self.pw_rate, self.pw_srv, self.pw_mass]:
            pw.enableAutoRange()

    def _sync_alt_ax2(self):
        self._alt_ax2.setGeometry(self.pw_alt.getViewBox().sceneBoundingRect())
        self._alt_ax2.linkedViewChanged(self.pw_alt.getViewBox(),
                                         self._alt_ax2.XAxis)
