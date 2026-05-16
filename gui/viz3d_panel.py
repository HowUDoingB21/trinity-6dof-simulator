"""
gui/viz3d_panel.py — 3D Post-Flight Visualisation with Animation
Trinity 6-DOF Simulator | Orbital Dynamics

Features:
  • Animated playback: cohete se mueve a lo largo de la trayectoria con
    orientación correcta en cada instante (cuaternión)
  • Estela parcial que crece conforme avanza la animación
  • Ejes de cuerpo del cohete (X rojo, Y verde, Z azul=nariz) visibles
  • Controles: Play/Pause, Stop, velocidad de reproducción (0.1×–20×)
  • Slider de tiempo sincronizado con la animación
  • HUD: tiempo, altitud, velocidad, fase
  • Camera-follow toggle: cámara sigue al cohete o libre
  • Marcadores estáticos: staging, apogeo
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider, QFrame, QSizePolicy, QCheckBox, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False


# ─── Rocket geometry ──────────────────────────────────────────────────────────
# Scale: real dimensions in metres, oriented with nose toward +Z (body frame)

def _build_rocket_base(diameter: float = 0.115, length: float = 2.8,
                        nose_length: float = 0.4,
                        fin_span: float = 0.14,
                        fin_chord: float = 0.20) -> "pv.PolyData | None":
    """Build procedural rocket mesh centred at CG (approx mid-body)."""
    if not _PYVISTA_OK:
        return None
    r = diameter / 2.0
    cg_offset = length / 2.0   # rough CG at mid-body for visual centering

    # Body cylinder (base at z=0 toward +z)
    body = pv.Cylinder(
        center=(0, 0, cg_offset - nose_length / 2),
        direction=(0, 0, 1),
        radius=r, height=length - nose_length, resolution=20, capping=True
    )

    # Nose cone
    cone = pv.Cone(
        center=(0, 0, length - nose_length / 2 - cg_offset + cg_offset),
        direction=(0, 0, 1),
        height=nose_length, radius=r, resolution=20, capping=True
    )
    cone.translate((0, 0, length - nose_length - cg_offset), inplace=True)

    # 4 fins at the base, in +X, -X, +Y, -Y
    fins = []
    for angle_deg in [0, 90, 180, 270]:
        a = np.deg2rad(angle_deg)
        cx = (r + fin_span / 2) * np.cos(a)
        cy = (r + fin_span / 2) * np.sin(a)
        fin = pv.Box(bounds=(
            cx - fin_span / 2, cx + fin_span / 2,
            cy - 0.003,        cy + 0.003,
            -cg_offset + 0.02, -cg_offset + fin_chord + 0.02,
        ))
        fins.append(fin)

    mesh = body + cone
    for f in fins:
        mesh = mesh + f

    return mesh


def _transform_to_enu(mesh: "pv.PolyData", pos_enu: np.ndarray,
                       quat: np.ndarray) -> "pv.PolyData":
    """Apply position + quaternion (body→ENU) to a rocket mesh."""
    from core.state import quat_to_rotmat
    R = quat_to_rotmat(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = pos_enu
    return mesh.transform(T, inplace=False)


def _body_axes(pos: np.ndarray, quat: np.ndarray,
               scale: float = 60.0) -> tuple:
    """Return (origin, x_tip, y_tip, z_tip) in ENU for body frame axes."""
    from core.state import quat_to_rotmat
    R = quat_to_rotmat(quat)
    return (pos,
            pos + R[:, 0] * scale,   # body X
            pos + R[:, 1] * scale,   # body Y
            pos + R[:, 2] * scale)   # body Z = nose


# ─── Viz3DPanel ───────────────────────────────────────────────────────────────

class Viz3DPanel(QWidget):
    """
    3D post-flight visualiser with full animation playback.
    Embedded inside the main PyQt6 window as a tab widget.
    """

    # Playback speeds (multiplier vs real time)
    _SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._n: int = 0
        self._frame: int = 0
        self._playing: bool = False
        self._speed: float = 1.0
        self._staging_t: float | None = None
        self._apogee_t:  float | None = None
        self._diam: float = 0.115
        self._len:  float = 2.8

        # STL / base mesh state
        self._stl_path: str = ""
        self._stl_mesh: "pv.PolyData | None" = None   # raw STL (user's frame)
        self._base_rot: np.ndarray = np.eye(3)         # offset rotation matrix

        # Animation timer  (fires every ~33 ms → ~30 fps)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_timer)

        # PyVista persistent actors
        self._rocket_actor  = None
        self._trail_actor   = None
        self._axes_actors: list = []
        self._base_mesh: "pv.PolyData | None" = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Top toolbar ───────────────────────────────────────────────────────
        top = QHBoxLayout()

        title = QLabel("3D Trajectory")
        title.setStyleSheet("font-weight:bold; font-size:13px; color:#ccc;")
        top.addWidget(title)
        top.addStretch(1)

        # Camera presets
        for label, fn in [("Side", self._cam_side),
                           ("Top",  self._cam_top),
                           ("Iso",  self._cam_iso)]:
            b = QPushButton(label); b.setFixedWidth(46)
            b.clicked.connect(fn); top.addWidget(b)

        # Camera follow checkbox
        self._follow_chk = QCheckBox("Follow")
        self._follow_chk.setChecked(True)
        self._follow_chk.setToolTip("Camera follows the rocket during playback")
        top.addWidget(self._follow_chk)

        # Body axes checkbox — connected to immediate refresh
        self._axes_chk = QCheckBox("Body axes")
        self._axes_chk.setChecked(True)
        self._axes_chk.setToolTip("Mostrar / ocultar flechas de ejes del cuerpo")
        self._axes_chk.stateChanged.connect(self._toggle_axes)
        top.addWidget(self._axes_chk)

        root.addLayout(top)

        # ── PyVista viewport ─────────────────────────────────────────────────
        if _PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background('#0d0d1a')
            root.addWidget(self._plotter.interactor, 1)
            self._ready = True
        else:
            lbl = QLabel("pyvista / pyvistaqt not installed.\n"
                         "pip install pyvista pyvistaqt")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#888;")
            root.addWidget(lbl, 1)
            self._ready = False

        # ── STL + orientation panel ───────────────────────────────────────────
        from PyQt6.QtWidgets import (QDoubleSpinBox, QFileDialog,
                                      QLineEdit, QGridLayout)

        stl_row = QHBoxLayout()
        stl_lbl = QLabel("Modelo STL:")
        stl_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        self._stl_le = QLineEdit()
        self._stl_le.setReadOnly(True)
        self._stl_le.setPlaceholderText(
            "Sin STL — usando cohete procedural")
        self._stl_le.setStyleSheet(
            "background:#1a1a2e; border:1px solid #333; color:#aaa;"
            "font-size:10px; padding:2px;")
        btn_stl = QPushButton("Cargar STL…")
        btn_stl.setFixedWidth(100)
        btn_stl.clicked.connect(self._load_stl)
        btn_proc = QPushButton("Procedural")
        btn_proc.setFixedWidth(80)
        btn_proc.setToolTip("Volver al cohete procedural")
        btn_proc.clicked.connect(self._use_procedural)

        # Scale factor
        scale_lbl = QLabel("Escala:")
        scale_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0001, 10000.0)
        self._scale_spin.setDecimals(5)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setSingleStep(0.1)
        self._scale_spin.setFixedWidth(90)
        self._scale_spin.setToolTip("Factor de escala aplicado al STL")

        btn_autoscale = QPushButton("Auto-escala")
        btn_autoscale.setFixedWidth(85)
        btn_autoscale.setToolTip(
            "Escala el STL automáticamente para que su longitud total\n"
            "coincida con la longitud del cohete configurada en Stages.\n"
            "La longitud se mide a lo largo del eje de la nariz (body Z)\n"
            "después de aplicar la rotación base.")
        btn_autoscale.clicked.connect(self._auto_scale)
        self._scale_spin.editingFinished.connect(self._apply_scale)

        stl_row.addWidget(stl_lbl)
        stl_row.addWidget(self._stl_le, 1)
        stl_row.addWidget(btn_stl)
        stl_row.addWidget(btn_proc)
        stl_row.addWidget(scale_lbl)
        stl_row.addWidget(self._scale_spin)
        stl_row.addWidget(btn_autoscale)
        root.addLayout(stl_row)

        # ── Base orientation offset ───────────────────────────────────────────
        # Allows correcting axis mismatch between CAD frame and simulator frame.
        # Applied BEFORE the simulation quaternion: R_final = R_sim @ R_offset
        ori_row = QHBoxLayout()
        ori_lbl = QLabel("Orientación base del modelo — Rx:")
        ori_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        ori_row.addWidget(ori_lbl)

        self._rx_spin = self._make_angle_spin()  # rotation about model X
        self._ry_spin = self._make_angle_spin()  # rotation about model Y
        self._rz_spin = self._make_angle_spin()  # rotation about model Z

        for axis, spin in [("Rx", self._rx_spin),
                            ("Ry", self._ry_spin),
                            ("Rz", self._rz_spin)]:
            ori_row.addWidget(QLabel(f"{axis}:"))
            ori_row.addWidget(spin)

        btn_apply = QPushButton("Aplicar")
        btn_apply.setFixedWidth(65)
        btn_apply.setToolTip(
            "Aplica la rotación base y actualiza el frame actual.\n"
            "Rx/Ry/Rz son rotaciones intrínsecas en el frame del modelo."
        )
        btn_apply.clicked.connect(self._apply_base_rotation)

        btn_reset = QPushButton("Reset")
        btn_reset.setFixedWidth(50)
        btn_reset.clicked.connect(self._reset_base_rotation)

        ori_row.addWidget(btn_apply)
        ori_row.addWidget(btn_reset)
        ori_row.addStretch(1)

        # Quick-preset buttons for common CAD→sim axis mismatches
        preset_lbl = QLabel("Preset:")
        preset_lbl.setStyleSheet("color:#888; font-size:10px;")
        ori_row.addWidget(preset_lbl)

        presets = [
            ("+Z→nose",  (0,   0,   0)),
            ("+X→nose",  (0,  90,   0)),
            ("+Y→nose",  (-90, 0,   0)),
            ("-Z→nose",  (180, 0,   0)),
        ]
        for label, angles in presets:
            b = QPushButton(label)
            b.setFixedWidth(65)
            b.setToolTip(
                f"Rota el modelo para que el eje {label.split('→')[0]} "
                f"de tu CAD apunte hacia la nariz del simulador (+body Z)."
            )
            b.clicked.connect(
                lambda _=False, a=angles: self._apply_preset(a))
            ori_row.addWidget(b)

        root.addLayout(ori_row)

        # ── Playback bar ──────────────────────────────────────────────────────
        pb = QHBoxLayout()

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.setToolTip("Play / Pause")
        self._btn_play.clicked.connect(self._toggle_play)

        self._btn_stop = QPushButton("⏹")
        self._btn_stop.setFixedWidth(36)
        self._btn_stop.setToolTip("Stop & reset to t=0")
        self._btn_stop.clicked.connect(self._stop)

        speed_lbl = QLabel("Speed:")
        self._speed_cb = QComboBox()
        for s in self._SPEEDS:
            self._speed_cb.addItem(f"{s}×")
        self._speed_cb.setCurrentIndex(3)   # 1×
        self._speed_cb.currentIndexChanged.connect(self._on_speed_change)
        self._speed_cb.setFixedWidth(70)

        pb.addWidget(self._btn_play)
        pb.addWidget(self._btn_stop)
        pb.addWidget(speed_lbl)
        pb.addWidget(self._speed_cb)

        # Prominent axes toggle button
        self._btn_axes = QPushButton("Ejes ✓")
        self._btn_axes.setFixedWidth(65)
        self._btn_axes.setCheckable(True)
        self._btn_axes.setChecked(True)
        self._btn_axes.setToolTip("Mostrar / ocultar flechas de ejes del cuerpo (X/Y/Z)")
        self._btn_axes.setStyleSheet(
            "QPushButton{background:#1e3a1e;color:#44ff44;border:1px solid #44ff44;"
            "border-radius:3px;font-size:11px;}"
            "QPushButton:checked{background:#1e3a1e;color:#44ff44;}"
            "QPushButton:!checked{background:#2a1a1a;color:#886666;"
            "border:1px solid #664444;}"
            "QPushButton:hover{border-color:#88ff88;}"
        )
        self._btn_axes.toggled.connect(self._on_axes_btn_toggled)
        pb.addWidget(self._btn_axes)

        pb.addStretch(1)

        # HUD info
        self._hud = QLabel("No data")
        self._hud.setStyleSheet(
            "color:#00d2ff; font-family:Consolas,monospace; font-size:11px;")
        pb.addWidget(self._hud)

        root.addLayout(pb)

        # ── Timeline slider ───────────────────────────────────────────────────
        sl_row = QHBoxLayout()
        sl_row.addWidget(QLabel("t ="))
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider)
        sl_row.addWidget(self._slider, 1)
        self._time_lbl = QLabel("0.0 s")
        self._time_lbl.setFixedWidth(55)
        self._time_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        sl_row.addWidget(self._time_lbl)
        root.addLayout(sl_row)

    @staticmethod
    def _make_angle_spin():
        from PyQt6.QtWidgets import QDoubleSpinBox
        s = QDoubleSpinBox()
        s.setRange(-360, 360)
        s.setSingleStep(90)
        s.setDecimals(1)
        s.setSuffix(" °")
        s.setValue(0.0)
        s.setFixedWidth(80)
        return s

    # ── STL loading ───────────────────────────────────────────────────────────

    def _load_stl(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar modelo STL", "",
            "STL Files (*.stl *.STL);;All Files (*)"
        )
        if not path:
            return
        if not _PYVISTA_OK:
            return
        try:
            mesh = pv.read(path)
            # Centre at own centroid
            mesh = mesh.translate(-np.array(mesh.center), inplace=False)
            self._stl_raw = mesh          # raw, unscaled, centred
            self._stl_mesh = mesh.copy()  # will be scaled
            self._stl_path = path
            from pathlib import Path
            self._stl_le.setText(Path(path).name)
            # Reset scale to 1 so user sees the true dimensions first
            self._scale_spin.setValue(1.0)
            self._update_base_mesh()
            if self._df is not None:
                self._render_frame(self._frame)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar STL", str(e))

    def _auto_scale(self):
        """
        Compute and apply a uniform scale factor so the STL's extent along
        body Z (nose direction, after base rotation) matches self._len.
        """
        if not hasattr(self, '_stl_raw') or self._stl_raw is None:
            return

        # Apply current base_rot to the raw mesh and measure Z extent
        T = np.eye(4)
        T[:3, :3] = self._base_rot
        oriented = self._stl_raw.transform(T, inplace=False)

        # Bounds: (xmin,xmax, ymin,ymax, zmin,zmax)
        b = oriented.bounds
        z_extent = abs(b[5] - b[4])   # extent along body Z (nose direction)

        if z_extent < 1e-9:
            return

        scale = self._len / z_extent
        self._scale_spin.setValue(round(scale, 5))
        self._apply_scale()

    def _apply_scale(self):
        """Apply the current scale_spin value to _stl_raw → _stl_mesh."""
        if not hasattr(self, '_stl_raw') or self._stl_raw is None:
            return
        s = self._scale_spin.value()
        scaled = self._stl_raw.scale([s, s, s], inplace=False)
        # Re-centre after scaling (centroid should still be at origin)
        scaled = scaled.translate(-np.array(scaled.center), inplace=False)
        self._stl_mesh = scaled
        self._update_base_mesh()
        if self._df is not None:
            self._render_frame(self._frame)

    def _use_procedural(self):
        """Discard STL and switch back to the procedural rocket."""
        self._stl_mesh = None
        self._stl_raw  = None
        self._stl_path = ""
        self._stl_le.setText("")
        self._scale_spin.setValue(1.0)
        self._update_base_mesh()
        if self._df is not None:
            self._render_frame(self._frame)

    def _update_base_mesh(self):
        """Rebuild _base_mesh by applying _base_rot to the raw STL or procedural."""
        if self._stl_mesh is not None:
            # Apply base rotation to the STL (in the model's own frame)
            T = np.eye(4)
            T[:3, :3] = self._base_rot
            self._base_mesh = self._stl_mesh.transform(T, inplace=False)
        else:
            self._base_mesh = _build_rocket_base(self._diam, self._len)

    # ── Base orientation controls ─────────────────────────────────────────────

    def _euler_to_rotmat(self, rx_deg: float, ry_deg: float,
                          rz_deg: float) -> np.ndarray:
        """
        Intrinsic Rx → Ry → Rz rotation matrix.
        Rx: rotation about model X axis (pitch in CAD)
        Ry: rotation about model Y axis
        Rz: rotation about model Z axis (yaw in CAD)
        """
        rx = np.deg2rad(rx_deg)
        ry = np.deg2rad(ry_deg)
        rz = np.deg2rad(rz_deg)

        Rx = np.array([[1,           0,            0],
                        [0,  np.cos(rx), -np.sin(rx)],
                        [0,  np.sin(rx),  np.cos(rx)]])

        Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                        [          0, 1,          0],
                        [-np.sin(ry), 0, np.cos(ry)]])

        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                        [np.sin(rz),  np.cos(rz), 0],
                        [         0,           0, 1]])

        return Rz @ Ry @ Rx

    def _apply_base_rotation(self):
        self._base_rot = self._euler_to_rotmat(
            self._rx_spin.value(),
            self._ry_spin.value(),
            self._rz_spin.value(),
        )
        self._update_base_mesh()
        if self._df is not None:
            self._render_frame(self._frame)

    def _reset_base_rotation(self):
        self._rx_spin.setValue(0.0)
        self._ry_spin.setValue(0.0)
        self._rz_spin.setValue(0.0)
        self._base_rot = np.eye(3)
        self._update_base_mesh()
        if self._df is not None:
            self._render_frame(self._frame)

    def _apply_preset(self, angles: tuple):
        """Apply a preset orientation (rx, ry, rz) in degrees."""
        rx, ry, rz = angles
        self._rx_spin.setValue(rx)
        self._ry_spin.setValue(ry)
        self._rz_spin.setValue(rz)
        self._apply_base_rotation()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_data(self, df: pd.DataFrame,
                    diameter_m: float = 0.115,
                    length_m:   float = 2.8,
                    staging_t:  float | None = None,
                    apogee_t:   float | None = None):
        """Load new flight data and rebuild the static scene."""
        if not self._ready or df is None or df.empty:
            return

        self._stop()
        self._df = df
        self._n  = len(df)
        self._diam = diameter_m
        self._len  = length_m
        self._staging_t = staging_t
        self._apogee_t  = apogee_t

        # Rebuild base mesh (STL or procedural) with current base_rot
        self._update_base_mesh()

        self._slider.setRange(0, self._n - 1)
        self._slider.setValue(0)

        self._rebuild_scene()
        self._render_frame(0)

    # ── Scene construction ────────────────────────────────────────────────────

    def _rebuild_scene(self):
        """Clear plotter and draw static elements (ground, full trajectory)."""
        if not self._ready:
            return
        self._plotter.clear()
        self._rocket_actor = None
        self._trail_actor  = None
        self._axes_actors  = []

        df = self._df
        pos = np.column_stack([df['pos_x'].values,
                                df['pos_y'].values,
                                df['alt_msl'].values])

        # ── Ground plane ──────────────────────────────────────────────────────
        z0 = float(df['alt_msl'].iloc[0])
        size = max(abs(pos[:, :2]).max() * 4, 500.0)
        plane = pv.Plane(center=(0, 0, z0), direction=(0, 0, 1),
                          i_size=size, j_size=size,
                          i_resolution=12, j_resolution=12)
        self._plotter.add_mesh(plane, color='#141f14', opacity=0.7,
                                show_edges=False)

        # ── Full trajectory (faint) ───────────────────────────────────────────
        step = max(1, self._n // 2000)
        pts  = pos[::step]
        if len(pts) > 2:
            try:
                spline = pv.Spline(pts, n_points=min(len(pts), 3000))
                tube   = spline.tube(radius=max(size * 0.002, 4.0))
                # Altitude-coloured
                alts = df['alt_agl'].values[::step]
                tube['alt'] = np.interp(
                    np.linspace(0, 1, tube.n_points),
                    np.linspace(0, 1, len(alts)), alts)
                self._plotter.add_mesh(
                    tube, scalars='alt', cmap='Blues',
                    opacity=0.25, show_scalar_bar=False)
            except Exception:
                pass

        # ── Staging marker ────────────────────────────────────────────────────
        if self._staging_t is not None:
            idx = int(np.argmin(np.abs(df['t'].values - self._staging_t)))
            sph = pv.Sphere(radius=max(size * 0.008, 20), center=pos[idx])
            self._plotter.add_mesh(sph, color='#ff6600', opacity=0.9)
            self._plotter.add_point_labels(
                [pos[idx]], ['STAGING'], font_size=10,
                text_color='#ff6600', point_size=0, shape=None)

        # ── Apogee marker ─────────────────────────────────────────────────────
        if self._apogee_t is not None:
            idx = int(np.argmin(np.abs(df['t'].values - self._apogee_t)))
            sph = pv.Sphere(radius=max(size * 0.010, 25), center=pos[idx])
            self._plotter.add_mesh(sph, color='#00ff99', opacity=0.9)
            apo_m = float(df['alt_agl'].max())
            self._plotter.add_point_labels(
                [pos[idx]],
                [f"APOGEO\n{apo_m/1000:.2f} km"],
                font_size=10, text_color='#00ff99',
                point_size=0, shape=None)

        # ── ENU axes at origin ────────────────────────────────────────────────
        ax_len = size * 0.08
        self._plotter.add_arrows(
            np.array([[0, 0, z0]]),
            np.array([[ax_len, 0, 0]]),
            color='red',   label='E')
        self._plotter.add_arrows(
            np.array([[0, 0, z0]]),
            np.array([[0, ax_len, 0]]),
            color='lime',  label='N')
        self._plotter.add_arrows(
            np.array([[0, 0, z0]]),
            np.array([[0, 0, ax_len]]),
            color='cyan',  label='U')

        self._cam_iso()
        self._plotter.render()

    # ── Per-frame rendering ───────────────────────────────────────────────────

    def _render_frame(self, idx: int):
        """Update rocket mesh, partial trail, body axes and HUD at frame idx."""
        if not self._ready or self._df is None:
            return

        df  = self._df
        row = df.iloc[idx]
        t   = float(row['t'])

        pos  = np.array([float(row['pos_x']),
                          float(row['pos_y']),
                          float(row['alt_msl'])])
        quat = np.array([float(row['qw']), float(row['qx']),
                          float(row['qy']), float(row['qz'])])

        # ── Rocket mesh ───────────────────────────────────────────────────────
        if self._rocket_actor is not None:
            try:
                self._plotter.remove_actor(self._rocket_actor,
                                            render=False)
            except Exception:
                pass

        if self._base_mesh is not None:
            try:
                # Full transform:
                #   R_final = R_sim @ base_rot  (base_rot already baked into mesh)
                # _base_mesh already has base_rot applied, so we only apply R_sim
                mesh_t = _transform_to_enu(self._base_mesh, pos, quat)
                self._rocket_actor = self._plotter.add_mesh(
                    mesh_t, color='#c8c8d8', opacity=0.95,
                    smooth_shading=True, render=False)
            except Exception:
                self._rocket_actor = None

        # ── Growing trail ─────────────────────────────────────────────────────
        if self._trail_actor is not None:
            try:
                self._plotter.remove_actor(self._trail_actor, render=False)
            except Exception:
                pass
            self._trail_actor = None

        if idx > 1:
            step = max(1, idx // 800)
            trail_idx = list(range(0, idx + 1, step))
            if trail_idx[-1] != idx:
                trail_idx.append(idx)
            pts = np.column_stack([
                df['pos_x'].values[trail_idx],
                df['pos_y'].values[trail_idx],
                df['alt_msl'].values[trail_idx],
            ])
            if len(pts) >= 2:
                try:
                    line = pv.Spline(pts, n_points=min(len(pts), 500))
                    self._trail_actor = self._plotter.add_mesh(
                        line, color='#00d2ff', line_width=2.5,
                        opacity=0.9, render=False)
                except Exception:
                    pass

        # ── Body frame axes ───────────────────────────────────────────────────
        for a in self._axes_actors:
            try:
                self._plotter.remove_actor(a, render=False)
            except Exception:
                pass
        self._axes_actors = []

        if self._axes_chk.isChecked():
            try:
                from core.state import quat_to_rotmat
                R = quat_to_rotmat(quat)
                ax_len = max(self._len * 2.5, 80.0)
                origins = np.array([pos, pos, pos])
                dirs = np.array([R[:, 0], R[:, 1], R[:, 2]]) * ax_len
                colors = ['#ff4444', '#44ff44', '#4488ff']  # X, Y, Z(nose)
                for i, (col, label) in enumerate(zip(colors,
                                                      ['X', 'Y', 'Z(nariz)'])):
                    a = self._plotter.add_arrows(
                        origins[i:i+1], dirs[i:i+1],
                        color=col, render=False)
                    self._axes_actors.append(a)
            except Exception:
                pass

        # ── HUD ───────────────────────────────────────────────────────────────
        alt_agl = float(row.get('alt_agl', 0))
        speed   = float(row.get('speed',   0))
        mach    = float(row.get('mach',    0))
        thrust  = float(row.get('thrust',  0))
        self._hud.setText(
            f"t = {t:7.2f} s   "
            f"Alt = {alt_agl/1000:6.3f} km   "
            f"V = {speed:6.1f} m/s   "
            f"M = {mach:.3f}   "
            f"T = {thrust:6.0f} N"
        )

        # Update slider without triggering callback
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._time_lbl.setText(f"{t:.1f} s")

        # ── Camera follow ─────────────────────────────────────────────────────
        if self._follow_chk.isChecked() and self._playing:
            self._plotter.set_focus(pos)

        self._plotter.render()

    # ── Animation timer ───────────────────────────────────────────────────────

    def _on_timer(self):
        if self._df is None or not self._playing:
            return

        df    = self._df
        dt_ms = self._timer.interval()            # ms per UI tick
        dt_s  = dt_ms / 1000.0 * self._speed     # simulated seconds per tick

        # Advance by dt_s in simulation time
        t_now  = float(df['t'].iloc[self._frame])
        t_next = t_now + dt_s
        times  = df['t'].values

        if t_next >= times[-1]:
            # End of playback
            self._frame  = self._n - 1
            self._render_frame(self._frame)
            self._playing = False
            self._timer.stop()
            self._btn_play.setText("▶")
            return

        # Find closest frame to t_next
        new_frame = int(np.searchsorted(times, t_next))
        new_frame = min(max(new_frame, 0), self._n - 1)
        if new_frame != self._frame:
            self._frame = new_frame
            self._render_frame(self._frame)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self._df is None:
            return
        self._playing = not self._playing
        if self._playing:
            self._btn_play.setText("⏸")
            # If at end, restart
            if self._frame >= self._n - 1:
                self._frame = 0
            self._timer.start()
        else:
            self._btn_play.setText("▶")
            self._timer.stop()

    def _stop(self):
        self._playing = False
        self._timer.stop()
        self._btn_play.setText("▶")
        self._frame = 0
        if self._df is not None:
            self._render_frame(0)

    def _on_speed_change(self, idx: int):
        self._speed = self._SPEEDS[idx]

    def _on_slider(self, value: int):
        """Manual scrub: stop playback and jump to frame."""
        if self._df is None:
            return
        if self._playing:
            self._playing = False
            self._timer.stop()
            self._btn_play.setText("▶")
        self._frame = int(np.clip(value, 0, self._n - 1))
        self._render_frame(self._frame)

    # ── Axes toggle ───────────────────────────────────────────────────────────

    def _toggle_axes(self, state: int):
        """Called when the checkbox in the top toolbar changes."""
        # Keep button in sync
        self._btn_axes.blockSignals(True)
        self._btn_axes.setChecked(bool(state))
        self._btn_axes.setText("Ejes ✓" if state else "Ejes ✗")
        self._btn_axes.blockSignals(False)
        self._apply_axes_visibility()

    def _on_axes_btn_toggled(self, checked: bool):
        """Called when the playback-bar button is toggled."""
        # Keep checkbox in sync
        self._axes_chk.blockSignals(True)
        self._axes_chk.setChecked(checked)
        self._axes_chk.blockSignals(False)
        self._btn_axes.setText("Ejes ✓" if checked else "Ejes ✗")
        self._apply_axes_visibility()

    def _apply_axes_visibility(self):
        """Remove or re-add axes actors based on current toggle state."""
        if not self._ready or self._df is None:
            return

        show = self._axes_chk.isChecked()

        if not show:
            # Remove all existing axes actors immediately
            for a in self._axes_actors:
                try:
                    self._plotter.remove_actor(a, render=False)
                except Exception:
                    pass
            self._axes_actors = []
            self._plotter.render()
        else:
            # Re-render current frame (will re-add axes)
            self._render_frame(self._frame)

    # ── Camera presets ────────────────────────────────────────────────────────

    def _cam_iso(self):
        if not self._ready: return
        self._plotter.camera_position = 'iso'
        self._plotter.reset_camera()

    def _cam_side(self):
        if not self._ready: return
        self._plotter.view_yz()

    def _cam_top(self):
        if not self._ready: return
        self._plotter.view_xy()
