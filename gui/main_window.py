"""
gui/main_window.py — Main Application Window
Trinity 6-DOF Simulator | Orbital Dynamics

Layout
──────
┌─────────────────────────────────────────────────────┐
│  Config Panel (left, 400 px)  │  Results Area (right) │
│  ─ Stages                     │  ┌ Telemetry / 3D tab ┤
│  ─ Aero                       │  │  PyQtGraph plots   │
│  ─ Control                    │  │  or PyVista 3D     │
│  ─ Environment                │  └────────────────────┤
│  [Run] [Export CSV]           │  Status / summary bar │
└─────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTabWidget, QProgressBar, QLabel,
    QStatusBar, QMessageBox, QFileDialog, QTextEdit,
    QFrame, QGroupBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QAction

from gui.config_panel   import ConfigPanel
from gui.telemetry_panel import TelemetryPanel
from gui.viz3d_panel    import Viz3DPanel


# ─── Worker thread for simulation ─────────────────────────────────────────────

class SimWorker(QObject):
    finished  = pyqtSignal(object)   # SimResults
    progress  = pyqtSignal(float, str)
    error     = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._runner = None

    def run(self):
        from simulation.runner import SimulationRunner
        try:
            self._runner = SimulationRunner(
                progress_cb=lambda f, m: self.progress.emit(f, m)
            )
            results = self._runner.run(self._config)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")

    def abort(self):
        if self._runner:
            self._runner.abort()


# ─── Summary widget ────────────────────────────────────────────────────────────

class SummaryWidget(QWidget):
    """Small panel below the results tab showing key flight statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self._labels: dict[str, QLabel] = {}
        for key, title in [
            ('apogee',  'Apogee AGL'),
            ('max_mach','Max Mach'),
            ('max_q',   'Max q'),
            ('staging', 'Staging t'),
            ('runtime', 'Wall time'),
        ]:
            col = QVBoxLayout()
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("color:#888; font-size:10px;")
            t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v_lbl = QLabel("—")
            v_lbl.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold;")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(t_lbl); col.addWidget(v_lbl)
            layout.addLayout(col, 1)
            self._labels[key] = v_lbl

    def update(self, results):
        self._labels['apogee'].setText(
            f"{results.apogee_alt_agl_m/1000:.2f} km")
        self._labels['max_mach'].setText(f"{results.max_mach:.3f}")
        self._labels['max_q'].setText(
            f"{results.max_q_pa/1000:.2f} kPa")
        self._labels['staging'].setText(f"{results.staging_time_s:.2f} s")
        self._labels['runtime'].setText(f"{results.runtime_s:.1f} s")


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trinity 6-DOF Simulator — Orbital Dynamics")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 950)

        self._results = None
        self._worker_thread: QThread | None = None
        self._worker: SimWorker | None = None

        self._build_menu()
        self._build_ui()
        self._build_status_bar()
        self._apply_dark_theme()

        # Restore last session automatically
        self._cfg_panel.auto_load_last()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self):
        bar = self.menuBar()

        # File
        file_menu = bar.addMenu("File")
        act_save_cfg = QAction("💾  Guardar setup…", self)
        act_save_cfg.triggered.connect(lambda: self._cfg_panel._save_config())
        act_save_cfg.setShortcut("Ctrl+Shift+S")
        act_load_cfg = QAction("📂  Cargar setup…", self)
        act_load_cfg.triggered.connect(lambda: self._cfg_panel._load_config())
        act_load_cfg.setShortcut("Ctrl+O")
        act_export = QAction("⬇  Export telemetry CSV…", self)
        act_export.triggered.connect(self._export_csv)
        act_export.setShortcut("Ctrl+S")
        file_menu.addAction(act_save_cfg)
        file_menu.addAction(act_load_cfg)
        file_menu.addSeparator()
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        act_quit.setShortcut("Ctrl+Q")
        file_menu.addAction(act_quit)

        # Simulation
        sim_menu = bar.addMenu("Simulation")
        act_run  = QAction("▶  Run", self)
        act_run.setShortcut("F5")
        act_run.triggered.connect(self._on_run)
        act_abort = QAction("■  Abort", self)
        act_abort.triggered.connect(self._on_abort)
        sim_menu.addAction(act_run)
        sim_menu.addAction(act_abort)

        # View
        view_menu = bar.addMenu("View")
        act_tel = QAction("Telemetry tab", self)
        act_tel.triggered.connect(lambda: self._result_tabs.setCurrentIndex(0))
        act_3d  = QAction("3D tab", self)
        act_3d.triggered.connect(lambda: self._result_tabs.setCurrentIndex(1))
        act_log = QAction("Log tab", self)
        act_log.triggered.connect(lambda: self._result_tabs.setCurrentIndex(2))
        view_menu.addAction(act_tel)
        view_menu.addAction(act_3d)
        view_menu.addAction(act_log)

    # ── Central UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Left: Config panel ─────────────────────────────────────────────
        self._cfg_panel = ConfigPanel()
        self._cfg_panel.run_requested.connect(self._on_run)
        self._cfg_panel.export_requested.connect(self._export_csv)

        # ── Right: Results area ────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Result tabs
        self._result_tabs = QTabWidget()

        # Telemetry tab
        self._tel_panel = TelemetryPanel()
        self._result_tabs.addTab(self._tel_panel, "📈  Telemetry")

        # 3D tab
        self._viz3d_panel = Viz3DPanel()
        self._result_tabs.addTab(self._viz3d_panel, "🚀  3D View")

        # Log tab
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setFont(QFont("Consolas", 9))
        self._log_edit.setStyleSheet("background:#0d0d0d; color:#aaffaa;")
        self._result_tabs.addTab(self._log_edit, "📋  Log")

        right_layout.addWidget(self._result_tabs, 1)

        # Summary bar
        self._summary = SummaryWidget()
        self._summary.setFixedHeight(60)
        self._summary.setFrameShape = lambda _: None   # cosmetic
        right_layout.addWidget(self._summary)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(
            "QProgressBar{border:none;background:#1a1a1a;border-radius:3px;}"
            "QProgressBar::chunk{background:#2a7aff;border-radius:3px;}"
        )
        right_layout.addWidget(self._progress)

        # ── Splitter ───────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._cfg_panel)
        splitter.addWidget(right)
        splitter.setSizes([400, 1200])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        root.addWidget(splitter, 1)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._sb_lbl = QLabel("Ready.")
        sb.addPermanentWidget(self._sb_lbl)

    # ── Run / Abort ───────────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker_thread and self._worker_thread.isRunning():
            QMessageBox.warning(self, "Simulation running",
                                "A simulation is already in progress. "
                                "Abort it first.")
            return

        # Build config (validates required files)
        try:
            config = self._cfg_panel.build_sim_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))
            return

        self._log_edit.clear()
        self._log("Simulation starting…")
        self._cfg_panel.btn_run.setEnabled(False)
        self._progress.setValue(0)
        self._sb_lbl.setText("Running…")

        # Spin up worker
        self._worker = SimWorker(config)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_sim_error)
        self._worker_thread.start()

    def _on_abort(self):
        if self._worker:
            self._worker.abort()
            self._log("Abort requested…")

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._sb_lbl.setText(message)
        self._log(message)

    def _on_sim_finished(self, results):
        self._worker_thread.quit()
        self._worker_thread.wait()
        self._cfg_panel.btn_run.setEnabled(True)
        self._progress.setValue(100)

        if not results.success:
            self._log(f"ERROR: {results.message}")
            QMessageBox.critical(self, "Simulation failed", results.message[:500])
            self._sb_lbl.setText("Simulation failed.")
            return

        self._results = results
        df = results.telemetry_df

        # ── Update telemetry panel ─────────────────────────────────────────
        self._tel_panel.update_data(
            df,
            staging_t = results.staging_time_s  if results.staging_time_s > 0 else None,
            apogee_t  = results.apogee_time_s   if results.apogee_time_s  > 0 else None,
        )

        # ── Update 3D panel ────────────────────────────────────────────────
        s2 = self._cfg_panel.s2_widget
        self._viz3d_panel.update_data(
            df,
            diameter_m = s2.diameter.value(),
            length_m   = s2.length.value(),
            staging_t  = results.staging_time_s if results.staging_time_s > 0 else None,
            apogee_t   = results.apogee_time_s  if results.apogee_time_s  > 0 else None,
        )

        # ── Update summary ─────────────────────────────────────────────────
        self._summary.update(results)

        # ── Log ───────────────────────────────────────────────────────────
        self._log(f"\n{'─'*60}")
        self._log(f"  Apogee AGL   : {results.apogee_alt_agl_m/1000:.3f} km")
        self._log(f"  Apogee MSL   : {results.apogee_alt_msl_m/1000:.3f} km")
        self._log(f"  Apogee time  : {results.apogee_time_s:.2f} s")
        self._log(f"  Staging time : {results.staging_time_s:.2f} s")
        self._log(f"  Max Mach     : {results.max_mach:.4f}")
        self._log(f"  Max q        : {results.max_q_pa/1000:.2f} kPa")
        self._log(f"  Wall time    : {results.runtime_s:.2f} s")
        self._log(f"  Data points  : {len(df)}")
        self._log(f"{'─'*60}\n")

        self._sb_lbl.setText(
            f"Done — Apogee {results.apogee_alt_agl_m/1000:.2f} km  |  "
            f"Mach {results.max_mach:.3f}  |  "
            f"Wall time {results.runtime_s:.1f} s"
        )
        self._result_tabs.setCurrentIndex(0)

    def _on_sim_error(self, message: str):
        self._worker_thread.quit()
        self._worker_thread.wait()
        self._cfg_panel.btn_run.setEnabled(True)
        self._progress.setValue(0)
        self._log(f"FATAL ERROR:\n{message}")
        QMessageBox.critical(self, "Simulation error", message[:600])
        self._sb_lbl.setText("Error — see log.")

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        if self._results is None or self._results.telemetry_df.empty:
            QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export telemetry", "trinity_telemetry.csv",
            "CSV Files (*.csv)"
        )
        if path:
            self._results.telemetry_df.to_csv(path, index=False)
            self._sb_lbl.setText(f"Exported → {path}")
            self._log(f"Telemetry exported to: {path}")

    # ── Log helper ────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self._log_edit.append(msg)

    # ── Dark theme ────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #12121e;
                color: #d0d0d0;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 11px;
            }
            QTabWidget::pane { border: 1px solid #2a2a3a; }
            QTabBar::tab {
                background: #1e1e2e; color: #888; padding: 6px 14px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #2a2a3a; color: #ddd; }
            QGroupBox {
                border: 1px solid #2a2a3a; border-radius: 4px;
                margin-top: 6px; padding-top: 6px;
                color: #888; font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; top: -1px;
            }
            QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
                background: #1e1e2e; border: 1px solid #333;
                border-radius: 3px; padding: 2px 4px; color: #ddd;
            }
            QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
                border: 1px solid #2a7aff;
            }
            QPushButton {
                background: #1e1e2e; border: 1px solid #333;
                border-radius: 4px; padding: 4px 10px; color: #ccc;
            }
            QPushButton:hover { background: #2a2a3a; border-color: #555; }
            QScrollBar:vertical {
                background: #1a1a2e; width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #333; border-radius: 4px; min-height: 20px;
            }
            QSplitter::handle { background: #2a2a3a; width: 2px; }
            QProgressBar { background: #1a1a1a; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #2a7aff; border-radius: 3px; }
            QMenuBar { background: #12121e; color: #ccc; }
            QMenuBar::item:selected { background: #2a2a3a; }
            QMenu { background: #1e1e2e; border: 1px solid #333; }
            QMenu::item:selected { background: #2a7aff; }
        """)
