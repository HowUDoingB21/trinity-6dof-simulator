"""
utils/config_io.py — Simulation Configuration Save / Load
Trinity 6-DOF Simulator | Orbital Dynamics

Serialises every GUI widget value to a human-readable JSON file and
restores it. All numeric fields, file paths, combo selections, and
checkboxes are captured so a session can be continued exactly as left.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui.config_panel import ConfigPanel


# ── Serialise ─────────────────────────────────────────────────────────────────

def save_config(panel: "ConfigPanel", path: str | Path) -> None:
    """Write all GUI values to a JSON file."""
    data = {
        "trinity_6dof_config": True,
        "version": 2,
        "stage1": _dump_stage(panel.s1_widget),
        "stage2": _dump_stage(panel.s2_widget),
        "aero":   _dump_aero(panel.aero_widget),
        "control":_dump_control(panel.ctrl_widget),
        "env":    _dump_env(panel.env_widget),
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_config(panel: "ConfigPanel", path: str | Path) -> None:
    """Restore GUI values from a JSON file previously saved by save_config."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    if not data.get("trinity_6dof_config"):
        raise ValueError("Not a valid Trinity 6-DOF configuration file.")

    _restore_stage(panel.s1_widget, data.get("stage1", {}))
    _restore_stage(panel.s2_widget, data.get("stage2", {}))
    _restore_aero(panel.aero_widget, data.get("aero", {}))
    _restore_control(panel.ctrl_widget, data.get("control", {}))
    _restore_env(panel.env_widget, data.get("env", {}))


# ── Stage widget ──────────────────────────────────────────────────────────────

def _dump_stage(w) -> dict:
    return {
        "dry_mass":   w.dry_mass.value(),
        "prop_mass":  w.prop_mass.value(),
        "diameter":   w.diameter.value(),
        "length":     w.length.value(),
        "cg_dry":     w.cg_dry.value(),
        "cg_prop":    w.cg_prop.value(),
        "grain_len":  w.grain_len.value(),
        "grain_od":   w.grain_od.value(),
        "grain_id":   w.grain_id.value(),
        "Ixx": w.Ixx.value(), "Iyy": w.Iyy.value(), "Izz": w.Izz.value(),
        "Ixy": w.Ixy.value(), "Ixz": w.Ixz.value(), "Iyz": w.Iyz.value(),
        "eng_path":   w._eng_path,
    }


def _restore_stage(w, d: dict) -> None:
    if not d:
        return
    for attr, key in [
        ("dry_mass", "dry_mass"), ("prop_mass", "prop_mass"),
        ("diameter", "diameter"), ("length", "length"),
        ("cg_dry",   "cg_dry"),   ("cg_prop",  "cg_prop"),
        ("grain_len","grain_len"), ("grain_od", "grain_od"),
        ("grain_id", "grain_id"),
        ("Ixx","Ixx"),("Iyy","Iyy"),("Izz","Izz"),
        ("Ixy","Ixy"),("Ixz","Ixz"),("Iyz","Iyz"),
    ]:
        if key in d:
            getattr(w, attr).setValue(d[key])

    eng = d.get("eng_path", "")
    if eng and Path(eng).exists():
        w._eng_path = eng
        from pathlib import Path as _P
        w._eng_le.setText(_P(eng).name)
        w._update_eng_label()


# ── Aero widget ───────────────────────────────────────────────────────────────

def _dump_aero(w) -> dict:
    return {
        "s1_csv":   w._s1_csv_path,
        "s2_csv":   w._s2_csv_path,
        "at_s1":    w._at_s1_path,
        "at_s2":    w._at_s2_path,
        "cp_s1":    w.cp_s1.value(),
        "cp_s2":    w.cp_s2.value(),
        "fin_sref": w.fin_sref.value(),
        "fin_d":    w.fin_d.value(),
        "fin_r":    w.fin_r.value(),
        "damp_lat": w.damp_lat.value(),
        "damp_rol": w.damp_rol.value(),
    }


def _restore_aero(w, d: dict) -> None:
    if not d:
        return
    for attr, key in [
        ("cp_s1","cp_s1"),("cp_s2","cp_s2"),
        ("fin_sref","fin_sref"),("fin_d","fin_d"),("fin_r","fin_r"),
        ("damp_lat","damp_lat"),("damp_rol","damp_rol"),
    ]:
        if key in d:
            getattr(w, attr).setValue(d[key])

    for path_attr, le_attr, key in [
        ("_s1_csv_path", "_s1_csv_le", "s1_csv"),
        ("_s2_csv_path", "_s2_csv_le", "s2_csv"),
        ("_at_s1_path",  "_at_s1_le",  "at_s1"),
        ("_at_s2_path",  "_at_s2_le",  "at_s2"),
    ]:
        p = d.get(key, "")
        if p:
            setattr(w, path_attr, p)
            from pathlib import Path as _P
            le = getattr(w, le_attr)
            le.setText(_P(p).name if Path(p).exists() else f"⚠ {_P(p).name} (not found)")


# ── Control widget ────────────────────────────────────────────────────────────

def _dump_control(w) -> dict:
    return {
        "ctrl_en": w.ctrl_en.isChecked(),
        "pid_pitch_kp": w.pid_pitch_kp.value(),
        "pid_pitch_ki": w.pid_pitch_ki.value(),
        "pid_pitch_kd": w.pid_pitch_kd.value(),
        "pid_yaw_kp":   w.pid_yaw_kp.value(),
        "pid_yaw_ki":   w.pid_yaw_ki.value(),
        "pid_yaw_kd":   w.pid_yaw_kd.value(),
        "pid_roll_kp":  w.pid_roll_kp.value(),
        "pid_roll_ki":  w.pid_roll_ki.value(),
        "pid_roll_kd":  w.pid_roll_kd.value(),
        "d_max":    w.d_max.value(),
        "slew":     w.slew.value(),
        "latency":  w.latency.value(),
        "lever_d":  w.lever_d.value(),
        "Q_alt":    w.Q_alt.value(),
        "Q_vz":     w.Q_vz.value(),
        "Q_att":    w.Q_att.value(),
        "R_baro":   w.R_baro.value(),
        "sn_en":    w.sn_en.isChecked(),
        "accel_sigma": w.accel_sigma.value(),
        "gyro_sigma":  w.gyro_sigma.value(),
        "baro_sigma":  w.baro_sigma.value(),
    }


def _restore_control(w, d: dict) -> None:
    if not d:
        return
    if "ctrl_en" in d:
        w.ctrl_en.setChecked(d["ctrl_en"])
    if "sn_en" in d:
        w.sn_en.setChecked(d["sn_en"])
    for attr in [
        "pid_pitch_kp","pid_pitch_ki","pid_pitch_kd",
        "pid_yaw_kp","pid_yaw_ki","pid_yaw_kd",
        "pid_roll_kp","pid_roll_ki","pid_roll_kd",
        "d_max","slew","latency","lever_d",
        "Q_alt","Q_vz","Q_att","R_baro",
        "accel_sigma","gyro_sigma","baro_sigma",
    ]:
        if attr in d:
            getattr(w, attr).setValue(d[attr])


# ── Environment widget ────────────────────────────────────────────────────────

def _dump_env(w) -> dict:
    return {
        "launch_alt":   w.launch_alt.value(),
        "tilt_pitch":   w.tilt_pitch.value(),
        "tilt_yaw":     w.tilt_yaw.value(),
        "staging_mode": w.staging_mode.currentIndex(),
        "staging_alt":  w.staging_alt.value(),
        "staging_time": w.staging_time.value(),
        "s2_delay":     w.s2_delay.value(),
        "max_time":     w.max_time.value(),
        "rtol":         w.rtol.value(),
        "atol":         w.atol.value(),
    }


def _restore_env(w, d: dict) -> None:
    if not d:
        return
    if "staging_mode" in d:
        w.staging_mode.setCurrentIndex(d["staging_mode"])
    for attr in [
        "launch_alt", "tilt_pitch", "tilt_yaw",
        "staging_alt", "staging_time", "s2_delay",
        "max_time", "rtol", "atol",
    ]:
        if attr in d:
            getattr(w, attr).setValue(d[attr])
