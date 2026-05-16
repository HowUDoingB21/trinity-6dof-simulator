# Trinity 6-DOF Flight Simulator

**Orbital Dynamics** — High-fidelity 6-DOF post-flight analysis and pre-flight validation tool for the **Trinity** two-stage experimental sounding rocket (target apogee: 50–100 km).

> Built to validate the Caronte V1 flight computer algorithms before hardware deployment.

---

## Overview

Trinity 6-DOF is a full-stack flight simulation environment that models the physics, aerodynamics, mass properties, and active fin control of a two-stage experimental rocket. It runs a faithful software port of the **Caronte V1** flight computer (EKF, PID, FSM, ServoController) against a physics engine, allowing hardware-validated control gains to be tested in simulation before flight.

```
Simulation Engine          Control System (Caronte V1 port)
──────────────────         ──────────────────────────────────
ISA Atmosphere             EKF9 State Estimator
Dryden Wind Model          PID with Dynamic Inversion
6-DOF Quaternion EOM       Aerodynamic Inversion (LUT)
Variable Mass Model        Flight State Machine (FSM)
RK45 Integrator            4-Fin Servo Mixer
Sensor Noise Injection     IMU / Barometer / GPS Models
```

---

## Features

- **Two-stage flight** with configurable staging trigger (altitude or time) and S2 ignition delay
- **Active fin control** — 4 independently actuated fins, full aerodynamic inversion from force commands to servo angles
- **Dual-state mass model** — CG interpolation between two known states (liftoff and burnout), matches OpenRocket output directly
- **Motor import** from standard `.eng` files (OpenRocket / RASP format)
- **Aerodynamic table import** from SimSweep CSV or `aero_table.h` LUT
- **PyQt6 GUI** with three tabs:
  - **Telemetry** — 6 synchronized live plots (altitude, speed, attitude, rates, servos, mass/thrust)
  - **3D View** — animated post-flight replay with correct orientation at every frame, STL rocket model support, growing trail, body-frame axes
  - **Log** — simulation log output
- **Save / Load** JSON configuration for repeatable simulation runs
- **CSV export** of full telemetry

---

## Architecture

```
trinity_6dof/
├── main.py                      # Entry point
├── core/
│   ├── atmosphere.py            # ISA atmosphere model (0–86 km)
│   ├── aerodynamics.py          # Body normal force, fin forces/moments, damping
│   ├── mass_model.py            # MassModel (S2 solo), TwoStageMassModel (S1+S2)
│   ├── physics.py               # 6-DOF ODE, RK45 integrator, ZOH control loop
│   └── state.py                 # Quaternion algebra, state vector conventions
├── control/
│   ├── ekf9.py                  # Port of EKF9.cpp — dual-IMU 9-state EKF
│   ├── pid.py                   # Port of PIDController.cpp — dynamic inversion PID
│   ├── fsm.py                   # Flight State Machine + ServoController
│   └── sensors.py               # IMU / barometer / GPS noise injection
├── simulation/
│   └── runner.py                # SimConfig, SimResults, two-stage orchestration
├── gui/
│   ├── main_window.py           # Main PyQt6 window, worker thread
│   ├── config_panel.py          # Stage / Aero / Control / Environment config
│   ├── telemetry_panel.py       # PyQtGraph live telemetry plots
│   └── viz3d_panel.py           # PyVista 3D animated replay
└── utils/
    ├── eng_parser.py            # .eng motor file parser
    ├── aero_table_parser.py     # SimSweep CSV / aero_table.h parser
    └── config_io.py             # JSON save/load
```

---

## Coordinate System

| Frame | Convention |
|---|---|
| Inertial | ENU (East-North-Up) |
| Body | body-Z = nose, body-X = fin 1/3 radial, body-Y = fin 2/4 radial |
| Launch (vertical) | body-Z = ENU-Z → quaternion = [1, 0, 0, 0] |
| Quaternion | Hamilton [w, x, y, z], body → ENU |
| CG / CP positions | From S2 nose tip (matches OpenRocket convention) |

---

## Mass Model — Entering OpenRocket Data

All CG positions are measured from the **nose tip of Stage 2** (the nose of the complete rocket). This matches OpenRocket's output directly.

| GUI Field | What to enter | Source |
|---|---|---|
| Stage 1 — CG lleno | Complete rocket CG at liftoff | OpenRocket static view |
| Stage 1 — CG vacío | Complete rocket CG at S1 burnout | OpenRocket simulation plot |
| Stage 2 — CG lleno | S2 CG at ignition (S2 full) | OpenRocket S2-only view |
| Stage 2 — CG vacío | S2 CG at burnout (S2 empty) | Estimate or OpenRocket sim |
| CP Stage 1 (Aero) | CP of complete rocket | OpenRocket static view |
| CP Stage 2 (Aero) | CP of S2 alone | OpenRocket S2-only view |

The model linearly interpolates CG between the two states proportional to propellant consumed. At t=0, the telemetry CG matches OpenRocket exactly.

---

## Control System

The control loop runs at **100 Hz** (zero-order hold). The Caronte V1 mapping for a vertical rocket with body-Z = nose:

| PID Channel | EKF Input | Rate Input | Actuator | Physical Effect |
|---|---|---|---|---|
| **Roll** | `EKF.yaw` (body-Z spin) | `gyro_z` | d_roll | τ_Z — corrects spin |
| **Pitch** | `-EKF.roll` (tilt toward Y) | `gyro_x` | d_pitch | τ_X — corrects Y-tilt |
| **Yaw** | `EKF.pitch` (tilt toward X) | `gyro_y` | d_yaw | τ_Y — corrects X-tilt |

Fin mixing (Caronte V1 convention):

```
δ1 = +d_pitch + d_roll    δ2 = +d_yaw - d_roll
δ3 = -d_pitch + d_roll    δ4 = -d_yaw - d_roll
```

---

## Installation

**Requirements:** Python 3.10+ on Windows (tested), macOS/Linux should work.

```bash
pip install numpy scipy pyqt6 pyqtgraph pyvista pyvistaqt pandas matplotlib
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

> **Note:** PyQt6 6.7.1 is recommended on Windows. Version 6.11.0 has known DLL load issues.

---

## Running

```bash
python main.py
```

1. Configure **Stages** (mass, geometry, motor `.eng` file)
2. Configure **Aero** (CP positions, fin geometry)
3. Configure **Control** (PID gains, servo limits)
4. Configure **Environment** (launch altitude, staging trigger)
5. Click **Run Simulation**
6. View results in **Telemetry** and **3D View** tabs
7. **Export CSV** for post-processing

---

## 3D Visualization

The 3D tab supports animated post-flight replay:

- **Play / Pause / Stop** — controls playback
- **Speed** — 0.1× to 20× real time
- **Slider** — manual scrub to any instant
- **Body axes** — toggle red/green/blue body-frame arrows
- **Follow** — camera tracks the rocket during playback
- **Load STL** — use your own CAD model
- **Auto-scale** — scales STL to match configured rocket length
- **Base orientation presets** — correct axis mismatch between CAD and simulator (+Z→nose, +X→nose, +Y→nose, -Z→nose)

---

## Project Context

This simulator was built to support the development of **Trinity**, a two-stage experimental sounding rocket developed by **Orbital Dynamics** at Universidad Marista de Guadalajara. Trinity targets a 50–100 km apogee using active aerodynamic fin control in the transonic and supersonic regimes.

The **Caronte V1** flight computer (STM32F405RGT6) runs the EKF, PID, and FSM algorithms ported here. Simulation-before-flight validation is a core development principle: control gains and staging logic are tested in this simulator before deployment to hardware.

---

## License

MIT License — see `LICENSE` for details.

---

*Orbital Dynamics — Guadalajara, México*
