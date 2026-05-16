"""
utils/eng_parser.py — OpenRocket / RASP .eng Thrust Curve Parser
Trinity 6-DOF Simulator | Orbital Dynamics

Parses the standard .eng file format used by OpenRocket, RASAero, and most
commercial motor certification databases (Thrustcurve.org).

FILE FORMAT
───────────
Lines starting with ';' are comments.
The first non-comment line is the HEADER:

  <Name> <Diameter_mm> <Length_mm> <Delays> <PropMass_kg> <TotalMass_kg> <Mfr>

Subsequent non-comment lines are data points:
  <Time_s> <Thrust_N>

The data set ends at EOF or at a line with Time < previous Time (not standard
but seen in the wild). An implicit Thrust=0 is added at t=0 if missing, and
at t=t_end if the last thrust value is non-zero.
"""

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# numpy 2.x renamed trapz → trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


@dataclass
class EngData:
    """Parsed motor data from a single .eng file."""
    name: str
    diameter_mm: float
    length_mm: float
    delays: str
    propellant_mass_kg: float
    total_mass_kg: float
    manufacturer: str

    times: np.ndarray         # [s]  thrust time series
    thrusts: np.ndarray       # [N]  thrust at each time

    @property
    def dry_mass_kg(self) -> float:
        return self.total_mass_kg - self.propellant_mass_kg

    @property
    def total_impulse_Ns(self) -> float:
        """Numerically integrated total impulse [N·s]."""
        return float(_trapz(self.thrusts, self.times))

    @property
    def burn_time_s(self) -> float:
        """Time at which thrust returns to zero."""
        return float(self.times[-1])

    @property
    def average_thrust_N(self) -> float:
        return self.total_impulse_Ns / self.burn_time_s if self.burn_time_s > 0 else 0.0

    @property
    def peak_thrust_N(self) -> float:
        return float(np.max(self.thrusts))

    @property
    def motor_class(self) -> str:
        """
        Classify by total impulse (standard NFPA classification).
        """
        I = self.total_impulse_Ns
        thresholds = [
            (0.3125, '1/8A'), (0.625, '1/4A'), (1.25, '1/2A'),
            (2.5, 'A'), (5, 'B'), (10, 'C'), (20, 'D'), (40, 'E'),
            (80, 'F'), (160, 'G'), (320, 'H'), (640, 'I'), (1280, 'J'),
            (2560, 'K'), (5120, 'L'), (10240, 'M'), (20480, 'N'),
            (40960, 'O'), (81920, 'P'),
        ]
        for limit, letter in thresholds:
            if I <= limit:
                return letter
        return 'O+'

    def get_thrust(self, t: float) -> float:
        """Interpolate thrust at time t [s]. Returns 0 outside burn window."""
        if t < 0.0 or t > self.times[-1]:
            return 0.0
        return float(np.interp(t, self.times, self.thrusts))

    def get_propellant_remaining(self, t: float) -> float:
        """
        Remaining propellant mass at time t [kg].
        Computed from impulse fraction:
          m_prop(t) = m_prop_total × (1 - I_used(t) / I_total)
        """
        if t <= 0.0:
            return self.propellant_mass_kg
        if t >= self.times[-1]:
            return 0.0

        # Integrate thrust from 0 to t
        mask = self.times <= t
        t_trunc = np.append(self.times[mask], t)
        F_trunc = np.append(self.thrusts[mask], self.get_thrust(t))
        impulse_used = float(_trapz(F_trunc, t_trunc))

        fraction_remaining = max(0.0, 1.0 - impulse_used / self.total_impulse_Ns)
        return self.propellant_mass_kg * fraction_remaining

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.manufacturer})  |  "
            f"Class {self.motor_class}  |  "
            f"I_total = {self.total_impulse_Ns:.1f} N·s  |  "
            f"T_avg = {self.average_thrust_N:.1f} N  |  "
            f"T_peak = {self.peak_thrust_N:.1f} N  |  "
            f"t_burn = {self.burn_time_s:.3f} s  |  "
            f"m_prop = {self.propellant_mass_kg:.3f} kg"
        )


def parse_eng(source: str | Path) -> list[EngData]:
    """
    Parse one or more motors from a .eng file.

    A single file may contain multiple motor definitions separated by the
    header line pattern (some databases bundle entire motor families).

    Parameters
    ----------
    source : str or Path
        File path or raw text content of the .eng file.

    Returns
    -------
    list[EngData]
        One EngData object per motor found in the file.
    """
    # Allow raw text or file path
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        text = Path(source).read_text(encoding='utf-8', errors='replace')
    else:
        text = str(source)

    motors: list[EngData] = []
    current_header: dict | None = None
    current_data: list[tuple[float, float]] = []

    HEADER_RE = re.compile(
        r'^\s*([A-Za-z0-9\-_.]+)'   # name
        r'\s+([\d.]+)'               # diameter mm
        r'\s+([\d.]+)'               # length mm
        r'\s+(\S+)'                  # delays
        r'\s+([\d.eE+\-]+)'         # propellant mass kg
        r'\s+([\d.eE+\-]+)'         # total mass kg
        r'\s+(\S+)',                 # manufacturer
        re.ASCII,
    )

    def _flush():
        """Convert accumulated data into EngData and append."""
        nonlocal current_header, current_data
        if current_header is None or not current_data:
            return
        times   = np.array([p[0] for p in current_data], dtype=float)
        thrusts = np.array([p[1] for p in current_data], dtype=float)

        # Ensure starts and ends at zero thrust for clean interpolation
        if times[0] > 1e-9:
            times   = np.insert(times, 0, 0.0)
            thrusts = np.insert(thrusts, 0, 0.0)
        if thrusts[-1] > 1e-3:
            dt = times[-1] - times[-2] if len(times) > 1 else 0.01
            times   = np.append(times, times[-1] + dt)
            thrusts = np.append(thrusts, 0.0)

        motors.append(EngData(
            name               = current_header['name'],
            diameter_mm        = current_header['diameter_mm'],
            length_mm          = current_header['length_mm'],
            delays             = current_header['delays'],
            propellant_mass_kg = current_header['propellant_mass_kg'],
            total_mass_kg      = current_header['total_mass_kg'],
            manufacturer       = current_header['manufacturer'],
            times              = times,
            thrusts            = thrusts,
        ))
        current_header = None
        current_data   = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip blank lines and comments
        if not line or line.startswith(';'):
            continue

        # Try to match a data point first (most common line)
        parts = line.split()
        if len(parts) == 2:
            try:
                t_val = float(parts[0])
                f_val = float(parts[1])
                if current_header is not None:
                    current_data.append((t_val, f_val))
                    continue
            except ValueError:
                pass

        # Try to match a header line
        m = HEADER_RE.match(line)
        if m:
            _flush()  # Save any previous motor
            current_header = {
                'name'               : m.group(1),
                'diameter_mm'        : float(m.group(2)),
                'length_mm'          : float(m.group(3)),
                'delays'             : m.group(4),
                'propellant_mass_kg' : float(m.group(5)),
                'total_mass_kg'      : float(m.group(6)),
                'manufacturer'       : m.group(7),
            }
            current_data = []

    _flush()  # Save last motor

    if not motors:
        raise ValueError(
            "No valid motor definition found in the supplied .eng content."
        )

    return motors


def load_eng(path: str | Path) -> EngData:
    """
    Load the first motor from a .eng file.
    Raises ValueError if the file contains no valid motor.
    """
    motors = parse_eng(path)
    return motors[0]
