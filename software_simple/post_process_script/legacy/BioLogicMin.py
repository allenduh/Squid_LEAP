# biologic_min.py
# Minimal Bio-Logic potentiostat control: OCV + CA + CSV/NPZ saving + optional .mps loader
# Requires: pip install PyExpLabSys eclabfiles (for .mps loading, optional)

from __future__ import annotations
import time, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

# High-level driver (EC-Lab OEM DLL wrapper)
from PyExpLabSys.drivers.bio_logic import SP150, OCV, CA  # Works for SP-150; see docs for other models

# Optional: parse .mps settings for consistency
try:
    import eclabfiles as ecf
    HAS_ECLABFILES = True
except Exception:
    HAS_ECLABFILES = False


@dataclass
class RunResult:
    t: List[float]
    Ewe: List[float]
    I: Optional[List[float]] = None
    meta: Dict[str, Any] = None


class BioLogicMin:
    """Super-small controller for OCV + CA (square wave) and saving traces."""
    def __init__(self, ip: str, dll_path: Optional[str] = None, channel: int = 0):
        self.ip = ip
        self.dll_path = dll_path
        self.ch = int(channel)
        self.dev = None

    # ---- connection ---------------------------------------------------------
    def connect(self):
        self.dev = SP150(self.ip, EClib_dll_path=self.dll_path)
        self.dev.connect()

    def disconnect(self):
        if self.dev:
            try:
                self.dev.stop_channel(self.ch)
            except Exception:
                pass
            self.dev.disconnect()
            self.dev = None

    # ---- OCV ----------------------------------------------------------------
    def ocv_once(self,
                 rest_time_s: float = 0.2,
                 record_every_dE: float = 10.0,
                 record_every_dT: float = 0.01) -> RunResult:
        """
        Start OCV briefly and return the latest OCV with its short trace.
        """
        ocv = OCV(rest_time_T=rest_time_s,
                  record_every_dE=record_every_dE,
                  record_every_dT=record_every_dT)
        self.dev.load_technique(self.ch, ocv)
        self.dev.start_channel(self.ch)

        t, ewe = [], []
        # Poll until technique finishes (get_data returns None)
        while True:
            data = self.dev.get_data(self.ch)
            if data is None:
                break
            # According to docs: .time and .Ewe fields are exposed by OCV technique
            t.extend(list(data.time))
            ewe.extend(list(data.Ewe))
            time.sleep(0.05)

        # Stop
        self.dev.stop_channel(self.ch)

        return RunResult(t=t, Ewe=ewe, I=None, meta={"technique": "OCV"})

    # ---- CA (square wave helper) -------------------------------------------
    def ca_square(self,
                  v_plus: float,
                  v_minus: float,
                  half_period_s: float,
                  cycles: int,
                  record_every_dT: float = 0.001) -> RunResult:
        """
        Run a simple CA square wave: [V+ for T] -> [V- for T] repeated 'cycles' times.
        """
        # The CA technique in PyExpLabSys maps to the EC-Lab CA; we provide two steps per cycle.
        # Build a step table: [E1, t1, E2, t2, E1, t1, ...]
        steps_E = []
        steps_T = []
        for _ in range(int(cycles)):
            steps_E += [float(v_plus), float(v_minus)]
            steps_T += [float(half_period_s), float(half_period_s)]

        ca = CA( # Common useful args (others left default per driver)
            N_steps=len(steps_E),
            E=steps_E,
            t=steps_T,
            record_every_dT=record_every_dT
        )

        self.dev.load_technique(self.ch, ca)
        self.dev.start_channel(self.ch)

        t, ewe, cur = [], [], []
        while True:
            data = self.dev.get_data(self.ch)
            if data is None:
                break
            # CA data fields typically include time, Ewe, I, and step index
            t.extend(list(data.time))
            ewe.extend(list(data.Ewe))
            cur.extend(list(data.I))
            time.sleep(0.02)

        self.dev.stop_channel(self.ch)
        return RunResult(t=t, Ewe=ewe, I=cur, meta={
            "technique": "CA",
            "v_plus": v_plus, "v_minus": v_minus,
            "half_period_s": half_period_s, "cycles": cycles
        })

    # ---- saving -------------------------------------------------------------
    @staticmethod
    def save_npz_csv(out_base: Path, rr: RunResult):
        out_base = Path(out_base)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        # NPZ
        npz_path = out_base.with_suffix(".npz")
        try:
            import numpy as np
            np.savez_compressed(npz_path, t=rr.t, Ewe=rr.Ewe, I=rr.I if rr.I is not None else [])
        except Exception as e:
            print("NPZ save failed:", e)

        # CSV
        csv_path = out_base.with_suffix(".csv")
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            header = ["t_s", "Ewe_V"] + (["I_A"] if rr.I is not None else [])
            w.writerow(header)
            for i in range(len(rr.t)):
                row = [rr.t[i], rr.Ewe[i]]
                if rr.I is not None and i < len(rr.I):
                    row.append(rr.I[i])
                w.writerow(row)

        # META
        meta_path = out_base.with_suffix(".json")
        with meta_path.open("w") as f:
            json.dump(rr.meta or {}, f, indent=2)

        return {"npz": str(npz_path), "csv": str(csv_path), "json": str(meta_path)}

    # ---- .mps: optional parameter loader for consistency --------------------
    @staticmethod
    def ca_params_from_mps(mps_path: str):
        """
        Parse a .mps and pull the first CA technique steps (E, t).
        Requires `eclabfiles` (optional).
        """
        if not HAS_ECLABFILES:
            raise RuntimeError("Install 'eclabfiles' to load .mps (pip install eclabfiles)")

        techniques, meta = ecf.process(mps_path)   # .mps returns techniques
        # Find first CA-like entry
        for tech in techniques:
            name = (tech.get("technique") or "").upper()
            if "CA" in name:  # Chrono-Amperometry
                # Typical fields: E (list of floats), t (list of floats), maybe repeats
                steps_E = list(map(float, tech.get("E", [])))
                steps_T = list(map(float, tech.get("t", [])))
                if not steps_E or not steps_T or len(steps_E) != len(steps_T):
                    continue
                return steps_E, steps_T, tech
        raise ValueError("No CA technique with steps found in .mps")
