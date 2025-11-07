#!/usr/bin/env python3
"""
potentiostat_live_ca.py  —  concise CA + OCV live viewer (Bio-Logic EC-Lab OEM, Python)

Assumes your get_experiment_data(api, data, tech_name, board_type) yields dictionaries:
    OCV: {"t", "Ewe"} or {"t","Ewe","Ece"}
    CP : {"t","Ewe","Iwe","cycle"}  (kept here for completeness)
    CA : {"t","Ewe","Iwe","cycle"}  (same shape as CP in practice)

Features:
  • Chrono-Amperometry (CA): steps = [0 V (1 s), -10 mV (1 s)], repeat 10 cycles
  • OCV: continuous Ewe (and Ece if available)
  • Live plots (Ewe/Ece and Iwe)
  • VMP300 (PREMIUM) hardware Get/Set: connection and grounded/floating
"""

import os
import time
import argparse
from collections import deque

import matplotlib.pyplot as plt

import kbio.kbio_types as KBIO
from kbio.c_utils import c_is_64b
from kbio.kbio_api import KBIO_api
from kbio.kbio_tech import ECC_parm, make_ecc_parm, make_ecc_parms
from kbio.kbio_tech import get_experiment_data, get_info_data


# -------------------------------
# Potentiostat (concise)
# -------------------------------

class Potentiostat:
    def __init__(self, address="USB0", channel=1, eclib_dir=None, verbosity=1):
        self.address = address
        self.channel = int(channel)
        self.verbosity = verbosity
        if eclib_dir is None:
            eclib_dir = os.environ.get("ECLIB_DIR", f"C:{os.sep}EC-Lab Development Package{os.sep}lib")
        self.dll_path = os.path.join(eclib_dir, "EClib64.dll" if c_is_64b else "EClib.dll")
        self.api = None; self.id_ = None; self.board_type = None
        self.ca_ecc = None; self.ocv_ecc = None
        # Simple in-memory buffers (optional)
        self.ivCurve = {"t": [], "V_we": [], "I": [], "cycle": []}


    def connect(self):
        self.api = KBIO_api(self.dll_path)
        if self.verbosity: print("[EC-Lab] Lib:", self.api.GetLibVersion())
        self.id_, _ = self.api.Connect(self.address)
        self.board_type = self.api.GetChannelBoardType(self.id_, self.channel)
        if self.board_type == KBIO.BOARD_TYPE.ESSENTIAL.value:
            fw, fpga, suf = "kernel.bin", "Vmp_ii_0437_a6.xlx", ""
        elif self.board_type == KBIO.BOARD_TYPE.PREMIUM.value:
            fw, fpga, suf = "kernel4.bin","vmp_iv_0395_aa.xlx","4"
        elif self.board_type == KBIO.BOARD_TYPE.DIGICORE.value:
            fw, fpga, suf = "kernel.bin","", "5"
        else:
            raise RuntimeError("Unsupported board")
        self.api.LoadFirmware(self.id_, self.api.channel_map({self.channel}), firmware=fw, fpga=fpga, force=False)
        info = self.api.GetChannelInfo(self.id_, self.channel)
        if not info.is_kernel_loaded: raise RuntimeError("Kernel not loaded")
        self.ca_ecc  = f"ca{suf}.ecc"
        self.ocv_ecc = f"ocv{suf}.ecc"

    # --- PREMIUM only (VMP300 family) ---
    def get_hardware_conf(self):
        if self.board_type != KBIO.BOARD_TYPE.PREMIUM.value: return None
        hw = self.api.GetHardConf(self.id_, self.channel)
        return {"connection": hw.connection, "mode": hw.mode}
    def set_hardware_conf(self, connection: KBIO.HW_CNX, mode: KBIO.HW_MODE):
        if self.board_type != KBIO.BOARD_TYPE.PREMIUM.value: return False
        self.api.SetHardConf(self.id_, self.channel, connection.value, mode.value); return True

    # --- Techniques ---
    def load_ca(self):
        """CA: steps = [0 V (1 s), -10 mV (1 s)], repeat 10, record dT=0.1s, dI=0.1A"""
        CA = {
            "Voltage_step":     ECC_parm("Voltage_step", float),
            "Duration_step":    ECC_parm("Duration_step", float),
            "vs_initial":       ECC_parm("vs_initial", bool),
            "Step_number":      ECC_parm("Step_number", int),
            "Record_every_dT":  ECC_parm("Record_every_dT", float),
            "Record_every_dI":  ECC_parm("Record_every_dI", float),
            "N_Cycles":         ECC_parm("N_Cycles", int),
        }
        p = []
        # step 0: 0 V, 1 s
        p += [ make_ecc_parm(self.api, CA["Voltage_step"], 0.0, 0),
               make_ecc_parm(self.api, CA["Duration_step"], 1.0, 0),
               make_ecc_parm(self.api, CA["vs_initial"],  False, 0) ]
        # step 1: -10 mV, 1 s
        p += [ make_ecc_parm(self.api, CA["Voltage_step"], -0.010, 1),
               make_ecc_parm(self.api, CA["Duration_step"], 1.0, 1),
               make_ecc_parm(self.api, CA["vs_initial"],  False, 1) ]

        ecc_parms = make_ecc_parms(self.api,
            *p,
            make_ecc_parm(self.api, CA["Step_number"], 1),
            make_ecc_parm(self.api, CA["Record_every_dT"], 0.1),
            make_ecc_parm(self.api, CA["Record_every_dI"], 0.1),
            make_ecc_parm(self.api, CA["N_Cycles"], 10),
        )
        self.api.LoadTechnique(self.id_, self.channel, self.ca_ecc, ecc_parms, first=True, last=True, display=False)

    def load_ocv(self):
        OCV = { "Record_every_dT": ECC_parm("Record_every_dT", float) }
        ecc_parms = make_ecc_parms(self.api, make_ecc_parm(self.api, OCV["Record_every_dT"], 0.1))
        self.api.LoadTechnique(self.id_, self.channel, self.ocv_ecc, ecc_parms, first=True, last=True, display=False)

    def start(self): self.api.StartChannel(self.id_, self.channel)

    # --- Raw snapshot reader (no get_experiment_data) ---
    def record_snapshot(self):
        """
        Pull one GetData() buffer and decode rows directly.
        Returns a list of parsed rows: [{"t","Ewe","Iwe","cycle","Ece"}...], and status.
        Works for CA and OCV by inspecting NbCols.
        """
        data = self.api.GetData(self.id_, self.channel)
        current_values, data_info, data_record = data
        parsed_rows = []
        idx = 0
        # Technique name mainly for info; decoding uses NbCols
        status, tech_name = get_info_data(self.api, data)

        # Each row layout is: [t_high, t_low, ...payload...]
        # payload size = NbCols - 2
        payload_words = max(0, data_info.NbCols - 2)

        for _ in range(data_info.NbRows):
            inx = idx + data_info.NbCols
            t_high, t_low, *row = data_record[idx:inx]
            # timestamp
            t_rel = (t_high << 32) + t_low
            t = current_values.TimeBase * t_rel

            out = {"t": t, "Ewe": None, "Iwe": None, "cycle": None, "Ece": None}

            # Decode by payload length
            if payload_words == 1:  # OCV (Ewe)
                Ewe = self.api.ConvertNumericIntoSingle(row[0])
                out.update({"Ewe": Ewe})
            elif payload_words == 2:  # OCV on VMP3 (Ewe, Ece)
                Ewe = self.api.ConvertNumericIntoSingle(row[0])
                Ece = self.api.ConvertNumericIntoSingle(row[1])
                out.update({"Ewe": Ewe, "Ece": Ece})
            elif payload_words == 3:  # CA/CP (Ewe, I, cycle)
                Ewe = self.api.ConvertNumericIntoSingle(row[0])
                Iwe = self.api.ConvertNumericIntoSingle(row[1])
                cycle = row[2]
                out.update({"Ewe": Ewe, "Iwe": Iwe, "cycle": cycle})
            else:
                # Unknown payload size; skip gracefully
                pass

            # store to ivCurve (names aligned with your controller)
            self.ivCurve["t"].append(out["t"])
            self.ivCurve["V_we"].append(out["Ewe"])
            self.ivCurve["I"].append(out["Iwe"] if out["Iwe"] is not None else 0.0)
            self.ivCurve["cycle"].append(out["cycle"])

            parsed_rows.append(out)
            idx = inx

        # Derive program state name
        prog_state = KBIO.PROG_STATE(current_values.State).name if hasattr(KBIO, "PROG_STATE") else "RUN"
        return parsed_rows, prog_state


    def stream(self, poll_s=0.1):
        """Yield dict rows parsed directly from GetData() using record_snapshot()."""
        while True:
            rows, status = self.record_snapshot()
            for r in rows:
                yield r, status
            if status == "STOP":
                break
            if not rows:
                time.sleep(poll_s)

    def disconnect(self):
        try:
            if self.id_ is not None: self.api.Disconnect(self.id_)
        except Exception: pass


# -------------------------------
# Simple live plots
# -------------------------------

class LivePlotter:
    def __init__(self, show_ece=True, max_points=5000):
        self.t = deque(maxlen=max_points); self.ewe = deque(maxlen=max_points)
        self.iwe = deque(maxlen=max_points); self.ece = deque(maxlen=max_points) if show_ece else None
        self.fig = plt.figure(figsize=(9,6))
        self.ax_v = self.fig.add_subplot(2,1,1)
        self.ax_i = self.fig.add_subplot(2,1,2)
        self.lewe, = self.ax_v.plot([], [], lw=1.2, label="Ewe")
        self.lece = None
        if self.ece is not None:
            self.lece, = self.ax_v.plot([], [], lw=1.0, alpha=0.75, label="Ece")
        self.liwe, = self.ax_i.plot([], [], lw=1.2, label="Iwe")
        self.ax_v.set_ylabel("Voltage (V)"); self.ax_i.set_ylabel("Current (A)")
        self.ax_v.set_xlabel("t (s)"); self.ax_i.set_xlabel("t (s)")
        self.ax_v.legend(); self.ax_i.legend()
        self.ax_v.grid(True, alpha=0.3); self.ax_i.grid(True, alpha=0.3)
        plt.tight_layout()

    def update(self, t, Ewe, Iwe, Ece=None):
        if t is None: return
        self.t.append(t)
        self.ewe.append(Ewe if Ewe is not None else float('nan'))
        self.iwe.append(Iwe if Iwe is not None else float('nan'))
        if self.ece is not None and Ece is not None: self.ece.append(Ece)
        self.lewe.set_data(self.t, self.ewe); self.liwe.set_data(self.t, self.iwe)
        if self.lece is not None and len(self.ece) > 1: self.lece.set_data(self.t, self.ece)
        if len(self.t) > 1:
            t0, t1 = self.t[0], self.t[-1]
            self.ax_v.set_xlim(t0, t1); self.ax_i.set_xlim(t0, t1)
        if len(self.ewe) > 1:
            ymin, ymax = min(self.ewe), max(self.ewe)
            if ymin == ymax: ymin, ymax = ymin-0.01, ymax+0.01
            self.ax_v.set_ylim(ymin, ymax)
        if len(self.iwe) > 1:
            ymin, ymax = min(self.iwe), max(self.iwe)
            if ymin == ymax: ymin, ymax = ymin-1e-6, ymax+1e-6
            self.ax_i.set_ylim(ymin, ymax)
        plt.pause(0.001)


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="CA / OCV live viewer (concise)")
    ap.add_argument("--address", default="USB0")
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--mode", choices=["ca","ocv"], default="ca")
    args = ap.parse_args()

    pot = Potentiostat(address=args.address, channel=args.channel, verbosity=1)
    pot.connect()

    # PREMIUM: show and example-set hardware (comment these lines if not desired)
    hw = pot.get_hardware_conf()
    if hw:
        print("[HW] current:", hw)
        pot.set_hardware_conf(KBIO.HW_CNX.WE_TO_GND, KBIO.HW_MODE.FLOATING)
        print("[HW] set ->", pot.get_hardware_conf())
        pot.set_hardware_conf(KBIO.HW_CNX.STANDARD, KBIO.HW_MODE.GROUNDED)
        print("[HW] revert ->", pot.get_hardware_conf())

    if args.mode == "ocv":
        pot.load_ocv(); plot = LivePlotter(show_ece=False)
    else:
        pot.load_ca();  plot = LivePlotter(show_ece=True)

    pot.start()
    for row, status in pot.stream(poll_s=0.1):
        plot.update(row["t"], row["Ewe"], row["Iwe"], row.get("Ece"))
        if status == "STOP": break
    pot.disconnect()


if __name__ == "__main__":
    main()
