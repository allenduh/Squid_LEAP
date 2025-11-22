# potentiostat_panel.py
# Controller + Widget with live plots (PyQtGraph)
# Features:
#  • Continuous OCV display (and optionally plot)
#  • dT (Record_every_dT) & dI (Record_every_dI) controls
#  • Asymmetric t+ / t- durations for CA
#  • Option to offset V+ / V- by the most recent OCV (relative-to-OCV)
#  • Show HW connection/mode (PREMIUM boards)

import os
os.environ.setdefault("QT_API", "pyqt5")

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
from qtpy.QtCore import Signal, QTimer
from qtpy.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFileDialog, QCheckBox
)

# Fast plotting
try:
    import pyqtgraph as pg
except ImportError:
    pg = None

import drivers.ECLab.kbio_types as KBIO
from drivers.ECLab.kbio_api import KBIO_api


# ----------------------------
# Controller (modeled after your original)
# ----------------------------

@dataclass
class voltage_step:
    voltage: float
    duration: float
    vs_init: bool = False

class potentialstat_controller:
    def __init__(self, address="USB0", dll_path=None, channel=1):
        self.address = address
        self.channel = int(channel)
        if dll_path is None:
            # Adjust if your DLL path differs
            dll_path = "drivers\\ECLab\\lib\\EClib64.dll"
        self.api = KBIO_api(dll_path)
        self.id_, self.device_info = self.api.Connect(self.address)

        # Defaults (you can override from the widget)
        self.steps: List[voltage_step] = [voltage_step(0.0, 1.0), voltage_step(-0.010, 1.0)]
        self.record_dt = 0.1
        self.record_dI = 0.1
        self.repeat_count = 10

        self.p_steps: List[KBIO.EccParam] = []
        self.final_parameters = None

        # buffers
        self.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    # --- HW helpers ---
    def get_hardware_conf(self):
        try:
            return self.api.GetHardConf(self.id_, self.channel)  # PREMIUM only
        except Exception:
            return None

    # --- ECC build like your original ---
    def make_single_parameter(self, name, idx, value):
        parm = KBIO.EccParam()
        self.api.DefineParameter(name, value, idx, parm)
        return parm

    def append_CA_parameter_list(self):
        self.p_steps.clear()
        for idx, step in enumerate(self.steps):
            self.p_steps.append(self.make_single_parameter('Voltage_step', idx, step.voltage))
            self.p_steps.append(self.make_single_parameter('Duration_step', idx, step.duration))
            self.p_steps.append(self.make_single_parameter('vs_initial',   idx, step.vs_init))
        self.p_steps.append(self.make_single_parameter('Step_number',    0, len(self.steps) - 1))
        self.p_steps.append(self.make_single_parameter('Record_every_dT',0, self.record_dt))
        self.p_steps.append(self.make_single_parameter('Record_every_dI',0, self.record_dI))
        self.p_steps.append(self.make_single_parameter('N_Cycles',       0, self.repeat_count))

    def make_final_parameters(self):
        nb = len(self.p_steps)
        arr = KBIO.ECC_PARM_ARRAY(nb)
        for i, p in enumerate(self.p_steps): arr[i] = p
        self.final_parameters = KBIO.EccParams(nb, arr)

    def load_technique_CA(self):
        tech_file = "drivers\\ECLab\\lib\\ca4.ecc"  # adjust per board family if needed
        self.api.LoadTechnique(self.id_, self.channel, tech_file, self.final_parameters,
                               first=True, last=True, display=False)

    def load_technique_OCV(self, record_dt=0.1):
        tech_file = "drivers\\ECLab\\lib\\ocv4.ecc"
        parm = KBIO.EccParam()
        self.api.DefineParameter('Record_every_dT', float(record_dt), 0, parm)
        arr = KBIO.ECC_PARM_ARRAY(1); arr[0] = parm
        parms = KBIO.EccParams(1, arr)
        self.api.LoadTechnique(self.id_, self.channel, tech_file, parms,
                               first=True, last=True, display=False)

    def start(self):
        self.api.StartChannel(self.id_, self.channel)
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    def stop(self):
        try:
            if hasattr(self.api, "StopChannel"):
                self.api.StopChannel(self.id_, self.channel)
        except Exception:
            pass

    # --- Raw buffer decode (works for OCV/CA) ---
    def record_snapshot(self) -> Tuple[List[dict], str]:
        """
        Returns (rows, status_name), rows: [{"t","Ewe","Iwe","cycle","Ece"}...]
        """
        data = self.api.GetData(self.id_, self.channel)
        current_values, data_info, data_record = data
        rows = []; idx = 0
        payload_words = max(0, data_info.NbCols - 2)
        status = KBIO.PROG_STATE(current_values.State).name if hasattr(KBIO, "PROG_STATE") else "RUN"

        for _ in range(data_info.NbRows):
            inx = idx + data_info.NbCols
            t_high, t_low, *payload = data_record[idx:inx]
            t_rel = (t_high << 32) + t_low
            t = current_values.TimeBase * t_rel
            out = {"t": t, "Ewe": None, "Iwe": None, "cycle": None, "Ece": None}

            if payload_words == 1:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])         # OCV
            elif payload_words == 2:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])         # OCV (VMP3)
                out["Ece"] = self.api.ConvertNumericIntoSingle(payload[1])
            elif payload_words == 3:
                out["Ewe"]  = self.api.ConvertNumericIntoSingle(payload[0])        # CA/CP
                out["Iwe"]  = self.api.ConvertNumericIntoSingle(payload[1])
                out["cycle"]= payload[2]
                # accumulate IV buffers for CA
                self.ivCurve["t"].append(out["t"])
                self.ivCurve["V_we"].append(out["Ewe"])
                self.ivCurve["I"].append(out["Iwe"] if out["Iwe"] is not None else 0.0)
                self.ivCurve["cycle"].append(out["cycle"])

            rows.append(out); idx = inx
        return rows, status


# ----------------------------
# Widget (plots + controls)
# ----------------------------

class potentialstatControlWidget(QFrame):
    # keep your legacy signal that emits the full curve when stopping
    iv_dict_from_gate_once = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.ctrl = potentialstat_controller()
        self._last_ocv: Optional[float] = None
        self._mode = "idle"  # "ocv" | "ca" | "idle"

        self._build_ui()
        self._connect()

        self.timer = QTimer(self); self.timer.setInterval(100)   # 100 ms polling
        self.timer.timeout.connect(self._on_poll)

        self._t: List[float] = []; self._v: List[float] = []; self._i: List[float] = []

        # show hardware conf once
        self._refresh_hw_label()

        self._start_ocv_background()

    def _build_ui(self):
        # --- Controls ---
        self.v_plus  = QDoubleSpinBox();  self.v_plus.setRange(-500, 500);  self.v_plus.setValue(0.0);   self.v_plus.setSingleStep(1)
        self.v_minus = QDoubleSpinBox();  self.v_minus.setRange(-500, 500); self.v_minus.setValue(-10.0);self.v_minus.setSingleStep(1)
        self.cycles  = QDoubleSpinBox();  self.cycles.setRange(0, 1000);    self.cycles.setValue(5);    self.cycles.setSingleStep(1)
        self.t_plus  = QDoubleSpinBox();  self.t_plus.setRange(0.001, 1000.0); self.t_plus.setDecimals(3); self.t_plus.setSingleStep(0.001); self.t_plus.setValue(1.0)
        self.t_minus = QDoubleSpinBox();  self.t_minus.setRange(0.001, 1000.0); self.t_minus.setDecimals(3); self.t_minus.setSingleStep(0.001); self.t_minus.setValue(1.0)
        self.dt_min  = QDoubleSpinBox();  self.dt_min.setRange(0.0, 10.0);  self.dt_min.setDecimals(3);  self.dt_min.setValue(0.001); self.dt_min.setSingleStep(0.01)
        self.dI_min  = QDoubleSpinBox();  self.dI_min.setRange(0.0, 10.0);  self.dI_min.setDecimals(6);  self.dI_min.setValue(0.001); self.dI_min.setSingleStep(0.01)
        self.chk_rel_ocv = QCheckBox("V relative to OCV")

        row0 = QHBoxLayout()
        row0.addWidget(QLabel('Protocol'))
        row0.addWidget(QLabel('V+ (mV)')); row0.addWidget(self.v_plus)
        row0.addWidget(QLabel('V- (mV)')); row0.addWidget(self.v_minus)
        row0.addWidget(QLabel('t+ (s)'));  row0.addWidget(self.t_plus)
        row0.addWidget(QLabel('t- (s)'));  row0.addWidget(self.t_minus)
        row0.addWidget(QLabel('cycles'));  row0.addWidget(self.cycles)
        row0.addWidget(QLabel('dT (s)'));  row0.addWidget(self.dt_min)
        row0.addWidget(QLabel('dI (A)'));  row0.addWidget(self.dI_min)
        row0.addWidget(self.chk_rel_ocv)

        self.btn_start = QPushButton("Start CA")
        self.btn_stop  = QPushButton("Stop")
        self.btn_ocv   = QPushButton("Measure OCV")
        self.btn_clear = QPushButton("Clear")
        self.btn_save  = QPushButton("Save IV (.npz)")
        self.lbl_ocv   = QLabel(); self.lbl_ocv.setNum(0.0); self.lbl_ocv.setFrameStyle(QFrame.Panel|QFrame.Sunken)
        self.lbl_hw    = QLabel("HW: n/a")

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_start); row1.addWidget(self.btn_stop); row1.addWidget(self.btn_ocv)
        row1.addWidget(QLabel("OCV (V):")); row1.addWidget(self.lbl_ocv)
        row1.addWidget(self.lbl_hw)
        row1.addWidget(self.btn_clear); row1.addWidget(self.btn_save)

        # --- Plots ---
        grid = QGridLayout(); grid.addLayout(row0, 0, 0); grid.addLayout(row1, 1, 0)

        if pg is None:
            grid.addWidget(QLabel("Install 'pyqtgraph' to enable live plots."), 2, 0)
            self.curve_v = self.curve_i = None
        else:
            self.plot_v = pg.PlotWidget()
            self.plot_i = pg.PlotWidget()
            self.plot_v.setLabel('left','V_we (V)'); self.plot_v.setLabel('bottom','t (s)')
            self.plot_i.setLabel('left','I (A)');    self.plot_i.setLabel('bottom','t (s)')
            self.plot_v.showGrid(x=True,y=True,alpha=0.3); self.plot_i.showGrid(x=True,y=True,alpha=0.3)
            self.curve_v = self.plot_v.plot([], [], pen=pg.mkPen(width=2))
            self.curve_i = self.plot_i.plot([], [], pen=pg.mkPen(width=2))
            self.plot_v.setDownsampling(mode='peak'); self.plot_i.setDownsampling(mode='peak')
            self.plot_v.setClipToView(True);         self.plot_i.setClipToView(True)
            grid.addWidget(self.plot_v, 2, 0)
            grid.addWidget(self.plot_i, 3, 0)

        self.setLayout(grid)

    def _connect(self):
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_ocv.clicked.connect(self._on_ocv)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_save.clicked.connect(self._on_save_npz)

    def _start_ocv_background(self):
        """Start/keep OCV running in the background (live label + optional plot)."""
        self.ctrl.load_technique_OCV(record_dt=float(self.dt_min.value()))
        self.ctrl.start()
        self._mode = "ocv"
        self._clear_buffers()              # keep the V trace clean if you don't want OCV drawn
        if not self.timer.isActive():
            self.timer.start()
        
    # --- HW status label ---
    def _refresh_hw_label(self):
        hw = self.ctrl.get_hardware_conf()
        if not hw:
            self.lbl_hw.setText("HW: n/a")
            return
        try:
            cnx = KBIO.HW_CNX(hw.connection).name
        except Exception:
            cnx = str(hw.connection)
        try:
            mode = KBIO.HW_MODE(hw.mode).name
        except Exception:
            mode = str(hw.mode)
        self.lbl_hw.setText(f"HW: {cnx}/{mode}")


    # --- Actions ---
    def _on_ocv(self):
        # Continuous OCV (live label, can also plot on V_we curve)
        self._start_ocv_background()

    def _on_start(self):
        # pause background OCV so it doesn't fight CA
        try:
            self.ctrl.stop()
        except Exception:
            pass
        self._mode = "idle"
        
        # Prepare CA protocol
        v_plus  = float(self.v_plus.value())  / 1000.0  # mV→V
        v_minus = float(self.v_minus.value()) / 1000.0
        t_plus  = float(self.t_plus.value())
        t_minus = float(self.t_minus.value())
        cycles  = int(self.cycles.value()) 

        if self.chk_rel_ocv.isChecked() and self._last_ocv is not None:
            v_plus  += self._last_ocv
            v_minus += self._last_ocv

        self.ctrl.steps = [voltage_step(v_plus, t_plus), voltage_step(v_minus, t_minus)]
        self.ctrl.repeat_count = cycles -1
        self.ctrl.record_dt = float(self.dt_min.value())
        self.ctrl.record_dI = float(self.dI_min.value())

        self.ctrl.append_CA_parameter_list()
        self.ctrl.make_final_parameters()
        self.ctrl.load_technique_CA()

        self._clear_buffers()
        self.ctrl.start()
        self._mode = "ca"
        self.timer.start()

    def _on_stop(self):
        # dump buffers for downstream
        to_emit = {"t": self._t[:], "V_we": self._v[:], "I": self._i[:]}
        print(to_emit)
        self.iv_dict_from_gate_once.emit(to_emit)

        self.timer.stop()
        self.ctrl.stop()

        # resume background OCV
        # self._start_ocv_background()

    def _on_clear(self):
        self._clear_buffers()
        if pg and self.curve_v and self.curve_i:
            self.curve_v.setData([], [])
            self.curve_i.setData([], [])

    def _on_save_npz(self):
        if not self._t:
            return
        default = f"ivcurve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Save IV (.npz)", default, "NumPy Zip (*.npz)")
        if not path:
            return
        meta = dict(
            address=self.ctrl.address, channel=self.ctrl.channel,
            steps=[(s.voltage, s.duration, s.vs_init) for s in self.ctrl.steps],
            record_dt=self.ctrl.record_dt, record_dI=self.ctrl.record_dI,
            repeat_count=self.ctrl.repeat_count, started=self.ctrl.timeStart,
        )
        np.savez(
            path,
            t=np.asarray(self._t, dtype=np.float64),
            V_we=np.asarray(self._v, dtype=np.float64),
            I=np.asarray(self._i, dtype=np.float64),
            cycle=np.asarray(self.ctrl.ivCurve["cycle"], dtype=np.int64) if self.ctrl.ivCurve["cycle"] else np.array([]),
            meta=str(meta),
        )

    def _clear_buffers(self):
        self._t.clear(); self._v.clear(); self._i.clear()

    # --- Poller: draws plots + updates OCV label ---
    def _on_poll(self):
        rows, status = self.ctrl.record_snapshot()

        # Update OCV label (and optionally plot) in OCV mode
        if rows and self._mode == "ocv":
            last_ewe = next((r["Ewe"] for r in reversed(rows) if r.get("Ewe") is not None), None)
            if last_ewe is not None:
                self._last_ocv = float(last_ewe)
                self.lbl_ocv.setNum(self._last_ocv)
                # If you also want to see OCV on the voltage plot live:
                if pg and self.curve_v:
                    t_vals = [float(r["t"]) for r in rows if r.get("t") is not None]
                    v_vals = [float(r["Ewe"]) for r in rows if r.get("Ewe") is not None]
                    self._t.extend(t_vals); self._v.extend(v_vals)
                    self.curve_v.setData(self._t, self._v)

        # Accumulate CA data & plot
        if self._mode == "ca":
            new_any = False
            for r in rows:
                t = r.get("t"); Ewe = r.get("Ewe"); Iwe = r.get("Iwe")
                if t is None: continue
                self._t.append(float(t))
                self._v.append(float(Ewe) if Ewe is not None else 0.0)
                self._i.append(float(Iwe) if Iwe is not None else 0.0)
                new_any = True
            if pg and new_any and self.curve_v and self.curve_i:
                self.curve_v.setData(self._t, self._v)
                self.curve_i.setData(self._t, self._i)

        if status == "STOP":
            self._on_stop()
