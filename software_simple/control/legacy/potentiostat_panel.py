# potentiostat_panel.py
# Combined controller + widget (minimal).
# Updates:
#  - Live plotting inside the widget (PyQtGraph) for V_we (Ewe) and I
#  - Save IV curve to NPZ on Stop (t, V_we, I, cycle + protocol/meta)

import os
os.environ.setdefault("QT_API", "pyqt5")

from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from qtpy.QtCore import Signal, QTimer
from qtpy.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFileDialog
)

# For plotting (fast & simple in Qt)
try:
    import pyqtgraph as pg
except ImportError:
    pg = None  # widget will show a label instructing to install pyqtgraph

import kbio.kbio_types as KBIO
from kbio.kbio_api import KBIO_api


# ----------------------------
# Controller (modeled on original)
# ----------------------------

@dataclass
class voltage_step:
    voltage: float
    duration: float
    vs_init: bool = False

class potentialstat_controller:
    """
    Same spirit as your original:
      - explicit CA parameter build via DefineParameter → EccParams
      - direct GetData() parsing (record_snapshot)
      - helpers to load CA/OCV and start/stop
    """

    def __init__(self, address="USB0", dll_path=None, channel=1):
        self.address = address
        self.channel = int(channel)
        if dll_path is None:
            # Default DLL path (adjust if needed)
            dll_path = "kbio\\lib\\EClib64.dll"
        self.api = KBIO_api(dll_path)
        self.id_, self.device_info = self.api.Connect(self.address)

        # Default CA: 0 V ↔ -10 mV, 1 s each, 10 cycles
        self.steps: List[voltage_step] = [voltage_step(0.0, 1.0), voltage_step(-0.010, 1.0)]
        self.record_dt = 0.1
        self.record_dI = 0.1
        self.repeat_count = 10

        self.p_steps: List[KBIO.EccParam] = []
        self.final_parameters = None

        # buffers
        self.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    # ---- Firmware / info ----
    def get_channel_info(self):
        return self.api.GetChannelInfo(self.id_, self.channel)

    def firmware_is_running(self):
        return self.get_channel_info().is_kernel_loaded

    # ---- CA parameter build (same pattern as your original) ----
    def make_single_parameter(self, name, idx, value):
        parm = KBIO.EccParam()
        self.api.DefineParameter(name, value, idx, parm)
        return parm

    def append_CA_parameter_list(self):
        self.p_steps.clear()
        for idx, step in enumerate(self.steps):
            self.p_steps.append(self.make_single_parameter('Voltage_step', idx, step.voltage))
            self.p_steps.append(self.make_single_parameter('Duration_step', idx, step.duration))
            self.p_steps.append(self.make_single_parameter('vs_initial', idx, step.vs_init))
        self.p_steps.append(self.make_single_parameter('Step_number', 0, len(self.steps) - 1))
        self.p_steps.append(self.make_single_parameter('Record_every_dT', 0, self.record_dt))
        self.p_steps.append(self.make_single_parameter('Record_every_dI', 0, self.record_dI))
        self.p_steps.append(self.make_single_parameter('N_Cycles', 0, self.repeat_count))

    def make_final_parameters(self):
        nb = len(self.p_steps)
        arr = KBIO.ECC_PARM_ARRAY(nb)
        for i, p in enumerate(self.p_steps):
            arr[i] = p
        self.final_parameters = KBIO.EccParams(nb, arr)

    # ---- Load techniques ----
    def load_technique_CA(self):
        # Note: for VMP300 family use ca4.ecc. Adjust per board if needed.
        tech_file = "kbio\\lib\\ca4.ecc"
        self.api.LoadTechnique(self.id_, self.channel, tech_file, self.final_parameters,
                               first=True, last=True, display=False)

    def load_technique_OCV(self, record_dt=0.1):
        tech_file = "kbio\\lib\\ocv4.ecc"  # adjust suffix per board if needed
        parm = KBIO.EccParam()
        self.api.DefineParameter('Record_every_dT', float(record_dt), 0, parm)
        arr = KBIO.ECC_PARM_ARRAY(1); arr[0] = parm
        parms = KBIO.EccParams(1, arr)
        self.api.LoadTechnique(self.id_, self.channel, tech_file, parms,
                               first=True, last=True, display=False)

    # ---- Start/Stop ----
    def start(self):
        self.api.StartChannel(self.id_, self.channel)
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    def stop(self):
        try:
            if hasattr(self.api, "StopChannel"):
                self.api.StopChannel(self.id_, self.channel)
        except Exception:
            pass

    # ---- Raw data read (single snapshot) ----
    def record_snapshot(self) -> Tuple[List[dict], str]:
        """
        Decode one GetData() buffer, appending to ivCurve.
        Returns (rows, status_name) where rows are [{t,Ewe,Iwe,cycle,Ece}?].
        """
        data = self.api.GetData(self.id_, self.channel)
        current_values, data_info, data_record = data
        rows = []
        idx = 0

        payload_words = max(0, data_info.NbCols - 2)
        status = KBIO.PROG_STATE(current_values.State).name if hasattr(KBIO, "PROG_STATE") else "RUN"

        for _ in range(data_info.NbRows):
            inx = idx + data_info.NbCols
            t_high, t_low, *payload = data_record[idx:inx]
            t_rel = (t_high << 32) + t_low
            t = current_values.TimeBase * t_rel

            out = {"t": t, "Ewe": None, "Iwe": None, "cycle": None, "Ece": None}

            if payload_words == 1:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])            # OCV
            elif payload_words == 2:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])            # OCV (VMP3)
                out["Ece"] = self.api.ConvertNumericIntoSingle(payload[1])
            elif payload_words == 3:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])            # CA/CP
                out["Iwe"] = self.api.ConvertNumericIntoSingle(payload[1])
                out["cycle"] = payload[2]
                # store in ivCurve
                self.ivCurve["t"].append(out["t"])
                self.ivCurve["V_we"].append(out["Ewe"])
                self.ivCurve["I"].append(out["Iwe"] if out["Iwe"] is not None else 0.0)
                self.ivCurve["cycle"].append(out["cycle"])

            rows.append(out)
            idx = inx

        return rows, status


# ----------------------------
# Widget (with plotting + save)
# ----------------------------

class potentialstatControlWidget(QFrame):
    preview = Signal(dict)
    iv_dict_from_gate_once = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.ctrl = potentialstat_controller()
        self._build_ui()
        self._connect()

        self.timer = QTimer(self); self.timer.setInterval(100)
        self.timer.timeout.connect(self._on_poll)

        # buffers (used for plots and save)
        self._t: List[float] = []; self._v: List[float] = []; self._i: List[float] = []

    def _build_ui(self):
        # ----- Controls row
        self.v_plus = QDoubleSpinBox();  self.v_plus.setRange(-500, 500); self.v_plus.setValue(0);   self.v_plus.setSingleStep(1)
        self.v_minus = QDoubleSpinBox(); self.v_minus.setRange(-500, 500); self.v_minus.setValue(-10); self.v_minus.setSingleStep(1)
        self.cycles = QDoubleSpinBox();  self.cycles.setRange(0, 100);     self.cycles.setValue(10);  self.cycles.setSingleStep(1)
        self.duration = QDoubleSpinBox();self.duration.setRange(0, 100);    self.duration.setValue(1); self.duration.setSingleStep(0.1)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel('Protocol'))
        row0.addWidget(QLabel('V+ (mV)')); row0.addWidget(self.v_plus)
        row0.addWidget(QLabel('V- (mV)')); row0.addWidget(self.v_minus)
        row0.addWidget(QLabel('cycles'));  row0.addWidget(self.cycles)
        row0.addWidget(QLabel('dur (s)')); row0.addWidget(self.duration)

        self.btn_ocv = QPushButton("Measure OCV")
        self.btn_start = QPushButton("Start CA")
        self.btn_stop = QPushButton("Stop")
        self.btn_clear = QPushButton("Clear")
        self.lbl_ocv = QLabel(); self.lbl_ocv.setNum(0.0); self.lbl_ocv.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.btn_save = QPushButton("Save IV (.npz)")

        row1 = QHBoxLayout()
        row1.addWidget(self.btn_start); row1.addWidget(self.btn_stop); row1.addWidget(self.btn_ocv)
        row1.addWidget(QLabel('OCV (V):')); row1.addWidget(self.lbl_ocv)
        row1.addWidget(self.btn_clear); row1.addWidget(self.btn_save)

        # ----- Plots (V and I stacked)
        grid = QGridLayout(); grid.addLayout(row0, 0, 0); grid.addLayout(row1, 1, 0)

        if pg is None:
            self._no_pg_label = QLabel("Install 'pyqtgraph' to enable live plots.")
            grid.addWidget(self._no_pg_label, 2, 0)
        else:
            self.plot_v = pg.PlotWidget()
            self.plot_i = pg.PlotWidget()
            self.plot_v.setLabel('left', 'V_we (V)'); self.plot_v.setLabel('bottom', 't (s)')
            self.plot_i.setLabel('left', 'I (A)');    self.plot_i.setLabel('bottom', 't (s)')
            self.plot_v.showGrid(x=True,y=True,alpha=0.3); self.plot_i.showGrid(x=True,y=True,alpha=0.3)
            self.curve_v = self.plot_v.plot([], [], pen=pg.mkPen(width=2))
            self.curve_i = self.plot_i.plot([], [], pen=pg.mkPen(width=2))
            # improve performance for live
            self.plot_v.setDownsampling(mode='peak'); self.plot_i.setDownsampling(mode='peak')
            self.plot_v.setClipToView(True);          self.plot_i.setClipToView(True)
            grid.addWidget(self.plot_v, 2, 0)
            grid.addWidget(self.plot_i, 3, 0)

        self.setLayout(grid)

    def _connect(self):
        self.btn_ocv.clicked.connect(self._on_ocv)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_save.clicked.connect(self._on_save_npz)

    # ---- actions ----
    def _on_ocv(self):
        self.ctrl.load_technique_OCV(record_dt=0.1)
        self.ctrl.start()
        # Pre-read to populate label
        rows, _ = self.ctrl.record_snapshot()
        if rows:
            self.lbl_ocv.setNum(float(rows[-1].get("Ewe") or 0.0))
        # Keep timer off unless you want continuous OCV plotting
        # If you want continuous OCV plot, uncomment:
        # self._clear_buffers(); self.timer.start()

    def _on_start(self):
        # Convert UI → controller CA recipe
        v_plus  = float(self.v_plus.value())/1000.0
        v_minus = float(self.v_minus.value())/1000.0
        cycles  = int(self.cycles.value()); dur = float(self.duration.value())

        self.ctrl.steps = [voltage_step(v_plus, dur), voltage_step(v_minus, dur)]
        self.ctrl.repeat_count = cycles
        self.ctrl.record_dt = 0.1; self.ctrl.record_dI = 0.1

        self.ctrl.append_CA_parameter_list()
        self.ctrl.make_final_parameters()
        self.ctrl.load_technique_CA()
        self._clear_buffers()
        self.ctrl.start()
        self.timer.start()

    def _on_stop(self):
        self.timer.stop()
        self.ctrl.stop()
        # emit single dict for downstream consumers
        self.iv_dict_from_gate_once.emit({"t": self._t[:], "V_we": self._v[:], "I": self._i[:]})

    def _on_clear(self):
        self._clear_buffers()
        if pg:
            self.curve_v.setData([], [])
            self.curve_i.setData([], [])

    def _on_save_npz(self):
        # Save current buffers + simple protocol/meta
        if not self._t:
            return
        default_name = f"ivcurve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Save IV (.npz)", default_name, "NumPy Zip (*.npz)")
        if not path:
            return
        meta = dict(
            address=self.ctrl.address,
            channel=self.ctrl.channel,
            steps=[(s.voltage, s.duration, s.vs_init) for s in self.ctrl.steps],
            record_dt=self.ctrl.record_dt,
            record_dI=self.ctrl.record_dI,
            repeat_count=self.ctrl.repeat_count,
            started=self.ctrl.timeStart,
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

    # ---- polling ----
    def _on_poll(self):
        rows, status = self.ctrl.record_snapshot()
        new_t = []; new_v = []; new_i = []
        for r in rows:
            t = r.get("t"); Ewe = r.get("Ewe"); Iwe = r.get("Iwe")
            if t is None: 
                continue
            self._t.append(float(t)); new_t.append(float(t))
            self._v.append(float(Ewe) if Ewe is not None else 0.0); new_v.append(self._v[-1])
            self._i.append(float(Iwe) if Iwe is not None else 0.0); new_i.append(self._i[-1])

        # Update plots
        if pg and new_t:
            # For simplicity, set full buffers (still fast enough). If needed, switch to setPos + setData of small windows.
            self.curve_v.setData(self._t, self._v)
            self.curve_i.setData(self._t, self._i)

        # Update OCV label if running OCV continuously
        if rows and self.btn_ocv.isEnabled() and self.ctrl.repeat_count == 0:
            self.lbl_ocv.setNum(float(self._v[-1]))

        if status == "STOP":
            self._on_stop()
