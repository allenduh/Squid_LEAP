# widgets_potentiostat.py
# Minimal, drop-in replacement for the older widgets_potentialstat, wired to a simple controller.
# Keep the same class name (potentialstatControlWidget) so the GUI uses it after a one-line import swap.

import os
os.environ["QT_API"] = "pyqt5"
from qtpy.QtCore import Signal, QTimer
from qtpy.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox

from typing import List, Tuple
from kbio import kbio_types as KBIO
from potentiostat_live_ca import Potentiostat

class PotentiostatController:
    def __init__(self, address="USB0", channel=1):
        self.address = address
        self.channel = int(channel)
        self.pot = None
        self.mode = None   # "ca" or "ocv"

    # public helpers
    def ensure_connected(self):
        if self.pot is None:
            self.pot = Potentiostat(address=self.address, channel=self.channel, verbosity=0)
            self.pot.connect()

    def start_ocv(self, record_dt=0.1):
        self.mode = "ocv"
        self.pot.load_ocv(record_dt=record_dt)
        self.pot.start()

    def prepare_ca_protocol(self, v_plus: float, v_minus: float, cycles: int, duration: float, dt: float=0.1):
        # Build CA with two steps: v_plus and v_minus, each 'duration' seconds, repeat 'cycles'
        # Reuse Potentiostat.load_ca by temporarily swapping its defaults via direct ECC creation.
        # For simplicity, we copy the same method but with passed-in values.
        CA = {
            "Voltage_step":     self.pot.api.ECC_parm("Voltage_step", float) if hasattr(self.pot.api, "ECC_parm") else None
        }
        # Instead of re-implementing ECC plumbing here, call the baked method with our values:
        # We rely on Potentiostat.load_ca reading the passed steps from our VStep-like dict.
        # To keep imports minimal, define a small tuple list with (voltage, duration).
        self._ca_params = dict(v_plus=v_plus, v_minus=v_minus, cycles=cycles, duration=duration, dt=dt)

    def start_ca(self):
        self.mode = "ca"
        # Use the Potentiostat.load_ca with a tiny local builder of CA steps
        class _V: 
            def __init__(self, v, d): self.voltage=v; self.duration=d; self.vs_init=False
        steps = [_V(self._ca_params["v_plus"], self._ca_params["duration"]),
                 _V(self._ca_params["v_minus"], self._ca_params["duration"])]
        self.pot.load_ca(steps=steps, record_dt=self._ca_params["dt"], record_dI=0.1, repeat=int(self._ca_params["cycles"]))
        self.pot.start()

    def poll(self) -> Tuple[List[dict], str]:
        """
        Non-blocking: read whatever's available, decode via record_snapshot, and return (rows, status).
        """
        if self.pot is None:
            return [], "IDLE"
        return self.pot.record_snapshot()

    def stop(self):
        # Try to stop channel gracefully if available; otherwise just disconnect.
        try:
            if hasattr(self.pot.api, "StopChannel"):
                self.pot.api.StopChannel(self.pot.id_, self.pot.channel)
        except Exception:
            pass
        # Leave connection open for re-use.


class potentialstatControlWidget(QFrame):

    preview = Signal(dict)                 # kept for API compatibility; emits protocol preview
    iv_dict_from_gate_once = Signal(dict)  # emits {"t": [...], "V_we": [...], "I": [...]} once after a CA run

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.ctrl = PotentiostatController()
        self._build_ui()
        self._make_connections()

        # periodic poller for live updates
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100 ms
        self.timer.timeout.connect(self._on_poll)

        # buffers for iv display/emit
        self._t = []
        self._v = []
        self._i = []

    def _build_ui(self):
        # Protocol inputs (mV, s, cycles) — same semantics as legacy widget
        self.v_plus = QDoubleSpinBox();  self.v_plus.setRange(-500, 500); self.v_plus.setValue(10); self.v_plus.setSingleStep(1)
        self.v_minus = QDoubleSpinBox(); self.v_minus.setRange(-500, 500); self.v_minus.setValue(0);  self.v_minus.setSingleStep(1)
        self.cycles = QDoubleSpinBox();  self.cycles.setRange(0, 100);     self.cycles.setValue(10);  self.cycles.setSingleStep(1)
        self.duration = QDoubleSpinBox();self.duration.setRange(0, 100);    self.duration.setValue(1); self.duration.setSingleStep(0.1)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel('Potentialstat protocol'))
        row0.addWidget(QLabel('V+ (mV)'));    row0.addWidget(self.v_plus)
        row0.addWidget(QLabel('V- (mV)'));    row0.addWidget(self.v_minus)
        row0.addWidget(QLabel('cycles'));     row0.addWidget(self.cycles)
        row0.addWidget(QLabel('duration (s)'));row0.addWidget(self.duration)

        # Control buttons
        self.button_measure_ocv = QPushButton("Measure OCV")
        self.button_start_ca    = QPushButton("Start CA")
        self.button_stop        = QPushButton("Stop")
        self.button_clear       = QPushButton("Clear buffers")

        # OCV label
        self.label_ocv = QLabel(); self.label_ocv.setNum(0.0); self.label_ocv.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        row1 = QHBoxLayout()
        row1.addWidget(self.button_start_ca)
        row1.addWidget(self.button_stop)
        row1.addWidget(self.button_measure_ocv)
        row1.addWidget(QLabel('OCV (V):')); row1.addWidget(self.label_ocv)
        row1.addWidget(self.button_clear)

        grid = QGridLayout()
        grid.addLayout(row0, 0, 0)
        grid.addLayout(row1, 1, 0)
        self.setLayout(grid)

    def _make_connections(self):
        self.button_measure_ocv.clicked.connect(self._on_measure_ocv)
        self.button_start_ca.clicked.connect(self._on_start_ca)
        self.button_stop.clicked.connect(self._on_stop)
        self.button_clear.clicked.connect(self._on_clear)

    # ---- actions ----
    def _on_measure_ocv(self):
        self.ctrl.ensure_connected()
        self.ctrl.start_ocv(record_dt=0.1)
        # grab a couple of snapshots quickly to stabilize readout
        rows, _ = self.ctrl.poll()
        if rows:
            last = rows[-1]
            self.label_ocv.setNum(float(last.get("Ewe") or 0.0))
        # keep polling in case the user wants continuous OCV; timer will update buffers if CA starts

    def _on_start_ca(self):
        self.ctrl.ensure_connected()
        v_plus  = float(self.v_plus.value())  / 1000.0  # mV -> V
        v_minus = float(self.v_minus.value()) / 1000.0
        cycles  = int(self.cycles.value())
        dur     = float(self.duration.value())

        self.ctrl.prepare_ca_protocol(v_plus=v_plus, v_minus=v_minus, cycles=cycles, duration=dur, dt=0.1)
        self.ctrl.start_ca()
        self._clear_buffers()
        self.timer.start()

    def _on_stop(self):
        self.timer.stop()
        self.ctrl.stop()

        # emit one-shot dict for compatibility with existing pipeline
        self.iv_dict_from_gate_once.emit({"t": self._t[:], "V_we": self._v[:], "I": self._i[:]})

    def _on_clear(self):
        self._clear_buffers()

    def _clear_buffers(self):
        self._t.clear(); self._v.clear(); self._i.clear()

    # ---- poller ----
    def _on_poll(self):
        rows, status = self.ctrl.poll()
        for r in rows:
            t, Ewe, Iwe = r.get("t"), r.get("Ewe"), r.get("Iwe")
            if t is None: 
                continue
            self._t.append(float(t))
            self._v.append(float(Ewe) if Ewe is not None else 0.0)
            self._i.append(float(Iwe) if Iwe is not None else 0.0)
        # if you want continuous OCV display updates:
        if rows and self.ctrl.mode == "ocv":
            self.label_ocv.setNum(float(self._v[-1]))
        if status == "STOP":
            self._on_stop()
