# potentiostat_panel_v3.py
# Upgrades:
#  • Continuous OCV when idle (auto-restart after any Stop)
#  • Auto-save on Stop (optional checkbox). Saves .npz + tiny .json next to it
#  • "Load NPZ" to visualize prior IV files
#  • Clearer state machine: mode in {"idle","ocv","ca"}; STOP event funnels through _finish_ca()
#
# Notes:
#  • If self.last_exp_dir exists in your app, it's used as the save directory.
#    Otherwise we prompt once and remember it for the session.

import os, json
os.environ.setdefault("QT_API", "pyqt5")
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
from qtpy.QtCore import Signal, QTimer
from qtpy.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QFileDialog, QCheckBox, QMessageBox
)

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

import drivers.ECLab.kbio_types as KBIO
from drivers.ECLab.kbio_api import KBIO_api


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
            dll_path = "drivers\\ECLab\\lib\\EClib64.dll"
        self.api = KBIO_api(dll_path)
        self.id_, self.device_info = self.api.Connect(self.address)

        self.steps: List[voltage_step] = [voltage_step(0.0, 1.0), voltage_step(-0.010, 1.0)]
        self.record_dt = 0.1
        self.record_dI = 0.1
        self.repeat_count = 10

        self.p_steps: List[KBIO.EccParam] = []
        self.final_parameters = None

        self.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    def get_hardware_conf(self):
        try:
            return self.api.GetHardConf(self.id_, self.channel)
        except Exception:
            return None

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

    def append_CV_parameter_list(
        self,
        Ei: float,
        E1: float,
        E2: float,
        Ef: float,
        scan_rate_mV_s: float,
        dE_V: float,
        n_cycles: int,
        average_over_dE: bool = False,
        vs_initial_flags=None,
        begin_I: float = 0.0,
        end_I: float = 1.0,
    ):
        """
        Build parameter list for CV (cv4.ecc) exactly as defined in the
        EC-Lab Development Package.

        Ei, E1, E2, Ef  : potentials in Volts
        scan_rate_mV_s  : scan rate (mV/s) – same for each segment here
        dE_V            : Record_every_dE in Volts
        n_cycles        : N_Cycles
        average_over_dE : Average_over_dE
        vs_initial_flags: list of 5 booleans for vs_initial
        begin_I, end_I  : Begin_measuring_I / End_measuring_I in [0..1]
        """
        self.p_steps.clear()

        # --- vs_initial: 5 booleans ---
        if vs_initial_flags is None:
            # default: all steps are "vs initial"
            vs_initial_flags = [False] * 5
        if len(vs_initial_flags) != 5:
            raise ValueError("vs_initial_flags must be length 5")

        for idx, flag in enumerate(vs_initial_flags):
            self.p_steps.append(
                self.make_single_parameter("vs_initial", idx, bool(flag))
            )

        # --- Voltage_step + Scan_Rate: 5 entries each ---
        # Manual: Voltage_step = [Ei, E1, E2, Ei, Ef]
        v_steps = [Ei, E1, E2, Ei, Ef]
        for idx, v in enumerate(v_steps):
            self.p_steps.append(
                self.make_single_parameter("Voltage_step", idx, float(v))
            )
            self.p_steps.append(
                self.make_single_parameter("Scan_Rate", idx, float(scan_rate_mV_s/1000))
            )

        # --- Scalar parameters (index 0) ---
        # Scan_number must be 2 for CV
        self.p_steps.append(
            self.make_single_parameter("Scan_number", 0, int(2))
        )
        self.p_steps.append(
            self.make_single_parameter("Record_every_dE", 0, float(dE_V))
        )
        self.p_steps.append(
            self.make_single_parameter("Average_over_dE", 0, bool(average_over_dE))
        )
        self.p_steps.append(
            self.make_single_parameter("N_Cycles", 0, int(n_cycles))
        )
        self.p_steps.append(
            self.make_single_parameter("Begin_measuring_I", 0, float(begin_I))
        )
        self.p_steps.append(
            self.make_single_parameter("End_measuring_I", 0, float(end_I))
        )


    def make_final_parameters(self):
        nb = len(self.p_steps)
        arr = KBIO.ECC_PARM_ARRAY(nb)
        for i, p in enumerate(self.p_steps):
            arr[i] = p
        self.final_parameters = KBIO.EccParams(nb, arr)

    def load_technique_CA(self):
        tech_file = "drivers\\ECLab\\lib\\ca4.ecc"
        self.api.LoadTechnique(self.id_, self.channel, tech_file, self.final_parameters,
                               first=True, last=True, display=False)

    def load_technique_CV(self):
        """
        Load CV technique (cv4.ecc for VMP-300).

        If your install uses a different filename, adjust tech_file.
        """
        tech_file = "drivers\\ECLab\\lib\\cv4.ecc"
        self.api.LoadTechnique(
            self.id_,
            self.channel,
            tech_file,
            self.final_parameters,
            first=True,
            last=True,
            display=False,
        )

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

    def record_snapshot(self) -> Tuple[List[dict], str]:
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
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])  # OCV
            elif payload_words == 2:
                out["Ewe"] = self.api.ConvertNumericIntoSingle(payload[0])  # OCV (VMP3)
                out["Ece"] = self.api.ConvertNumericIntoSingle(payload[1])
            elif payload_words == 3:
                out["Ewe"]  = self.api.ConvertNumericIntoSingle(payload[0])  # CA/CP
                out["Iwe"]  = self.api.ConvertNumericIntoSingle(payload[1])
                out["cycle"]= payload[2]
                self.ivCurve["t"].append(out["t"])
                self.ivCurve["V_we"].append(out["Ewe"])
                self.ivCurve["I"].append(out["Iwe"] if out["Iwe"] is not None else 0.0)
                self.ivCurve["cycle"].append(out["cycle"])

            rows.append(out); idx = inx
        return rows, status


class potentialstatControlWidget(QFrame):
    iv_dict_from_gate_once = Signal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.ctrl = potentialstat_controller()
        self._last_ocv: Optional[float] = None
        self._mode = "idle"  # {"idle","ocv","ca","cv"}

        # If parent app set this, we'll use it. Otherwise we fill it on first save.
        self.experiment_ID= getattr(self, "experiment_ID", None)

        self._build_ui()
        self._connect()

        self.timer = QTimer(self); self.timer.setInterval(100)   # 100 ms
        self.timer.timeout.connect(self._on_poll)

        self._t: List[float] = []; self._v: List[float] = []; self._i: List[float] = []
        self._refresh_hw_label()
        self._start_ocv_background()

    # ---------- UI ----------
    def _build_ui(self):
        # --- Existing CA controls (unchanged) ---
        self.v_plus  = QDoubleSpinBox();  self.v_plus.setRange(-500, 500);  self.v_plus.setValue(0.0);    self.v_plus.setSingleStep(1)
        self.v_minus = QDoubleSpinBox();  self.v_minus.setRange(-500, 500); self.v_minus.setValue(-10.0); self.v_minus.setSingleStep(1)
        self.cycles  = QDoubleSpinBox();  self.cycles.setRange(1, 1000);    self.cycles.setValue(5);      self.cycles.setSingleStep(1)
        self.t_plus  = QDoubleSpinBox();  self.t_plus.setRange(0.001, 1000.0); self.t_plus.setDecimals(3); self.t_plus.setSingleStep(0.001); self.t_plus.setValue(1.0)
        self.t_minus = QDoubleSpinBox();  self.t_minus.setRange(0.001, 1000.0); self.t_minus.setDecimals(3); self.t_minus.setSingleStep(0.001); self.t_minus.setValue(1.0)
        self.dt_min  = QDoubleSpinBox();  self.dt_min.setRange(0.0, 10.0);  self.dt_min.setDecimals(3);  self.dt_min.setValue(0.001); self.dt_min.setSingleStep(0.01)
        self.dI_min  = QDoubleSpinBox();  self.dI_min.setRange(0.0, 10.0);  self.dI_min.setDecimals(6);  self.dI_min.setValue(0.001); self.dI_min.setSingleStep(0.01)
        self.chk_rel_ocv = QCheckBox("V relative to OCV")

        # --- NEW: CV (break-in) controls ------------------------------------
        self.cv_Ei   = QDoubleSpinBox();  self.cv_Ei.setRange(-1500.0, 1500.0);  self.cv_Ei.setValue(0.0);    self.cv_Ei.setSingleStep(10.0)
        self.cv_E1   = QDoubleSpinBox();  self.cv_E1.setRange(-1500.0, 1500.0);  self.cv_E1.setValue(500.0);  self.cv_E1.setSingleStep(10.0)
        self.cv_E2   = QDoubleSpinBox();  self.cv_E2.setRange(-1500.0, 1500.0);  self.cv_E2.setValue(0.0);    self.cv_E2.setSingleStep(10.0)
        self.cv_Ef   = QDoubleSpinBox();  self.cv_Ef.setRange(-1500.0, 1500.0);  self.cv_Ef.setValue(0.0);    self.cv_Ef.setSingleStep(10.0)
        self.cv_scan = QDoubleSpinBox();  self.cv_scan.setRange(1.0, 10000.0);   self.cv_scan.setDecimals(1); self.cv_scan.setSingleStep(10.0);  self.cv_scan.setValue(50.0)   # mV/s
        self.cv_dE   = QDoubleSpinBox();  self.cv_dE.setRange(0.1, 500.0);       self.cv_dE.setDecimals(2);   self.cv_dE.setSingleStep(1.0);     self.cv_dE.setValue(1.0)      # mV
        self.cv_cycles = QDoubleSpinBox(); self.cv_cycles.setRange(1.0, 1000.0); self.cv_cycles.setSingleStep(1.0); self.cv_cycles.setValue(10.0)
        self.chk_cv_rel_ocv = QCheckBox("V relative to OCV (CV)")

        row0 = QHBoxLayout()
        for w in [QLabel('Protocol'),
                  QLabel('V+ (mV)'), self.v_plus, QLabel('V- (mV)'), self.v_minus,
                  QLabel('t+ (s)'), self.t_plus, QLabel('t- (s)'), self.t_minus,
                  QLabel('cycles'), self.cycles, QLabel('dT (s)'), self.dt_min,
                  QLabel('dI (A)'), self.dI_min, self.chk_rel_ocv]:
            row0.addWidget(w)

        # --- NEW: CV row with its own Start button --------------------------
        self.btn_start_cv = QPushButton("Start CV (break-in)")
        row2 = QHBoxLayout()
        for w in [
            QLabel("CV break-in"),
            QLabel("Ei (mV)"), self.cv_Ei,
            QLabel("E1 (mV)"), self.cv_E1,
            QLabel("E2 (mV)"), self.cv_E2,
            QLabel("Ef (mV)"), self.cv_Ef,
            QLabel("scan (mV/s)"), self.cv_scan,
            QLabel("dE (mV)"), self.cv_dE,
            QLabel("cycles"), self.cv_cycles,
            self.chk_cv_rel_ocv,
            self.btn_start_cv,
        ]:
            row2.addWidget(w)

        self.btn_start = QPushButton("Start CA")
        self.btn_stop  = QPushButton("Stop")
        self.btn_ocv   = QPushButton("Measure OCV")
        self.btn_clear = QPushButton("Clear")
        self.btn_save  = QPushButton("Save IV (.npz)")
        self.btn_load  = QPushButton("Load NPZ…")   # NEW
        self.chk_autosave = QCheckBox("Auto-save on Stop")  # NEW
        self.chk_autosave.setChecked(False)

        self.lbl_ocv   = QLabel(); self.lbl_ocv.setNum(0.0); self.lbl_ocv.setFrameStyle(QFrame.Panel|QFrame.Sunken)
        self.lbl_hw    = QLabel("HW: n/a")

        row1 = QHBoxLayout()
        for w in [self.btn_start, self.btn_stop, self.btn_ocv,
                  QLabel("OCV (V):"), self.lbl_ocv, self.lbl_hw,
                  self.btn_clear, self.btn_save, self.btn_load, self.chk_autosave]:
            row1.addWidget(w)

        grid = QGridLayout()
        grid.addLayout(row0, 0, 0)
        grid.addLayout(row1, 1, 0)
        grid.addLayout(row2, 2, 0)

        if pg is None:
            grid.addWidget(QLabel("Install 'pyqtgraph' to enable live plots."), 3, 0)
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
            grid.addWidget(self.plot_v, 3, 0)
            grid.addWidget(self.plot_i, 4, 0)

        self.setLayout(grid)


    def _connect(self):
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_ocv.clicked.connect(self._on_ocv)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_save.clicked.connect(self._on_save_npz_dialog)
        self.btn_load.clicked.connect(self._on_load_npz_dialog)
        self.btn_start_cv.clicked.connect(self._on_start_cv)

    # ---------- State helpers ----------
    def _refresh_hw_label(self):
        hw = self.ctrl.get_hardware_conf()
        if not hw:
            self.lbl_hw.setText("HW: n/a"); return
        try:   cnx = KBIO.HW_CNX(hw.connection).name
        except Exception: cnx = str(hw.connection)
        try:   mode = KBIO.HW_MODE(hw.mode).name
        except Exception: mode = str(hw.mode)
        self.lbl_hw.setText(f"HW: {cnx}/{mode}")

    def _clear_buffers(self):
        self._t.clear(); self._v.clear(); self._i.clear()
        self.ctrl.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}

    def _on_clear(self):
        self._clear_buffers()
        if pg and self.curve_v and self.curve_i:
            self.curve_v.setData([], [])
            self.curve_i.setData([], [])
            
    def _set_exp_dir_if_needed(self):
        # prefer attribute passed by parent app
        if getattr(self, "experiment_ID", None) and os.path.isdir(self.experiment_ID):
            return True
        d = QFileDialog.getExistingDirectory(self, "Select experiment directory")
        if not d:
            return False
        self.experiment_ID = d
        return True

    # ---------- OCV control ----------
    def _start_ocv_background(self):
        """Ensure OCV is running (for monitoring) when we're not doing CA."""
        try:
            self.ctrl.stop()
        except Exception:
            pass
        self.ctrl.load_technique_OCV(record_dt=float(self.dt_min.value()))
        self.ctrl.start()
        self._mode = "ocv"
        # For monitor-only feel, don't accumulate OCV into IV buffers:
        # keep V label live; if you DO want it plotted, uncomment next line
        # self._clear_buffers()
        if not self.timer.isActive():
            self.timer.start()

    def _on_ocv(self):
        self._start_ocv_background()

    # ---------- CA control ----------
    def _on_start(self):
        # halt OCV so techniques don't clash
        try: self.ctrl.stop()
        except Exception: pass
        self._mode = "idle"

        v_plus  = float(self.v_plus.value())  / 1000.0
        v_minus = float(self.v_minus.value()) / 1000.0
        t_plus  = float(self.t_plus.value())
        t_minus = float(self.t_minus.value())
        cycles  = int(self.cycles.value())

        if self.chk_rel_ocv.isChecked() and self._last_ocv is not None:
            v_plus  += self._last_ocv
            v_minus += self._last_ocv

        self.ctrl.steps = [voltage_step(v_plus, t_plus), voltage_step(v_minus, t_minus)]
        self.ctrl.repeat_count = max(0, cycles - 1)
        self.ctrl.record_dt = float(self.dt_min.value())
        self.ctrl.record_dI = float(self.dI_min.value())

        self.ctrl.append_CA_parameter_list()
        self.ctrl.make_final_parameters()
        self.ctrl.load_technique_CA()

        self._clear_buffers()
        self.ctrl.start()
        self._mode = "ca"
        self.timer.start()

    def _on_start_cv(self):
        # Stop any running technique first
        try:
            self.ctrl.stop()
        except Exception:
            pass
        self._mode = "idle"

        # Potentials are entered in mV in the GUI → convert to V
        Ei = float(self.cv_Ei.value()) / 1000.0
        E1 = float(self.cv_E1.value()) / 1000.0
        E2 = float(self.cv_E2.value()) / 1000.0
        Ef = float(self.cv_Ef.value()) / 1000.0

        # Optional "relative to OCV" behavior
        if self.chk_cv_rel_ocv.isChecked() and self._last_ocv is not None:
            Ei += self._last_ocv
            E1 += self._last_ocv
            E2 += self._last_ocv
            Ef += self._last_ocv

        scan_rate_mV_s = float(self.cv_scan.value())      # mV/s
        dE_V           = float(self.cv_dE.value()) / 1000.0  # mV → V
        n_cycles       = int(self.cv_cycles.value())

        # Build CV parameters as the manual specifies
        self.ctrl.append_CV_parameter_list(
            Ei=Ei,
            E1=E1,
            E2=E2,
            Ef=Ef,
            scan_rate_mV_s=scan_rate_mV_s,
            dE_V=dE_V,
            n_cycles=n_cycles,
            average_over_dE=False,        # or expose in UI if you want
            vs_initial_flags=None,        # default [True]*5
            begin_I=0.0,
            end_I=1.0,
        )
        self.ctrl.make_final_parameters()
        self.ctrl.load_technique_CV()

        self._clear_buffers()
        self.ctrl.start()
        self._mode = "cv"
        self.timer.start()


    def _on_stop_clicked(self):
        """User pressed Stop; end CA (or OCV) and finalize."""
        self.ctrl.stop()             # ask hardware to stop
        # We will funnel completion via _finish_ca() in the poller when status turns STOP.

    def _finish_ca(self):
        """Common path whenever CA actually ends (STOP from hardware or manual)."""
        # 1) Prepare IV dict and emit
        to_emit = {"t": self._t[:], "V_we": self._v[:], "I": self._i[:]}
        self.iv_dict_from_gate_once.emit(to_emit)

        # 2) Optional auto-save
        if self.chk_autosave.isChecked() and self._t:
            self._save_iv_npz(auto=True)

        # 3) Resume background OCV for monitoring
        self._start_ocv_background()

    # ---------- Persistence ----------
    def _save_iv_npz(self, auto=False, path=None):
        """Save NPZ + tiny JSON in last_exp_dir (or prompt once)."""
        if not self._t:
            if not auto:
                QMessageBox.information(self, "Save IV", "No data to save.")
            return

        # resolve directory (prefer experiment dir if available)
        if not self._set_exp_dir_if_needed():
            if not auto:
                QMessageBox.warning(self, "Save IV", "No directory selected; save cancelled.")
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f"iv_{ts}"
        if path is None:
            path = os.path.join(self.experiment_ID, base + ".npz")

        meta = dict(
            address=self.ctrl.address, channel=self.ctrl.channel,
            steps=[(s.voltage, s.duration, s.vs_init) for s in self.ctrl.steps],
            record_dt=self.ctrl.record_dt, record_dI=self.ctrl.record_dI,
            repeat_count=self.ctrl.repeat_count, started=self.ctrl.timeStart,
            mode=self._mode
        )
        np.savez(
            path,
            t=np.asarray(self._t, dtype=np.float64),
            V_we=np.asarray(self._v, dtype=np.float64),
            I=np.asarray(self._i, dtype=np.float64),
            cycle=np.asarray(self.ctrl.ivCurve["cycle"], dtype=np.int64) if self.ctrl.ivCurve["cycle"] else np.array([]),
            meta=str(meta),
        )
        # tiny JSON sidecar
        with open(os.path.join(self.experiment_ID, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if not auto:
            QMessageBox.information(self, "Saved", f"Saved:\n{path}")

    def _on_save_npz_dialog(self):
        if not self._t:
            QMessageBox.information(self, "Save IV", "No data to save.")
            return
        default = f"ivcurve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        path, _ = QFileDialog.getSaveFileName(self, "Save IV (.npz)", default, "NumPy Zip (*.npz)")
        if not path:
            return
        # Still produce the tiny JSON next to the chosen path
        self._save_iv_npz(auto=False, path=path)

    def _on_load_npz_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load IV (.npz)", "", "NumPy Zip (*.npz)")
        if not path:
            return
        try:
            with np.load(path, allow_pickle=True) as z:
                t = z.get("t"); v = z.get("V_we"); i = z.get("I")
                if t is None or v is None:
                    QMessageBox.warning(self, "Load NPZ", "File missing t/V_we arrays."); return
                self._t = list(np.asarray(t, dtype=float))
                self._v = list(np.asarray(v, dtype=float))
                self._i = list(np.asarray(i, dtype=float)) if i is not None else [0.0]*len(self._t)
        except Exception as e:
            QMessageBox.critical(self, "Load NPZ", f"Failed to load:\n{e}")
            return

        if pg and getattr(self, "curve_v", None) and getattr(self, "curve_i", None):
            self.curve_v.setData(self._t, self._v)
            self.curve_i.setData(self._t, self._i)

    # ---------- Poller ----------
    def _on_poll(self):
        rows, status = self.ctrl.record_snapshot()

        # OCV monitoring (label + optional plot of V)
        if rows and self._mode == "ocv":
            last_ewe = next((r["Ewe"] for r in reversed(rows) if r.get("Ewe") is not None), None)
            if last_ewe is not None:
                self._last_ocv = float(last_ewe)
                self.lbl_ocv.setNum(self._last_ocv)
            # If you want to see OCV trend on the V plot, uncomment below:
            # if pg and self.curve_v:
            #     t_vals = [float(r["t"]) for r in rows if r.get("t") is not None]
            #     v_vals = [float(r["Ewe"]) for r in rows if r.get("Ewe") is not None]
            #     self._t.extend(t_vals); self._v.extend(v_vals)
            #     self.curve_v.setData(self._t, self._v)

        # CA data accumulation + plotting
        if self._mode in ("ca", "cv"):
            new_any = False
            for r in rows:
                t = r.get("t"); Ewe = r.get("Ewe"); Iwe = r.get("Iwe")
                if t is None:
                    continue
                self._t.append(float(t))
                self._v.append(float(Ewe) if Ewe is not None else 0.0)
                self._i.append(float(Iwe) if Iwe is not None else 0.0)
                new_any = True
            if pg and new_any and self.curve_v and self.curve_i:
                self.curve_v.setData(self._t, self._v)
                self.curve_i.setData(self._t, self._i)

        # Hardware STOP → finalize CA/CV, autosave, resume OCV
        if status == "STOP" and self._mode in ("ca", "cv", "ocv", "idle"):
            # If we were in CA or CV, finalize; if OCV, just make sure state resets.
            if self._mode in ("ca") and self._t:
                self._finish_ca()
            else:
                # ensure we are in OCV after any non-CA/CV stop
                self._start_ocv_background()
