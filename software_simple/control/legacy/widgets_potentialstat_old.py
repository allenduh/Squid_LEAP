"""
 Widget in GUI -- control potentialstat.
 
 Input: client.potentialstat(self.leap_win_server)
		& self.ivStreamHandler

 Example: 
 # Linux client initiate potentialstat, NKT, spectrometer from "server_read_in_linux"
 self.potentialstat = client.potentialstat(self.leap_win_server)
 self.potentialstatControlWidget = widgets_potentialstat.potentialstatControlWidget(self.potentialstat, self.ivStreamHandler)

 Date: 09/16/2022
 Author: Yi-Shiou Duh (allenduh@stanford.edu)
"""
# set QT_API environment variable
import os 
os.environ["QT_API"] = "pyqt5"
import qtpy
import sys

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *
import pyqtgraph as pg
from datetime import datetime



import easy_biologic as ebl
import easy_biologic.base_programs as ebp




class potentialstatControlWidget(QFrame):

	preview = Signal(dict)
	iv_dict_from_gate_once = Signal(dict)

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.potentialstat = ebl.BiologicDevice('USB0')
		self.add_components()
		self.setFrameStyle(QFrame.Panel | QFrame.Raised)

	def add_components(self):
		# Line 0: Protocol
		self.button_preview_protocol = QPushButton("Preview protocol")
		self.button_preview_protocol.setDefault(False)

		# Line 0: Protocol
		self.button_connect_server = QPushButton("Connect server")
		self.button_connect_server.setDefault(False)

		# Range of V+
		self.v_plus = QDoubleSpinBox()
		self.v_plus.setMinimum(-500) 
		self.v_plus.setMaximum(500)  
		self.v_plus.setSingleStep(1)
		self.v_plus.setValue(10)
		self.v_plus.setKeyboardTracking(False)

		# Range of V-
		self.v_minus = QDoubleSpinBox()
		self.v_minus.setMinimum(-500) 
		self.v_minus.setMaximum(500)  
		self.v_minus.setSingleStep(1)
		self.v_minus.setValue(0)
		self.v_minus.setKeyboardTracking(False)

		# Range of cycles
		self.cycles = QDoubleSpinBox()
		self.cycles.setMinimum(0) 
		self.cycles.setMaximum(100)  
		self.cycles.setSingleStep(1)
		self.cycles.setValue(5)
		self.cycles.setKeyboardTracking(False)

		# Range of duration
		self.duration = QDoubleSpinBox()
		self.duration.setMinimum(0) 
		self.duration.setMaximum(100)  
		self.duration.setSingleStep(0.1)
		self.duration.setValue(1)
		self.duration.setKeyboardTracking(False)

		grid_line0 = QHBoxLayout()
		grid_line0.addWidget(QLabel('Potentialstat protocol'))
		grid_line0.addWidget(QLabel('V+ (mV)'))
		grid_line0.addWidget(self.v_plus)
		grid_line0.addWidget(QLabel('V- (mV)'))
		grid_line0.addWidget(self.v_minus)
		grid_line0.addWidget(QLabel('cycles'))
		grid_line0.addWidget(self.cycles)
		grid_line0.addWidget(QLabel('durations (s)'))
		grid_line0.addWidget(self.duration)
		

		# Line 1
		self.button_measure_ocv = QPushButton("Measure ocv")
		self.button_measure_ocv.setDefault(False)

		self.button_clear_plots = QPushButton("Clear plots")
		self.button_clear_plots.setDefault(False)

		self.button_gate_once = QPushButton("Gate once")
		self.button_gate_once.setCheckable(True)
		self.button_gate_once.setChecked(False)
		self.button_gate_once.setDefault(False)

		self.label_ocv = QLabel()
		self.label_ocv.setNum(0)
		self.label_ocv.setFrameStyle(QFrame.Panel | QFrame.Sunken)

		grid_line1 = QHBoxLayout()
		grid_line1.addWidget(self.button_connect_server)
		grid_line1.addWidget(self.button_preview_protocol)
		grid_line1.addWidget(self.button_gate_once)
		grid_line1.addWidget(self.button_measure_ocv)
		grid_line1.addWidget(QLabel('Technique: CA'))
		grid_line1.addWidget(QLabel('OCV = (mV)'))
		grid_line1.addWidget(self.label_ocv)


		# connections
		self.button_gate_once.clicked.connect(self.toggle_gate_once)
		self.button_measure_ocv.clicked.connect(self.toggle_measure_ocv)
		self.button_preview_protocol.clicked.connect(self.toggle_output_protocol)
		self.button_connect_server.clicked.connect(self.toggle_connect_server)
		self.v_plus.valueChanged.connect(self.any_value_change)
		self.v_minus.valueChanged.connect(self.any_value_change)
		self.cycles.valueChanged.connect(self.any_value_change)
		self.duration.valueChanged.connect(self.any_value_change)

		# layout
		self.grid = QGridLayout()
		self.grid.addLayout(grid_line0, 0, 0)
		self.grid.addLayout(grid_line1, 1, 0)
		self.setLayout(self.grid)



	def toggle_gate_once(self, pressed):
		if pressed:
			params = { 
				'voltages':  [ 0, 1 ]* 2,
				'durations': [ 1 ]* 4, 
				'time_interval': 0.1
			}

			save_path = 'testt.csv'
			prg = ebp.CA(potentialstat, params, channels = [0])
			prg.run()
			prg.save_data(save_path)


			self.any_value_change()
			self.potentialstat.measure_ocv()  # measure ocv before gating start
			startTime = self.potentialstat.start_gating() # include prepare_CA_gating and start channel
			iv_dict = self.potentialstat.after_gating_read_iv()
			self.iv_dict_from_gate_once.emit(iv_dict)
			self.button_gate_once.setChecked(False) # So we can press again 
		else:
			self.potentialstat.abort_gating_without_reading()

	def toggle_measure_ocv(self):
		self.label_ocv.setNum(self.potentialstat.measure_ocv())
		

	def toggle_output_protocol(self):  # preview protocol
		self.any_value_change()
		self.potentialstat.output_protocol_parameters()
		self.preview.emit(self.potentialstat.preview_v_protocol())



	def any_value_change(self):
		input_info = {
			 'v_plus': self.v_plus.value() / 1000,  # convert mV (GUI)to V(potentialstat client)
			 'v_minus': self.v_minus.value() / 1000, 
			 'cycles':self.cycles.value(), 
			 'duration':self.duration.value()
			}
		self.potentialstat.update_protocol_parameters(input_info)


class ivDisplay(QFrame):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.plot_I_Widget = Plot_IV_Widget(yLable = 'I')
		self.plot_V_Widget = Plot_IV_Widget(yLable = 'V')
		self.setFrameStyle(QFrame.Panel | QFrame.Raised)

		layout = QGridLayout() 
		layout.addWidget(self.plot_V_Widget, 0, 0)
		layout.addWidget(self.plot_I_Widget, 1, 0)
		self.setLayout(layout)

	def plot(self,data):  # called in GUI: self.ivStreamHandler.spectrum_to_display.connect(self.ivDisplay.plot)
		self.plot_I_Widget.update(data["t"], data["I"])
		self.plot_V_Widget.update(data["t"], data["V_we"])


class Plot_IV_Widget(pg.GraphicsLayoutWidget):
	
	def __init__(self, yLable, parent=None):
		super().__init__(parent)
		self.plotWidget = self.addPlot()
		self.curve = self.plotWidget.plot()
		
		self.plotWidget.setLabel(axis='left', text=yLable)

	def update(self, t ,I):
		self.curve.setData(t, I, pen = 'g')
