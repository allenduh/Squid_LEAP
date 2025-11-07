import os
import sys
import subprocess


# set QT_API environment variable
os.environ["QT_API"] = "pyqt5"

# qt libraries
import qtpy
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

import pyqtgraph as pg
import pandas as pd
import napari
from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS
import re
import cv2
import math
import locale
import time
from datetime import datetime
import itertools
import numpy as np
from scipy.spatial import Delaunay
import shutil
from control._def import *
from PIL import Image, ImageDraw, ImageFont

class LiveControlWidget(QFrame):

    signal_newExposureTime = Signal(float)
    signal_newAnalogGain = Signal(float)
    signal_autoLevelSetting = Signal(bool)
    signal_live_configuration = Signal(object)
    signal_start_live = Signal()

    def __init__(self, streamHandler, liveController, show_trigger_options=True, show_display_options=False, show_autolevel = False, autolevel=False, stretch=True, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.liveController = liveController
        self.streamHandler = streamHandler
        
        self.fps_trigger = 10
        self.fps_display = 10
        self.liveController.set_trigger_fps(self.fps_trigger)
        self.streamHandler.set_display_fps(self.fps_display)

        #self.triggerMode = TriggerMode.SOFTWARE
        self.triggerMode = TriggerMode.CONTINUOUS
        # note that this references the object in self.configurationManager.configurations
        #self.currentConfiguration = self.configurationManager.configurations[0]

        self.add_components(show_trigger_options,show_display_options,show_autolevel,autolevel,stretch)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        #self.update_microscope_mode_by_name(self.currentConfiguration.name)

        self.is_switching_mode = False # flag used to prevent from settings being set by twice - from both mode change slot and value change slot; another way is to use blockSignals(True)

    def add_components(self,show_trigger_options,show_display_options,show_autolevel,autolevel,stretch):
        # line 0: trigger mode
        self.triggerMode = None
        self.dropdown_triggerManu = QComboBox()
        self.dropdown_triggerManu.addItems([TriggerMode.SOFTWARE,TriggerMode.HARDWARE,TriggerMode.CONTINUOUS])
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dropdown_triggerManu.setSizePolicy(sizePolicy)

        # line 1: fps
        self.entry_triggerFPS = QDoubleSpinBox()
        self.entry_triggerFPS.setMinimum(0.02)
        self.entry_triggerFPS.setMaximum(1000)
        self.entry_triggerFPS.setSingleStep(1)
        self.entry_triggerFPS.setValue(self.fps_trigger)
        self.entry_triggerFPS.setDecimals(0)

        # line 2: choose microscope mode / toggle live mode


        self.btn_live = QPushButton("Start Live")
        self.btn_live.setCheckable(True)
        self.btn_live.setChecked(False)
        self.btn_live.setDefault(False)
        self.btn_live.setStyleSheet("background-color: #C2C2FF")
        self.btn_live.setSizePolicy(sizePolicy)

        # line 3: exposure time and analog gain associated with the current mode
        self.entry_exposureTime = QDoubleSpinBox()
        self.entry_exposureTime.setMinimum(self.liveController.camera.EXPOSURE_TIME_MS_MIN)
        self.entry_exposureTime.setMaximum(self.liveController.camera.EXPOSURE_TIME_MS_MAX)
        self.entry_exposureTime.setSingleStep(1)
        self.entry_exposureTime.setSuffix(' ms')
        self.entry_exposureTime.setValue(0)
        self.entry_exposureTime.setSizePolicy(sizePolicy)

        self.entry_analogGain = QDoubleSpinBox()
        self.entry_analogGain = QDoubleSpinBox()
        self.entry_analogGain.setMinimum(0)
        self.entry_analogGain.setMaximum(24)
        # self.entry_analogGain.setSuffix('x')
        self.entry_analogGain.setSingleStep(0.1)
        self.entry_analogGain.setValue(0)
        self.entry_analogGain.setSizePolicy(sizePolicy)

        self.slider_illuminationIntensity = QSlider(Qt.Horizontal)
        self.slider_illuminationIntensity.setTickPosition(QSlider.TicksBelow)
        self.slider_illuminationIntensity.setMinimum(0)
        self.slider_illuminationIntensity.setMaximum(100)
        self.slider_illuminationIntensity.setValue(100)
        self.slider_illuminationIntensity.setSingleStep(2)

        self.entry_illuminationIntensity = QDoubleSpinBox()
        self.entry_illuminationIntensity.setMinimum(0)
        self.entry_illuminationIntensity.setMaximum(100)
        self.entry_illuminationIntensity.setSingleStep(1)
        self.entry_illuminationIntensity.setSuffix('%')
        self.entry_illuminationIntensity.setValue(100)

        # line 4: display fps and resolution scaling
        self.entry_displayFPS = QDoubleSpinBox()
        self.entry_displayFPS.setMinimum(1)
        self.entry_displayFPS.setMaximum(240)
        self.entry_displayFPS.setSingleStep(1)
        self.entry_displayFPS.setDecimals(0)
        self.entry_displayFPS.setValue(self.fps_display)

        self.slider_resolutionScaling = QSlider(Qt.Horizontal)
        self.slider_resolutionScaling.setTickPosition(QSlider.TicksBelow)
        self.slider_resolutionScaling.setMinimum(10)
        self.slider_resolutionScaling.setMaximum(100)
        self.slider_resolutionScaling.setValue(DEFAULT_DISPLAY_CROP)
        self.slider_resolutionScaling.setSingleStep(10)

        self.label_resolutionScaling = QSpinBox()
        self.label_resolutionScaling.setMinimum(10)
        self.label_resolutionScaling.setMaximum(100)
        self.label_resolutionScaling.setValue(self.slider_resolutionScaling.value())
        self.label_resolutionScaling.setSuffix(" %")
        self.slider_resolutionScaling.setSingleStep(5)

        self.slider_resolutionScaling.valueChanged.connect(lambda v: self.label_resolutionScaling.setValue(round(v)))
        self.label_resolutionScaling.valueChanged.connect(lambda v: self.slider_resolutionScaling.setValue(round(v)))

        # autolevel
        self.btn_autolevel = QPushButton('Autolevel')
        self.btn_autolevel.setCheckable(True)
        self.btn_autolevel.setChecked(autolevel)

        # Determine the maximum width needed
        self.entry_illuminationIntensity.setMinimumWidth(self.btn_live.sizeHint().width())
        self.btn_autolevel.setMinimumWidth(self.btn_autolevel.sizeHint().width())

        max_width = max(
            self.btn_autolevel.minimumWidth(),
            self.entry_illuminationIntensity.minimumWidth()
        )

        # Set the fixed width for all three widgets
        self.entry_illuminationIntensity.setFixedWidth(max_width)
        self.btn_autolevel.setFixedWidth(max_width)

        # connections
        self.entry_triggerFPS.valueChanged.connect(self.liveController.set_trigger_fps)
        self.entry_displayFPS.valueChanged.connect(self.streamHandler.set_display_fps)
        self.slider_resolutionScaling.valueChanged.connect(self.streamHandler.set_display_resolution_scaling)
        self.slider_resolutionScaling.valueChanged.connect(self.liveController.set_display_resolution_scaling)
        #self.dropdown_modeSelection.currentTextChanged.connect(self.update_microscope_mode_by_name)
        self.dropdown_triggerManu.currentIndexChanged.connect(self.update_trigger_mode)
        self.btn_live.clicked.connect(self.toggle_live)
        self.entry_exposureTime.valueChanged.connect(self.update_config_exposure_time)
        self.entry_analogGain.valueChanged.connect(self.update_config_analog_gain)
        self.entry_illuminationIntensity.valueChanged.connect(self.update_config_illumination_intensity)
        self.entry_illuminationIntensity.valueChanged.connect(lambda x: self.slider_illuminationIntensity.setValue(int(x)))
        self.slider_illuminationIntensity.valueChanged.connect(self.entry_illuminationIntensity.setValue)
        self.btn_autolevel.toggled.connect(self.signal_autoLevelSetting.emit)

        # layout
        grid_line1 = QHBoxLayout()

        grid_line1.addWidget(self.btn_live, 1)

        grid_line2 = QHBoxLayout()
        grid_line2.addWidget(QLabel('Exposure Time'))
        grid_line2.addWidget(self.entry_exposureTime)
        gain_label = QLabel(' Analog Gain')
        gain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_line2.addWidget(gain_label)
        grid_line2.addWidget(self.entry_analogGain)
        if show_autolevel:
            grid_line2.addWidget(self.btn_autolevel)



        grid_line0 = QHBoxLayout()
        if show_trigger_options:
            grid_line0.addWidget(QLabel('Trigger Mode'))
            grid_line0.addWidget(self.dropdown_triggerManu)
            grid_line0.addWidget(QLabel('Trigger FPS'))
            grid_line0.addWidget(self.entry_triggerFPS)

        grid_line05 = QHBoxLayout()
        show_dislpay_fps = False
        if show_display_options:
            resolution_label = QLabel('Display Resolution')
            resolution_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid_line05.addWidget(resolution_label)
            grid_line05.addWidget(self.slider_resolutionScaling)
            if show_dislpay_fps:
                grid_line05.addWidget(QLabel('Display FPS'))
                grid_line05.addWidget(self.entry_displayFPS)
            else:
                grid_line05.addWidget(self.label_resolutionScaling)

        self.grid = QVBoxLayout()
        if show_trigger_options:
            self.grid.addLayout(grid_line0)
        self.grid.addLayout(grid_line1)
        self.grid.addLayout(grid_line2)

        if not stretch:
            self.grid.addStretch()
        self.setLayout(self.grid)


    def toggle_live(self,pressed):
        if pressed:
            self.update_trigger_mode()
            self.liveController.start_live()
            self.btn_live.setText('Stop Live')
            self.signal_start_live.emit()
        else:
            self.liveController.stop_live()
            self.btn_live.setText('Start Live')

    def toggle_autolevel(self,autolevel_on):
        self.btn_autolevel.setChecked(autolevel_on)

    def update_camera_settings(self):
        self.signal_newAnalogGain.emit(self.entry_analogGain.value())
        self.signal_newExposureTime.emit(self.entry_exposureTime.value())

    def update_microscope_mode_by_name(self,current_microscope_mode_name):
        self.is_switching_mode = True
        # identify the mode selected (note that this references the object in self.configurationManager.configurations)
        self.currentConfiguration = next((config for config in self.configurationManager.configurations if config.name == current_microscope_mode_name), None)
        self.signal_live_configuration.emit(self.currentConfiguration)
        # update the microscope to the current configuration
        self.liveController.set_microscope_mode(self.currentConfiguration)
        # update the exposure time and analog gain settings according to the selected configuration
        self.entry_exposureTime.setValue(self.currentConfiguration.exposure_time)
        self.entry_analogGain.setValue(self.currentConfiguration.analog_gain)
        self.entry_illuminationIntensity.setValue(self.currentConfiguration.illumination_intensity)
        self.is_switching_mode = False

    def update_trigger_mode(self):
        self.liveController.set_trigger_mode(self.dropdown_triggerManu.currentText())

    def update_config_exposure_time(self,new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.exposure_time = new_value
            self.configurationManager.update_configuration(self.currentConfiguration.id,'ExposureTime',new_value)
            self.signal_newExposureTime.emit(new_value)

    def update_config_analog_gain(self,new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.analog_gain = new_value
            self.configurationManager.update_configuration(self.currentConfiguration.id,'AnalogGain',new_value)
            self.signal_newAnalogGain.emit(new_value)

    def update_config_illumination_intensity(self,new_value):
        if self.is_switching_mode == False:
            self.currentConfiguration.illumination_intensity = new_value
            self.configurationManager.update_configuration(self.currentConfiguration.id,'IlluminationIntensity',new_value)
            #self.liveController.set_illumination(self.currentConfiguration.illumination_source, self.currentConfiguration.illumination_intensity)

    def set_microscope_mode(self,config):
        # self.liveController.set_microscope_mode(config)
        self.dropdown_modeSelection.setCurrentText(config.name)

    def set_trigger_mode(self,trigger_mode):
        self.dropdown_triggerManu.setCurrentText(trigger_mode)
        self.liveController.set_trigger_mode(self.dropdown_triggerManu.currentText())


class RecordingWidget(QFrame):
    def __init__(self, streamHandler, imageSaver, main=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageSaver = imageSaver # for saving path control
        self.streamHandler = streamHandler
        self.base_path_is_set = False
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        self.btn_setSavingDir = QPushButton('Browse')
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon('icon/folder.png'))

        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText('Choose a base saving directory')

        self.lineEdit_savingDir.setText("D:/")
        self.imageSaver.set_base_path("D:/")
        self.base_path_is_set = True

        self.lineEdit_experimentID = QLineEdit()

        self.entry_saveFPS = QDoubleSpinBox()
        self.entry_saveFPS.setMinimum(0.02)
        self.entry_saveFPS.setMaximum(4000)
        self.entry_saveFPS.setSingleStep(1)
        self.entry_saveFPS.setValue(10)


        self.entry_timeLimit = QSpinBox()
        self.entry_timeLimit.setMinimum(-1)
        self.entry_timeLimit.setMaximum(60*60*24*30)
        self.entry_timeLimit.setSingleStep(1)
        self.entry_timeLimit.setValue(5)

        self.btn_record = QPushButton("Start Recording")
        self.btn_record.setCheckable(True)
        self.btn_record.setChecked(False)
        self.btn_record.setDefault(False)

        self.btn_record_5000fps = QPushButton("5000fps Record")
        self.btn_record_5000fps.setCheckable(True)
        self.btn_record_5000fps.setChecked(False)
        self.btn_record_5000fps.setDefault(False)

        self.btn_run_sensitivity = QPushButton("Run sensitivity script")
        self.btn_run_sensitivity.setCheckable(True)
        self.btn_run_sensitivity.setChecked(False)
        self.btn_run_sensitivity.setDefault(False)

        grid_line1 = QGridLayout()
        grid_line1.addWidget(QLabel('Saving Path'))
        grid_line1.addWidget(self.lineEdit_savingDir, 0,1)
        grid_line1.addWidget(self.btn_setSavingDir, 0,2)

        grid_line2 = QGridLayout()
        grid_line2.addWidget(QLabel('Experiment ID'), 0,0)
        grid_line2.addWidget(self.lineEdit_experimentID,0,1)

        grid_line3 = QGridLayout()
        grid_line3.addWidget(QLabel('Saving FPS'), 0,0)
        grid_line3.addWidget(self.entry_saveFPS, 0,1)
        grid_line3.addWidget(QLabel('Time Limit (s)'), 0,2)
        grid_line3.addWidget(self.entry_timeLimit, 0,3)

        self.grid = QVBoxLayout()
        self.grid.addLayout(grid_line1)
        self.grid.addLayout(grid_line2)
        self.grid.addLayout(grid_line3)
        self.grid.addWidget(self.btn_record)
        self.grid.addWidget(self.btn_record_5000fps)
        self.grid.addWidget(self.btn_run_sensitivity)
        self.setLayout(self.grid)

 

        # connections
        self.btn_setSavingDir.clicked.connect(self.set_saving_dir)
        self.btn_record.clicked.connect(self.toggle_recording)
        self.btn_record_5000fps.clicked.connect(self.toggle_recording_5000fps)
        self.btn_run_sensitivity.clicked.connect(self.run_sensitivity_script)
        self.entry_saveFPS.valueChanged.connect(self.streamHandler.set_save_fps)
        self.entry_timeLimit.valueChanged.connect(self.imageSaver.set_recording_time_limit)
        self.imageSaver.stop_recording.connect(self.stop_recording)

    def set_live_widget(self, live_widget):
        """
        Inject the LiveControlWidget so toggle_recording_5000fps can
        set trigger mode to CONTINUOUS and start/stop Live.
        """
        self.liveWidget = live_widget

    def set_camera(self, camera):
        """
        Inject the Emergent Camera instance so we can point it at the
        correct output folder before starting Continuous+Live.
        """
        self.camera = camera

    def set_saving_dir(self):
        dialog = QFileDialog()
        save_dir_base = dialog.getExistingDirectory(None, "Select Folder")
        self.imageSaver.set_base_path(save_dir_base)
        self.lineEdit_savingDir.setText(save_dir_base)
        self.base_path_is_set = True

        self.camera.set_output_path(save_dir_base)

    def toggle_recording(self,pressed):
        if self.base_path_is_set == False:
            self.btn_record.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return
        if pressed:
            self.lineEdit_experimentID.setEnabled(False)
            self.btn_setSavingDir.setEnabled(False)
            self.btn_record.setText('Stop Recording')
            self.last_exp_dir = self.imageSaver.start_new_experiment(self.lineEdit_experimentID.text())
            self.streamHandler.start_recording()
            
        else:
            self.stop_recording()

    # stop_recording can be called by imageSaver
    def stop_recording(self):
        self.lineEdit_experimentID.setEnabled(True)
        self.btn_record.setChecked(False)
        self.btn_record.setText('Start Recording')
        self.btn_record_5000fps.setChecked(False)
        self.streamHandler.stop_recording()
        self.btn_setSavingDir.setEnabled(True)
        self.camera.high_performance_recording = False

    def run_sensitivity_script(self):
        
        print(self.last_exp_dir)

        # Path to the script you attached (adjust if it's elsewhere)
        script_path = "C:/Users/user/Documents/Github/Squid_LEAP/software_simple\post_process_script/leap_bmp_postprocess_v5.py" 

        # Optional: you can pre-seed rows/cols/box/fps here if you want (script still shows the grid adjuster)
        # Example uses defaults; add flags like "--rows 8 --cols 9 --box 32 --fps 5000" if desired.
        cmd = [sys.executable, str(script_path),
            "--exp", str(self.last_exp_dir),
            "--init-quad", "auto"]  # seeds TL/TR/BL/BR at image border insets

        # Launch and let the script open its own ROI + grid UI
        try:
            subprocess.Popen(cmd)  # or run(..., check=True) if you want to wait/block
        except Exception as e:
            print("Launch failed")




    def toggle_recording_5000fps(self, pressed):
        # Require base path
        if self.base_path_is_set == False:
            self.btn_record_5000fps.setChecked(False)
            msg = QMessageBox()
            msg.setText("Please choose base saving directory first")
            msg.exec_()
            return

        if pressed:
            try:
                self.experiment_ID = self.lineEdit_experimentID.text() + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
                
                self.last_exp_dir = os.path.join(self.lineEdit_savingDir.text(),self.experiment_ID)
                os.mkdir(self.last_exp_dir)
                
                self.camera.set_output_path(self.last_exp_dir)
                self.camera.set_max_frame(5000 * self.entry_timeLimit.value()) ## TO DO
                self.camera.high_performance_recording = True
                self.camera.start_high_performance_recording() # had start_strem effect included
                print("widgets -- toggle_recording_5000fps Done")
                self.btn_record_5000fps.setChecked(False)

            except Exception as e:
                print("5000fps recording fail")
                return


        else:
            self.stop_recording()
            




class NavigationWidget(QFrame):
    def __init__(self, navigationController, slidePositionController=None, main=None, widget_configuration = 'full', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigationController = navigationController
        self.slidePositionController = slidePositionController
        self.widget_configuration = widget_configuration
        self.slide_position = None
        self.flag_click_to_move = False
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

    def add_components(self):
        x_label = QLabel('X :')
        x_label.setFixedWidth(20)
        self.label_Xpos = QLabel()
        self.label_Xpos.setNum(0)
        self.label_Xpos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dX = QDoubleSpinBox()
        self.entry_dX.setMinimum(0)
        self.entry_dX.setMaximum(25)
        self.entry_dX.setSingleStep(0.2)
        self.entry_dX.setValue(0)
        self.entry_dX.setDecimals(3)
        self.entry_dX.setSuffix(' mm')
        self.entry_dX.setKeyboardTracking(False)
        self.btn_moveX_forward = QPushButton('Forward')
        self.btn_moveX_forward.setDefault(False)
        self.btn_moveX_backward = QPushButton('Backward')
        self.btn_moveX_backward.setDefault(False)

        self.btn_home_X = QPushButton('Home X')
        self.btn_home_X.setDefault(False)
        self.btn_home_X.setEnabled(HOMING_ENABLED_X)
        self.btn_zero_X = QPushButton('Zero X')
        self.btn_zero_X.setDefault(False)

        self.checkbox_clickToMove = QCheckBox('Click to Move')
        self.checkbox_clickToMove.setChecked(False)
        self.checkbox_clickToMove.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed))

        y_label = QLabel('Y :')
        y_label.setFixedWidth(20)
        self.label_Ypos = QLabel()
        self.label_Ypos.setNum(0)
        self.label_Ypos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dY = QDoubleSpinBox()
        self.entry_dY.setMinimum(0)
        self.entry_dY.setMaximum(25)
        self.entry_dY.setSingleStep(0.2)
        self.entry_dY.setValue(0)
        self.entry_dY.setDecimals(3)
        self.entry_dY.setSuffix(' mm')

        self.entry_dY.setKeyboardTracking(False)
        self.btn_moveY_forward = QPushButton('Forward')
        self.btn_moveY_forward.setDefault(False)
        self.btn_moveY_backward = QPushButton('Backward')
        self.btn_moveY_backward.setDefault(False)

        self.btn_home_Y = QPushButton('Home Y')
        self.btn_home_Y.setDefault(False)
        self.btn_home_Y.setEnabled(HOMING_ENABLED_Y)
        self.btn_zero_Y = QPushButton('Zero Y')
        self.btn_zero_Y.setDefault(False)

        z_label = QLabel('Z :')
        z_label.setFixedWidth(20)
        self.label_Zpos = QLabel()
        self.label_Zpos.setNum(0)
        self.label_Zpos.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.entry_dZ = QDoubleSpinBox()
        self.entry_dZ.setMinimum(0)
        self.entry_dZ.setMaximum(1000)
        self.entry_dZ.setSingleStep(0.2)
        self.entry_dZ.setValue(0)
        self.entry_dZ.setDecimals(3)
        self.entry_dZ.setSuffix(' μm')
        self.entry_dZ.setKeyboardTracking(False)
        self.btn_moveZ_forward = QPushButton('Forward')
        self.btn_moveZ_forward.setDefault(False)
        self.btn_moveZ_backward = QPushButton('Backward')
        self.btn_moveZ_backward.setDefault(False)

        self.btn_home_Z = QPushButton('Home Z')
        self.btn_home_Z.setDefault(False)
        self.btn_home_Z.setEnabled(HOMING_ENABLED_Z)
        self.btn_zero_Z = QPushButton('Zero Z')
        self.btn_zero_Z.setDefault(False)

        self.btn_load_slide = QPushButton('Move To Loading Position')
        self.btn_load_slide.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        grid_line0 = QGridLayout()
        grid_line0.addWidget(x_label, 0,0)
        grid_line0.addWidget(self.label_Xpos, 0,1)
        grid_line0.addWidget(self.entry_dX, 0,2)
        grid_line0.addWidget(self.btn_moveX_forward, 0,3)
        grid_line0.addWidget(self.btn_moveX_backward, 0,4)

        grid_line0.addWidget(y_label, 1,0)
        grid_line0.addWidget(self.label_Ypos, 1,1)
        grid_line0.addWidget(self.entry_dY, 1,2)
        grid_line0.addWidget(self.btn_moveY_forward, 1,3)
        grid_line0.addWidget(self.btn_moveY_backward, 1,4)

        grid_line0.addWidget(z_label, 2,0)
        grid_line0.addWidget(self.label_Zpos, 2,1)
        grid_line0.addWidget(self.entry_dZ, 2,2)
        grid_line0.addWidget(self.btn_moveZ_forward, 2,3)
        grid_line0.addWidget(self.btn_moveZ_backward, 2,4)

        grid_line3 = QHBoxLayout()

        if self.widget_configuration == 'full':
            grid_line3.addWidget(self.btn_home_X)
            grid_line3.addWidget(self.btn_home_Y)
            grid_line3.addWidget(self.btn_home_Z)
            grid_line3.addWidget(self.btn_zero_X)
            grid_line3.addWidget(self.btn_zero_Y)
            grid_line3.addWidget(self.btn_zero_Z)
        else:
            grid_line3.addWidget(self.btn_load_slide, 1)
            grid_line3.addWidget(self.btn_home_Z, 1)
            grid_line3.addWidget(self.btn_zero_Z, 1)

        grid_line3.addWidget(self.checkbox_clickToMove, 1)

        self.grid = QVBoxLayout()
        self.grid.addLayout(grid_line0)
        self.grid.addLayout(grid_line3)
        self.setLayout(self.grid)

        self.entry_dX.valueChanged.connect(self.set_deltaX)
        self.entry_dY.valueChanged.connect(self.set_deltaY)
        self.entry_dZ.valueChanged.connect(self.set_deltaZ)

        self.btn_moveX_forward.clicked.connect(self.move_x_forward)
        self.btn_moveX_backward.clicked.connect(self.move_x_backward)
        self.btn_moveY_forward.clicked.connect(self.move_y_forward)
        self.btn_moveY_backward.clicked.connect(self.move_y_backward)
        self.btn_moveZ_forward.clicked.connect(self.move_z_forward)
        self.btn_moveZ_backward.clicked.connect(self.move_z_backward)

        self.btn_home_X.clicked.connect(self.home_x)
        self.btn_home_Y.clicked.connect(self.home_y)
        self.btn_home_Z.clicked.connect(self.home_z)
        self.btn_zero_X.clicked.connect(self.zero_x)
        self.btn_zero_Y.clicked.connect(self.zero_y)
        self.btn_zero_Z.clicked.connect(self.zero_z)

        self.checkbox_clickToMove.stateChanged.connect(self.navigationController.set_flag_click_to_move)

        self.btn_load_slide.clicked.connect(self.switch_position)
        self.btn_load_slide.setStyleSheet("background-color: #C2C2FF")

    def toggle_navigation_controls(self, started):
        if started:
            self.flag_click_to_move = self.navigationController.get_flag_click_to_move()
            self.setEnabled_all(False)
            self.checkbox_clickToMove.setChecked(False)
        else:
            self.setEnabled_all(True)
            self.checkbox_clickToMove.setChecked(self.flag_click_to_move)

    def setEnabled_all(self, enabled):
        self.checkbox_clickToMove.setEnabled(enabled)
        self.btn_home_X.setEnabled(enabled)
        self.btn_zero_X.setEnabled(enabled)
        self.btn_moveX_forward.setEnabled(enabled)
        self.btn_moveX_backward.setEnabled(enabled)
        self.btn_home_Y.setEnabled(enabled)
        self.btn_zero_Y.setEnabled(enabled)
        self.btn_moveY_forward.setEnabled(enabled)
        self.btn_moveY_backward.setEnabled(enabled)
        self.btn_home_Z.setEnabled(enabled)
        self.btn_zero_Z.setEnabled(enabled)
        self.btn_moveZ_forward.setEnabled(enabled)
        self.btn_moveZ_backward.setEnabled(enabled)
        self.btn_load_slide.setEnabled(enabled)

    def move_x_forward(self):
        self.navigationController.move_x(self.entry_dX.value())
    def move_x_backward(self):
        self.navigationController.move_x(-self.entry_dX.value())
    def move_y_forward(self):
        self.navigationController.move_y(self.entry_dY.value())
    def move_y_backward(self):
        self.navigationController.move_y(-self.entry_dY.value())
    def move_z_forward(self):
        self.navigationController.move_z(self.entry_dZ.value()/1000)
    def move_z_backward(self):
        self.navigationController.move_z(-self.entry_dZ.value()/1000)

    def set_deltaX(self,value):
        mm_per_ustep = self.navigationController.get_mm_per_ustep_X()
        deltaX = round(value/mm_per_ustep)*mm_per_ustep
        self.entry_dX.setValue(deltaX)
    def set_deltaY(self,value):
        mm_per_ustep = self.navigationController.get_mm_per_ustep_Y()
        deltaY = round(value/mm_per_ustep)*mm_per_ustep
        self.entry_dY.setValue(deltaY)
    def set_deltaZ(self,value):
        mm_per_ustep = self.navigationController.get_mm_per_ustep_Z()
        deltaZ = round(value/1000/mm_per_ustep)*mm_per_ustep*1000
        self.entry_dZ.setValue(deltaZ)

    def home_x(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_x()

    def home_y(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_y()

    def home_z(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Confirm your action")
        msg.setInformativeText("Click OK to run homing")
        msg.setWindowTitle("Confirmation")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        retval = msg.exec_()
        if QMessageBox.Ok == retval:
            self.navigationController.home_z()

    def zero_x(self):
        self.navigationController.zero_x()

    def zero_y(self):
        self.navigationController.zero_y()

    def zero_z(self):
        self.navigationController.zero_z()

    def slot_slide_loading_position_reached(self):
        self.slide_position = 'loading'
        self.btn_load_slide.setStyleSheet("background-color: #C2FFC2")
        self.btn_load_slide.setText('Move to Scanning Position')
        self.btn_moveX_forward.setEnabled(False)
        self.btn_moveX_backward.setEnabled(False)
        self.btn_moveY_forward.setEnabled(False)
        self.btn_moveY_backward.setEnabled(False)
        self.btn_moveZ_forward.setEnabled(False)
        self.btn_moveZ_backward.setEnabled(False)
        self.btn_load_slide.setEnabled(True)

    def slot_slide_scanning_position_reached(self):
        self.slide_position = 'scanning'
        self.btn_load_slide.setStyleSheet("background-color: #C2C2FF")
        self.btn_load_slide.setText('Move to Loading Position')
        self.btn_moveX_forward.setEnabled(True)
        self.btn_moveX_backward.setEnabled(True)
        self.btn_moveY_forward.setEnabled(True)
        self.btn_moveY_backward.setEnabled(True)
        self.btn_moveZ_forward.setEnabled(True)
        self.btn_moveZ_backward.setEnabled(True)
        self.btn_load_slide.setEnabled(True)

    def switch_position(self):
        if self.slide_position != 'loading':
            self.slidePositionController.move_to_slide_loading_position()
        else:
            self.slidePositionController.move_to_slide_scanning_position()
        self.btn_load_slide.setEnabled(False)

    def replace_slide_controller(self, slidePositionController):
        self.slidePositionController = slidePositionController
        self.slidePositionController.signal_slide_loading_position_reached.connect(self.slot_slide_loading_position_reached)
        self.slidePositionController.signal_slide_scanning_position_reached.connect(self.slot_slide_scanning_position_reached)
