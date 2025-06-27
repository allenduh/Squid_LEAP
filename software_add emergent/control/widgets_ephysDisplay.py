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
import re
import numpy as np

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *
import pyqtgraph as pg
from datetime import datetime
from control._def import *
import cv2
from datetime import datetime
import csv
import control.utils as utils
from scipy import signal
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque

#import skimage.io as io


class RawViewerWidget(QtWidgets.QWidget):
    def __init__(self, folder, bin_size=128, history=20, update_ms=50):
        super().__init__()

        self.folder = folder
        self.bin = bin_size
        self.history = history
        self.update_ms = update_ms

        # Load and bin all frames
        self.binned_frames = self.load_and_bin_raw()
        self.total_frames, self.H, self.W = self.binned_frames.shape

        # Layout for live plots
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plot_grid = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.plot_grid)

        self.buffers = [[deque(maxlen=self.history) for _ in range(self.W)] for _ in range(self.H)]
        self.curves = [[None]*self.W for _ in range(self.H)]

        for i in range(self.H):
            for j in range(self.W):
                p = self.plot_grid.addPlot(row=i, col=j)
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.setXRange(0, self.history-1, padding=0)
                p.setYRange(0, 255)
                p.hideAxis("bottom")
                p.hideAxis("left")
                self.curves[i][j] = p.plot(pen=pg.mkPen("#00e200", width=1))

        # Start playback timer
        self.frame_idx = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.update_ms)

    def load_and_bin_raw(self):
        height = 608
        width = 1024
        raw_files = sorted(glob.glob(os.path.join(self.folder, "*.raw")))
        frames = []

        for file in raw_files:
            with open(file, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            num_frames = data.size // (height * width)
            if num_frames == 0:
                continue
            frame_array = data[:num_frames * height * width].reshape((num_frames, height, width))
            frames.append(frame_array)

        all_frames = np.concatenate(frames, axis=0)  # shape: (T, H, W)
        T = all_frames.shape[0]
        H_b = (height // self.bin) * self.bin
        W_b = (width // self.bin) * self.bin
        all_frames = all_frames[:, :H_b, :W_b]

        binned = all_frames.reshape(
            T, H_b // self.bin, self.bin,
            W_b // self.bin, self.bin
        ).mean(axis=(2, 4))  # shape: (T, H_binned, W_binned)

        return binned

    def update_frame(self):
        if self.frame_idx >= self.total_frames:
            print("Done replaying all frames.")
            self.timer.stop()
            return

        frame = self.binned_frames[self.frame_idx]
        for i in range(self.H):
            for j in range(self.W):
                buf = self.buffers[i][j]
                buf.append(frame[i, j])
                self.curves[i][j].setData(range(len(buf)), list(buf))

        self.frame_idx += 1

class ROIControlWidget(QFrame):

    trace_ROI = Signal(np.ndarray)
    image_to_display = Signal(np.ndarray)
    R_ROI_to_display = Signal(np.ndarray, np.ndarray)
    multi_channel_ephys = Signal(str, np.ndarray)
    iv_dict_to_display = Signal(dict)
    clear_ROI_signal = Signal()

    def __init__(self, imageDisplay, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.imageDisplay = imageDisplay
        
        self.current_folder = ''
        self.last_exp_saving_path = ''
        self.demo_folder_path = '/Users/allen/Downloads/09272023_Brainslice/brain5_Add_drug_during_recording_30s_Add_acetone_2023-09-26_18-40-55.106836'
        self.sorted_filenames = []
        self.num_of_files= 0
        self.R_base = None

        self.ephys_unbinned = None
        self.ephys_binned = None
        self.full_pixels_x = 0
        self.full_pixels_y = 0



        self.ephys = None
  
        self.add_components()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)


    def add_components(self):
        self.btn_setSavingDir = QPushButton('Browse')
        self.btn_setSavingDir.setDefault(False)
        self.btn_setSavingDir.setIcon(QIcon('icon/folder.png'))
        
        self.lineEdit_savingDir = QLineEdit()
        self.lineEdit_savingDir.setReadOnly(True)
        self.lineEdit_savingDir.setText('User select folder / or last exp')

        # Operation Buttons
        self.button_load_unbinned = QPushButton("Load")
        self.button_load_unbinned.setDefault(False)

        self.button_bin_ephys = QPushButton("Bin")
        self.button_bin_ephys.setDefault(False)

        self.button_load_binned = QPushButton("Load + Bin")
        self.button_load_binned.setDefault(False)

        self.button_plot_ephys = QPushButton("Plot ephys")
        self.button_plot_ephys.setDefault(False)

        self.button_band_pass = QPushButton("Band pass")
        self.button_band_pass.setDefault(False)

        self.button_remove_artifact = QPushButton("Remove artifact")
        self.button_remove_artifact.setDefault(False)

        self.button_process_all = QPushButton("Process all")
        self.button_process_all.setDefault(False)

        # Bin range and num of bin
        self.x_binned = QSpinBox()
        self.x_binned.setMinimum(0) 
        self.x_binned.setMaximum(100) 
        self.x_binned.setSingleStep(1)
        self.x_binned.setValue(10)
        

        self.y_binned = QSpinBox()
        self.y_binned.setMinimum(0) 
        self.y_binned.setMaximum(50) 
        self.y_binned.setSingleStep(1)
        self.y_binned.setValue(2)

        self.frame_show = QSpinBox()
        self.frame_show.setMinimum(0) 
        self.frame_show.setMaximum(100000) 
        self.frame_show.setSingleStep(1)
        self.frame_show.setValue(0)

        self.pixels_x_binned = QLabel()
        self.pixels_x_binned.setNum(0)
        self.pixels_x_binned.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.pixels_y_binned = QLabel()
        self.pixels_y_binned.setNum(0)
        self.pixels_y_binned.setFrameStyle(QFrame.Panel | QFrame.Sunken)


        # Time component (Band pass, frame index) 
        self.frame_index_min = QSpinBox()
        self.frame_index_min.setMinimum(0) 
        self.frame_index_min.setMaximum(10000000) 
        self.frame_index_min.setSingleStep(1)
        self.frame_index_min.setValue(0)

        self.frame_index_max = QSpinBox()
        self.frame_index_max.setMinimum(0) 
        self.frame_index_max.setMaximum(10000000) 
        self.frame_index_max.setSingleStep(1)
        self.frame_index_max.setValue(0)


        self.bandpass_low = QSpinBox()
        self.bandpass_low.setMinimum(0)  # (Unit Hz)
        self.bandpass_low.setMaximum(1000) 
        self.bandpass_low.setSingleStep(1)
        self.bandpass_low.setValue(100)

        self.bandpass_high = QSpinBox()
        self.bandpass_high.setMinimum(0) 
        self.bandpass_high.setMaximum(10000) 
        self.bandpass_high.setSingleStep(1)
        self.bandpass_high.setValue(750)

        self.frame_rate = QSpinBox()
        self.frame_rate.setMinimum(0) 
        self.frame_rate.setMaximum(100000) 
        self.frame_rate.setSingleStep(1)
        self.frame_rate.setValue(0)

        self.frame_total = QLabel()
        self.frame_total.setNum(0)
        self.frame_total.setFrameStyle(QFrame.Panel | QFrame.Sunken)


        # Layout --
        grid_line0 = QGridLayout()
        grid_line0.addWidget(QLabel('Load folder...'))
        grid_line0.addWidget(self.lineEdit_savingDir, 0,1)
        grid_line0.addWidget(self.btn_setSavingDir, 0,2)

        grid_line1 = QHBoxLayout()
        grid_line1.addWidget(self.button_load_unbinned)
        grid_line1.addWidget(self.button_bin_ephys)
        grid_line1.addWidget(self.button_load_binned)
        grid_line1.addWidget(self.button_band_pass)
        grid_line1.addWidget(self.button_plot_ephys)
        grid_line1.addWidget(self.button_process_all)
        

        grid_line2 = QHBoxLayout()
        grid_line2.addWidget(QLabel('x binned')) 
        grid_line2.addWidget(self.x_binned)
        grid_line2.addWidget(QLabel('y binned'))
        grid_line2.addWidget(self.y_binned)
        grid_line2.addWidget(QLabel('Pixels x'))
        grid_line2.addWidget(self.pixels_x_binned)
        grid_line2.addWidget(QLabel('Pixels y'))
        grid_line2.addWidget(self.pixels_y_binned)
        grid_line2.addWidget(QLabel('Frame to show'))
        grid_line2.addWidget(self.frame_show)

        grid_line3 = QHBoxLayout()
        grid_line3.addWidget(QLabel('Frame start'))
        grid_line3.addWidget(self.frame_index_min)
        grid_line3.addWidget(QLabel('Frame end'))
        grid_line3.addWidget(self.frame_index_max)
        grid_line3.addWidget(QLabel('Frame total'))
        grid_line3.addWidget(self.frame_total)
        grid_line3.addWidget(QLabel('Low pass'))
        grid_line3.addWidget(self.bandpass_low)
        grid_line3.addWidget(QLabel('High pass'))
        grid_line3.addWidget(self.bandpass_high)
        grid_line3.addWidget(QLabel('Frame rate'))
        grid_line3.addWidget(self.frame_rate)
        

        # layout
        self.grid = QGridLayout()
        self.grid.addLayout(grid_line0, 0, 0)
        self.grid.addLayout(grid_line1, 1, 0)
        self.grid.addLayout(grid_line2, 2, 0)
        self.grid.addLayout(grid_line3, 3, 0)
        self.setLayout(self.grid)


        # connections
        self.btn_setSavingDir.clicked.connect(self.set_user_load_dir)
        self.button_load_unbinned.clicked.connect(self.load_unbinned)
        self.button_load_binned.clicked.connect(self.load_binned)
        self.button_bin_ephys.clicked.connect(self.bin_ephys)
        self.button_plot_ephys.clicked.connect(self.emit_multi_channel)
        self.button_band_pass.clicked.connect(self.band_pass)
        self.button_remove_artifact.clicked.connect(self.remove_artifact)
        self.button_process_all.clicked.connect(self.process_all)
        self.frame_show.valueChanged.connect(self.change_frame)
        self.x_binned.valueChanged.connect(self.binning_change_clicked)
        self.y_binned.valueChanged.connect(self.binning_change_clicked)

        #self.initialize_load_images() DEBUG
        
    #########
    ## Function of buttons effect / request from other widget
    #########
    def set_user_load_dir(self):
        dialog = QFileDialog()
        user_load_dir_base = dialog.getExistingDirectory(None, "Select Folder", DEFAULT_SAVING_PATH)
        self.update_load_file_path(user_load_dir_base)
        self.initialize_load_images()
        
    
    def initialize_load_images(self):
        # Step 1: Collect filenames
        self.current_folder = self.decide_folder_to_read()

        all_filenames = utils.collect_filenames(self.current_folder) 
        self.sorted_filenames = utils.sort_filenames_by_last_digit(all_filenames)

        self.num_of_files = len(self.sorted_filenames)

        # Step 2: Read one image 
        if self.num_of_files:
            self.first_img = utils.read_one_image(filename_to_read = self.sorted_filenames[0])
            self.image_to_display.emit(self.first_img) 
            self.full_pixels_y, self.full_pixels_x = self.first_img.shape

            # Step 3: Update class parameter
            # update image shape
            if self.full_pixels_y == 10: # height = 40 before bin
                self.frame_rate.setValue(1586) 
            if self.full_pixels_y == 30: # height = 30
                self.frame_rate.setValue(1788) 
            # update frame total
            self.frame_total.setNum(self.num_of_files)
            self.frame_show.setMaximum(self.num_of_files) 
            # update binning size    
            self.binning_change_clicked()
        

    def change_frame(self):
        frame = utils.read_one_image(filename_to_read = self.sorted_filenames[self.frame_show.value()])
        self.image_to_display.emit(frame - self.first_img + 128) 

    def load_unbinned(self):
        self.initialize_load_images()
        # Step 3: Read images
        self.ephys_unbinned = utils.read_images(self.sorted_filenames,
                                                frame_start=self.frame_index_min.value(), 
                                                frame_end=self.frame_index_max.value(), 
                                                desired_y_after_binning=self.full_pixels_y, 
                                                desired_x_after_binning=self.full_pixels_x)

        self.R_base = self.ephys_unbinned[0, :, :]
        print("Read in " + str(self.ephys_unbinned.shape[0]) + " unbinned files")

      

    def load_binned(self):   
        self.initialize_load_images()     
        # Step 3: Read images
        self.ephys_binned = utils.read_images(self.sorted_filenames,
                                                frame_start=self.frame_index_min.value(), 
                                                frame_end=self.frame_index_max.value(), 
                                                desired_y_after_binning=self.y_binned.value(), 
                                                desired_x_after_binning=self.x_binned.value())

        print("Read in " + str(self.ephys_binned.shape[0]) + " binned files")


    def bin_ephys(self):
        start_time = datetime.now()
        self.ephys_binned = utils.bin_image_full(self.ephys_unbinned, desired_y_after_binning=self.y_binned.value(), 
                                               desired_x_after_binning=self.x_binned.value())

        end_time = datetime.now()
        print('Binned all files in: {}'.format(end_time - start_time))


    def emit_multi_channel(self): 
        if not self.ephys:
            self.load_binned()
            self.band_pass()
        self.multi_channel_ephys.emit(self.pdf_filename, self.ephys)


    def band_pass(self): 
        # offset
        self.ephys = self.ephys_binned - self.ephys_binned[0, :, :]
        lowpass = self.bandpass_low.value()
        highpass = self.bandpass_high.value()
        frame_index_min = self.frame_index_min.value()
        frame_index_max = self.frame_index_max.value()
        fps = self.frame_rate.value()
        # band pass
        sos = signal.butter(1, [lowpass, highpass], 'band', fs=fps, output='sos')
        self.ephys = signal.sosfilt(sos, self.ephys, axis=0)

        # Step 4: Plot_multichannel_ephys
        folder_name = os.path.basename(self.current_folder)
        parent_directory = os.path.dirname(self.current_folder)

        self.pdf_filename = os.path.join(parent_directory, f"{folder_name}_[{lowpass}_{highpass}]_[{frame_index_min}_{frame_index_max}].pdf")
        utils.plot_multichannel_ephys(self.ephys, pdf_filename=self.pdf_filename + '_ephys.pdf')

    def remove_artifact(self):
        self.ephys

    def process_all(self): 

        dialog = QFileDialog()
        parent_path = dialog.getExistingDirectory(None, "Select Folder", DEFAULT_SAVING_PATH)

        pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
        folder_list = [folder for folder in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, folder)) if pattern.search(folder)]
        folder_list = sorted(folder_list, key=utils.sort_key_from_date_and_time)
        print(folder_list)

        for folder in folder_list:
            try:
                self.current_folder = os.path.join(parent_path, folder) 
                self.initialize_load_images()

                
                self.frame_index_max.setValue(self.num_of_files)   # Load all

                # if self.full_pixels_x < 300: # small image
                #     self.load_unbinned()
                #     self.bin_ephys()
                # else: # Large image need to bin when loading
                #     self.load_binned()

                self.load_binned()
                self.band_pass()
                self.emit_multi_channel()
            except:
                print(folder + " has problem")


    # Load -> Bin -> filter -> plot
    def quick_data_process(self):
        self.load_binned()
        self.emit_multi_channel()
        

    def update_load_file_path(self, path_user_load): 
        self.current_folder = path_user_load
        self.lineEdit_savingDir.setText(path_user_load)
        self.base_path_is_set = True


    def update_last_exp_file_path(self, saving_path): 
        self.last_exp_saving_path = saving_path
        self.lineEdit_savingDir.setText(self.last_exp_saving_path)
        self.base_path_is_set = True


    #########
    ## Helper Function 
    #########
    def decide_folder_to_read(self):
    # Determine the folder path to read images
        if bool(self.current_folder): 
            print("Use user selected folder = " + self.current_folder)
            return self.current_folder
        elif bool(self.last_exp_saving_path): 
            print("Use last exp saving folder" + self.last_exp_saving_path)
            return self.last_exp_saving_path
        else:
            print("Use demo example folder" + self.demo_folder_path)
            return self.demo_folder_path

    def binning_change_clicked(self):
        self.pixels_x_binned.setNum(int(np.ceil((self.full_pixels_x / self.x_binned.value()))))
        self.pixels_y_binned.setNum(int(np.ceil((self.full_pixels_y / self.y_binned.value()))))



class Display(QFrame):

    def __init__(self, imageDisplay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setting up the frame style, similar to your old code
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.imageDisplay = imageDisplay

        # Create a GraphicsLayoutWidget for the ephysDisplay
        self.ephysDisplay = pg.GraphicsLayoutWidget(self)  # Set the parent widget as self

        # Set up the grid of PlotWidgets (3x10)
        rows, cols = 2, 10
        for i in range(rows):
            for j in range(cols):
                # Generate random data (replace this with your actual data)
                x = np.linspace(0, 10, 100)
                y = np.random.random(100)/ 100 + (i * cols + j)  # Offset each plot for clarity

                # Create a plot
                plot = self.ephysDisplay.addPlot(row=i, col=j)
                plot.plot(x, y)

   
        layout = QGridLayout(self)  # Set the parent widget as self
        layout.addWidget(self.ephysDisplay, 0, 0)  # Add the ephysDisplay to the layout
        self.setLayout(layout)

    def plot_multichannel_ephys(self, filename, ephys):
        """ Update ephysDisplay using ephys i.e. shape (100, 11, 17)"""
        # Clear all items from ephysDisplay
        self.ephysDisplay.clear()

        # Create a time axis for the plots
        num_samples = ephys.shape[0]
        x = np.linspace(0, num_samples, num_samples)

        # Get the shape of the ephys data
        rows, cols = ephys.shape[1], ephys.shape[2]

        # Iterate over the data and set up the plots
        for i in range(rows):
            for j in range(cols):
                y = ephys[:, i, j]  # Get the data for this channel

                # Create a plot for this location
                plot = self.ephysDisplay.addPlot(row=i, col=j)
                plot.plot(x, y, pen={'width': 0.1})
        utils.plt_plot_multichannel_ephys(filename + '_map.pdf', ephys)
