# set QT_API environment variable
import os
import sys

# qt libraries
os.environ["QT_API"] = "pyqt5"
import qtpy
import pyqtgraph as pg
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

# control
from control._def import *



try:
    from control.multipoint_custom_script_entry_v2 import *
    print('custom multipoint script found')
except:
    pass

from queue import Queue
from threading import Thread, Lock
from pathlib import Path
from datetime import datetime
import time
import subprocess
import shutil
from lxml import etree
import json
import math
import random
import numpy as np
import pandas as pd
import scipy.signal
import cv2
import imageio as iio


class StreamHandler(QObject):

    image_to_display = Signal(np.ndarray)
    packet_image_to_write = Signal(np.ndarray, int, float)

    def __init__(self,crop_width=Acquisition.CROP_WIDTH,crop_height=Acquisition.CROP_HEIGHT,display_resolution_scaling=1):
        QObject.__init__(self)
        self.fps_display = 1
        self.fps_save = 10
        self.fps_track = 1
        self.timestamp_last_display = 0
        self.timestamp_last_save = 0
        self.timestamp_last_track = 0

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.display_resolution_scaling = display_resolution_scaling

        self.save_image_flag = False
        self.track_flag = False
        self.handler_busy = False

        # for fps measurement
        self.timestamp_last = time.perf_counter()
        self.counter = 0
        self.fps_real = 0
        self.wrte_idx = 0

    def start_recording(self):
        self.save_image_flag = True

    def stop_recording(self):
        self.save_image_flag = False

    def start_tracking(self):
        self.tracking_flag = True

    def stop_tracking(self):
        self.tracking_flag = False

    def set_display_fps(self,fps):
        self.fps_display = fps

    def set_save_fps(self,fps):
        self.fps_save = fps

    def set_crop(self,crop_width,crop_height):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling/100
        print(self.display_resolution_scaling)

    def on_new_frame(self, camera):   # ToDo not take whole camera

        if camera.is_live:
            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter+1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                print('real camera fps is ' + str(self.fps_real))

            time_now = time.perf_counter()

            # send image to display
            if time_now-self.timestamp_last_display >= 1/self.fps_display:
                # self.image_to_display.emit(cv2.resize(image_cropped,(round(self.crop_width*self.display_resolution_scaling), round(self.crop_height*self.display_resolution_scaling)),cv2.INTER_LINEAR))
                self.image_to_display.emit(camera.current_frame)
                self.timestamp_last_display = time_now

            # send image to write (using index)
            camera_fps = 5000
            save_interval = int(camera_fps/ self.fps_save)  # e.g., 5000 / 100 = 50

            if self.save_image_flag and self.wrte_idx % save_interval == 0:
                self.packet_image_to_write.emit(camera.current_frame, camera.frame_ID, camera.timestamp)
                print(f"t = {camera.timestamp:.6f}s  ID = {camera.frame_ID:.0f}")
            self.wrte_idx += 1



class ImageSaver(QObject):

    stop_recording = Signal()

    def __init__(self,image_format=Acquisition.IMAGE_FORMAT):
        QObject.__init__(self)
        self.base_path = 'D:'
        self.experiment_ID = ''
        self.image_format = image_format
        self.max_num_image_per_folder = 1000
        self.queue = Queue(1000) # max 10 items in the queue
        self.image_lock = Lock()
        self.stop_signal_received = False
        self.thread = Thread(target=self.process_queue)
        self.thread.start()
        self.counter = 0
        self.recording_start_time = 0
        self.recording_time_limit = 5

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image,frame_ID,timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                folder_ID = int(self.counter/self.max_num_image_per_folder)
                file_ID = int(self.counter%self.max_num_image_per_folder)
                # create a new folder
                if file_ID == 0:
                    os.mkdir(os.path.join(self.base_path,self.experiment_ID,str(folder_ID)))

                if image.dtype == np.uint16:
                    # need to use tiff when saving 16 bit images
                    saving_path = os.path.join(self.base_path,self.experiment_ID,str(folder_ID),str(file_ID) + '_' + str(frame_ID) + '.tiff')
                    iio.imwrite(saving_path,image)
                else:
                    saving_path = os.path.join(self.base_path,self.experiment_ID,str(folder_ID),str(file_ID) + '_' + str(frame_ID) + '.' + self.image_format)
                    cv2.imwrite(saving_path, image)


                self.counter = self.counter + 1
                self.queue.task_done()
                self.image_lock.release()
            except:
                pass

    def enqueue(self,image,frame_ID,timestamp):
        try:
            self.queue.put_nowait([image,frame_ID,timestamp])
            if ( self.recording_time_limit>0 ) and ( time.time()-self.recording_start_time >= self.recording_time_limit ):
                self.stop_recording.emit()
            # when using self.queue.put(str_), program can be slowed down despite multithreading because of the block and the GIL
        except:
            print('imageSaver queue is full, image discarded')

    def set_base_path(self,path):
        self.base_path = path

    def set_recording_time_limit(self,time_limit):
        self.recording_time_limit = time_limit

    def start_new_experiment(self,experiment_ID,add_timestamp=True):
        if add_timestamp:
            # generate unique experiment ID
            self.experiment_ID = experiment_ID + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        else:
            self.experiment_ID = experiment_ID
        self.recording_start_time = time.time()
        # create a new folder
        try:
            exp_dir = os.path.join(self.base_path,self.experiment_ID)
            os.mkdir(exp_dir)
            # to do: save configuration
        except:
            pass
        # reset the counter
        self.counter = 0
        return exp_dir

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()



class ImageDisplay(QObject):

    image_to_display = Signal(np.ndarray)

    def __init__(self):
        QObject.__init__(self)
        self.queue = Queue(10) # max 10 items in the queue
        self.image_lock = Lock()
        self.stop_signal_received = False
        self.thread = Thread(target=self.process_queue)
        self.thread.start()

    def process_queue(self):
        while True:
            # stop the thread if stop signal is received
            if self.stop_signal_received:
                return
            # process the queue
            try:
                [image,frame_ID,timestamp] = self.queue.get(timeout=0.1)
                self.image_lock.acquire(True)
                self.image_to_display.emit(image)
                self.image_lock.release()
                self.queue.task_done()
            except:
                pass

    # def enqueue(self,image,frame_ID,timestamp):
    def enqueue(self,image):
        try:
            self.queue.put_nowait([image,None,None])
            # when using self.queue.put(str_) instead of try + nowait, program can be slowed down despite multithreading because of the block and the GIL
            pass
        except:
            print('imageDisplay queue is full, image discarded')

    def emit_directly(self,image):
        self.image_to_display.emit(image)

    def close(self):
        self.queue.join()
        self.stop_signal_received = True
        self.thread.join()

class Configuration:
    def __init__(self,mode_id=None,name=None,color=None,camera_sn=None,exposure_time=None,analog_gain=None,illumination_source=None,illumination_intensity=None,z_offset=None,pixel_format=None,_pixel_format_options=None,emission_filter_position=None):
        self.id = mode_id
        self.name = name
        self.color = color
        self.exposure_time = exposure_time
        self.analog_gain = analog_gain
        self.illumination_source = illumination_source
        self.illumination_intensity = illumination_intensity
        self.camera_sn = camera_sn
        self.z_offset = z_offset
        self.pixel_format = pixel_format
        if self.pixel_format is None:
            self.pixel_format = "default"
        self._pixel_format_options = _pixel_format_options
        if _pixel_format_options is None:
            self._pixel_format_options = self.pixel_format
        self.emission_filter_position = emission_filter_position


class LiveController(QObject):
    def __init__(self,camera,microcontroller,parent=None,control_illumination=True,use_internal_timer_for_hardware_trigger=True,for_displacement_measurement=False):
        QObject.__init__(self)
        self.microscope = parent
        self.camera = camera
        self.microcontroller = microcontroller
        self.currentConfiguration = None
        self.trigger_mode = TriggerMode.SOFTWARE # @@@ change to None
        self.control_illumination = control_illumination
        self.illumination_on = False
        self.use_internal_timer_for_hardware_trigger = use_internal_timer_for_hardware_trigger # use QTimer vs timer in the MCU
        self.for_displacement_measurement = for_displacement_measurement

        self.fps_trigger = 10
        self.timer_trigger_interval = (1/self.fps_trigger)*1000

        self.timer_trigger = QTimer()
        self.timer_trigger.setInterval(int(self.timer_trigger_interval))
        self.timer_trigger.timeout.connect(self.trigger_acquisition)

        self.trigger_ID = -1

        self.fps_real = 0
        self.counter = 0
        self.timestamp_last = 0

        self.display_resolution_scaling = DEFAULT_DISPLAY_CROP/100

        self.enable_channel_auto_filter_switching = True


    # illumination control
    def turn_on_illumination(self):
        self.microcontroller.turn_on_illumination()
        self.illumination_on = True

    def turn_off_illumination(self):
        self.microcontroller.turn_off_illumination()
        self.illumination_on = False

    def start_live(self):
        self.camera.start_streaming()
        self.camera.is_live = True
        if self.trigger_mode == TriggerMode.SOFTWARE:
            self.camera.start_cont_acquisition_thread()

        if self.trigger_mode == TriggerMode.CONTINUOUS:
            self.camera.start_high_performance_recording() 

    def stop_live(self):
        if self.camera.is_live == True:
            if self.trigger_mode == TriggerMode.SOFTWARE:
                self._stop_triggerred_acquisition()
                self.camera.stop_streaming()
                
            if self.trigger_mode == TriggerMode.CONTINUOUS:
                self.camera.stop_streaming()

            self.camera.is_live = False

            if self.control_illumination:
                self.turn_off_illumination()

    # software trigger related
    def trigger_acquisition(self):
        if self.trigger_mode == TriggerMode.SOFTWARE:
            if self.control_illumination and self.illumination_on == False:
                self.turn_on_illumination()
            self.trigger_ID = self.trigger_ID + 1
            self.camera.send_trigger()
            # measure real fps
            timestamp_now = round(time.time())
            if timestamp_now == self.timestamp_last:
                self.counter = self.counter+1
            else:
                self.timestamp_last = timestamp_now
                self.fps_real = self.counter
                self.counter = 0
                # print('real trigger fps is ' + str(self.fps_real))
        elif self.trigger_mode == TriggerMode.HARDWARE:
            self.trigger_ID = self.trigger_ID + 1
            if ENABLE_NL5 and NL5_USE_DOUT:
                self.microscope.nl5.start_acquisition()
            else:
                self.microcontroller.send_hardware_trigger(control_illumination=True,illumination_on_time_us=self.camera.exposure_time*1000)

    def _start_triggerred_acquisition(self):
        self.timer_trigger.start()

    def _set_trigger_fps(self,fps_trigger):
        self.fps_trigger = fps_trigger
        self.timer_trigger_interval = (1/self.fps_trigger)*1000
        self.timer_trigger.setInterval(int(self.timer_trigger_interval))

    def _stop_triggerred_acquisition(self):
        self.timer_trigger.stop()

    # trigger mode and settings
    def set_trigger_mode(self,mode):
        if mode == TriggerMode.SOFTWARE:
            self.camera.set_software_triggered_acquisition()
            self._start_triggerred_acquisition()

        if mode == TriggerMode.CONTINUOUS:
            if ( self.trigger_mode == TriggerMode.SOFTWARE ) or ( self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger ):
                self._stop_triggerred_acquisition()
            self.camera.set_continuous_acquisition()
        self.trigger_mode = mode

    def set_trigger_fps(self,fps):
        if ( self.trigger_mode == TriggerMode.SOFTWARE ) or ( self.trigger_mode == TriggerMode.HARDWARE and self.use_internal_timer_for_hardware_trigger ):
            self._set_trigger_fps(fps)

    # set microscope mode
    # @@@ to do: change softwareTriggerGenerator to TriggerGeneratror
    def set_microscope_mode(self,configuration):

        self.currentConfiguration = configuration
        print("setting microscope mode to " + self.currentConfiguration.name)

        # temporarily stop live while changing mode

        self.timer_trigger.stop()
        if self.control_illumination:
            self.turn_off_illumination()

        # set camera exposure time and analog gain
        self.camera.set_exposure_time(self.currentConfiguration.exposure_time)
        self.camera.set_analog_gain(self.currentConfiguration.analog_gain)

        # set illumination
        # if self.control_illumination:
        #     self.set_illumination(self.currentConfiguration.illumination_source,self.currentConfiguration.illumination_intensity)

        # restart live

        if self.control_illumination:
            self.turn_on_illumination()
        self.timer_trigger.start()

    def get_trigger_mode(self):
        return self.trigger_mode

    # slot
    def on_new_frame(self):
        if self.fps_trigger <= 5:
            if self.control_illumination and self.illumination_on == True:
                self.turn_off_illumination()

    def set_display_resolution_scaling(self, display_resolution_scaling):
        self.display_resolution_scaling = display_resolution_scaling/100

    def reset_strobe_arugment(self):
        # re-calculate the strobe_delay_us value
        try:
            self.camera.calculate_hardware_trigger_arguments()
        except AttributeError:
            pass
        self.microcontroller.set_strobe_delay_us(self.camera.strobe_delay_us)


class NavigationController(QObject):

    xPos = Signal(float)
    yPos = Signal(float)
    zPos = Signal(float)
    new_zPos_mm = Signal(float)
    thetaPos = Signal(float)
    xyPos = Signal(float,float)
    signal_joystick_button_pressed = Signal()

    # x y z axis pid enable flag
    pid_enable_flag = [False, False, False]


    def __init__(self,microcontroller, parent=None):
        # parent should be set to OctopiGUI instance to enable updates
        # to camera settings, e.g. binning, that would affect click-to-move
        QObject.__init__(self)
        self.microcontroller = microcontroller
        self.parent = parent
        
        self.x_pos_mm = 0
        self.y_pos_mm = 0
        self.z_pos_mm = 0
        self.z_pos = 0
        self.theta_pos_rad = 0
        self.x_microstepping = MICROSTEPPING_DEFAULT_X
        self.y_microstepping = MICROSTEPPING_DEFAULT_Y
        self.z_microstepping = MICROSTEPPING_DEFAULT_Z
        self.click_to_move = False
        self.theta_microstepping = MICROSTEPPING_DEFAULT_THETA
        self.enable_joystick_button_action = True

        # to be moved to gui for transparency
        self.microcontroller.set_callback(self.update_pos)

        # self.timer_read_pos = QTimer()
        # self.timer_read_pos.setInterval(PosUpdate.INTERVAL_MS)
        # self.timer_read_pos.timeout.connect(self.update_pos)
        # self.timer_read_pos.start()

        # scan start position
        self.scan_begin_position_x = 0
        self.scan_begin_position_y = 0
    
    def get_mm_per_ustep_X(self):
        return SCREW_PITCH_X_MM/(self.x_microstepping*FULLSTEPS_PER_REV_X)

    def get_mm_per_ustep_Y(self):
        return SCREW_PITCH_Y_MM/(self.y_microstepping*FULLSTEPS_PER_REV_Y)

    def get_mm_per_ustep_Z(self):
        return SCREW_PITCH_Z_MM/(self.z_microstepping*FULLSTEPS_PER_REV_Z)

    def set_flag_click_to_move(self, flag):
        self.click_to_move = flag

    def get_flag_click_to_move(self):
        return self.click_to_move

    def scan_preview_move_from_click(self, click_x, click_y, image_width, image_height, Nx=1, Ny=1, dx_mm=0.9, dy_mm=0.9):
        """
        napariTiledDisplay uses the Nx, Ny, dx_mm, dy_mm fields to move to the correct fov first
        imageArrayDisplayWindow assumes only a single fov (default values do not impact calculation but this is less correct)
        """
        # check if click to move enabled
        if not self.click_to_move:
            print("allow click to move")
            return
        # restore to raw coordicate
        click_x = image_width / 2.0 + click_x
        click_y = image_height / 2.0 - click_y
        print("click - (x, y):", (click_x, click_y))
        fov_col = click_x * Nx // image_width
        fov_row = click_y * Ny // image_height
        end_position_x = Ny % 2 # right side or left side
        fov_col = Nx - (fov_col + 1) if end_position_x else fov_col
        fov_row = fov_row
        pixel_sign_x = (-1)**end_position_x # inverted
        pixel_sign_y = -1 if INVERTED_OBJECTIVE else 1

        # move to selected fov
        self.move_x_to(self.scan_begin_position_x+dx_mm*fov_col*pixel_sign_x)
        self.microcontroller.wait_till_operation_is_completed()
        self.move_y_to(self.scan_begin_position_y+dy_mm*fov_row*pixel_sign_y)
        self.microcontroller.wait_till_operation_is_completed()

        # move to actual click, offset from center fov
        tile_width = (image_width / Nx) * PRVIEW_DOWNSAMPLE_FACTOR
        tile_height = (image_height / Ny) * PRVIEW_DOWNSAMPLE_FACTOR
        offset_x = (click_x * PRVIEW_DOWNSAMPLE_FACTOR) % tile_width
        offset_y = (click_y * PRVIEW_DOWNSAMPLE_FACTOR) % tile_height
        offset_x_centered = int(offset_x - tile_width / 2)
        offset_y_centered = int(tile_height / 2 - offset_y)
        self.move_from_click(offset_x_centered, offset_y_centered, tile_width, tile_height)

    def move_from_click(self, click_x, click_y, image_width, image_height):
        if self.click_to_move:
            pixel_size_um = self.objectiveStore.get_pixel_size()

            pixel_sign_x = 1
            pixel_sign_y = 1 if INVERTED_OBJECTIVE else -1

            delta_x = pixel_sign_x * pixel_size_um * click_x / 1000.0
            delta_y = pixel_sign_y * pixel_size_um * click_y / 1000.0

            self.move_x(delta_x)
            self.microcontroller.wait_till_operation_is_completed()
            self.move_y(delta_y)
            self.microcontroller.wait_till_operation_is_completed()

    def move_from_click_mosaic(self, x_mm, y_mm):
        if self.click_to_move:
            self.move_to(x_mm, y_mm)

    def move_to_cached_position(self):
        if not os.path.isfile("cache/last_coords.txt"):
            return
        with open("cache/last_coords.txt","r") as f:
            for line in f:
                try:
                    x,y,z = line.strip("\n").strip().split(",")
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    self.move_to(x,y)
                    self.move_z_to(z)
                    break
                except:
                    pass
                break

    def cache_current_position(self):
        with open("cache/last_coords.txt","w") as f:
            f.write(",".join([str(self.x_pos_mm),str(self.y_pos_mm),str(self.z_pos_mm)]))

    def move_x(self,delta):
        self.microcontroller.move_x_usteps(int(delta/self.get_mm_per_ustep_X()))

    def move_y(self,delta):
        self.microcontroller.move_y_usteps(int(delta/self.get_mm_per_ustep_Y()))

    def move_z(self,delta):
        self.microcontroller.move_z_usteps(int(delta/self.get_mm_per_ustep_Z()))

    def move_x_to(self,delta):
        self.microcontroller.move_x_to_usteps(STAGE_MOVEMENT_SIGN_X*int(delta/self.get_mm_per_ustep_X()))

    def move_y_to(self,delta):
        self.microcontroller.move_y_to_usteps(STAGE_MOVEMENT_SIGN_Y*int(delta/self.get_mm_per_ustep_Y()))

    def move_z_to(self,delta):
        self.microcontroller.move_z_to_usteps(STAGE_MOVEMENT_SIGN_Z*int(delta/self.get_mm_per_ustep_Z()))

    def move_x_usteps(self,usteps):
        self.microcontroller.move_x_usteps(usteps)

    def move_y_usteps(self,usteps):
        self.microcontroller.move_y_usteps(usteps)

    def move_z_usteps(self,usteps):
        self.microcontroller.move_z_usteps(usteps)

    def move_x_to_usteps(self,usteps):
        self.microcontroller.move_x_to_usteps(usteps)

    def move_y_to_usteps(self,usteps):
        self.microcontroller.move_y_to_usteps(usteps)

    def move_z_to_usteps(self,usteps):
        self.microcontroller.move_z_to_usteps(usteps)

    def update_pos(self,microcontroller):
        # get position from the microcontroller
        x_pos, y_pos, z_pos, theta_pos = microcontroller.get_pos()
        self.z_pos = z_pos
        # calculate position in mm or rad
        if USE_ENCODER_X:
            self.x_pos_mm = x_pos*ENCODER_POS_SIGN_X*ENCODER_STEP_SIZE_X_MM
        else:
            self.x_pos_mm = x_pos*STAGE_POS_SIGN_X*self.get_mm_per_ustep_X()
        if USE_ENCODER_Y:
            self.y_pos_mm = y_pos*ENCODER_POS_SIGN_Y*ENCODER_STEP_SIZE_Y_MM
        else:
            self.y_pos_mm = y_pos*STAGE_POS_SIGN_Y*self.get_mm_per_ustep_Y()
        if USE_ENCODER_Z:
            self.z_pos_mm = z_pos*ENCODER_POS_SIGN_Z*ENCODER_STEP_SIZE_Z_MM
        else:
            self.z_pos_mm = z_pos*STAGE_POS_SIGN_Z*self.get_mm_per_ustep_Z()
        if USE_ENCODER_THETA:
            self.theta_pos_rad = theta_pos*ENCODER_POS_SIGN_THETA*ENCODER_STEP_SIZE_THETA
        else:
            self.theta_pos_rad = theta_pos*STAGE_POS_SIGN_THETA*(2*math.pi/(self.theta_microstepping*FULLSTEPS_PER_REV_THETA))
        # emit the updated position
        self.xPos.emit(self.x_pos_mm)
        self.yPos.emit(self.y_pos_mm)
        self.zPos.emit(self.z_pos_mm*1000)
        self.thetaPos.emit(self.theta_pos_rad*360/(2*math.pi))
        self.xyPos.emit(self.x_pos_mm,self.y_pos_mm)

        if microcontroller.signal_joystick_button_pressed_event:
            if self.enable_joystick_button_action:
                self.signal_joystick_button_pressed.emit()
            print('joystick button pressed')
            microcontroller.signal_joystick_button_pressed_event = False

    def home_x(self):
        self.microcontroller.home_x()

    def home_y(self):
        self.microcontroller.home_y()

    def home_z(self):
        self.microcontroller.home_z()

    def home_theta(self):
        self.microcontroller.home_theta()

    def home_xy(self):
        self.microcontroller.home_xy()

    def zero_x(self):
        self.microcontroller.zero_x()

    def zero_y(self):
        self.microcontroller.zero_y()

    def zero_z(self):
        self.microcontroller.zero_z()

    def zero_theta(self):
        self.microcontroller.zero_theta()

    def home(self):
        pass

    def set_x_limit_pos_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_X > 0:
            self.microcontroller.set_lim(LIMIT_CODE.X_POSITIVE,int(value_mm/self.get_mm_per_ustep_X()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.X_NEGATIVE,STAGE_MOVEMENT_SIGN_X*int(value_mm/self.get_mm_per_ustep_X()))

    def set_x_limit_neg_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_X > 0:
            self.microcontroller.set_lim(LIMIT_CODE.X_NEGATIVE,int(value_mm/self.get_mm_per_ustep_X()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.X_POSITIVE,STAGE_MOVEMENT_SIGN_X*int(value_mm/self.get_mm_per_ustep_X()))

    def set_y_limit_pos_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_Y > 0:
            self.microcontroller.set_lim(LIMIT_CODE.Y_POSITIVE,int(value_mm/self.get_mm_per_ustep_Y()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.Y_NEGATIVE,STAGE_MOVEMENT_SIGN_Y*int(value_mm/self.get_mm_per_ustep_Y()))

    def set_y_limit_neg_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_Y > 0:
            self.microcontroller.set_lim(LIMIT_CODE.Y_NEGATIVE,int(value_mm/self.get_mm_per_ustep_Y()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.Y_POSITIVE,STAGE_MOVEMENT_SIGN_Y*int(value_mm/self.get_mm_per_ustep_Y()))

    def set_z_limit_pos_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_Z > 0:
            self.microcontroller.set_lim(LIMIT_CODE.Z_POSITIVE,int(value_mm/self.get_mm_per_ustep_Z()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.Z_NEGATIVE,STAGE_MOVEMENT_SIGN_Z*int(value_mm/self.get_mm_per_ustep_Z()))

    def set_z_limit_neg_mm(self,value_mm):
        if STAGE_MOVEMENT_SIGN_Z > 0:
            self.microcontroller.set_lim(LIMIT_CODE.Z_NEGATIVE,int(value_mm/self.get_mm_per_ustep_Z()))
        else:
            self.microcontroller.set_lim(LIMIT_CODE.Z_POSITIVE,STAGE_MOVEMENT_SIGN_Z*int(value_mm/self.get_mm_per_ustep_Z()))

    def move_to(self,x_mm,y_mm):
        self.move_x_to(x_mm)
        self.microcontroller.wait_till_operation_is_completed()
        self.move_y_to(y_mm)
        self.microcontroller.wait_till_operation_is_completed()

    def configure_encoder(self, axis, transitions_per_revolution,flip_direction):
        self.microcontroller.configure_stage_pid(axis, transitions_per_revolution=int(transitions_per_revolution), flip_direction=flip_direction)

    def set_pid_control_enable(self, axis, enable_flag):
        self.pid_enable_flag[axis] = enable_flag;
        if self.pid_enable_flag[axis] is True:
            self.microcontroller.turn_on_stage_pid(axis)
        else:
            self.microcontroller.turn_off_stage_pid(axis)

    def turnoff_axis_pid_control(self):
        for i in range(len(self.pid_enable_flag)):
            if self.pid_enable_flag[i] is True:
                self.microcontroller.turn_off_stage_pid(i)

    def get_pid_control_flag(self, axis):
        return self.pid_enable_flag[axis]

    def keep_scan_begin_position(self, x, y):
        self.scan_begin_position_x = x
        self.scan_begin_position_y = y

    def set_axis_PID_arguments(self, axis, pid_p, pid_i, pid_d):
        self.microcontroller.set_pid_arguments(axis, pid_p, pid_i, pid_d)

    def set_piezo_um(self, z_piezo_um):
        dac = int(65535 * (z_piezo_um / OBJECTIVE_PIEZO_RANGE_UM))
        dac = 65535 - dac if OBJECTIVE_PIEZO_FLIP_DIR else dac
        self.microcontroller.analog_write_onboard_DAC(7, dac)






class ImageDisplayWindow(QMainWindow):

    image_click_coordinates = Signal(int, int, int, int)

    def __init__(self, liveController=None, contrastManager=None, window_title='', draw_crosshairs = False, show_LUT=False, autoLevels=False):
        super().__init__()
        self.liveController = liveController
        self.contrastManager = contrastManager
        self.setWindowTitle(window_title)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self.widget = QWidget()
        self.show_LUT = show_LUT
        self.autoLevels = autoLevels

        # interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')

        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.view = self.graphics_widget.addViewBox()
        self.graphics_widget.view.invertY()

        ## lock the aspect ratio so pixels are always square
        self.graphics_widget.view.setAspectLocked(True)

        ## Create image item
        if self.show_LUT:
            self.graphics_widget.view = pg.ImageView()
            self.graphics_widget.img = self.graphics_widget.view.getImageItem()
            self.graphics_widget.img.setBorder('w')
            self.graphics_widget.view.ui.roiBtn.hide()
            self.graphics_widget.view.ui.menuBtn.hide()
            self.LUTWidget = self.graphics_widget.view.getHistogramWidget()
            self.LUTWidget.region.sigRegionChanged.connect(self.update_contrast_limits)
            self.LUTWidget.region.sigRegionChangeFinished.connect(self.update_contrast_limits)
            # self.LUTWidget = self.graphics_widget.view.getHistogramWidget()
            # self.LUTWidget.autoHistogramRange()
            # self.graphics_widget.view.autolevels()
        else:
            self.graphics_widget.img = pg.ImageItem(border='w')
            self.graphics_widget.view.addItem(self.graphics_widget.img)

        ## Create ROI
        self.roi_pos = (500,500)
        self.roi_size = (500,500)
        self.ROI = pg.ROI(self.roi_pos, self.roi_size, scaleSnap=True, translateSnap=True)
        self.ROI.setZValue(10)
        self.ROI.addScaleHandle((0,0), (1,1))
        self.ROI.addScaleHandle((1,1), (0,0))
        self.graphics_widget.view.addItem(self.ROI)
        self.ROI.hide()
        self.ROI.sigRegionChanged.connect(self.update_ROI)
        self.roi_pos = self.ROI.pos()
        self.roi_size = self.ROI.size()

        ## Variables for annotating images
        self.draw_rectangle = False
        self.ptRect1 = None
        self.ptRect2 = None
        self.DrawCirc = False
        self.centroid = None
        self.DrawCrossHairs = False
        self.image_offset = np.array([0, 0])

        # ## flag of setting scaling level
        # self.flag_image_scaling_level_init = False

        ## Layout
        layout = QGridLayout()
        if self.show_LUT:
            layout.addWidget(self.graphics_widget.view, 0, 0)
        else:
            layout.addWidget(self.graphics_widget, 0, 0)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)

        # set window size
        desktopWidget = QDesktopWidget();
        width = min(desktopWidget.height()*0.9,1000) #@@@TO MOVE@@@#
        height = width
        self.setFixedSize(int(width),int(height))
        if self.show_LUT:
            self.graphics_widget.view.getView().scene().sigMouseClicked.connect(self.mouse_clicked)
        else:
            self.graphics_widget.view.scene().sigMouseClicked.connect(self.mouse_clicked)

    def is_within_image(self, coordinates):
        try:
            image_width = self.graphics_widget.img.width()
            image_height = self.graphics_widget.img.height()

            return 0 <= coordinates.x() < image_width and 0 <= coordinates.y() < image_height
        except:
            return False

    def mouse_clicked(self, evt):
        try:
            pos = evt.pos()
            if self.show_LUT:
                view_coord = self.graphics_widget.view.getView().mapSceneToView(pos)
            else:
                view_coord = self.graphics_widget.view.mapSceneToView(pos)
            image_coord = self.graphics_widget.img.mapFromView(view_coord)
        except:
            return

        if self.is_within_image(image_coord):
            x_pixel_centered = int(image_coord.x() - self.graphics_widget.img.width()/2)
            y_pixel_centered = int(image_coord.y() - self.graphics_widget.img.height()/2)
            self.image_click_coordinates.emit(x_pixel_centered, y_pixel_centered, self.graphics_widget.img.width(), self.graphics_widget.img.height())

    def display_image(self, image):

        info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else np.finfo(image.dtype)
        min_val, max_val = info.min, info.max

        if self.liveController is not None and self.contrastManager is not None:
            channel_name = self.liveController.currentConfiguration.name
            if self.contrastManager.acquisition_dtype != None and self.contrastManager.acquisition_dtype != np.dtype(image.dtype):
                self.contrastManager.scale_contrast_limits(np.dtype(image.dtype))
            min_val, max_val = self.contrastManager.get_limits(channel_name, image.dtype)

        self.graphics_widget.img.setImage(image, autoLevels=self.autoLevels, levels=(min_val, max_val))

        if not self.autoLevels:
            if self.show_LUT:
                self.LUTWidget.setLevels(min_val, max_val)
                self.LUTWidget.setHistogramRange(info.min, info.max)
            else:
                self.graphics_widget.img.setLevels((min_val, max_val))

        self.graphics_widget.img.updateImage()

    def update_contrast_limits(self):
        if self.show_LUT and self.contrastManager and self.contrastManager.acquisition_dtype:
            min_val, max_val = self.LUTWidget.region.getRegion()
            self.contrastManager.update_limits(self.liveController.currentConfiguration.name, min_val, max_val)

    def update_ROI(self):
        self.roi_pos = self.ROI.pos()
        self.roi_size = self.ROI.size()

    def show_ROI_selector(self):
        self.ROI.show()

    def hide_ROI_selector(self):
        self.ROI.hide()

    def get_roi(self):
        return self.roi_pos,self.roi_size

    def update_bounding_box(self,pts):
        self.draw_rectangle=True
        self.ptRect1=(pts[0][0],pts[0][1])
        self.ptRect2=(pts[1][0],pts[1][1])

    def get_roi_bounding_box(self):
        self.update_ROI()
        width = self.roi_size[0]
        height = self.roi_size[1]
        xmin = max(0, self.roi_pos[0])
        ymin = max(0, self.roi_pos[1])
        return np.array([xmin, ymin, width, height])

    def set_autolevel(self,enabled):
        self.autoLevels = enabled
        print('set autolevel to ' + str(enabled))


class NavigationViewer(QFrame):

    signal_update_live_scan_grid = Signal(float, float)
    signal_update_well_coordinates = Signal(bool)

    def __init__(self, objectivestore, sample = 'glass slide', invertX = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.sample = sample
        self.objectiveStore = objectivestore
        self.well_size_mm = WELL_SIZE_MM
        self.well_spacing_mm = WELL_SPACING_MM
        self.number_of_skip = NUMBER_OF_SKIP
        self.a1_x_mm = A1_X_MM
        self.a1_y_mm = A1_Y_MM
        self.a1_x_pixel = A1_X_PIXEL
        self.a1_y_pixel = A1_Y_PIXEL
        self.location_update_threshold_mm = 0.2
        self.box_color = (255, 0, 0)
        self.box_line_thickness = 2
        self.acquisition_size = Acquisition.CROP_HEIGHT
        self.x_mm = None
        self.y_mm = None
        self.acquisition_started = False
        self.image_paths = {
            'glass slide': 'images/slide carrier_828x662.png',
            '4 glass slide': 'images/4 slide carrier_1509x1010.png',
            '6 well plate': 'images/6 well plate_1509x1010.png',
            '12 well plate': 'images/12 well plate_1509x1010.png',
            '24 well plate': 'images/24 well plate_1509x1010.png',
            '96 well plate': 'images/96 well plate_1509x1010.png',
            '384 well plate': 'images/384 well plate_1509x1010.png',
            '1536 well plate': 'images/1536 well plate_1509x1010.png'
        }

        print("navigation viewer:", sample)
        self.init_ui(invertX)

        self.load_background_image(self.image_paths.get(sample, 'images/slide carrier_828x662.png'))
        self.create_layers()
        self.update_display_properties(sample)
        # self.update_display()

    def init_ui(self, invertX):
        # interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground("w")

        self.view = self.graphics_widget.addViewBox(invertX=invertX, invertY=True)
        self.view.setAspectLocked(True)

        self.grid = QVBoxLayout()
        self.grid.addWidget(self.graphics_widget)
        self.setLayout(self.grid)

    def load_background_image(self, image_path):
        self.view.clear()
        self.background_image = cv2.imread(image_path)
        if self.background_image is None:
            #raise ValueError(f"Failed to load image from {image_path}")
             self.background_image = cv2.imread(self.image_paths.get('glass slide'))
        
        if len(self.background_image.shape) == 2:  # Grayscale image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_GRAY2RGBA)
        elif self.background_image.shape[2] == 3:  # BGR image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGBA)
        elif self.background_image.shape[2] == 4:  # BGRA image
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGRA2RGBA)

        self.background_image_copy = self.background_image.copy()
        self.image_height, self.image_width = self.background_image.shape[:2]
        self.background_item = pg.ImageItem(self.background_image)
        self.view.addItem(self.background_item)

    def create_layers(self):
        self.scan_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        self.fov_overlay = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)

        self.scan_overlay_item = pg.ImageItem()
        self.fov_overlay_item = pg.ImageItem()

        self.view.addItem(self.scan_overlay_item)
        self.view.addItem(self.fov_overlay_item)

    def update_display_properties(self, sample):
        if sample == 'glass slide':
            self.location_update_threshold_mm = 0.2
            self.mm_per_pixel = 0.1453
            self.origin_x_pixel = 200
            self.origin_y_pixel = 120
        elif sample == '4 glass slide':
            self.location_update_threshold_mm = 0.2
            self.mm_per_pixel = 0.084665
            self.origin_x_pixel = 50
            self.origin_y_pixel = 0
            self.view.invertX(False)
            self.view.invertY(True)
        else:
            self.location_update_threshold_mm = 0.05
            self.mm_per_pixel = 0.084665
            self.origin_x_pixel = self.a1_x_pixel - (self.a1_x_mm)/self.mm_per_pixel
            self.origin_y_pixel = self.a1_y_pixel - (self.a1_y_mm)/self.mm_per_pixel
            self.view.invertX(False)
            self.view.invertY(True)
        self.update_fov_size()

    def update_fov_size(self):
        pixel_size_um = self.objectiveStore.get_pixel_size()
        self.fov_size_mm = self.acquisition_size * pixel_size_um / 1000

    def on_objective_changed(self):
        self.clear_overlay()
        self.update_fov_size()
        if self.x_mm is not None and self.y_mm is not None:
            if 'glass slide'in self.sample:
                self.signal_update_live_scan_grid.emit(self.x_mm, self.y_mm)
            self.draw_current_fov(self.x_mm, self.y_mm)

    def on_acquisition_start(self, acquisition_started):
        self.acquisition_started = acquisition_started

    def update_wellplate_settings(self, sample_format, a1_x_mm, a1_y_mm, a1_x_pixel, a1_y_pixel, well_size_mm, well_spacing_mm, number_of_skip, rows, cols):
        if isinstance(sample_format, QVariant):
            sample_format = sample_format.value()

        if sample_format == 0:
            if IS_HCS:
                sample = '4 glass slide'
            else:
                sample = 'glass slide'
        elif isinstance(sample_format, int):
            sample = f'{sample_format} well plate'
        else:  # Custom wellplate
            sample = sample_format

        self.sample = sample
        self.a1_x_mm = a1_x_mm
        self.a1_y_mm = a1_y_mm
        self.a1_x_pixel = a1_x_pixel
        self.a1_y_pixel = a1_y_pixel
        self.well_size_mm = well_size_mm
        self.well_spacing_mm = well_spacing_mm
        self.number_of_skip = number_of_skip
        self.rows = rows
        self.cols = cols

        # Try to find the image for the wellplate
        image_path = self.image_paths.get(sample)
        if image_path is None or not os.path.exists(image_path):
            # Look for a custom wellplate image
            custom_image_path = os.path.join('images', f'{sample.replace(" ", "_")}.png')
            if os.path.exists(custom_image_path):
                image_path = custom_image_path
            else:
                print(f"Warning: Image not found for {sample}. Using default image.")
                image_path = self.image_paths.get('glass slide')  # Use a default image

        self.load_background_image(image_path)
        self.create_layers()
        self.update_display_properties(sample)
        self.draw_current_fov(self.x_mm, self.y_mm)

    def update_current_location(self, x_mm=None, y_mm=None):
        if x_mm is None and y_mm is None:
            self.draw_current_fov(self.x_mm, self.y_mm)

        elif self.x_mm is not None and self.y_mm is not None:
            # update only when the displacement has exceeded certain value
            if abs(x_mm - self.x_mm) > self.location_update_threshold_mm or abs(y_mm - self.y_mm) > self.location_update_threshold_mm:
                self.draw_current_fov(x_mm, y_mm)
                self.x_mm = x_mm
                self.y_mm = y_mm
                # update_live_scan_grid
                if 'glass slide'in self.sample and not self.acquisition_started:
                    self.signal_update_live_scan_grid.emit(x_mm, y_mm)
        else:
            self.draw_current_fov(x_mm, y_mm)
            self.x_mm = x_mm
            self.y_mm = y_mm
            # update_live_scan_grid
            if 'glass slide'in self.sample and not self.acquisition_started:
                self.signal_update_live_scan_grid.emit(x_mm, y_mm)

    def get_FOV_pixel_coordinates(self, x_mm, y_mm):
        if self.sample == 'glass slide':
            current_FOV_top_left = (
                round(self.origin_x_pixel + x_mm/self.mm_per_pixel - self.fov_size_mm/2/self.mm_per_pixel),
                round(self.image_height - (self.origin_y_pixel + y_mm/self.mm_per_pixel) - self.fov_size_mm/2/self.mm_per_pixel)
            )
            current_FOV_bottom_right = (
                round(self.origin_x_pixel + x_mm/self.mm_per_pixel + self.fov_size_mm/2/self.mm_per_pixel),
                round(self.image_height - (self.origin_y_pixel + y_mm/self.mm_per_pixel) + self.fov_size_mm/2/self.mm_per_pixel)
            )
        else:
            current_FOV_top_left = (
                round(self.origin_x_pixel + x_mm/self.mm_per_pixel - self.fov_size_mm/2/self.mm_per_pixel),
                round((self.origin_y_pixel + y_mm/self.mm_per_pixel) - self.fov_size_mm/2/self.mm_per_pixel)
            )
            current_FOV_bottom_right = (
                round(self.origin_x_pixel + x_mm/self.mm_per_pixel + self.fov_size_mm/2/self.mm_per_pixel),
                round((self.origin_y_pixel + y_mm/self.mm_per_pixel) + self.fov_size_mm/2/self.mm_per_pixel)
            )
        return current_FOV_top_left, current_FOV_bottom_right

    def draw_current_fov(self, x_mm, y_mm):
        self.fov_overlay.fill(0)
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(self.fov_overlay, current_FOV_top_left, current_FOV_bottom_right, (255, 0, 0, 255), self.box_line_thickness)
        self.fov_overlay_item.setImage(self.fov_overlay)

    def register_fov(self, x_mm, y_mm):
        color = (0, 0, 255, 255)  # Blue RGBA
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(self.background_image, current_FOV_top_left, current_FOV_bottom_right, color, self.box_line_thickness)
        self.background_item.setImage(self.background_image)

    def register_fov_to_image(self, x_mm, y_mm):
        color = (252, 174, 30, 128)  # Yellow RGBA
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(self.scan_overlay, current_FOV_top_left, current_FOV_bottom_right, color, self.box_line_thickness)
        self.scan_overlay_item.setImage(self.scan_overlay)

    def deregister_fov_to_image(self, x_mm, y_mm):
        current_FOV_top_left, current_FOV_bottom_right = self.get_FOV_pixel_coordinates(x_mm, y_mm)
        cv2.rectangle(self.scan_overlay, current_FOV_top_left, current_FOV_bottom_right, (0, 0, 0, 0), self.box_line_thickness)
        self.scan_overlay_item.setImage(self.scan_overlay)

    def clear_slide(self):
        self.background_image = self.background_image_copy.copy()
        self.background_item.setImage(self.background_image)
        self.draw_current_fov(self.x_mm, self.y_mm)

    def update_slide(self):
        self.background_image = self.background_image_copy.copy()
        self.background_item.setImage(self.background_image)
        self.draw_current_fov(self.x_mm, self.y_mm)
        self.signal_update_well_coordinates.emit(True)

    def clear_overlay(self):
        self.scan_overlay.fill(0)
        self.scan_overlay_item.setImage(self.scan_overlay)


