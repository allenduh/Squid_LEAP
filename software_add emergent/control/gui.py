# set QT_API environment variable
import os 
os.environ["QT_API"] = "pyqt5"
import qtpy

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

# app specific libraries
import control.widgets as widgets
import control.camera_emergent as camera
import control.core as core
import control.microcontroller as microcontroller
import control.widgets_ephysDisplay as widgets_ephysDisplay
from control._def import *
import squid.logging


import pyqtgraph.dockarea as dock
SINGLE_WINDOW = True # set to False if use separate windows for display and control

class OctopiGUI(QMainWindow):

    # variables
    fps_software_trigger = 100

    def __init__(self, is_simulation = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log = squid.logging.get_logger(self.__class__.__name__)

        # load window
        if ENABLE_TRACKING:
            self.imageDisplayWindow = core.ImageDisplayWindow(draw_crosshairs=True,autoLevels=AUTOLEVEL_DEFAULT_SETTING)
            self.imageDisplayWindow.show_ROI_selector()
        else:
            self.imageDisplayWindow = core.ImageDisplayWindow(draw_crosshairs=True,autoLevels=AUTOLEVEL_DEFAULT_SETTING)
        self.imageArrayDisplayWindow = core.ImageArrayDisplayWindow() 
        # self.imageDisplayWindow.show()
        # self.imageArrayDisplayWindow.show()

        # Live display
        self.ephysDisplay = widgets_ephysDisplay.Display(self.imageDisplayWindow)
        self.ROIControl = widgets_ephysDisplay.ROIControlWidget(self.imageDisplayWindow)

        # image display windows
        self.imageDisplayTabs = QTabWidget()
        self.imageDisplayTabs.addTab(self.imageDisplayWindow.widget, "Live View")
        self.imageDisplayTabs.addTab(self.imageArrayDisplayWindow.widget, "Multichannel Acquisition")
        self.imageDisplayTabs.addTab(self.ephysDisplay, "MEA live")

        viewer = widgets_ephysDisplay.RawViewerWidget(folder="output/EVT_Py_convert", bin_size=128)
        self.imageDisplayTabs.addTab(viewer, "viewer")

        # load objects
        if is_simulation:
            self.camera = camera.Camera_Simulation(rotate_image_angle=ROTATE_IMAGE_ANGLE,flip_image=FLIP_IMAGE)
            self.microcontroller = microcontroller.Microcontroller_Simulation()
        else:
            try:
                self.camera = camera.Camera(rotate_image_angle=ROTATE_IMAGE_ANGLE,flip_image=FLIP_IMAGE)
                self.camera.open()
            except:
                self.camera = camera.Camera_Simulation(rotate_image_angle=ROTATE_IMAGE_ANGLE,flip_image=FLIP_IMAGE)
                self.camera.open()
                self.log.error("camera not detected, using simulated camera")
            try:
                self.microcontroller = microcontroller.Microcontroller(version=CONTROLLER_VERSION)
                print(CONTROLLER_VERSION)
            except:
                self.log.error("Microcontroller not detected, using simulated microcontroller")
                self.microcontroller = microcontroller.Microcontroller_Simulation()

        # reset the MCU
        self.microcontroller.reset()

        # configure the actuators
        self.microcontroller.configure_actuators()
        
        self.objectiveStore = core.ObjectiveStore()
        self.configurationManager = core.ConfigurationManager('./channel_configurations.xml')
        self.objectiveStore = core.ObjectiveStore(parent=self) # todo: add widget to select/save objective save
        self.streamHandler = core.StreamHandler(display_resolution_scaling=DEFAULT_DISPLAY_CROP/100)
        self.liveController = core.LiveController(self.camera,self.microcontroller,self.configurationManager)
        self.navigationController = core.NavigationController(self.microcontroller,self.objectiveStore)
        self.autofocusController = core.AutoFocusController(self.camera,self.navigationController,self.liveController)
        self.scanCoordinates = core.ScanCoordinates()
        self.multipointController = core.MultiPointController(self.camera,self.navigationController,self.liveController,self.autofocusController,self.configurationManager,scanCoordinates=self.scanCoordinates,parent=self)
        if ENABLE_TRACKING:
            self.trackingController = core.TrackingController(self.camera,self.microcontroller,self.navigationController,self.configurationManager,self.liveController,self.autofocusController,self.imageDisplayWindow)
        self.imageSaver = core.ImageSaver(image_format=Acquisition.IMAGE_FORMAT)
        self.imageDisplay = core.ImageDisplay()
        
        # set up the camera		
        # self.camera.set_reverse_x(CAMERA_REVERSE_X) # these are not implemented for the cameras in use
        # self.camera.set_reverse_y(CAMERA_REVERSE_Y) # these are not implemented for the cameras in use
        
        self.camera.set_callback(self.streamHandler.on_new_frame)
        self.camera.enable_callback()
        if ENABLE_STROBE_OUTPUT:
            self.camera.set_line3_to_exposure_active()

        # load widgets:
        self.objectivesWidget=widgets.ObjectivesWidget(self.objectiveStore)

        self.cameraSettingWidget = widgets.CameraSettingsWidget(self.camera,include_gain_exposure_time=False)
        self.liveControlWidget = widgets.LiveControlWidget(self.streamHandler,self.liveController,self.configurationManager,show_trigger_options=True,show_display_options=True,show_autolevel=SHOW_AUTOLEVEL_BTN,autolevel=AUTOLEVEL_DEFAULT_SETTING)
        self.navigationWidget = widgets.NavigationWidget(self.navigationController)
        self.dacControlWidget = widgets.DACControWidget(self.microcontroller)
        self.autofocusWidget = widgets.AutoFocusWidget(self.autofocusController)
        self.recordingControlWidget = widgets.RecordingWidget(self.streamHandler,self.imageSaver)
        if ENABLE_TRACKING:
            self.trackingControlWidget = widgets.TrackingControllerWidget(self.trackingController,self.configurationManager,show_configurations=TRACKING_SHOW_MICROSCOPE_CONFIGURATIONS)
        self.multiPointWidget = widgets.MultiPointWidget(self.multipointController,self.configurationManager)
        self.navigationViewer = core.NavigationViewer(self.objectiveStore, sample=str(6)+' well plate')
        self.multiPointWidget2 = widgets.MultiPointWidget2(self.navigationController,self.navigationViewer,self.multipointController,self.configurationManager,scanCoordinates=None)

        self.recordTabWidget = QTabWidget()
        if ENABLE_TRACKING:
            self.recordTabWidget.addTab(self.trackingControlWidget, "Tracking")
        self.recordTabWidget.addTab(self.multiPointWidget2, "Flexible Multipoint")
        self.recordTabWidget.addTab(self.recordingControlWidget, "Simple Recording")
        # self.recordTabWidget.addTab(self.multiPointWidget, "Multipoint Acquisition")


        # layout widgets
        layout = QVBoxLayout() 
        layout.addWidget(self.cameraSettingWidget)
        #self.objectivesWidget.setFixedHeight(100)
        layout.addWidget(self.liveControlWidget)
        layout.addWidget(self.navigationWidget)
        if SHOW_DAC_CONTROL:
            layout.addWidget(self.dacControlWidget)
        layout.addWidget(self.autofocusWidget)
        layout.addWidget(self.recordTabWidget)
        layout.addWidget(self.objectivesWidget)
        layout.addWidget(self.ROIControl)
        layout.addStretch()

        # transfer the layout to the central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        # self.centralWidget.setFixedSize(self.centralWidget.minimumSize())
        # self.centralWidget.setFixedWidth(self.centralWidget.minimumWidth())
        # self.centralWidget.setMaximumWidth(self.centralWidget.minimumWidth())
        self.centralWidget.setFixedWidth(self.centralWidget.minimumSizeHint().width())

        if SINGLE_WINDOW:
            dock_display = dock.Dock('Image Display', autoOrientation = False)
            dock_display.showTitleBar()
            dock_display.addWidget(self.imageDisplayTabs)
            dock_display.setStretch(x=100,y=None)
            dock_controlPanel = dock.Dock('Controls', autoOrientation = False)
            # dock_controlPanel.showTitleBar()
            dock_controlPanel.addWidget(self.centralWidget)
            dock_controlPanel.setStretch(x=1,y=None)
            dock_controlPanel.setFixedWidth(dock_controlPanel.minimumSizeHint().width())

            # dock_MEA = dock.Dock('Live MEA', autoOrientation = False)
            # dock_MEA.showTitleBar()
            # dock_MEA.addWidget(self.ephysDisplay)
            # dock_MEA.setStretch(x=1, y=None)


            main_dockArea = dock.DockArea()
            main_dockArea.addDock(dock_display)
            main_dockArea.addDock(dock_controlPanel,'right')
            self.setCentralWidget(main_dockArea)
            desktopWidget = QDesktopWidget()
            height_min = 0.9*desktopWidget.height()
            width_min = 0.96*desktopWidget.width()
            self.setMinimumSize(int(width_min),int(height_min))


            #dock_ROI.setFixedHeight(4 * dock_ROI.minimumSizeHint().height())
        else:
            self.setCentralWidget(self.centralWidget)
            self.tabbedImageDisplayWindow = QMainWindow()
            self.tabbedImageDisplayWindow.setCentralWidget(self.imageDisplayTabs)
            self.tabbedImageDisplayWindow.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
            self.tabbedImageDisplayWindow.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
            desktopWidget = QDesktopWidget()
            width = 0.96*desktopWidget.height()
            height = width
            self.tabbedImageDisplayWindow.setFixedSize(width,height)
            self.tabbedImageDisplayWindow.show()

        # make connections
        self.streamHandler.signal_new_frame_received.connect(self.liveController.on_new_frame)
        self.streamHandler.image_to_display.connect(self.imageDisplay.enqueue)
        self.streamHandler.packet_image_to_write.connect(self.imageSaver.enqueue)
        # self.streamHandler.packet_image_for_tracking.connect(self.trackingController.on_new_frame)
        self.imageDisplay.image_to_display.connect(self.imageDisplayWindow.display_image) # may connect streamHandler directly to imageDisplayWindow
        self.navigationController.xPos.connect(self.navigationWidget.label_Xpos.setNum)
        self.navigationController.yPos.connect(self.navigationWidget.label_Ypos.setNum)
        self.navigationController.zPos.connect(self.navigationWidget.label_Zpos.setNum)
        if ENABLE_TRACKING:
            self.navigationController.signal_joystick_button_pressed.connect(self.trackingControlWidget.slot_joystick_button_pressed)
        else:
            self.navigationController.signal_joystick_button_pressed.connect(self.autofocusController.autofocus)
        self.autofocusController.image_to_display.connect(self.imageDisplayWindow.display_image)
        self.multipointController.image_to_display.connect(self.imageDisplayWindow.display_image)
        self.multipointController.signal_current_configuration.connect(self.liveControlWidget.set_microscope_mode)
        self.multipointController.image_to_display_multi.connect(self.imageArrayDisplayWindow.display_image)
        self.liveControlWidget.signal_newExposureTime.connect(self.cameraSettingWidget.set_exposure_time)
        self.liveControlWidget.signal_newAnalogGain.connect(self.cameraSettingWidget.set_analog_gain)
        self.liveControlWidget.update_camera_settings()
        self.liveControlWidget.signal_autoLevelSetting.connect(self.imageDisplayWindow.set_autolevel)
        self.liveControlWidget.signal_start_live.connect(self.onStartLive)
        self.multiPointWidget2.signal_acquisition_started.connect(self.navigationWidget.toggle_navigation_controls)

        if USE_NAPARI_FOR_MULTIPOINT:
            self.napariMultiChannelWidget = widgets.NapariMultiChannelWidget(self.objectiveStore)
            self.imageDisplayTabs.addTab(self.napariMultiChannelWidget, "Multichannel Acquisition")
            self.multiPointWidget.signal_acquisition_channels.connect(self.napariMultiChannelWidget.initChannels)
            self.multiPointWidget.signal_acquisition_shape.connect(self.napariMultiChannelWidget.initLayersShape)
            if ENABLE_FLEXIBLE_MULTIPOINT:
                self.multiPointWidget2.signal_acquisition_channels.connect(self.napariMultiChannelWidget.initChannels)
                self.multiPointWidget2.signal_acquisition_shape.connect(self.napariMultiChannelWidget.initLayersShape)

            self.multipointController.napari_layers_init.connect(self.napariMultiChannelWidget.initLayers)
            self.multipointController.napari_layers_update.connect(self.napariMultiChannelWidget.updateLayers)

        self.ROIControl.multi_channel_ephys.connect(self.ephysDisplay.plot_multichannel_ephys)


    def onStartLive(self):
        print(self.camera.exposure_time)
        #self.imageDisplayTabs.setCurrentIndex(0)

    def closeEvent(self, event):
        event.accept()
        # self.softwareTriggerGenerator.stop() @@@ => 
        self.navigationController.home()
        self.liveController.stop_live()
        self.camera.close()
        self.imageSaver.close()
        self.imageDisplay.close()
        if not SINGLE_WINDOW:
            self.imageDisplayWindow.close()
            self.imageArrayDisplayWindow.close()
            self.tabbedImageDisplayWindow.close()
        self.microcontroller.close()
