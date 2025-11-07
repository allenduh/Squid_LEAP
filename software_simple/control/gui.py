
import os 
os.environ["QT_API"] = "pyqt5"
import qtpy

# qt libraries
from qtpy.QtCore import *
from qtpy.QtWidgets import *
from qtpy.QtGui import *

# app specific libraries
import control.widgets as widgets
#import control.widgets_potentiostat as widgets_potentiostat
import control.potentiostat_panel_v2 as widgets_potentiostat
import control.camera_emergent_cleaned as camera
import control.core_simple as core
import control.microcontroller as microcontroller
from control._def import *
import pyqtgraph.dockarea as dock


class OctopiGUI(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # display windows
        self.imageDisplayWindow = core.ImageDisplayWindow(draw_crosshairs=True,autoLevels=AUTOLEVEL_DEFAULT_SETTING)
        self.imageDisplayTabs = QTabWidget()
        self.imageDisplayTabs.addTab(self.imageDisplayWindow.widget, "Live View")

        try:
            self.camera = camera.Camera(rotate_image_angle=ROTATE_IMAGE_ANGLE,flip_image=FLIP_IMAGE)
            self.camera.open()
        except:
            print("Camera has issues")
        
        try:
            self.microcontroller = microcontroller.Microcontroller(version=CONTROLLER_VERSION)
            print(CONTROLLER_VERSION)
        except:
            self.microcontroller = microcontroller.Microcontroller_Simulation()

        # reset the MCU
        self.microcontroller.reset()

        # configure the actuators
        self.microcontroller.configure_actuators()
        self.streamHandler = core.StreamHandler(display_resolution_scaling=DEFAULT_DISPLAY_CROP/100)
        self.liveController = core.LiveController(self.camera,self.microcontroller)
        self.navigationController = core.NavigationController(self.microcontroller)

        self.imageSaver = core.ImageSaver(image_format=Acquisition.IMAGE_FORMAT)
        
        # Camera will output the frame into streamHandler during software acquisition
        self.camera.set_callback(self.streamHandler.on_new_frame)
    
        # load widgets:

        self.liveControlWidget = widgets.LiveControlWidget(self.streamHandler,self.liveController,show_trigger_options=True,show_display_options=True,show_autolevel=SHOW_AUTOLEVEL_BTN,autolevel=AUTOLEVEL_DEFAULT_SETTING)
        self.navigationWidget = widgets.NavigationWidget(self.navigationController)

        self.recordingControlWidget = widgets.RecordingWidget(self.streamHandler,self.imageSaver)
        self.recordingControlWidget.set_live_widget(self.liveControlWidget) # enable changing from software to contineous
        self.recordingControlWidget.set_camera(self.camera)
        
        self.recordTabWidget = QTabWidget()
        self.recordTabWidget.addTab(self.recordingControlWidget, "Simple Recording")
        self.potentiostatWidget = widgets_potentiostat.potentialstatControlWidget()


        # layout widgets
        layout = QVBoxLayout() 
        layout.addWidget(self.liveControlWidget)
        layout.addWidget(self.navigationWidget)
        layout.addWidget(self.recordTabWidget)
        layout.addWidget(self.potentiostatWidget)
        layout.addStretch()

        # transfer the layout to the central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        self.centralWidget.setFixedWidth(self.centralWidget.minimumSizeHint().width())


        dock_display = dock.Dock('Image Display', autoOrientation = False)
        dock_display.showTitleBar()
        dock_display.addWidget(self.imageDisplayTabs)
        dock_display.setStretch(x=100,y=None)
        dock_controlPanel = dock.Dock('Controls', autoOrientation = False)

        dock_controlPanel.addWidget(self.centralWidget)
        dock_controlPanel.setStretch(x=1,y=None)
        dock_controlPanel.setFixedWidth(dock_controlPanel.minimumSizeHint().width())

        main_dockArea = dock.DockArea()
        main_dockArea.addDock(dock_display)
        main_dockArea.addDock(dock_controlPanel,'right')
        self.setCentralWidget(main_dockArea)
        desktopWidget = QDesktopWidget()
        height_min = 0.9*desktopWidget.height()
        width_min = 0.96*desktopWidget.width()
        self.setMinimumSize(int(width_min),int(height_min))

        # make connections
        self.streamHandler.image_to_display.connect(self.imageDisplayWindow.display_image)
        self.streamHandler.packet_image_to_write.connect(self.imageSaver.enqueue)

        self.navigationController.xPos.connect(self.navigationWidget.label_Xpos.setNum)
        self.navigationController.yPos.connect(self.navigationWidget.label_Ypos.setNum)
        self.navigationController.zPos.connect(self.navigationWidget.label_Zpos.setNum)
       
        self.liveControlWidget.signal_autoLevelSetting.connect(self.imageDisplayWindow.set_autolevel)



    def closeEvent(self, event):
        event.accept()
        self.navigationController.home()
        self.liveController.stop_live()
        self.camera.close()
        self.imageSaver.close()
        self.microcontroller.close()
