"""
potentialstat_controller class -- Automate potentialstat gating and acquisition


Remote access
Input: gating protocol from Linux
Output: Saved file in Windows and return dictionary of i(t), v(t) back to Linux

Date: 9/1/2022
Author: Yi-Shiou Duh (allenduh@stanford.edu)
"""
import Pyro4 # for remote access

from datetime import datetime
from dataclasses import dataclass

import sys
potentialStatDir = 'C://EC-Lab Development Package//Examples//Python'
sys.path.append(potentialStatDir)

# Biologic specific
import kbio.kbio_types as KBIO
from kbio.kbio_api import KBIO_api

# import numpy as np
# import time
# from kbio.kbio_tech import ECC_parm, make_ecc_parm, make_ecc_parms, print_experiment_data
# from kbio.c_utils import c_is_64b
# from kbio.utils import exception_brief
# from scipy.io import savemat


@dataclass
@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class voltage_step :
    voltage: float
    duration: float
    vs_init: bool = False 

@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class potentialstat_controller(object):
    def __init__(self):
        ## Potentialstat ID
        address = "USB0"
        DLL_path = "C:\\EC-Lab Development Package\\EC-Lab Development Package\\EClib64.dll"  # library used to communicate with the instrument
        self.api = KBIO_api(DLL_path) # Open DLL to initialize API
        self.id_, self.device_info = self.api.Connect(address) # connect to potentialstat using its address
        self.channel = 1

        ## Define Gating protocol
        self.steps = [
          voltage_step(0, 1), # 0mV during 3s
          voltage_step(0.1, 1), # 100mV during 3s
          voltage_step(-0.1, 1), # -100mV during 3s
          voltage_step(0, 1), # 0mV during 3s
          voltage_step(0.1, 1), # 100mV during 3s
          voltage_step(-0.1, 1), # -100mV during 3s
          voltage_step(0, 1, True), # 0.5mA delta during 3s
        ]
        self.record_dt = 0.1  # seconds
        self.record_dI = 0.1  # current
        self.repeat_count = 0
        
        # Summerized technique parameters into a list and feed intopotentialstat
        self.p_steps = list()   # append gating protocol into a list. Can visualize before gating by setting verbosity = 2
        self.final_parameters = [] # final technique parameter array shown by setting verbosity = 2
        
        # IV curve recording
        self.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

    def output_protocol_parameters(self):
        return self.steps, self.record_dt, self.record_dI, self.repeat_count

    def auto_start_gating(self):
        if self.firmware_is_running():
            self.append_CA_parameter_list()
            self.make_final_parameters()
            self.load_technique_CA()
            
            self.api.StartChannel(self.id_, self.channel)  
            self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
            print('Gating start at =' + self.timeStart)
            return True #self.start_channel_and_record()
        else:
            print("firmware not running. See load_firmware")

    def record_snapshot(self):        
        single_point_length = 5  # for CA. See P71
        
        ## Record data points and store into iv curve

        data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
        current_values, data_info, data_record = data  # retrieve data 
        idx_start = 0        # index inside the buffer 
        num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1. 2(rare)
        
        for _ in range(num_dataPoints_to_read_next) :
            # parse data
            idx_end = idx_start + single_point_length    
            t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
            
            # compute timestamp in seconds using t_high, t_low
            t_rel = (t_high << 32) + t_low
            t = current_values.TimeBase * t_rel   
            print(t)
            print(len(data_record))
            
            # Convert Ewe, I to float i.e 3902889481 into 0.00056
            Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
            I = self.api.ConvertNumericIntoSingle(I)
        
            # store data into ivCurve
            self.ivCurve["t"].append(t)
            self.ivCurve["V_we"].append(Ewe)
            self.ivCurve["I"].append(I)
            self.ivCurve["cycle"].append(cycle)
            
            idx_start = idx_end  # read next datapoint in the buffer by shifting index 
            
            # Status, monitoring STOP
            status = KBIO.PROG_STATE(current_values.State).name   
            
            return t, Ewe, I, cycle, status


        return num_dataPoints_to_read_next#, t, Ewe, I, cycle, status


    def auto_record_all(self):        
        single_point_length = 5  # for CA. See P71
        
        ## Record data points and store into iv curve
        while True :
            data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
            current_values, data_info, data_record = data  # retrieve data 
            idx_start = 0        # index inside the buffer 
            num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1. 2(rare)
            
            for _ in range(num_dataPoints_to_read_next) :
                # parse data
                idx_end = idx_start + single_point_length    
                t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
                
                # compute timestamp in seconds using t_high, t_low
                t_rel = (t_high << 32) + t_low
                t = current_values.TimeBase * t_rel   
                
                # Convert Ewe, I to float i.e 3902889481 into 0.00056
                Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                I = self.api.ConvertNumericIntoSingle(I)
            
                # store data into ivCurve
                self.ivCurve["t"].append(t)
                self.ivCurve["V_we"].append(Ewe)
                self.ivCurve["I"].append(I)
                self.ivCurve["cycle"].append(cycle)
                
                idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                
                # Status, monitoring STOP
                status = KBIO.PROG_STATE(current_values.State).name        

            if status == 'STOP' : # stop when channel reports it is no longer running
                break

        print("> experiment done")

        return self.ivCurve 

#####################################
## Helper function    
#####################################

    def start_channel_and_record(self):

        self.api.StartChannel(self.id_, self.channel)  
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        print('Gating start at =' + self.timeStart)
        
        single_point_length = 5  # for CA. See P71
        
        ## Record data points and store into iv curve
        while True :
            data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
            current_values, data_info, data_record = data  # retrieve data 
            idx_start = 0        # index inside the buffer 
            num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1. 2(rare)
            
            for _ in range(num_dataPoints_to_read_next) :
                # parse data
                idx_end = idx_start + single_point_length    
                t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
                
                # compute timestamp in seconds using t_high, t_low
                t_rel = (t_high << 32) + t_low
                t = current_values.TimeBase * t_rel   
                
                # Convert Ewe, I to float i.e 3902889481 into 0.00056
                Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                I = self.api.ConvertNumericIntoSingle(I)
            
                # store data into ivCurve
                self.ivCurve["t"].append(t)
                self.ivCurve["V_we"].append(Ewe)
                self.ivCurve["I"].append(I)
                self.ivCurve["cycle"].append(cycle)
                
                idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                
                # Status, monitoring STOP
                status = KBIO.PROG_STATE(current_values.State).name        

            if status == 'STOP' : # stop when channel reports it is no longer running
                break

        print("> experiment done")

        return self.ivCurve 


    def load_firmware(self):   
        """Load firmware into api"""                
        firmware_path = "kernel4.bin" # channel firmware for SP-300 series
        fpga_path     = "vmp_iv_0395_aa.xlx"  # channel firmware for SP-300 series
        channel_map = self.api.channel_map({self.channel}) # create a map from channel set
        
        self.api.LoadFirmware(self.id_, channel_map, firmware=firmware_path, fpga=fpga_path, force=True) # force = True if we load_firmware

    def get_channel_info(self):
        """retrieve the device Channel info -- Channel: 1, C340_IF2_Z board, S/N 216, 1 technique
        State: STOP, no amplifiers, IRange, MaxBandwidth: 9, memory, KERNEL (v5.35), FPGA (AA1C)"""  
        return self.api.GetChannelInfo(self.id_, self.channel)
    
    def firmware_is_running(self): 
        """ Check firmware is running """          
        channel_info = self.get_channel_info()
        return channel_info.is_kernel_loaded
        # If false, kernel (firmware) must be loaded in order to run the experiment. Run ex_api_misc.py first
    
    def append_CA_parameter_list(self):
        """ Append a list of (ECC_parameter format) """ 
        # append voltage step
        for idx, step in enumerate(self.steps) :
            self.p_steps.append(self.make_single_parameter('Voltage_step', idx, step.voltage))
            self.p_steps.append(self.make_single_parameter('Duration_step', idx, step.duration))
            self.p_steps.append(self.make_single_parameter('vs_initial', idx, step.vs_init))

        # append other recording parameters
        self.p_steps.append(self.make_single_parameter('Step_number', 0, len(self.steps) - 1))
        self.p_steps.append(self.make_single_parameter('Record_every_dT', 0, self.record_dt))
        self.p_steps.append(self.make_single_parameter('Record_every_dI', 0, self.record_dI))
        self.p_steps.append(self.make_single_parameter('N_Cycles', 0, self.repeat_count))

    def make_single_parameter(self, para_name, idx, value):
        """value is converted to its proper type (ECC_parameter format) 
        this convertion is done by DefineParameter"""  
        parm = KBIO.EccParam()
        self.api.DefineParameter(para_name, value, idx, parm)
        return parm

    def make_final_parameters (self) :
        """Create an EccParam array from an EccParam list, and return an EccParams refering to it."""  
        numOfParms = len(self.p_steps)

        # convert python list to list with ECC_parameter format
        parms_array = KBIO.ECC_PARM_ARRAY(numOfParms)
        for i, parm in enumerate(self.p_steps) :
            parms_array[i] = parm

        self.final_parameters = KBIO.EccParams(numOfParms, parms_array)
  
    def load_technique_CA(self):
        """ Load chrono-amperometry technique using self.final_parameters"""  
        # See P74 file:///C:/EC-Lab%20Development%20Package/EC-Lab%20Development%20Package.pdf
        tech_file = "ca4.ecc" # for VMP300 family, use ca4_tech_file        
        verbosity = 1  # 1: no window pop up. 2: window poped
        self.api.LoadTechnique(self.id_, self.channel, tech_file, self.final_parameters, first=True, last=True, display=(verbosity>1))
   
    def disconnect(self):
        self.api.Disconnect(self.id_)
        print("DisConnect successfully")
        return True
    


'''

"""

Example main :

  * open the DLL,
  * connect to the device using its address,
  * retrieve the device channel info,
  * test whether the proper firmware is running,
  * if it is, print all the messages this channel has accumulated so far,
  * create a CA parameter list (a subset of all possible parameters),
  * load the CA technique into the channel,
  * start the technique,
  * in a loop :
      * retrieve and display experiment data,
      * display messages,
      * stop when channel reports it is no longer running

Note: for each call to the DLL, the base API function is shown in a comment.

"""

try :




    
    ivCurveReshaped = np.reshape(ivCurve, [int(len(ivCurve) / 4), 4])
    
    ecLabData = {"ivCurve": ivCurveReshaped, "timeStamp": timeStamp}
    dirFolder = "E:\\user data\\Yi-Shiou\\07252022 (gatingProDot)"
    savemat(dirFolder + "\\test2.mat", ecLabData)
'''

"""
     def append_CA_parameter_list_(self):
         # append a list of 'Voltage_step', 'step_duration', 'vs_init' for each constant voltage step
             # BL_Define<xxx>Parameter
             # .. value is converted to its proper type, which DefineParameter will use
         for idx, step in enumerate(self.steps) :
             parm = KBIO.EccParam()
             self.api.DefineParameter('Voltage_step', step.voltage, idx, parm)
             self.p_steps.append(parm)
             
             parm = KBIO.EccParam()
             self.api.DefineParameter('step_duration', step.duration, idx, parm)
             self.p_steps.append(parm)

             parm = KBIO.EccParam()
             self.api.DefineParameter('vs_initial', step.vs_init, idx, parm)
             self.p_steps.append(parm)
    def make_ecc_parms_ (api, *ecc_parm_list) :
        "Create an EccParam array from an EccParam list, and return an EccParams refering to it.""
        nb_parms = len(ecc_parm_list)
        parms_array = KBIO.ECC_PARM_ARRAY(nb_parms)

        for i,parm in enumerate(ecc_parm_list) :
            parms_array[i] = parm

        parms = KBIO.EccParams(nb_parms, parms_array)
        return parms
    
    def append_CA_parameter_list_copy(self):
        # append a list of 'Voltage_step', 'step_duration', 'vs_init' for each constant voltage step
        for idx, step in enumerate(self.steps) :
            parm = make_ecc_parm(self.api, self.CA_parms['Voltage_step'], step.voltage, idx)
            self.p_steps.append(parm)
            parm = make_ecc_parm(self.api, self.CA_parms['step_duration'], step.duration, idx)
            self.p_steps.append(parm)
            parm = make_ecc_parm(self.api, self.CA_parms['vs_init'], step.vs_init, idx)
            self.p_steps.append(parm)
 
 
        self.CA_parms = { # dictionary of CA parameters (non exhaustive)
            'Voltage_step':  ECC_parm("Voltage_step", float),
            'vs_init':       ECC_parm("vs_initial", bool),
            'step_duration': ECC_parm("Duration_step", float),
            'nb_steps':      ECC_parm("Step_number", int),
            'record_dt':     ECC_parm("Record_every_dT", float),
            'record_dI':     ECC_parm("Record_every_dI", float),
            'repeat':        ECC_parm("N_Cycles", int),
        }
        
        
    def make_technique_parameter_array(self):
        # number of steps is one less than len(steps)
        p_nb_steps = make_ecc_parm(self.api, self.CA_parms['nb_steps'], len(self.steps) - 1) 

        # record parameters
        p_record_dt = make_ecc_parm(self.api, self.CA_parms['record_dt'], self.record_dt)
        p_record_dI = make_ecc_parm(self.api, self.CA_parms['record_dI'], self.record_dI)

        # repeating factor
        p_repeat = make_ecc_parm(self.api, self.CA_parms['repeat'], self.repeat_count)

        # make the technique parameter array
        self.final_parameters = make_ecc_parms(self.api, *self.p_steps, p_nb_steps, p_record_dt, p_record_dI, p_repeat)
"""  