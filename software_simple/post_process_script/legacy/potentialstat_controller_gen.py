"""
potentialstat_controller class -- Automate potentialstat gating and acquisition


Remote access
Input: gating protocol from Linux
Output: Saved file in Windows and return dictionary of i(t), v(t) back to Linux

See P10 for the gernal flow file:///C:/EC-Lab%20Development%20Package/EC-Lab%20Development%20Package.pdf

Date: 9/26/2022
Author: Yi-Shiou Duh (allenduh@stanford.edu)
"""

import Pyro4 # for remote access

from datetime import datetime
import time
from dataclasses import dataclass
from control._def_LEAP import *
import numpy as np
from scipy import signal
import sys
import csv
import glob
potentialStatDir = 'C://EC-Lab Development Package//Examples//Python'
sys.path.append(potentialStatDir)
USE_REAL_POTENTIALSTAT = False
# Biologic specific
if USE_REAL_POTENTIALSTAT:
    import kbio.kbio_types as KBIO
    from kbio.kbio_api import KBIO_api

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
        self.address = "USB0"
        self.DLL_path = "C:\\EC-Lab Development Package\\EC-Lab Development Package\\EClib64.dll"  # library used to communicate with the instrument
        if USE_REAL_POTENTIALSTAT:
            self.api = KBIO_api(self.DLL_path) # Open DLL to initialize API
            self.id_, self.device_info = self.api.Connect(self.address) # connect to potentialstat using its address
        self.channel = 1

        ## Define CA Gating protocol
        self.ocv = float(0.0) # unit V
        self.v_plus = float(0.1)  # unit V
        self.v_minus = float(0.0)  # unit V
        self.cycles = 4 # total cycles
        self.duration = float(1.0) # second  # for one phase , either v_plus or v_minus
        self.record_dt = 0.001  # seconds
        self.record_dI = 0.001  # current
        self.duration_ocv = float(0.5) # Define OCV measurement parameters
        
        self.update_steps() # Define self.timeSpan and self.steps

        self.filename_csv = '/Users/allen/octopi-research/software/Example_data/gating_100_-100mV_2022-12-04_15-58-9.791705/iv_comm.csv'

#####################################
## Start channel
#####################################
    def start_channel(self):
        """ start potentialstat channel after preparation i.e. prepare_ocv/ prepare_CA_gating"""  
        if USE_REAL_POTENTIALSTAT:        
            self.api.StartChannel(self.id_, self.channel)    
        self.timeStart = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        self.tic = datetime.now()
        print('Can harvest data after ' + str(self.timeSpan) + 'sec')
        return self.timeStart
            
        
    def abort_gating(self):
        """ abort channel operation early"""  
        if USE_REAL_POTENTIALSTAT:        
            self.api.StopChannel(self.id_, self.channel)  
            
#####################################
## Read iv / measure ocv  
#####################################
    def read_iv(self):
        """ After potentialstat finish gating, read the data stored inside api"""  
        if USE_REAL_POTENTIALSTAT:           
            ## Record data points and store into iv curve
            while True :
                data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
                current_values, data_info, data_record = data  # retrieve data 
                idx_start = 0        # index inside the buffer 
                num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1 when read live (if read after then is all iv curve)
                
                for _ in range(num_dataPoints_to_read_next) :
                    # parse data
                    single_point_length = 5  # for CA. See P71
                    idx_end = idx_start + single_point_length    
                    t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
                    
                    # compute timestamp in seconds using t_high, t_low
                    t_rel = (t_high << 32) + t_low
                    t = current_values.TimeBase * t_rel   
                    
                    # Convert Ewe, I to float i.e 3902889481 into 0.00056
                    Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                    I = self.api.ConvertNumericIntoSingle(I)
                    
                    # store datapoint into ivCurve
                    self.ivCurve["t"].append(t)
                    self.ivCurve["V_we"].append(Ewe)
                    self.ivCurve["I"].append(I)
                    self.ivCurve["cycle"].append(cycle)
                    
                    idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                    
                    # Status, monitoring STOP
                    status = KBIO.PROG_STATE(current_values.State).name  
                    toc = datetime.now()
                    running_time = (toc - self.tic).total_seconds()
                    #print((t, Ewe, I, status))

                if status == 'STOP' or running_time > self.timeSpan : # stop when channel reports it is no longer running
                    break
            return self.ivCurve 
        else:
            return self.read_iv_sim()

    def read_iv_in_memory(self):
        """ During gating, read the data stored inside api"""  
        if USE_REAL_POTENTIALSTAT:           
            ## Record data points and store into iv curve
 
            data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
            current_values, data_info, data_record = data  # retrieve data 
            idx_start = 0        # index inside the buffer 
            num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1 when read live (if read after then is all iv curve)
            
            for _ in range(num_dataPoints_to_read_next) :
                # parse data
                single_point_length = 5  # for CA. See P71
                idx_end = idx_start + single_point_length    
                t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
                
                # compute timestamp in seconds using t_high, t_low
                t_rel = (t_high << 32) + t_low
                t = current_values.TimeBase * t_rel   
                
                # Convert Ewe, I to float i.e 3902889481 into 0.00056
                Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                I = self.api.ConvertNumericIntoSingle(I)
                
                # store datapoint into ivCurve
                self.ivCurve["t"].append(t)
                self.ivCurve["V_we"].append(Ewe)
                self.ivCurve["I"].append(I)
                self.ivCurve["cycle"].append(cycle)
                
                idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                
                # Status, monitoring STOP
                status = KBIO.PROG_STATE(current_values.State).name  
                toc = datetime.now()
                running_time = (toc - self.tic).total_seconds()
                #print((t, Ewe, I, status))

            return self.ivCurve 
        else:
            return self.load_iv_in_csv_after_gating(self.filename_csv)
            #return self.read_iv_sim()

    def read_iv_in_memory(self):
        """ During gating, read the data stored inside api"""  
        if USE_REAL_POTENTIALSTAT:           
            ## Record data points and store into iv curve
 
            data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming          
            current_values, data_info, data_record = data  # retrieve data 
            idx_start = 0        # index inside the buffer 
            num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1 when read live (if read after then is all iv curve)
            
            for _ in range(num_dataPoints_to_read_next) :
                # parse data
                single_point_length = 5  # for CA. See P71
                idx_end = idx_start + single_point_length    
                t_high, t_low, Ewe, I, cycle = data_record[idx_start : idx_end] 
                
                # compute timestamp in seconds using t_high, t_low
                t_rel = (t_high << 32) + t_low
                t = current_values.TimeBase * t_rel   
                
                # Convert Ewe, I to float i.e 3902889481 into 0.00056
                Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                I = self.api.ConvertNumericIntoSingle(I)
                
                # store datapoint into ivCurve
                self.ivCurve["t"].append(t)
                self.ivCurve["V_we"].append(Ewe)
                self.ivCurve["I"].append(I)
                self.ivCurve["cycle"].append(cycle)
                
                idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                
                # Status, monitoring STOP
                status = KBIO.PROG_STATE(current_values.State).name  
                toc = datetime.now()
                running_time = (toc - self.tic).total_seconds()
                #print((t, Ewe, I, status))

            return self.ivCurve 
        else:
            return self.load_iv_in_csv_after_gating()
            #self.read_iv_sim()        

    def read_iv_sim(self):
        """ Simulation counterpart of read_iv. Can use as preview protocol""" 
        numOfPointHalfCycle = 200
        numOfHalfCycle = int((self.cycles + 0.5) * 2)
        totalSamplePoint = numOfPointHalfCycle * numOfHalfCycle
        total_duration = self.duration * numOfHalfCycle  # counts the number of steps * duration
        t = np.linspace(0, total_duration, totalSamplePoint) 
        t_inner = t[numOfPointHalfCycle : totalSamplePoint - numOfPointHalfCycle]

        v_baseline = self.ocv + (self.v_plus + self.v_minus) / 2
        v_amplitude = (self.v_plus - self.v_minus) / 2
        frequency = 1 / (2 * self.duration) #Hz

        v =  np.linspace(self.ocv, self.ocv, totalSamplePoint) # 1024 sample points
        v[numOfPointHalfCycle : totalSamplePoint - numOfPointHalfCycle] = v_baseline - v_amplitude * signal.square(2 * np.pi * frequency * t_inner)

        I =  np.linspace(self.ocv, self.ocv, totalSamplePoint) # 1024 sample points

        # v = 10000 * signal.square(2 * np.pi * 0.2 * t)
        data = {'t': t.tolist(), 'V_we': v.tolist(), 'I': I.tolist(), 'cycle':[5]}
        #print(data)

        return data   
    


    def measure_ocv(self):
        """ Read the ocv data stored in api and update self.ocv, self.step"""  
        if USE_REAL_POTENTIALSTAT:        
            while True :
                data = self.api.GetData(self.id_, self.channel)  # data contineouslt streaming            
                
                current_values, data_info, data_record = data  # retrieve data 
                idx_start = 0        # index inside the buffer 
                num_dataPoints_to_read_next = data_info.NbRows # Usually 0, or 1 when read live (if read after then is all iv curve)
                       
                for _ in range(num_dataPoints_to_read_next) :
                    # parse data
                    single_point_length = 3
                    idx_end = idx_start + single_point_length    
                    t_high, t_low, Ewe = data_record[idx_start : idx_end] 
                    
                    # compute timestamp in seconds using t_high, t_low
                    t_rel = (t_high << 32) + t_low
                    t = current_values.TimeBase * t_rel   
                    
                    # Convert Ewe, I to float i.e 3902889481 into 0.00056
                    Ewe = self.api.ConvertNumericIntoSingle(Ewe) # working electrode potential
                    
                    # store data into ivCurve
                    self.ivCurve["t"].append(t)
                    self.ivCurve["V_we"].append(Ewe)

                    idx_start = idx_end  # read next datapoint in the buffer by shifting index 
                    
                    # Status, monitoring STOP

                    status = KBIO.PROG_STATE(current_values.State).name  
                    toc = datetime.now()
                    running_time = (toc - self.tic).total_seconds()
                    #print((t, Ewe, status))

                if status == 'STOP' or running_time > self.duration_ocv : # stop when channel reports it is no longer running
                    break
                
            # update ocv, self.step with new measured ocv     
            ocv_ave = np.mean(self.ivCurve["V_we"])
            print("ocv measured = " + str(ocv_ave))
            self.ocv = float(ocv_ave)
            self.update_steps()  
            return ocv_ave  ## return an average          


        else:
            self.ocv = float(-0.5)
            self.update_steps()  
            print("Assume ocv = -0.5 (SIMULATION)")
            
        return self.ocv
 
#####################################
## Prepare gating protocol    
#####################################       
    def prepare_ocv(self):
        """ Preparation step before measuring ocv. prepare paramter array"""   
        if USE_REAL_POTENTIALSTAT:           
            if self.firmware_is_running():
                self.p_steps = list()  # append gating protocol into a list. 
                # Summerized technique parameters into a list and feed into potentialstat
                self.p_steps.append(self.make_single_parameter('Rest_time_T', 0, self.duration_ocv))
                self.p_steps.append(self.make_single_parameter('Record_every_dT', 0, self.record_dt))
                self.e_range = 'ERANGE_AUTO'# 'E_RANGE_2_5V' #'E_RANGE_10V' 
                self.p_steps.append(self.make_single_parameter('E_Range', 0, 3)) #see whether this work 3: 'ERANGE_AUTO'
                #self.p_steps.append(self.make_single_parameter('E_Range', 0, KBIO.E_RANGE[self.e_range].value))
                # Summerized into a fianl list and feed into potentialstat
                self.make_final_parameters()
                self.load_technique_CA(tech_file = "ocv4.ecc") # load ocv tech file
                self.ivCurve = {'t':[], 'V_we':[]}  # make sure to initialize ivCurve

                return True 
            else:
                print("firmware not running. See load_firmware")


    def prepare_CA_gating(self):
        """ Preparation step before measuring read_iv. prepare paramter array"""   
        if USE_REAL_POTENTIALSTAT:    
            if self.firmware_is_running():
                self.p_steps = list() # append gating protocol into a list.
                # Summerized technique parameters into a list and feed into potentialstat
                self.append_CA_parameter_list()       
                # Summerized into a fianl list and feed into potentialstat
                #self.p_steps.append(self.make_single_parameter('I_Range', 0, 3))
                self.p_steps.append(self.make_single_parameter('E_Range', 0, 3)) #see whether this work 3: 'ERANGE_AUTO'  ## New added 0207
                #self.p_steps.append(self.make_single_parameter('Bandwidth', 0, integer)) #see whether this work 3: 'ERANGE_AUTO'  ## New added 0207
                self.make_final_parameters()
                self.load_technique_CA()  # tech file default for ocv
                self.ivCurve = {'t':[], 'V_we':[], 'I':[], 'cycle':[]}  # make sure to initialize ivCurve

                return True 
            else:
                print("firmware not running. See load_firmware")

#####################################
## Editing gating protocol    
#####################################
    def output_protocol_parameters(self):
        """ Output gating protocol"""   
        output_info = {'v_plus':self.v_plus, 'v_minus':self.v_minus, 'cycles':self.cycles, 'duration':self.duration, 'ocv': self.ocv, 'timeSpan': self.timeSpan}
        return output_info

    def update_protocol_parameters(self, input_info):
        """ Update gating protocol with new input dict"""   
        print("update_protocol_parameters as:")
        print(input_info)
        self.v_plus = float(input_info['v_plus'])
        self.v_minus = float(input_info['v_minus'])
        self.cycles = float(input_info['cycles'])
        self.duration = float(input_info['duration'])

        self.update_steps()  


    def update_steps(self):
        """ append gating protocol into a list. Update self.step and self.timeSpan"""    
        # Example of steps: VERY important all numbers should be float, not int!
        # self.steps = [  
        #   voltage_step(0.0, 0.0), # 0mV during 3s
        #   voltage_step(0.0, 3.0), # 0mV during 3s
        #   voltage_step(0.1, 3.0), # 100mV during 3s
        #   voltage_step(-0.1, 2.0), # 100mV during 3s
        #   voltage_step(0.0, 0.0, True), # 0.5mA delta during 3s
        # ]
        
        self.steps = [voltage_step(self.ocv, self.duration)]

        for cycle in range (int(self.cycles)):
            self.steps.append(voltage_step(self.ocv + self.v_plus, self.duration))
            self.steps.append(voltage_step(self.ocv + self.v_minus, self.duration))

        self.steps.append(voltage_step(self.ocv, 0.0, True)) # Last one: truncate = True
        self.timeSpan = (self.cycles + 0.5) * 2 * self.duration


#####################################
## Helper function    
#####################################

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
        self.p_steps.append(self.make_single_parameter('N_Cycles', 0, 0)) # let's always assume additional cycles = 0

    def make_single_parameter(self, para_name, idx, value):
        """value is converted to its proper type (ECC_parameter format) 
        this convertion is done by DefineParameter"""  
        parm = KBIO.EccParam()
        self.api.DefineParameter(para_name, value, idx, parm)
        return parm

    def make_final_parameters (self) :
        """Create an EccParam array from an EccParam list, and return an EccParams refering to it."""  
        # Summerized technique parameters into a final parameters list and feed into potentialstat
        self.final_parameters = []
        numOfParms = len(self.p_steps)

        # convert python list to list with ECC_parameter format
        parms_array = KBIO.ECC_PARM_ARRAY(numOfParms)
        for i, parm in enumerate(self.p_steps) :
            parms_array[i] = parm

        self.final_parameters = KBIO.EccParams(numOfParms, parms_array)
  
    def load_technique_CA(self, tech_file = "ca4.ecc", verbosity = 1):
        """ Load chrono-amperometry technique using self.final_parameters"""  
        # See P74 file:///C:/EC-Lab%20Development%20Package/EC-Lab%20Development%20Package.pdf
        #tech_file = "ca4.ecc" # for VMP300 family, use ca4_tech_file        
        #verbosity = 1  # 1: no window pop up. 2: window poped  Can visualize p_step before gating 
        self.api.LoadTechnique(self.id_, self.channel, tech_file, self.final_parameters, first=True, last=True, display=(verbosity>1))
   
    def disconnect(self):
        self.api.Disconnect(self.id_)
        print("DisConnect successfully")
        return True

    def load_iv_in_csv_after_gating(self):
        with open(self.filename_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')

            # get the starting time from the first line
            starting_time_str = next(reader)[1]
            starting_time = datetime.strptime(starting_time_str, '%m/%d/%Y %H:%M:%S.%f')
            starting_time_str_new = starting_time.strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
            print("potentialstat start = "+ starting_time_str_new)

            # dict to output
            data_dict = {'t':[], 'V_we':[], 'I':[], 'starting_time': starting_time_str_new}

            next(reader) # skip the second line

            # read the data into a list of dictionaries
            for row in reader:
                t, Ewe, I = row

                data_dict["t"].append(float(t))
                data_dict["V_we"].append(float(Ewe))
                data_dict["I"].append(float(I))

        return data_dict
    
    def find_latest_csv_filename(self):

        download_dir = "C:\\Users\\allen\Downloads"  # Replace with ECLab_saving_Dir 

        csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
        try:
            latest_csv_file = max(csv_files, key=os.path.getmtime)

            print("Latest CSV file:", latest_csv_file)
            self.filename_csv = latest_csv_file
        except:
            print("Caan't find CSV file")

        

#Minitest:   
def main():
    pc = potentialstat_controller()
    
    # measure ocv
    pc.prepare_ocv()
    pc.start_channel()
    pc.measure_ocv()
    
    # run CA gating 
    pc.prepare_CA_gating()
    pc.start_channel()
    print("started gating and during reading")
    time.sleep(5)
    pc.abort_gating()
    print("after 5 sec, pause gating")
    print("restart and start reading")
    pc.prepare_CA_gating()
    pc.start_channel()
    time.sleep(10)
    print("wait long enough to make sure gating finish")
    data = pc.read_iv()

    import matplotlib.pyplot as plt
    plt.plot(data["t"], data["V_we"])
    plt.show()





if __name__ == "__main__":
    pc = potentialstat_controller()
    data_dict = pc.load_iv_in_csv_after_gating()
    print(data_dict['starting_time'])
    #print(data_dict['t'])
    #print(data_dict['V_we'])
    #print(data_dict['I'])
    #main()
