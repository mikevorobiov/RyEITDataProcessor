#%%
import os
import re
import datetime as dt

from flir_daq import FLIRdaq
from hardware_manager import HardwareManager
from stark_preprocessor import StarkPreprocessor
from stark_calibrator import StarkCalibrator
from stark_preprocessor import StarkPreprocessor
import logging

import pandas as pd
import pyperclip as pyp

#%%
# DEFINE FUNCTIONS
def increment_filename_index(file_path: str) -> str:
    """
    Checks if a file exists and, if it does, increments a numerical index 
    in the filename until a unique path is found.

    The function looks for a pattern like '_XXX' or '(XXX)' at the end of the 
    filename (before the extension), where XXX is a number. If no pattern 
    is found, it starts with index '_001'.

    Parameters
    ----------
    file_path : str
        The initial path to check (e.g., 'data/run_test.h5').

    Returns
    -------
    str
        A unique, non-existent file path (e.g., 'data/run_test_003.h5').
    """
    
    # Check if the file already exists. If not, return the original path.
    if not os.path.exists(file_path):
        return file_path, 0

    # 1. Separate the directory, name, and extension
    dirname, basename = os.path.split(file_path)
    filename, ext = os.path.splitext(basename)

    # Regex to find a number at the end, possibly preceded by '_' or enclosed in '()'
    # Captures the prefix and the index number.
    # Group 1: Prefix (e.g., 'run_A')
    # Group 2: The numerical index (e.g., '001')
    pattern = re.compile(r'^(.*)[_\-]?(\d+)$') 
    
    match = pattern.search(filename)
    
    # 2. Determine the base name and starting index
    if match:
        # If an index exists: use the non-numeric prefix and the found index.
        base_name = match.group(1).rstrip('_-')
        start_index = int(match.group(2))
        
        # Determine the number of leading zeros for formatting
        padding = len(match.group(2))
    else:
        # If no index exists: use the full filename as the base and start at 0.
        base_name = filename
        start_index = 0
        padding = 3 # Default padding, e.g., '_001'

    current_index = start_index + 1
    
    # 3. Loop until a unique file path is found
    while True:
        # Format the new index with the determined padding
        new_index_str = str(current_index).zfill(padding)
        
        # Construct the new filename (using '_' as a separator)
        new_filename = f"{base_name}_{new_index_str}{ext}"
        new_path = os.path.join(dirname, new_filename)

        if not os.path.exists(new_path):
            return new_path, current_index

        current_index += 1
        
        # If the index outgrows the original padding (e.g., goes from 999 to 1000), 
        # update the padding dynamically.
        if len(str(current_index)) > padding:
             padding = len(str(current_index))


#%%
# -------------- CONFIGURE LOGGER ----------------- 
# Logging Configuration 
# 1. Get the module logger
logger = logging.getLogger() 
logger.setLevel(logging.INFO) 
# Add a console handler and formatter IF one isn't already present
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#%%
# -------------- INITIALIZE EQUIPMENT ----------------- 
address_dict = {
    'scope2': ('SDS800XHD', 'TCPIP0::192.168.110.198::inst0::INSTR'),
    'Grid Current': ('HP3457A', 'visa://192.168.194.15/GPIB1::22::INSTR'),
    'Anode Current': ('HP_34401', 'visa://192.168.194.15/ASRL12::INSTR'),
    'Chamber Current': ('BK_5491', 'visa://192.168.194.15/ASRL9::INSTR'),
    'Pressure': ('MKS390', 'visa://192.168.194.15/ASRL13::INSTR'),
    'Anode Source': ('PSW25045', 'visa://192.168.194.15/ASRL10::INSTR'),
    'Filaments Bias': ('GPP3610H', 'visa://192.168.194.15/ASRL11::INSTR'),
    'Heating': ('KeysightE36231A', 'visa://192.168.194.15/USB0::0x2A8D::0x2F02::MY61003701::INSTR'),
}
plasma_setup = HardwareManager(address_dict)
instruments = plasma_setup.get_resources()


#%%
# -------------- DATA ACQUISITION -----------------
#    
# Create DAQ object for acquisition and initial pre-processing
daq = FLIRdaq(blue_laser_sweep_hz=0.02,
              fps_hz=14.67,
              full_resolution=(1536,1024)) 
# Since Camera is not necessarily available, acquire_frames will use dummy data
daq.acquire_frames(camera_index=0) 
daq.acquire_reference_trace(instruments['scope2'])
# Get plasma auxillary parameters like pressure, currents and voltages!
plasma_state_dict = plasma_setup.get_readings()
plasma_state_dict["Timestamp"] = dt.datetime.now().time().strftime('%H:%M:%S.%f')
daq.print_info()

#%%
# -------------- CALIBRATION AND PREPROCESSING  -----------------


daq_dict = daq.get_data_dict() # Store daq results in a dictionary
# Create calibrator object and initialze it with DAQ data dictionary
smc = StarkCalibrator(daq_dict,
                      eit_ref_channel=1,
                      trig_ref_channel=4)
# Make `Calibrator` perform reference curve basilining 
# using SNIP algorithm
# from `pybaselines` (see package docs for parameters)
smc.correct_ref_baseline(max_half_window = 120, 
                         decreasing = False,
                         smooth_half_window = 5)
# Calculate map frequency samples in MHz and 
# store relevant calibration parameters in
# the dictionary for later access
cal_dict = smc.calibrate_axis(sigma_scale=7e-2,
                              height=0.02,
                              width=[0,700],
                              prominence=0.00, 
                              distance=300)
# Print `cal_dict` content
for c in cal_dict:
    print(f'{c}: {cal_dict[c]}')

# Plot calibration results
fig, ax = smc.plot_calibration()
fig.set_size_inches(6, 10)

calibrated_images = smc.get_data_dict()


#%%
# Get Stark Map
preproc = StarkPreprocessor(calibrated_images)
smap = preproc.get_single_stark_map()
fig_map, ax_map = smap.plot() # plot stark map





# %%
# -------------- SAVE DATA -----------------
# Create the 'data' directory if it doesn't exist
working_dir = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-09-30'
date = dt.datetime.now().date() 
data_labels = ['imgs', 'smap', 'params']
saving_dir = os.path.join(working_dir, 'data')

os.makedirs(saving_dir, exist_ok=True)
file_params_path = os.path.join(saving_dir, f'parameters-{date}.csv')
file_daq_path, index = increment_filename_index(os.path.join(saving_dir, f'data-imgs-{date}_0.h5'))
file_smap_path = os.path.join(saving_dir, f'data-smap-{date}_{index}.h5')
file_plot_path = os.path.join(saving_dir, f'plot-smap-{date}_{index}.png')

smap.set_file_id(os.path.basename(file_daq_path))
smap.save_hdf5(file_smap_path)
fig_map.savefig(file_plot_path)
daq.save_hdf5(file_daq_path)

df = pd.DataFrame()
state_dict = {"File #": index}
state_dict.update(plasma_state_dict)
if os.path.exists(file_params_path):
    new_row = pd.DataFrame([state_dict])
    df = pd.read_csv(file_params_path, index_col=False)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_params_path, index=False)
else:
    dict = {k: [v] for k,v in state_dict.items()}
    df = pd.DataFrame(dict)
    df.to_csv(file_params_path, index=False)

# Generate MD table entry
smap_fname = os.path.basename(file_smap_path)
imgs_fname = os.path.basename(file_daq_path)
plot_fname = os.path.basename(file_plot_path)
md_table_entry = "|"+"|".join([str(index), f"[[{smap_fname}\|link]]", f"![[{plot_fname}\|200]]"])+"||"
pyp.copy(md_table_entry)
print(md_table_entry)

# %%
