#%%
from flir_daq import FLIRdaq
from stark_calibrator import StarkCalibrator
from stark_preprocessor import StarkPreprocessor
from stark_processor import StarkProcessor

from qolab.hardware.scope.sds800xhd import SDS800XHD
import pyvisa

import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
counter = 0

#%%
date = datetime.date.today()
dir_path = f'G:/My Drive/Vaults/WnM-AMO/__Data/{date}/data/'
file_prefix = 'data'
file_suffix_list = ['img', 'smap']

fname_images = f'{file_prefix}_{file_suffix_list[0]}[{date}]-{counter}.npz'
fname_smap = f'{file_prefix}_{file_suffix_list[1]}[{date}]-{counter}.npz'


check_path = os.path.join(dir_path, fname_images)
if os.path.exists(check_path):
    print(f"File '{check_path}' exists.")
    counter += 1  # Increment the counter
    fname_images = f'{file_prefix}_{file_suffix_list[0]}[{date}]-{counter}.npz'
    fname_smap = f'{file_prefix}_{file_suffix_list[1]}[{date}]-{counter}.npz'
else:
    print(f"File '{check_path}' does not exist yet.")

print(f"Counter value: {counter}")



# %%
# ----------- Data acquisition ---------------
# Create DAQ object for frames and reference trace acquisition
rm = pyvisa.ResourceManager()
resource = rm.open_resource('TCPIP0::192.168.110.198::inst0::INSTR')
scope = SDS800XHD(resource)

# %%
# Create DAQ object to get camera frames and reference traces from the scope
daq = FLIRdaq(blue_laser_sweep_hz=0.02,
              fps_hz=14.67,
              full_resolution=(1536,1024),
              file_id=fname_images) 
daq.acquire_frames(camera_index=0) # request frames from camera
daq.acquire_reference_trace(scope_visa_resource=scope) # get the EIT reference trace
daq.add_comment('PINQUED project...') # Add comment to the image batch
daq.print_info() # Print information on the data just taken

daq_dict = daq.get_data_dict() # Store daq results in a dictionary
#%%
daq.save_hdf5('test.h5')
# %%
# ----------- Frequency axis calibration ---------------
# Create calibrator object and initialze it with DAQ data dictionary
smc = StarkCalibrator(daq_dict,
                      eit_ref_channel=1,
                      trig_ref_channel=4)
#%%
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
                              width=[10,700],
                              prominence=0.00, 
                              distance=300)
# Print `cal_dict` content
for c in cal_dict:
    print(f'{c}: {cal_dict[c]}')


# Plot calibration results
fig, ax = smc.plot_calibration()
fig.set_size_inches(6, 10)

calibrated_images = smc.get_data_dict()
# %%
smpp = StarkPreprocessor(calibrated_images)
smap = smpp.get_single_stark_map()

# %%
smap.plot() # plot stark map

smap_processor = StarkProcessor(smap)
# smap_processor.baseline_map(mode='snip',
#                             max_half_window = 70, 
#                             decreasing = True,
#                             smooth_half_window = 20)
# smap_processor.baseline_map(mode='arpls',
#                             lam=4e7)

map_tmp = smap_processor.get_processed_map()

map_control, f, d = map_tmp.get_map_data()
print(map_control.shape)
fig, ax = plt.subplots()
for trace, dist in zip(map_control.T[::300], d[::300]):
    ax.plot(f, trace, label=f'Dist: {dist:.2f} mm')
ax.plot(f, np.mean(map_control, axis=1), 'o', color='r', label='Total Mean')
ax.set_xlabel('Blue detuning (MHz)')
ax.set_ylabel('EIT signal (%)')
ax.legend()


smap_processor.horizontal_binning(0)
smap_corrected = smap_processor.get_processed_map()
smap_corrected.plot()
smap_corrected.add_comment('Raw Stark map without processing.')

map_control_proc, f, d = smap_corrected.get_map_data()
print(map_control_proc.shape)
fig, ax = plt.subplots()
for trace, dist in zip(map_control_proc.T[::200], d[::200]):
    ax.plot(f, trace, label=f'Dist: {dist:.2f} mm')
ax.plot(f, np.mean(map_control_proc, axis=1), 'o', color='r', label='Total Mean')
ax.set_xlabel('Blue detuning (MHz)')
ax.set_ylabel('EIT signal (%)')
ax.legend()


#%%
# ----------- Save image stack/calbrated Stark map (frequency calibrated) ---------------
overwrite = False
if os.path.exists(check_path) and overwrite:
    print(f"Warning: File '{check_path}' already exists.\nOVERWRITING!")
    smc.save_images(os.path.join(dir_path, fname_images))
    smap_corrected.save(os.path.join(dir_path, fname_smap))
elif os.path.exists(check_path) and not overwrite:
    print(f"Warning: File '{check_path}' already exists.\nChange flag if overwriting...")
else:
    print(os.path.join(dir_path, fname_images))
    smc.save_images(os.path.join(dir_path, fname_images))
    smap_corrected.save(os.path.join(dir_path, fname_smap))
# %%

sss_map = smap.get_map_data()