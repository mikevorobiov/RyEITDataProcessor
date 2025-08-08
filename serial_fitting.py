#%%
from field_reconstructor import FLStarkMapProcessor, read_files

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
#ommit_files = [1,2,3,4,5,6,7,10,16,18,19,20,23,28,30,48]
omit_files = [0,1,2,3,4,5,6,7,8,10,11,12,13,21,108, 68]
#dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-06-26\\processed'
dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-07-01\\processed'
fl_data, fl_files = read_files(dirpath, key='spec', files_idx_ommit=omit_files)
ref_data, ref_files = read_files(dirpath, key='sync', files_idx_ommit=omit_files)
#%%
print(fl_files)
fl_idx = [int(f.split('\\')[-1].split('-')[2]) for f in fl_files]


#%%
fps = 12.8 # Hz
sweep = 0.05 # Hz
fov = 10.8 # mm

#%%
file_number = 46

n_files = len(fl_files)
n_par = 3
n_peaks_max = 5
fit_parameters = np.full((n_files, n_peaks_max), np.nan)
amp_dict = {}
peaks_labels = ['D[5/2][1/2]', 'D[5/2][3/2]', 'D[3/2][1/2]', 'D[5/2][5/2]', 'D[3/2][3/2]']
par_names = ['centers', 'amplitudes', 'fwhm', 'centers_err', 'amplitudes_err', 'fwhm_err']
expected_positions = [43.59715211,-50.79838943,-185.02709302,-414.56650897,-585.25113147] # MHz
tested_peaks_models = [2,3,4,5]
tolerance = 10.0
for i,(f,r) in enumerate(zip(fl_data, ref_data)):
    print(fl_idx[i])
    if i:
        fsm = FLStarkMapProcessor(f,
                                  r.T,
                                  fps,
                                  sweep,
                                  fov,
                                  bin_power=7)
        # fsm.plot_calibration(figsize=(6,6))
        # fsm.plot_stark_map(False)
        # fsm.plot_corrected_stark_map(figsize=(6,5))
        # fsm.plot_image_slice(8)
        fsm.fit_stark_map(tested_peaks_models, 
                          verbose=False, 
                          peak_height=0.5, 
                          peak_prominence=0.5, 
                          peak_distance=1)

        x_data, results_dict, n_peaks_max = fsm.get_fit_results()
        centers = results_dict['centers'].T[0]
        amp = results_dict['amplitudes'].T[0]
        print(centers)
        for id, c in enumerate(centers):
            print(c)
            if not c==np.nan:
                idx = fsm._nearest_index(c, expected_positions)
                print(idx)
                if np.abs(expected_positions[idx]-c) < tolerance:
                    fit_parameters[i,idx] = amp[id]

        
#%%
fig, ax = plt.subplots()
ax.plot(fit_parameters)


# %%
# x_data, results_dict, _ = fsm.get_fit_results()
# print(results_dict['centers'].T[0])
# print(results_dict['centers_err'].T[0])
# print(results_dict['amplitudes'].T[0])
# print(results_dict['amplitudes_err'].T[0])
# print(results_dict['fwhm'].T[0])
# print(results_dict['fwhm_err'].T[0])

# %%
