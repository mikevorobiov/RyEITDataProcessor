'''
Mykhailo Vorobiov

Created: 2025-08-20
Modified: 2025-08-20
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import pandas as pd
import re

from lmfit import CompositeModel # Ensure CompositeModel is imported
from lmfit.models import GaussianModel, ConstantModel, LorentzianModel


from stark_map import StarkMap
from pybaselines import Baseline
import copy
from os.path import join
from scipy.signal import find_peaks
#%%
class StarkCalibrator():
    '''
    Class for calibrating StarkMaps.
    '''
    def __init__(self,
                 daq_data_dict: dict | str,
                 eit_ref_channel: int = 1,
                 trig_ref_channel: int = 4,
                 known_peak_separation: float = 137.54814724194335, # MHz
                 trigger_threshold: float = 2.5 # Volts
                ):

        self.known_peak_separation = known_peak_separation
        self.eit_ref_channel = eit_ref_channel
        self.trig_ref_channel = trig_ref_channel
        
        self.data_dict = daq_data_dict

        self.time_stop = daq_data_dict['time_stop']
        self.map_time = np.linspace(0, daq_data_dict['time_stop'], daq_data_dict['images_stack'].shape[0])
        self.map_freq = None



        trigger_signal = daq_data_dict['reference_signals_volt'][trig_ref_channel]
        self.reference_trig = trigger_signal
        trigger_index = self._detect_trigger(trigger_signal, trigger_threshold)
        self.reference_time = daq_data_dict['reference_signals_volt'][0, trigger_index:]
        self.reference_eit = daq_data_dict['reference_signals_volt'][eit_ref_channel, trigger_index:]
        
        self.reference_corrected = None
        self.reference_background = None
        self.calibrations_dict = None

        self.map_frequency_mhz = None
       

    def _detect_trigger(self, trace, threshold: float = 2.5):
        print(trace.shape)
        above_threshold = trace > threshold
        crossings = np.diff(above_threshold.astype(int))
        upward_crossings_indices = np.where(crossings == 1)[0]
        print(upward_crossings_indices)
        return upward_crossings_indices[0]

    def correct_ref_baseline(self,**kwargs):
        # Assuming raw_reference_trace is structured as [[x1, x2, ...], [y1, y2, ...]]
        x_data = self.reference_time
        y_data = self.reference_eit

        # Initialize the Baseline fitter
        baseline_fitter = Baseline(x_data=x_data)

        # Apply the SNIP algorithm
        # The second return value (params) is discarded as it's not used
        background, _ = baseline_fitter.snip(y_data,**kwargs)

        # Correct the signal by subtracting the baseline
        y_corrected = y_data - background

        self.reference_corrected = y_corrected
        self.refrence_background = background

        return y_corrected, background
    
    
    def calibrate_axis(self, sigma_scale = 1e-2, **kwargs):
        """
        """
        x = self.reference_time
        y = self.reference_corrected
           
        ppos, _ = find_peaks(y,**kwargs)
        print(ppos)

        # Fit two-gaussians model        
        gmodel = GaussianModel(prefix='d52_') + GaussianModel(prefix='d32_') + ConstantModel(prefix='c_')

        time_separation = np.abs(x[ppos[0]] - x[ppos[1]])
        sigma_init = sigma_scale * time_separation

        pars = gmodel.make_params(
            d52_center = x[ppos[0]],
            d52_sigma = sigma_init,
            d52_amplitude = y[ppos[0]] * sigma_init * np.sqrt(2*np.pi),
            d32_center = x[ppos[1]],
            d32_sigma = sigma_init,
            d32_amplitude = y[ppos[1]] * sigma_init * np.sqrt(2*np.pi),
            c_c = 0.0
        )
        pars.set(d52_sigma = {"expr": 'd32_sigma'})

        ref_fit_results = gmodel.fit(y,pars,x=x)

        best_pars = ref_fit_results.best_values
        main_center = best_pars['d52_center']
        subs_center = best_pars['d32_center']

        sec_to_mhz = self.known_peak_separation / np.abs(main_center-subs_center)

        print(self.map_time)
        self.map_freq = (self.map_time - main_center) * sec_to_mhz

        out_calibrations_dict = {"seconds_to_mhz": sec_to_mhz,
                                 "main_peak_sec": main_center,
                                 "subsid_peak_sec": subs_center,
                                 "peak_indices": ppos[:2],
                                 "lmfit_result": ref_fit_results,
                                 "bg_samples": self.reference_background,
                                 "map_frequency_mhz": self.map_freq,
                                 }
        self.calibrations_dict = out_calibrations_dict
        
        return out_calibrations_dict
    
    def get_data_dict(self) -> dict:
        self.data_dict['map_frequency_mhz'] = self.map_freq
        return self.data_dict
    
    def save_images(self, 
                    file_path:str
                    ):
        '''
        Save images, reference trace and other information for further 
        processing and calibration.
        '''
        print("Saving images to: %s" % file_path)

        file_name = re.split(r'[\\/]', file_path)[-1]
        self.file_id = file_name
    
        output_dict = self.get_data_dict()

        print('Saving imags to \'.npz\' file...')
        try:
            np.savez(file_path, **output_dict)
        except IOError as e:
                print(f"Error saving dictionary to file: {e}")
        print(f"Dictionary successfully saved to {file_path}")

    def plot_calibration(self,
                     figsize: tuple = (4, 6),
                     height_ratios: tuple = (3, 3, 1),
                     main_title: str = "Reference EIT Calibration Plot",
                     y_label_corrected_signal: str = 'Corr. signal (V)',
                     y_label_raw_signal: str = 'Raw signal (V)',
                     y_label_residual: str = 'Resid. (V)',
                     x_label_time: str = 'Time (sec)'
                     ):
        # --- 1. Data Retrieval and Validation ---
        x_data = self.reference_time
        y_data_raw = self.reference_eit

        bkg = self.refrence_background
        pidx = self.calibrations_dict["peak_indices"]
        fit_result = self.calibrations_dict["lmfit_result"]
        y_data_corrected = self.reference_eit - self.refrence_background
        # --- 2. Figure and Axes Setup ---
        fig, ax = plt.subplots(3, 1,
                               figsize=figsize,
                               height_ratios=height_ratios,
                               sharex=True
                               )
        fig.suptitle(main_title, fontsize=14, y=1.02) # Adds a main title to the figure

        # --- 3. Plotting Subplot 1: Raw, Baseline, and Corrected Signal with Peaks ---
        ax[0].plot(x_data, y_data_corrected, 'C0', label='Corrected Signal')
        if pidx.size > 0: # Only plot peaks if any were found
            ax[0].plot(x_data[pidx], y_data_corrected[pidx], 'o', color='red', markersize=6, label='Detected Peaks')
        ax[0].set_ylabel(y_label_corrected_signal)
        ax[0].legend(loc='upper left', frameon=False) # frameon=False for cleaner look

        # Twin axis for raw signal and baseline
        ax0_twin = ax[0].twinx()
        ax0_twin.plot(x_data, y_data_raw, 'C1', label='Raw Signal')
        ax0_twin.plot(x_data, bkg, 'k', linestyle='--', label='Baseline')
        ax0_twin.set_ylabel(y_label_raw_signal)
        ax0_twin.legend(loc='upper right', frameon=False)


        # --- 4. Plotting Subplot 2: Corrected Signal with Fit ---
        ax[1].plot(x_data, y_data_corrected, 'C0', label='Corrected Signal')
        ax[1].plot(x_data, fit_result.init_fit, '--', color='orange', label='Initial Guess') # Use specific color for clarity
        ax[1].plot(x_data, fit_result.best_fit, 'r-', label='Best Fit', linewidth=1.5)
        ax[1].set_ylabel(y_label_corrected_signal)
        ax[1].legend(frameon=False)


        # --- 5. Plotting Subplot 3: Residuals ---
        ax[2].plot(x_data, fit_result.residual, 'g', label='Residuals')
        ax[2].set_ylabel(y_label_residual)
        ax[2].set_xlabel(x_label_time)
        ax[2].legend(frameon=False)
        ax[2].axhline(0, color='black', linestyle=':', linewidth=1) # Add a zero line for residuals

        # --- 6. Final Touches ---
        # Adjust layout to prevent labels from overlapping
        fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to make space for suptitle

        return fig, ax

#%%
if __name__=='__main__':
    ...
# %%
