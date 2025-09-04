'''
Mykhailo Vorobiov

Created: 2025-08-26
Modified: 2025-08-26
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from stark_map import StarkMap
import copy
#%%
class StarkPreprocessor():
    '''
    Class for processing image stack from FLIRdaq and StarkCalibrator 
    into StarkMaps.
    '''
    def __init__(self, daq_dict: dict):
        
        self.data_dict = daq_dict

        self.images_stack_original = daq_dict['images_stack']
        self.images_stack_processed = daq_dict['images_stack'].copy()

        self.horizontal_dist_mm = np.linspace(0,daq_dict['horizontal_mm'],daq_dict['images_stack'].shape[1])
        self.vertical_dist_mm = np.linspace(0,daq_dict['vertical_mm'],daq_dict['images_stack'].shape[2])

    def _apply_binning_to_trace(self, signal, bin_power):
        '''
        Bin single array of size integer multiple of 2
        '''
        bin_size = 2**bin_power
        binned_signal = signal.reshape(-1, bin_size).mean(axis=1)
        return binned_signal

    def vertical_binning(self, bin_power=1):       
        # Determine the map to use for processing
        images_to_process = self.images_stack_original
        
        images_binned = np.apply_along_axis(self._apply_binning_to_trace, 
                                         1, 
                                         images_to_process, 
                                         bin_power=bin_power)
        

        # Add the new comment to the processed map
        self.data_dict['comments'].append(f'[{dt.date.today()}] Vertical binning: `2**{bin_power}`')
        self.images_stack_processed = images_binned
        return self.images_stack_processed
    
    def horizontal_binning(self, bin_power=1):       
        # Determine the map to use for processing
        images_to_process = self.images_stack_original
        
        images_binned = np.apply_along_axis(self._apply_binning_to_trace, 
                                         2, 
                                         images_to_process, 
                                         bin_power=bin_power)
        
        # Add the new comment to the processed map
        self.data_dict['comments'].append(f'[{dt.date.today()}] Vertical binning: `2**{bin_power}`')
        self.images_stack_processed = images_binned
        return self.images_stack_processed
    
    def detect_overexposed(self, roi:int=-1):
        images_to_process = self.images_stack_processed
        nx, _, _ = images_to_process.shape
        medians = np.mean(images_to_process[:,:,:roi].reshape(nx, -1),axis=1)
        medians_divisor = medians[:, np.newaxis, np.newaxis]/medians.mean()
        self.images_stack_processed = images_to_process/medians_divisor
        return medians

    def get_single_stark_map(self):
        n_freq, _, _ = self.images_stack_processed.shape
        vertical_mean = self.images_stack_processed.mean(axis=2)
        ones_mat = np.ones(vertical_mean.shape)
        base_intensity_trace = vertical_mean[0,:]
        base_intensity = np.repeat(base_intensity_trace[np.newaxis,:], 
                                   repeats=n_freq,
                                   axis=0)
        stark_map = (ones_mat - vertical_mean/base_intensity)*100

        output_map = StarkMap(file_id=self.data_dict['file_id'],
                              map_data=stark_map,
                              frequency_mhz=self.data_dict['map_frequency_mhz'],
                              distance_mm=self.horizontal_dist_mm)

        return output_map
    
