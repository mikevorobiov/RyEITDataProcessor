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

from stark_map import StarkMap
from pybaselines import Baseline
import copy
#%%
class StarkMapAnalyzer():
    '''
    Class for analyzing StarkMaps.
    '''
    def __init__(self, map: StarkMap):
        self.map = map

        self.theory_stark_shifts = None
        self.e_field_theory = None
        self.stark_components_names = None

    def lineshape(self, x, loc, scale, amp):
        return amp * np.exp(-0.5*((x-loc)/scale)**2) / (scale*np.sqrt(2*np.pi))
    
    def model_signal(self, x, *args):
        signal = np.zeros(len(x))
        for m, s, a in zip(*args):
            signal += self.lineshape(x, m, s, a)
        return signal

    def matched_filter(self, signal, signal_ref, mode='same'):
        convolve = np.convolve(signal, np.flip(signal_ref), mode)
        return convolve
    
    def infer_e_field_value(self, spec, freq, *args, s=2, sg=0.1):
        e_field = self.e_field_theory
        shifts = self.theory_stark_shifts
        gradients = np.abs(np.gradient(shifts, axis=1))
        width =s*(1+sg*(gradients - np.repeat(gradients.T[0][:,np.newaxis], 
                             repeats=gradients.shape[1], 
                             axis=1))**2)
        
        model_map = np.zeros((len(e_field), len(freq)))
        for i, (loc, w) in enumerate(zip(shifts.T, width.T)):
            model_trace = self.model_signal(freq, loc, w, *args)
            model_map[i] = model_trace * (wiener(spec,10).max()/model_trace.max())

        mpfunc = self.matched_filter
        matched_filter_map = np.apply_along_axis(mpfunc, 1,
                                                 model_map,
                                                 signal_ref=spec)

        spec_map_repeated = np.repeat(spec[np.newaxis,:], model_map.shape[0], axis=0)
        loss_map = (matched_filter_map - 10*np.abs(spec_map_repeated - model_map))**2
        
        flattened_argmax = loss_map.argmax()
        # flattened_argmax = matched_filter_map.argmax()
        max_indx_efield, max_indx_freq  = np.unravel_index(flattened_argmax, matched_filter_map.shape)
        max_indxs = [max_indx_efield, max_indx_freq]
        max_location = [e_field[max_indx_efield], freq[max_indx_freq]]

        plt.figure()
        plt.plot(matched_filter_map.sum(axis=1))
        plt.figure()
        plt.pcolormesh(e_field, freq, loss_map.T, cmap='jet')
        plt.figure()
        plt.pcolormesh(e_field, freq, matched_filter_map.T, cmap='jet')
        print(max_location)
        output_dict = {
            "matched_map": matched_filter_map,
            "model_map": model_map,
            "max_location": max_location,
            "max_indexes": max_indxs
        }
        return output_dict

    def infer_e_field_profile(self, freq, *args, s=2, sg=0.1):

        map = self.map
        inference_func = self.infer_e_field_value
        np.apply_along_axis(inference_func, 1, map, freq, np.array([2.5,2,2,1,1]), s=1, sg=2)


    def load_theory_shifts(self, file_path, interpolate_n=800):
        shifts_df = pd.read_csv(file_path, index_col=0)
        self.e_field_theory = shifts_df['E'].to_numpy()
        self.stark_components_names = shifts_df.columns[1:] # Skip 'E' column
        self.theory_stark_shifts = shifts_df[self.stark_components_names].to_numpy().T
        
        if interpolate_n < len(self.e_field_theory):
            return True

        self.interpolate_theory_shifts(interpolate_n)
        return True
    
    def plot_theory_shifts(self):
        if self.theory_stark_shifts is None:
            print('ERROR: Load theoretical Stark shifts before trying to plot!')
            return None
        e_field = self.e_field_theory
        fig, ax = plt.subplots()
        for i,name in enumerate(self.stark_components_names):
            shift = self.theory_stark_shifts[i]
            ax.plot(e_field, shift, label=name)
        ax.set_xlabel('Electric field (V/cm)')
        ax.set_ylabel('Blue detunung (MHz)')
        ax.legend()
        ax.grid(True)
        return fig, ax
        
    def _interpolate_trace(self, y, x, xi):
         return np.interp(xi, x, y)

    def interpolate_theory_shifts(self, interpolate_n=800):
        
        e_field = self.e_field_theory
        
        e_field_start = e_field[0]
        e_field_stop = e_field[-1]


        e_field_interp = np.linspace(e_field_start, e_field_stop, interpolate_n)

        shifts = self.theory_stark_shifts
        shifts_interp = np.apply_along_axis(self._interpolate_trace, 1, shifts, x=e_field, xi=e_field_interp)

        self.e_field_theory = e_field_interp
        self.theory_stark_shifts = shifts_interp



# %%
if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from stark_map import StarkMap
    from stark_processor import StarkMapProcessor

    from scipy.signal import wiener
    from scipy.ndimage import median_filter

    sm = StarkMap('data-smap-8-2025-08-21.npz').load('G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-21\\processed_maps\\data-smap-8-2025-08-21.npz')
    fig, ax = sm.plot()
#%%
    smp = StarkMapProcessor(sm)
    sm_new = smp.horizontal_binning(3)
    sm_new = smp.baseline_map(mode='arpls', lam=5e4)
    sm_new = smp.median_baseline()
    sm_new.plot()
    sm_new.print_info()

    sma = StarkMapAnalyzer(sm_new)
    sma.load_theory_shifts('G:\\My Drive\\Vaults\\WnM-AMO\\__Scripts\\calculated_sark_map.csv',
                           interpolate_n=1000)
#%%
    fig, ax = sma.plot_theory_shifts()
#%%
    plt.figure()
    d = 5.0
    _, spec_orig = sm.get_spectrum(d)
    print(spec_orig.shape)
    freq, spec = sm_new.get_spectrum(d)
    infer_dict = sma.infer_e_field_value(spec_orig, freq, np.array([2.5,2,2,1,1]), s=2, sg=4)
    plt.figure()
    #spec = wiener(spec, 10)
    plt.plot(freq, spec, 'o-', alpha=0.3, label='Corrected')
    plt.plot(freq, spec_orig, 'v-', alpha=0.2, label='Original')
    plt.legend()
    plt.axhline(y=0.0, color='k', linestyle='--')
    
    idx_e = infer_dict["max_indexes"][0]
    matched_trace = infer_dict["model_map"]
    print(matched_trace.shape)
    print(idx_e)

    plt.plot(freq, matched_trace[idx_e], c='C3')
    
# %%
