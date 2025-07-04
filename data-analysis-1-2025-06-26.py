'''
Mykhailo Vorobiov

Created 2025-06-26
Modified 2025-06-27
'''
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import wiener, find_peaks
from pybaselines import Baseline

import os
from os.path import join

from lmfit.models import GaussianModel

mpl.style.use('custom-style')
# %%
def read_files(dirpath_str: str, 
               key: str='spec', 
               files_idx_ommit: list[int]=[], 
               delimiter=',', 
               comments='#',
               fix_mismatch=True):
    if os.path.exists(dirpath_str):
        print("Provided path exists. ", dirpath_str)
        all_files_list = os.listdir(dirpath_str)
        files_list = [f for f in all_files_list if f.endswith('.csv') and os.path.isfile(os.path.join(dirpath_str, f)) and key in f]
        split_names = [f.split('-') for f in files_list]
        try:
            file_idxs = [int(f[2]) for f in split_names]
        except ValueError:
            print('Error: Cannot convert string file # to integer!')
        valid_files = [f for id,f in zip(file_idxs, files_list) if id not in files_idx_ommit]
        valid_files_sorted = sorted(list(zip(file_idxs,valid_files)))
        valid_files_paths = [join(dirpath_str, s[1]) for s in valid_files_sorted]
        
        data = None
        for f in valid_files_paths:
            current_data = np.genfromtxt(f,delimiter=delimiter,comments=comments)
            print(current_data.shape)
            if data is None:
                data = current_data[np.newaxis]
            else:
                data_dims = data.shape
                current_data_dims = current_data.shape
                mismatch = data_dims[1] - current_data_dims[0]
                if fix_mismatch and mismatch>0:
                    last_row = current_data[-1][np.newaxis].T
                    print('Last arr', last_row.shape)
                    extension_arr = np.repeat(last_row, mismatch, axis=1).T
                    print('Extnshn arr', extension_arr.shape)
                    extended_arr = np.vstack((current_data, extension_arr))
                    print('Extndd arr', extended_arr.shape)
                    data = np.append(data, extended_arr[np.newaxis] , axis=0)
                elif fix_mismatch and mismatch<0:
                    data = np.append(data, current_data[np.newaxis,:data_dims[1]] , axis=0)
                else:
                    data = np.append(data, current_data[np.newaxis] , axis=0)
            print(data.shape)
        return data, valid_files_paths
    else:
        print("Provided path does no exist! ", dirpath_str)
        return None

def calibrate_axis(reference_trace, 
                   peak_separation=137.54814724194335 , 
                   show_plot=False):
    x,y = reference_trace.T
    baseline_fitter = Baseline(x_data=x)
    bkg, _ = baseline_fitter.snip(
            y, max_half_window=90, decreasing=False, smooth_half_window=20
        )
    y_corrected = y - bkg

    # Find peaks
    pidx, pprop= find_peaks(y_corrected, width=[20, 300], distance=20)
    #pidx, pprop= find_peaks(y,height=0.05, distance=50, prominence=0.05, width=10)
    print(pidx)
    pmax =[]
    if len(pidx)>1:
        s = y_corrected[pidx].argsort()
        pmax = np.flip(pidx[s][-2:])
        print(pmax)
        print('Peaks found at:')
        for i,p in enumerate(pidx):
            print(f'{i}: {x[p]:.3} #{p}')
        print('Peaks selected:')
        for i,p in enumerate(pmax):
            print(f'{i}: {x[p]:.3} #{p}')
    else:
        raise ValueError("Could not locate enough peaks!")


    gmodel =  GaussianModel(prefix='D52_') + GaussianModel(prefix='D32_')
    pars = gmodel.make_params(
        D52_center = x[pmax[0]],
        D52_sigma = 0.15,
        D52_amplitude = 1e-2,
        D32_center = x[pmax[1]],
        D32_sigma = 0.15,
        D32_amplitude = 1e-3,
    )
    pars.set(D52_sigma=dict(expr='D32_sigma'))
    pars.set(D52_amplitude=dict(min=1e-3))
    pars.set(D32_amplitude=dict(min=1e-6))

    res = gmodel.fit(y_corrected,pars,x=x)

    best_pars = res.best_values
    main_center = best_pars['D52_center']
    subs_center = best_pars['D32_center']
    sec_to_mhz = peak_separation / np.abs(main_center-subs_center)
    
    if show_plot:
        fig, ax = plt.subplots(3,1,
                               figsize=(4,6),
                               height_ratios=(3,3,1),
                               sharex=True)

        ax[0].plot(x, y_corrected,'orange',label='bg corr.')
        ax[0].plot(x[pidx],y_corrected[pidx],'o',color='red')
        ax[0].set_ylabel('Corr. signal (V)')
        ax[0].legend(loc='upper left')
        ax2 = ax[0].twinx()
        ax2.plot(x,y,label='raw signal')
        ax2.plot(x,bkg,'r--',label='baseline')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Raw signal (V)')


        ax[1].plot(x, y_corrected,label='bg corrected')
        ax[1].plot(x, res.init_fit,'--',label='init guess')
        ax[1].plot(x, res.best_fit,'r-',label='best fit',linewidth=1.5)
        ax[1].set_ylabel('Corr. signal (V)')
        ax[1].legend()
        ax[2].plot(x,res.residual,'g')
        ax[2].set_ylabel('Resid. (V)')
        ax[2].set_xlabel('Time (sec)')
        #plt.plot(x, bkg,'r--')

    return sec_to_mhz, main_center

def bin_image(image, bin_pow):
    return np.array([c.reshape(-1, 2**bin_pow).mean(axis=1) for c in img.T])

def baseline_image(image, plot_rnd_trace=False):

    im_out = np.copy(image.T)
    for i,c in enumerate(image.T):
       baseline_fitter = Baseline(x_data=np.arange(c.shape[0]))
       bkg, _ = baseline_fitter.asls(c, lam=1e5, p=0.1)
       im_out[i] = c - bkg
    baseline_fitter = Baseline(x_data=np.arange(c.shape[0]))

    if plot_rnd_trace:
        nn = im_out.shape[0]
        irnd = np.random.randint(0,im_out.shape[0])
        cc = image.T[irnd]
        #bkg, _ = baseline_fitter.modpoly(c, poly_order=3)
        bkg, _ = baseline_fitter.asls(cc, lam=1e5, p=0.1)
        #bkg, _ = baseline_fitter.snip(
        #        c, max_half_window=12, decreasing=False, smooth_half_window=4
        #    )
        cout = cc - bkg

        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot(cc, label=f'raw trace #{irnd}/{nn}')
        ax.plot(bkg, label='bg ')
        ax.plot(cout, label='corrected')
        ax.set_ylabel('EIT signal (V)')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend()

    return im_out.T

def plot_map(img, sec_to_mhz, main_pos, sweep_freq, fov):
    X = np.linspace(0,fov,img.shape[1])
    Y = (main_pos-np.linspace(0,0.5/sweep_freq,img.shape[0])) * sec_to_mhz
    _, ax =plt.subplots(figsize=(5,4))
    p = ax.pcolormesh(X,Y,img_binned, cmap='jet')
    plt.colorbar(p, ax=ax, label='EIT Signal (%)')
    ax.set_xlabel('Distance along laser beam (mm)')
    ax.set_ylabel('Blue detuning (MHz)')
    return p

def field_reconstruct(image, sec_to_mhz, main_pos, sweep_freq, fov, show_plot=False):
    img = np.copy(image.T)

    X = np.linspace(0,fov,img.shape[0])
    Y = (main_pos-np.linspace(0,0.5/sweep_freq,img.shape[1])) * sec_to_mhz

    prefixes = ['d1_', 'd2_', 'd3_']
    gmodel =  GaussianModel(prefix='d1_') + GaussianModel(prefix='d2_') + GaussianModel(prefix='d3_')+GaussianModel(prefix='d4_')


    results_arr = []
    max_traces_dict = {}
    for i,trace in enumerate(img):
        # Find peaks
        pidx, pprop= find_peaks(trace, height=0.06, distance=2, prominence=0.4, width=2)
        pmax =[]
        if len(pidx)>0:
            s = trace[pidx].argsort()
            pmax = np.flip(pidx[s][-4:])
            max_traces_dict.update({i:pmax})
        else:
            raise ValueError("Could not locate enough peaks!")
        
        pars = gmodel.make_params(
            d1_center = Y[pmax[0]],
            d1_sigma = 8,
            d1_amplitude = 30,
            d2_center = Y[pmax[1]],
            d2_sigma = 8,
            d2_amplitude = 30,
            d3_center = Y[pmax[2]],
            d3_sigma = 8,
            d3_amplitude = 30,
            d4_center = Y[pmax[3]],
            d4_sigma = 8,
            d4_amplitude = 30,
        )
        pars.set(d1_sigma=dict(expr='d2_sigma'))
        pars.set(d1_sigma=dict(expr='d3_sigma'))
        pars.set(d3_sigma=dict(expr='d2_sigma'))
        pars.set(d4_sigma=dict(expr='d3_sigma'))

        res = gmodel.fit(trace,pars,x=Y)
        par_best = res.best_values
        results_arr.append(res)
        print(f'Iteration {i}/{img.shape[0]}\r')
        if show_plot:
            fig, ax = plt.subplots(2,1, sharex=True, height_ratios=(3,1))
            ax[0].plot(Y, trace, 'o-', alpha=0.5)
            ax[0].plot(Y, res.init_fit, '--')
            ax[0].plot(Y, res.best_fit, 'r', linewidth=2)
            ax[0].set_ylabel('EIT signal (%)')
            ax[1].plot(Y, trace-res.best_fit, 'g', linewidth=1)
            ax[1].set_xlabel('Blue detuning (MHz)')
            ax[1].set_ylabel('Resid.')
            print(res.fit_report())
            break

    return results_arr
    
def detect_outliers_zscore(data, threshold=3):
    """
    Detects outliers in a 1D dataset using the Z-score method.

    A data point is considered an outlier if its Z-score (number of standard
    deviations from the mean) exceeds a given threshold.

    Args:
        data (np.ndarray or list): The 1D dataset to analyze.
        threshold (float, optional): The Z-score threshold above which a point
                                     is considered an outlier. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Boolean array where True indicates an outlier.
            - np.ndarray: The Z-scores for each data point.
    """
    data_array = np.asarray(data) # Ensure it's a numpy array
    mean_val = np.mean(data_array)
    std_dev = np.std(data_array)

    if std_dev == 0:
        # Handle case where all data points are the same
        return np.zeros_like(data_array, dtype=bool), np.zeros_like(data_array, dtype=float)

    z_scores = np.abs((data_array - mean_val) / std_dev)
    outlier_mask = z_scores > threshold
    return outlier_mask, z_scores
# %%    
# Read files into memory
ommit_files = [1,2,3,4,5,6,7,10,16,18,19,20,23,28,30,48]
fl_data, fl_files = read_files('./processed/', key='spec', files_idx_ommit=ommit_files)
ref_data, ref_files = read_files('./processed/', key='sync', files_idx_ommit=ommit_files)


# %%
fps = 12.8 # Hz
sweep = 0.05 # Hz
image_number = 22
fov = 10.8

img = np.copy(fl_data[image_number])

stomhz, mainpos = calibrate_axis(ref_data[image_number], show_plot=True)


img_binned = bin_image(img, 3)
img_binned = baseline_image(img_binned, plot_rnd_trace=True)

#%%
X = np.linspace(0,fov,img_binned.shape[1])
#%%

plot_map(img_binned,stomhz,mainpos,sweep,fov)

# %%
reconstr_res = field_reconstruct(img_binned,stomhz,mainpos,sweep,fov)
#%%
plot_labels = ['$44D_{5/2}$: $|m_J|=5/2$', '$44D_{5/2}$: $|m_J|=1/2$', '$44D_{5/2}$: $|m_J|=3/2$', '$44D_{3/2}$: $|m_J|=1/2$']


fig, ax = plt.subplots(3,1, sharex=True, figsize=(4,7))
for i in range(4):
    peaks_pos = np.array([r.params[f'd{i+1}_center'].value for r in reconstr_res], dtype=float)
    pos_err = np.array([r.params[f'd{i+1}_center'].stderr for r in reconstr_res], dtype=float)
    ax[0].errorbar(x=X, y=peaks_pos, yerr=pos_err)

    heights = np.array([r.params[f'd{i+1}_height'].value for r in reconstr_res], dtype=float)
    heights_err = np.array([r.params[f'd{i+1}_height'].stderr for r in reconstr_res], dtype=float)
    ax[1].plot(X, heights, linewidth=1)
    ax[1].fill_between(X, heights-heights_err, heights+heights_err, alpha=0.4)

w = np.array([r.params[f'd1_sigma'].value for r in reconstr_res], dtype=float)
w_err = np.array([r.params[f'd1_sigma'].stderr for r in reconstr_res], dtype=float)
ax[2].plot(X, w, '.', color='k', alpha = 0.6)
ax[2].errorbar(x = X,
                y = w,
                yerr = w_err,
                linestyle='None',
                color='k',
                alpha = 0.3)
ax[2].set_xlabel('Distance (mm)')


fig, ax = plt.subplots(2,2, sharex=True)
axs = ax.flatten()
for i in range(4):
    peaks_pos = np.array([r.params[f'd{i+1}_center'].value for r in reconstr_res], dtype=float)
    pos_err = np.array([r.params[f'd{i+1}_center'].stderr for r in reconstr_res], dtype=float)

    db_mask, _ = detect_outliers_zscore(peaks_pos, threshold=1.5)

    axs[i].plot(X[~db_mask],
                peaks_pos[~db_mask],
                marker='.',
                alpha = 0.5,
                color = f'C{i}'
                )
    axs[i].errorbar(x = X[~db_mask],
                    y = peaks_pos[~db_mask],
                    yerr = pos_err[~db_mask],
                    color = f'C{i}',
                    linestyle='None',
                    alpha = 0.3)
    axs[i].set_title(plot_labels[i])
fig.supxlabel('Distance (mm)', fontsize=14)
fig.supylabel('Blue detuning (MHz)', fontsize=14)
fig.suptitle('Peak positions', fontsize=18)

# %%
print(img[::300].shape)
plt.plot(img_binned[:,::600],'o-', alpha=0.3)
plt.plot()

# %%
spectrum = np.copy(img_binned[:,1000])
plt.plot(Y,spectrum, 'o')
# %%
baseline_fitter = Baseline(x_data=Y)

bkg_1, params_1 = baseline_fitter.modpoly(spectrum, poly_order=8)
bkg_2, params_2 = baseline_fitter.asls(spectrum, lam=1e3, p=0.1)
bkg_3, params_3 = baseline_fitter.mor(spectrum, half_window=7)
bkg_4, params_4 = baseline_fitter.snip(
    spectrum, max_half_window=12, decreasing=False, smooth_half_window=8
)

plt.plot(Y, spectrum, label='raw data', lw=1.5)
plt.plot(Y, spectrum-bkg_4, label='raw data-asls', lw=1.5)
plt.plot(Y, bkg_1, '--', label='modpoly')
plt.plot(Y, bkg_2, '--', label='asls')
plt.plot(Y, bkg_3, '--', label='mor')
plt.plot(Y, bkg_4, '--', label='snip')

plt.legend()
plt.show()


# %%
img_blined = img
for i,r in enumerate(img.T):
    bkg_4, params_4 = baseline_fitter.snip(
        r, max_half_window=12, decreasing=False, smooth_half_window=4
    )
    img_blined[:,i] = r - bkg_4


plt.pcolormesh(X,Y,wiener(img_blined ,5), cmap='jet')

# %%
plt.plot(Y, img_binned[:,170], 'o-', label='raw data', lw=1.5)
# %%
print([(a,b) for a,b, in zip(range(len(fl_files)), fl_files)])
# %%
