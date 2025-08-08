# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('custom-style')

import ASDCache
from ASDCache import SpectraCache, BibCache

import Moose
import pandas as pd

from pybaselines import Baseline

# %%
nist  = SpectraCache()
# %%
lines_Ar_I = nist.fetch('Ar I')
lines_Ar_II = nist.fetch('Ar II')

        
# %%

ar_spectrum_file = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Documents\\IX_Spectrum-Ar-discharge-lamp-2025-01-20.csv'
ar_wl_raw, ar_spectrum = np.genfromtxt(ar_spectrum_file, delimiter=',', skip_header=1).T[:,100:]
ar_wl = ar_wl_raw - 0.8

eV_in_wn = 8065.544  # express energy in eV
Te = 0.7
asd_wl = lines_Ar_I['obs_wl_vac(nm)'].to_numpy()
weighted_lines = lines_Ar_I['Aki(s^-1)'] * (lines_Ar_I.g_k/lines_Ar_I.g_i) * np.exp(-lines_Ar_I["Ek(cm-1)"] / eV_in_wn / Te)


baseline_fitter = Baseline(x_data=ar_wl)
#bkg_2, params_2 = baseline_fitter.asls(ar_spectrum, lam=1e9, p=0.01)
bkg_2, params_2 = baseline_fitter.snip(
    ar_spectrum, max_half_window=40, decreasing=False, smooth_half_window=2
)
# %%
ar_spectrum_bg = ar_spectrum - bkg_2

fig, ax = plt.subplots()
ax.plot(ar_wl, ar_spectrum)
ax.plot(ar_wl, bkg_2, 'k--')
ax.set_yscale('log')

#%%

wl_interval = [760, 770]
mask_spec = (ar_wl >= wl_interval[0]) & (ar_wl <= wl_interval[1])
max_y_in_range = np.max(ar_spectrum_bg[mask_spec])
print(max_y_in_range)


mask_asd = (asd_wl >=wl_interval[0]) & (asd_wl <=wl_interval[1])
asd_idx = mask_asd.argmax()
asd_intensity = weighted_lines.to_numpy()[asd_idx]

# mask_int = (weighted_lines.to_numpy() >=1e-2)
# wl_show = ar_wl[mask_int]
# int_show = weighted_lines.to_numpy()[mask_int]
#term_show = [a]


ar_spectrum_norm = (asd_intensity/max_y_in_range)*ar_spectrum_bg


fig, ax = plt.subplots(figsize=(5,2))
fig.set_dpi(300)
markerline, stemlines, baselines = ax.stem(lines_Ar_I['obs_wl_vac(nm)'],
                                                    weighted_lines,
                                                    label='NIST Atomic Spec. Data')
plt.setp(stemlines, color=f'C1')
plt.setp(markerline, markerfacecolor=f'C1', markeredgecolor=f'C1')


ax.plot(ar_wl,
        ar_spectrum_norm,
        label=f'Experiment')
ax.fill_between(ar_wl, ar_spectrum_norm, alpha=0.3)
ax.plot(asd_wl[asd_idx], asd_intensity, 'o', color='g')
ax.axvspan(775, 790, alpha=0.3, color='grey')
ax.axvline(780, color='k', linestyle='dashed')
ax.set_xlim([680, 870])
ax.set_ylim([1e-5, 10])
ax.set_yscale('log')
ax.legend()
ax.set_xlabel(r"$\lambda$ (nm)")
ax.set_ylabel(r"A$_{ki}$ $(\mathrm{s}^{-1})$")
ax.grid(True)
# ax.set_title('Ar I emission vs NIST ASD', fontsize=16)
# %%


lines_Ar_I_sifted = lines_Ar_I[(lines_Ar_I['obs_wl_vac(nm)'] >= 770) & (lines_Ar_I['obs_wl_vac(nm)'] <= 800)]
lines_Ar_I_sifted

# %%

# %%
