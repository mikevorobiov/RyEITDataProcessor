#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import wiener
from scipy.integrate import quad

from os.path import join

#%%
class LangmuirProcessor():
    
    m_e = 9.1093837e-31 #(kg) Electron mass
    q = 1.60217663e-19 #(C) Electron charge
    M_Ar = 6.6335209e-26 #(kg) Argon atomic mass 
    kB = 1.380649e-23 #(J/K) Boltzman constant
    
    def __init__(self,
                 data: np.array, # Input Lamgnuir probe sweep data
                 probe_tips: int = 1, # Number of probe tips (currently only single-tip supported)
                 probe_length: float = 1, # Probe tip length (mm)
                 probe_diameter: float = 1 # Probe tip radius (mm)
                 ):
        '''
        Initialize Langmuir probe object for data analysis.
        Only cyllindrical probe is supported!
        '''
        # Ceck if the input number of probe tips is allowed
        allowed_tips_number = [1]
        if not probe_tips in allowed_tips_number:
            print(f'Error: Requested number of tips {probe_tips} is not allowed.\\Use one of the following {allowed_tips_number}.')
        
        # Ceck if the input `data` shape is appropriate for furcher unpacking
        if not data.shape[0] == 2:
            print(f'Error: Data must be a 2D array [[voltage], [current]] of size (2, N), but was given {data.shape}')

        self.voltage, self.current = data
        self.current_abs = np.abs(data[1])

        self.radius = 0.5 * probe_diameter * 1e3 # Convert from mm to m
        self.length = probe_length * 1e3 # Convert from mm to m


        # Parameters of the approximating curves initialized to 0
        # These parameters are stored here during manual fitting
        self.e_maxwell_amplitude = 0
        self.e_maxwell_temperature = 0

        self.e_saturation_amplitude = 0
        self.e_saturation_temperature = 0

        self.beam_energy = 0
        self.beam_temperature = 0
        self.plasma_potential = 0

        self.ion_slope = 0
        self.ion_v0 = 0


    def plot_sweep(self,
                   figsize=(6,5)):
        '''
        Plot raw Langmuir probe sweep.
        '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.voltage, self.current_abs)
        ax.set_title('Raw Langmuir probe sweep')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')
        ax.set_yscale('log')
        ax.grid(True)

        return fig, ax
    
    def plot_ion_part(self,
                      figsize=(6,5)
                      ):
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.voltage, self.current)
        ax.set_title('Ion part')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A)')
        ax.set_yscale('log')

        return fig, ax
    
    def get_raw_trace(self):
        return self.voltage, self.current
    
    def get_electron_part(self):
        idx = np.argmin(self.current_abs)
        return self.voltage[idx:], self.current[idx:]
    
    def get_ion_part(self):
        idx = np.argmin(self.current_abs)
        return self.voltage[:idx], self.current[:idx]
    
    def get_eedf(self):
        raise NotImplementedError
    
    def get_derivatives(self, wfilter_size = (2,2,2)):
        fsize_1, fsize_2, fsize_3 = wfilter_size
        v, c = self.get_electron_part()
        original = wiener(c,fsize_1)
        deriv_1 = []
        deriv_2 = []
        if np.abs(fsize_1)+np.abs(fsize_2) > 0:
            deriv_1 = wiener(np.gradient(c, v), fsize_2)
            deriv_2 = wiener(np.gradient(deriv_1, v), fsize_3)
        else:
            deriv_1 = np.gradient(c, v)
            deriv_2 = np.gradient(deriv_1, v)
        return v, deriv_1, deriv_2
    
    def get_crossing_voltage(self):
        index = np.argmin(self.current_abs)
        return index, self.voltage[index]
    
    def ebeam_correction(self,
                         I0: float = 1, #(A)
                         Vs: float = 20, #(V) 
                         Eb: float = 15, #(eV)
                         Tb: float = 0.3, #(eV) beam temperature
                         ):
        '''
        Calulate electron beam correction according to 
        `Lecture notes on Langmuir Probe Diagnostics` by Francis F. Chen 2003 (page 34)
        '''
        voltage, _ = self.get_electron_part()
        idx = np.where(voltage < Vs)
        v = voltage[idx]

        vc = np.sqrt(-2 * self.q * (v - Vs) / self.m_e)
        vb = np.sqrt(2 * self.q * Eb / self.m_e)
        vthb = np.sqrt(2 * self.q * Tb / self.m_e)

        uc = vc/vthb
        ub = vb/vthb

        beam_current = np.full(len(voltage), 0)
        for i, u in enumerate(uc):
            integral, _ = quad(lambda x: x*np.exp(-(x-ub)**2),
                                     u, np.inf)
            #print(integral)
            beam_current[i] = integral

        return beam_current * I0
    
    def _line(self, x, slope, x0):
        return slope*(x - x0)
    
    def _expf(self, x, A, kT, V0=0):
        return A*np.exp((x-V0)/kT)
    
    def approximate_ion_saturation(self,
                                   slope = 8.5e-8,
                                   V0 = 17,
                                   residuals_percent = 10,
                                   figsize=(6,5)):
        x, y = self.get_ion_part()
        yl = self._line(x, slope, V0)

        self.ion_slope = slope
        self.ion_v0 = V0

        fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=figsize, height_ratios=(4,1))
        fig.suptitle('Ion saturation current approximation', fontsize=16)
        ax[0].plot(x, y, '.', alpha=0.4, label='Data', color='C0')
        ax[0].plot(x, yl, label='Ion saturation', color='C3', linewidth=1.5)
        ax[0].set_ylabel('Ion current (A)')
        ax[0].grid(True)
        ax[0].legend()

        res_norm = (1-yl/y) * 100
        ax[1].plot(x, res_norm, 's', alpha=0.4, color='C2')
        ax[1].set_ylabel('Rel. Residual (%)')
        ax[1].set_ylim([-residuals_percent, residuals_percent])
        ax[1].grid(True)
        ax[1].set_xlabel('Voltage (V)')
        plt.subplots_adjust(hspace=0)

        return fig, ax

    def approximate_electron_current(self,
                                     emaxwell_params: dict = {},
                                     ebeam_params: dict = {},
                                     esaturation_params: dict = {},
                                     residuals_percent = 20,
                                     figsize = (6,8)
                                     ):

        x, y = self.get_electron_part()

        Ib = ebeam_params['Ib']
        Vplasma = ebeam_params['plasma potential']
        Eb = ebeam_params['beam energy']
        Tb = ebeam_params['beam temperature']

        A_emaxwell = emaxwell_params['amplitude']
        kT_emaxwell = emaxwell_params['temperature']

        A_esaturation = esaturation_params['amplitude']
        kT_esaturation = esaturation_params['temperature']


        self.e_maxwell_amplitude = A_emaxwell
        self.e_maxwell_temperature = kT_emaxwell
        self.e_saturation_amplitude = A_esaturation
        self.e_saturation_temperature = kT_esaturation

        exp_maxwell = self._expf(x, A_emaxwell, kT_emaxwell)
        exp_saturation = self._expf(x, A_esaturation, kT_esaturation)
        ebeam = self.ebeam_correction(I0=Ib, Vs=Vplasma, Eb=Eb, Tb=Tb)
        emaxwell_n_ebeam = exp_maxwell + ebeam
        
        y_ion_line = self._line(x, self.ion_slope, self.ion_v0)
        y_no_ion_current = y + y_ion_line
        y_no_ebeam = y_no_ion_current - ebeam

        NN = 40

        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True, height_ratios=(4,1))
        fig.suptitle('Langmuir probe sweep fit', fontsize=16)
        ax[0].plot(x[:NN], y[:NN], '-', alpha=0.4, label='Original data', color='gray')
        ax[0].plot(x[:NN], y_no_ion_current[:NN], 'o', alpha=0.4, label='$I - I_{sat}$')
        ax[0].plot(x[:NN], y_no_ebeam[:NN], 'v', alpha=0.4, label='$I - I_{sat}-I_{beam}$')
        ax[0].plot(x[:NN], emaxwell_n_ebeam[:NN], label='$e-beam + e-maxwell FIT')
        ax[0].plot(x[:NN], exp_maxwell[:NN], label='$I_{e}$ Maxwell')
        ax[0].plot(x[:NN], ebeam[:NN], label='$I_{beam}$ FIT')
        ax[0].plot(x[:NN], exp_saturation[:NN], label='$I_{beam}$ FIT')

        Vp, Isat = self._intersection()
        ax[0].axvline(x=Vp, alpha=0.3, linestyle='--')
        ax[0].axhline(y=Isat, alpha=0.3, linestyle='--')
        ax[0].plot(Vp, Isat, 'o', color='red')
        ax[0].set_ylabel('Current (A)')
        ax[0].set_yscale('log')
        ax[0].grid(True)
        # ax[0].text(0.05, 0.85, f'$V_p$={Vp:2.2f} (V)\\$I_s^e$={Isat:2.2} (V)')
        ax[0].legend()

        relative_residual = (1-emaxwell_n_ebeam/y_no_ion_current) * 100
        ax[1].plot(x[:NN], relative_residual[:NN], 'o-', alpha=0.4)
        ax[1].set_ylabel('Rel. Residual (%)')
        ax[1].set_ylim([-residuals_percent, residuals_percent])
        ax[1].grid(True)
        ax[1].set_xlabel('Voltage (V)')
        plt.subplots_adjust(hspace=0)

        return fig, ax
    
    def _intersection(self):
        d_slope = 1/self.e_maxwell_temperature - 1/self.e_saturation_temperature
        d_intercept = np.log(self.e_maxwell_amplitude) - np.log(self.e_saturation_amplitude)
        plasma_potential = -d_intercept / d_slope
        e_current_saturation = self._expf(plasma_potential,
                                          A=self.e_maxwell_amplitude,
                                          kT=self.e_maxwell_temperature)
        return plasma_potential, e_current_saturation
    
    def calculate_plasma_parameters(self):
        raise NotImplementedError

    def get_plasma_parameters(self):
        raise NotImplementedError
    
    def clipboard_plasma_parameters(self):
        raise NotImplementedError
    


#%%
if __name__=='__main__':
    mpl.style.use('custom-style') #Custom style for plots. Can be removed without harm!
    
    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-04-02\\data'
    file = 'langmuir_data-48-2025-04-02.csv'
    filepath = join(path, file)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1).T

    lp = LangmuirProcessor(data, probe_length=11, probe_diameter=0.4)

    lp.plot_sweep()

    # print(lp.get_crossing_voltage())



# %%
    def line(x, slope, x0):
        return slope*(x - x0)
    
    def expf(x, kT, V0, A):
        return A*np.exp((x-V0)/kT)
# %%
    xi, yi = lp.get_ion_part()
    slope = 8.5e-8
    x0 = 17
    yl = line(xi, slope, x0)
    fig, ax = plt.subplots()
    ax.plot(xi, yi, '.-', alpha=0.4)
    ax.plot(xi, yl)
#%%
    lp.approximate_ion_saturation(slope=8.4e-8,
                                  V0 = 17.2,
                                  residuals_percent=2)
# %%
    sat_pars = {'amplitude': 2e-5,
                'temperature': 20}

    maxwell_pars = {'amplitude': 0.72e-9,
                    'temperature': 1.81}
    
    ebeam_pars = {'Ib': 6.7e-8,
                 'plasma potential': 26,
                 'beam energy': 15.5,
                 'beam temperature': 0.08}

    #ebeam_pars = {'Ib': 7.4e-8,
    #              'plasma potential': 19.15,
    #              'beam energy': 9.0,
    #              'beam temperature': 0.08}

    lp.approximate_electron_current(emaxwell_params=maxwell_pars,
                                    ebeam_params=ebeam_pars,
                                    esaturation_params=sat_pars,
                                    residuals_percent=15)
# %%
    pp, isat = lp._intersection()
    print(f'plasma potential: {pp:.3f} (V)')
    print(f'e-saturation current: {isat:.4} (A)')
# %%
