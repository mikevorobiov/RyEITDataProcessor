#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import wiener
from scipy.integrate import quad

from os.path import join

from scipy.constants import physical_constants as pc

import pyperclip
#%%
class LangmuirProcessor():

    """
    A class to analyze Langmuir probe I-V characteristic data
    and derive plasma parameters.
    """
    # Define fundamental physical constants as class attributes (SI units)
    ELECTRON_MASS = pc['atomic unit of mass'][0]  # kg (electron rest mass)
    ELEMENTARY_CHARGE = pc['elementary charge'][0]  # C (elementary charge)
    VACUUM_PERMITTIVITY = pc['vacuum electric permittivity'][0] # F/m (epsilon_0, permittivity of free space)
    
    ARGON_MASS = 6.6335209e-26 #(kg) Argon atomic mass
    M_Ar = 6.6335209e-26 #(kg) Argon atomic mass
    kB = 1.380649e-23 #(J/K) Boltzman constant
    
    def __init__(self,
                 data: np.array, # Input Lamgnuir probe sweep data
                 file_name: str,
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

        self.filename = file_name

        self.voltage, self.current = data
        self.current_abs = np.abs(data[1])

        self.radius = 0.5 * probe_diameter * 1e-3 # Convert from mm to m
        self.length = probe_length * 1e-3 # Convert from mm to m


        # Parameters of the approximating curves initialized to 0
        # These parameters are stored here during manual fitting
        self.e_maxwell_amplitude = 0
        self.e_maxwell_temperature = 0
        self.e_maxwell_Vs = 0

        self.e_saturation_amplitude = 0
        self.e_saturation_temperature = 0

        self.beam_energy = 0
        self.beam_temperature = 0
        self.plasma_potential = 0

        self.ion_slope = 0
        self.ion_v0 = 0

        self.corrected_electron_part = []
        self.plasma_potential = 0


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

        vc = np.sqrt(-2 * self.ELEMENTARY_CHARGE * (v - Vs) / self.ELECTRON_MASS)
        vb = np.sqrt(2 * self.ELEMENTARY_CHARGE  * Eb / self.ELECTRON_MASS)
        vthb = np.sqrt(2 * self.ELEMENTARY_CHARGE * Tb / self.ELECTRON_MASS)

        uc = vc/vthb
        ub = vb/vthb

        beam_current = np.full(len(voltage), 0)
        for i, u in enumerate(uc):
            integral, _ = quad(lambda x: x*np.exp(-(x-ub)**2),
                                     u, np.inf,epsrel=1e-10)
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
                                     ylim = [],
                                     xlim = [],
                                     figsize = (6,8)
                                     ):

        x, y = self.get_electron_part()

        Ib = ebeam_params['Ib']
        Vplasma = ebeam_params['plasma potential']
        Eb = ebeam_params['beam energy']
        Tb = ebeam_params['beam temperature']

        A_emaxwell = emaxwell_params['amplitude']
        kT_emaxwell = emaxwell_params['temperature']
        Vs_emaxwell = emaxwell_params['Vs']

        A_esaturation = esaturation_params['amplitude']
        kT_esaturation = esaturation_params['temperature']


        self.e_maxwell_amplitude = A_emaxwell
        self.e_maxwell_temperature = kT_emaxwell
        self.e_maxwell_Vs = Vs_emaxwell
        self.e_saturation_amplitude = A_esaturation
        self.e_saturation_temperature = kT_esaturation

        exp_maxwell = self._expf(x, A_emaxwell, kT_emaxwell, V0=Vs_emaxwell)
        exp_saturation = self._expf(x, A_esaturation, kT_esaturation)
        ebeam = self.ebeam_correction(I0=Ib, Vs=Vplasma, Eb=Eb, Tb=Tb)
        emaxwell_n_ebeam = exp_maxwell + ebeam
        
        y_ion_line = self._line(x, self.ion_slope, self.ion_v0)
        y_no_ion_current = y + y_ion_line
        y_no_ebeam = y_no_ion_current - ebeam

        self.corrected_electron_part = []
        self.corrected_electron_part = y_no_ebeam


        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=True, height_ratios=(4,1))
        fig.suptitle('Langmuir probe sweep fit', fontsize=16)
        ax[0].plot(x, y, '-', alpha=0.4, label='Original data', color='gray')
        ax[0].plot(x, y_no_ion_current, 'o', alpha=0.4, label='$I - I_{sat}$')
        ax[0].plot(x, y_no_ebeam, 'v', alpha=0.4, label='$I - I_{sat}-I_{beam}$')
        ax[0].plot(x, emaxwell_n_ebeam, label='$I_{beam} + I_{e Maxwell}$ FIT')
        ax[0].plot(x, exp_maxwell, label='$I_{e-Maxwell}$ ')
        ax[0].plot(x, ebeam, label='$I_{beam}$ FIT')
        ax[0].plot(x, exp_saturation, label='$I_{e-sat}$ FIT')
        if len(ylim)==2:
            ax[0].set_ylim(ylim)
        if len(xlim)==2:
            ax[0].set_xlim(xlim)
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
        ax[1].plot(x, relative_residual, 'o-', alpha=0.4)
        ax[1].set_ylabel('Rel. Residual (%)')
        ax[1].set_ylim([-residuals_percent, residuals_percent])
        ax[1].grid(True)
        ax[1].set_xlabel('Voltage (V)')
        plt.subplots_adjust(hspace=0)

        return fig, ax
    
    def _intersection(self):
        d_slope = 1/self.e_maxwell_temperature - 1/self.e_saturation_temperature
        d_intercept = np.log(self.e_maxwell_amplitude) - np.log(self.e_saturation_amplitude) - self.e_maxwell_Vs/self.e_maxwell_temperature
        plasma_potential = -d_intercept / d_slope
        e_current_saturation = self._expf(plasma_potential,
                                          A=self.e_maxwell_amplitude,
                                          kT=self.e_maxwell_temperature,
                                          V0 = self.e_maxwell_Vs)
        self.plasma_potential = plasma_potential
        return plasma_potential, e_current_saturation
    
    def calculate_plasma_parameters(self):
        raise NotImplementedError
    
    def _eta_correction(self, xi: float) -> float:
        """
        Calculates the eta correction factor for Langmuir probe diagnostics.

        This function is based on F. Chen's Lecture Notes on Langmuir Probe Diagnostics,
        page 6, formula (6). It is typically used within the `get_plasma_parameters`
        function to refine the calculation of the floating potential.

        Args:
            xi (float): A dimensionless ratio of probe radius to Debye length.
                        Must be positive.

        Returns:
            float: The calculated eta correction factor.

        Raises:
            ValueError: If xi is not a positive number, as log(xi) is undefined or problematic.
        """
        if not isinstance(xi, (int, float)) or xi <= 0:
            raise ValueError("Input 'xi' must be a positive number for log(xi).")

        # Constants as per Chen's formula (6)
        A = 0.583
        B = 3.723
        C = -0.027
        D = 5.431

        log_xi = np.log(xi)

        term_1_denominator = (A * log_xi + B)
        term_2_denominator = (C * log_xi + D)

        term_1 = 1 / (term_1_denominator ** 6)
        term_2 = 1 / (term_2_denominator ** 6)

        eta = 1 / np.power(term_1 + term_2, 1/6)
        return eta
    
    def _calculate_probe_area(self) -> float:
        """
        Calculates the effective collection area of the Langmuir probe.

        Formula `L * R * 2*np.pi + np.pi * R**2` suggests the area of 
        the cylindrical side plus one circular end (e.g., a blunt tip probe,
        or assuming one end is not collecting).

        Returns:
            float: The calculated probe collection area in square meters (m^2).
        """
        # Area of cylindrical side + Area of one circular end
        return (2 * np.pi * self.radius * self.length) + (np.pi * self.radius**2)
    
    def get_plasma_parameters(self) -> dict:
        """
        Calculates key plasma parameters from the Langmuir probe data.

        This method computes the electron density, Debye length, and floating potential
        using the probe's dimensions and the extracted I-V curve characteristics.

        Formulas are primarily referenced from F. Chen's Lecture Notes on
        Langmuir Probe Diagnostics.

        Returns:
            dict: A dictionary containing the calculated plasma parameters:
                  - 'electron_density (m^-3)': Electron number density.
                  - 'electron_temperature (eV)': Electron temperature.
                  - 'plasma_potential (V)': Plasma potential.
                  - 'floating_potential (V)': Floating potential.
                  - 'debye_length (mm)': Electron Debye length.
                  - 'xi (dimensionless)': Dimensionless plasma parameter (radius/Debye length).
        """
        # --- 1. Probe Area Calculation ---
        probe_area = self._calculate_probe_area()

        # --- 2. Electron Density Calculation ---
        # Based on Chen Lectures, page 2, Eq. (2) for electron thermal current (I_e0)
        # I_e0 = A * n_e * q * sqrt(k * T_e / (2 * pi * m_e))
        # Rearranging for n_e: n_e = I_e0 / (A * q * sqrt(k * T_e / (2 * pi * m_e)))
        # Here, `e_maxwell_amplitude` is assumed to be I_e0 (total electron saturation current).
        # T_e is in eV, so k*T_e becomes `e_maxwell_temperature * ELEMENTARY_CHARGE` for Joules.
        electron_density = (
            self.e_maxwell_amplitude *
            np.sqrt(2 * np.pi * self.ELECTRON_MASS /
                    (self.ELEMENTARY_CHARGE * self.e_maxwell_temperature)) / # (Te in eV)
            (self.ELEMENTARY_CHARGE * probe_area)
        )

        # --- 3. Debye Length Calculation ---
        # Debye length formula: lambda_D = sqrt(eps0 * k * T_e / (n_e * q^2))
        # T_e in eV: `e_maxwell_temperature * ELEMENTARY_CHARGE`
        debye_length = np.sqrt(
            self.VACUUM_PERMITTIVITY * self.e_maxwell_temperature /
            (electron_density * self.ELEMENTARY_CHARGE)
        )

        # --- 4. Dimensionless Plasma Parameter (xi) ---
        xi = self.radius / debye_length

        # --- 5. Floating Potential Calculation ---
        # Formula from Chen p.5, Eq.(5): V_f = V_p - eta_correction * T_e
        # Note: T_e here is in eV. The `_eta_correction` factor itself is dimensionless,
        # so multiplying by T_e (in eV) directly gives a potential in Volts, as expected.
        floating_potential = self.plasma_potential - self._eta_correction(xi) * self.e_maxwell_temperature

        # --- 6. Compile Output Parameters ---
        output_parameters = {
            'electron_density (m^-3)': electron_density,
            'electron_temperature (eV)': self.e_maxwell_temperature,
            'plasma_potential (V)': self.plasma_potential,
            'floating_potential (V)': floating_potential,
            'debye_length (mm)': debye_length*1e3,
            'xi (dimensionless)': xi,
        }
        return output_parameters
    
    def plasma_parameters_as_markdown_row(self,
            plasma_parameters: dict,
            generate_header = False, # For future use (WARNING: not implemented yet!)
            copy_to_clipboard: bool = False
            ) -> str:
        """
        Formats a dictionary of plasma parameters into a Markdown table row.

        This function assumes the input dictionary contains specific keys for
        plasma parameters and formats their values for display in a Markdown table.
        It also defines a consistent order for the columns.

        Args:
            plasma_parameters (dict): A dictionary containing plasma parameters,
                                      e.g., as returned by `LangmuirProbeAnalyzer.get_plasma_parameters()`.
                                      Expected keys (case-sensitive, including units):
                                      - 'electron_density (m^-3)'
                                      - 'electron_temperature (eV)'
                                      - 'plasma_potential (V)'
                                      - 'floating_potential (V)'
                                      - 'debye_length (m)'
                                      - 'xi (dimensionless)'
            copy_to_clipboard (bool): If True, the generated Markdown row will also
                                      be copied to the system's clipboard. Defaults to False.

        Returns:
            str: A Markdown formatted string representing a single table row.
                 Example: "| 1.23e+18 | 3.50 | 15.2 | 8.1 | 1.23e-05 | 0.56 |"
        """
        # Define the desired order of columns and their formatting precision
        column_specs = [
            ('electron_density (m^-3)', ':.3e'),
            ('electron_temperature (eV)', ':.3f'),
            ('plasma_potential (V)', ':.3f'),
            ('floating_potential (V)', ':.3f'),
            ('debye_length (mm)', ':.3e'),
            ('xi (dimensionless)', ':.3f'),
        ]

        row_values = []
        for key, fmt_spec in column_specs:
            value = plasma_parameters.get(key, 'N/A')
            if isinstance(value, (float, np.floating)):
                row_entry = "{0"+fmt_spec+"}"
                row_values.append(row_entry.format(value))
            else:
                row_values.append(str(value))

        filename_column = "| [["+self.filename+"]]"
        markdown_row = filename_column + "| " + " | ".join(row_values) + " |"

        if generate_header:
            header = "|File| Electron Density (m^-3) | Electron Temp (eV) | Plasma Potential (V) | Floating Potential (V) | Debye Length (m) | Xi (dimensionless) |\n"
            separator = "|---|---|---|---|---|---|---|\n" 
            markdown_row = header + separator + markdown_row

        if copy_to_clipboard:
            try:
                pyperclip.copy(markdown_row)
                print("Markdown row copied to clipboard.")
            except pyperclip.PyperclipException as e:
                print(f"Could not copy to clipboard: {e}")
                print("Please ensure you have a copy/paste mechanism available")
                print("(e.g., xclip on Linux, or Tkinter installed).")
            
        return markdown_row


    def clipboard_plasma_parameters(self):
        raise NotImplementedError
    


#%%
if __name__=='__main__':
    mpl.style.use('custom-style') #Custom style for plots. Can be removed without harm!
    
    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-07-20'
    file = 'voltage_current_data(2024-12-19)_9.csv'
    filepath = join(path, file)
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1).T

    lp = LangmuirProcessor(data, file_name=file, probe_length=14.4, probe_diameter=0.81)
#%%
    lp.plot_sweep()

    # print(lp.get_crossing_voltage())

# %%
    def line(x, slope, x0):
        return slope*(x - x0)
    
    def expf(x, kT, V0, A):
        return A*np.exp((x-V0)/kT)
#%%
    lp.approximate_ion_saturation(slope=10e-7,
                                  V0 = 120,
                                  residuals_percent=2)
# %%
    sat_pars = {'amplitude': 3.8e-4,
                'temperature': 19}

    maxwell_pars = {'amplitude': 3.75e-3,
                    'temperature': 0.70,
                    'Vs': 42.616}
    
    ebeam_pars = {'Ib': 6.4e-6,
                 'plasma potential': 42.58,
                 'beam energy': 3.7,
                 'beam temperature': 0.1}

    #ebeam_pars = {'Ib': 7.4e-8,
    #              'plasma potential': 19.15,
    #              'beam energy': 9.0,
    #              'beam temperature': 0.08}

    lp.approximate_electron_current(emaxwell_params=maxwell_pars,
                                    ebeam_params=ebeam_pars,
                                    esaturation_params=sat_pars,
                                    residuals_percent=15,
                                    ylim=[1e-6, 1e-2],
                                    xlim=[35,45])

# %%
    pars = lp.get_plasma_parameters()
# %%
    for p in pars:
        v = pars[p]
        print(p+f': {v:.4e}')
# %%
    lp.plasma_parameters_as_markdown_row(pars, copy_to_clipboard=True, generate_header=True)
# %%
