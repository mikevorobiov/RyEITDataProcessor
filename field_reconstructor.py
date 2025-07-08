'''
Mykhailo Vorobiov

Created 2025-06-27
'''
#%%
import numpy as np
from pybaselines import Baseline
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import lmfit
from lmfit import CompositeModel # Ensure CompositeModel is imported
from lmfit.models import GaussianModel, ConstantModel
from scipy.stats import chi2
from scipy.linalg import cholesky

from skimage import feature
from sklearn.preprocessing import Binarizer
from skimage.morphology import skeletonize

import os
from os.path import join

mpl.style.use('custom-style')

class FLStarkMapProcessor():
    '''
    This calss contains data structures and functions to save, process and 
    perform data analysis on the Rydberg-EIT electrometry data from fluorescence
    images obtained using cameras.
    '''

    def __init__(self,
             image: np.ndarray, # Input fluorescence Stark map (image hereafter)
             reference_trace: np.ndarray, # Reference EIT trace outside chamber/cell under test
             fps: float, # (Hz) Frames per second of the image acquisition.
             blue_sweep_freq: float, # (Hz) Frequency of the laser sweep.
             fov: float = 10.8, # (mm) Physical field of view represented by the image's spatial dimension.
             known_peak_separation_mhz: float = 137.54814724194335, # (MHz) Known frequency separation between two EIT reference peaks.
             bin_power: int = 0 # Binning exponent (2^n samples, where n=1,2,3...).
             ):
        """
        Initializes the FieldReconstructor with raw data and experimental parameters.
        Args:
            image (np.ndarray): The 2D (spatial x spectral/time) EIT image data.
                                Expected shape: (spatial_pixels, spectral_pixels).
            reference_trace (np.ndarray): The 1D reference EIT trace (time-domain)
                                          used for initial axis calibration.
            fps (float): Frames per second of the image acquisition (Hz).
            sweep_freq (float): The laser sweeping frequency during acquisition (Hz).
            fov (float): The physical field of view along the spatial dimension (mm).
            known_peak_separation_mhz (float, optional): The accurately known frequency
                                                          separation between two EIT
                                                          peaks in the reference trace (MHz).
                                                          Defaults to 137.54814724194335 MHz.
            bin_power (int, optional): The exponent 'n' for binning (2^n samples)
                                       applied during processing. Defaults to 1.
        Raises:
            RuntimeError: If the initial frequency axis calibration fails, indicating
                          a fundamental issue with the input data or `calibrate_axis` method.
        """
        # Store raw input data
        self.raw_image = image # Renamed for clarity vs. processed images
        self.raw_reference_trace = reference_trace
        # Store basic experimental parameters (better setup dictionary)
        self.fps = fps
        self.blue_sweep_freq = blue_sweep_freq
        self.fov = fov
        self.bin_power = bin_power
        self.known_peak_separation_mhz = known_peak_separation_mhz
        # Store raw image shape (useful for debugging/validation)
        self.image_shape_raw = image.shape

        self.covariance_matrix = np.array([])
        self.fitted_models = []

        # --- Perform initial calibration and axis generation ---
        try:
            # Assume _calibrate_axis sets self.calibrations_dict
            self.calibrations_dict = self._calibrate_axis()
            # Spatial axis (X-axis)
            self.x_axis_mm = np.linspace(0, self.fov, self.raw_image.shape[0])
            # Spectral/Frequency axis (Y-axis), derived from calibration
            # Assuming main_peak is a main peak time value,
            # and sec_to_mhz is a conversion factor.
            main_peak = self.calibrations_dict["Main peak (sec)"]
            sec_to_mhz = self.calibrations_dict["Seconds to MHz"]
            half_period_sec = 0.5 / self.blue_sweep_freq
            self.spectral_axis_mhz = (
                            main_peak -
                            np.linspace(0, half_period_sec, self.raw_image.shape[1])
                            ) * sec_to_mhz
            if bin_power>0:
                self.image_binned = self._bin_raw_image()
                self.x_axis_mm_bin = np.linspace(0, self.fov, self.image_binned.shape[0])
                self.spectral_axis_mhz_bin = (main_peak -
                                              np.linspace(0, half_period_sec, self.raw_image.shape[1])
                                              ) * sec_to_mhz
                self.image_corrected = self._baseline_image(self.image_binned)
            else:
                self.image_binned = None
                print('Note: Binned image has not been initialized!')
            print('Stark map processor has been successfully calibrated!')
        except Exception: # Catch a broader Exception if calibrate_axis can raise various errors
            print("Error: Intial calibration failed miserably!")
            

    def _correct_ref_baseline(self,
                              max_half_window: int = 90,
                              decreasing: bool = False,
                              smooth_half_window: int = 20
                              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates and removes the baseline from the raw reference EIT spectrum.

        This method uses the 'snip' algorithm from the 'baseline' library to estimate
        the background (baseline) of the `self.raw_reference_trace`. The corrected
        spectrum is then obtained by subtracting this baseline.

        The `self.raw_reference_trace` is expected to be a 2D NumPy array where
        the first row/column contains the x-data (e.g., time/frequency axis)
        and the second row/column contains the y-data (e.g., signal amplitude).

        Args:
            max_half_window (int, optional): The maximum half-window size for the
                                             SNIP algorithm. Controls the extent of
                                             the baseline estimation. Defaults to 90.
            decreasing (bool, optional): If True, use a decreasing trend for SNIP.
                                         False for increasing. Defaults to False.
            smooth_half_window (int, optional): The half-window size for smoothing
                                                the baseline. Defaults to 20.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - y_corrected (np.ndarray): The baseline-corrected (flattened) EIT signal.
                - bkg (np.ndarray): The calculated baseline (background) signal.

        Raises:
            ValueError: If `self.raw_reference_trace` is not a 2D array with at least
                        two rows/columns for x and y data.
            Exception: Propagates any exceptions from the `baseline.snip` function.
        """
        if not hasattr(self, 'raw_reference_trace') or not isinstance(self.raw_reference_trace, np.ndarray):
            raise AttributeError("`self.raw_reference_trace` must be a NumPy array.")
        if self.raw_reference_trace.ndim != 2 or self.raw_reference_trace.shape[0] != 2:
             raise ValueError(
                 "Expected `self.raw_reference_trace` to be a 2D array "
                 "with 2 rows (x and y data), but got shape "
                 f"{self.raw_reference_trace.shape}. Did you mean to transpose it first?"
            )

        # Assuming raw_reference_trace is structured as [[x1, x2, ...], [y1, y2, ...]]
        x_data, y_data = self.raw_reference_trace

        # Initialize the Baseline fitter
        baseline_fitter = Baseline(x_data=x_data)

        # Apply the SNIP algorithm
        # The second return value (params) is discarded as it's not used
        background, _ = baseline_fitter.snip(y_data,
                                             max_half_window=max_half_window,
                                             decreasing=decreasing,
                                             smooth_half_window=smooth_half_window
                                             )

        # Correct the signal by subtracting the baseline
        y_corrected = y_data - background
        corrected_trace = np.vstack((x_data, y_corrected))

        return corrected_trace, background
    
    def _find_eit_peaks(self,
                       signal: np.ndarray,
                       number_of_peaks: int = 2,
                       min_peak_width: int = 20,
                       max_peak_width: int = 300,
                       min_peak_distance: int = 100,
                       return_peak_properties: bool = False
                       ) -> np.ndarray:
        """
        Identifies and selects the specified number of most prominent peaks from a signal.

        This function uses `scipy.signal.find_peaks` to locate peaks based on
        specified width and distance criteria. It then selects the 'number_of_peaks'
        strongest (highest amplitude) peaks from those found.

        Args:
            signal (np.ndarray): The 1D input signal (e.g., EIT spectrum) to find peaks in.
            number_of_peaks (int, optional): The desired number of most prominent peaks to return. Defaults to 2.
            min_peak_width (int, optional): Required width of peaks in samples. Defaults to 15.
            max_peak_width (int, optional): Maximum allowed width of peaks in samples. Defaults to 300.
            min_peak_distance (int, optional): Required minimum horizontal distance (in samples)
                                               between neighboring peaks. Defaults to 50.
            return_peak_properties (bool, optional): If True, also returns the peak properties
                                                     dictionary from `find_peaks`. Defaults to False.

        Returns:
            np.ndarray: An array of the indices of the selected peaks, sorted by position.
                        If `return_peak_properties` is True, returns a tuple:
                        (np.ndarray, dict) containing peak indices and the properties dictionary.

        Raises:
            ValueError: If fewer than 'number_of_peaks' are found based on the criteria.
        """
        # Find all peaks that satisfy the width and distance criteria
        peak_indices, properties = find_peaks(
            signal,
            width=[min_peak_width, max_peak_width],
            distance=min_peak_distance,
            # You might also consider 'height' or 'prominence' here if relevant
            # for initial filtering, but sorting by amplitude later is also fine.
        )

        if len(peak_indices) < number_of_peaks:
            raise ValueError(
                f"Could not locate enough peaks! Found {len(peak_indices)} peaks, "
                f"but {number_of_peaks} were requested. Adjust peak search parameters if necessary."
            )

        # Sort peaks by their amplitude in descending order
        # Get the amplitudes of the found peaks
        peak_amplitudes = signal[peak_indices]
        # Get the indices that would sort peak_amplitudes in descending order
        sorted_indices_by_amplitude = np.argsort(peak_amplitudes)[::-1]

        # Select the top 'number_of_peaks' strongest peaks
        top_peak_original_indices = peak_indices[sorted_indices_by_amplitude[:number_of_peaks]]

        # Sort the selected peaks by their original position for consistent output
        final_peak_indices = np.sort(top_peak_original_indices)

        if return_peak_properties:
            print("Note: Returning all properties from initial find_peaks call. " \
            "Consider re-evaluating if specific properties for only the selected peaks are needed.")
            return final_peak_indices, properties
        else:
            return final_peak_indices



    def _calibrate_axis(self):
        """Calculates the time-to-frequency conversion factor from EIT peaks.

        This function analyzes the time-domain data of two Electromagnetically Induced
        Transparency (EIT) peaks. It fits the EIT spectrum with two Gaussian functions
        to precisely extract their time-domain separation. Using this time separation (sec)
        and the known frequency separation (MHz) between the two EIT peaks, the
        time-to-frequency conversion factor is then calculated as their ratio.

        Args:
            args: freq_separation (float): The actual frequency separation
                                            between the two EIT peaks in Hz.

        Returns:
            float: The calculated time-to-frequency conversion factor (e.g., in MHz/s).
            float: The position of the main EIT peak (in sec)

        Raises:
            ValueError: If fitting fails or data is invalid.
        """
        signal, background = self._correct_ref_baseline()
        x, y = signal

        try:
            peak_indices = self._find_eit_peaks(y, number_of_peaks=2)
        except ValueError:
            print('Error: Less than 2 peaks found. Cannot continue fitting calibration trace!')

        # Fit two-gaussians model        
        gmodel = GaussianModel(prefix='d52_') + GaussianModel(prefix='d32_')
        pars = gmodel.make_params(
            d52_center = x[peak_indices[0]],
            d52_sigma = 0.15,
            d52_amplitude = 1e-2,
            d32_center = x[peak_indices[1]],
            d32_sigma = 0.15,
            d32_amplitude = 1e-3,
        )
        pars.set(d52_sigma = {"expr": 'd32_sigma'})
        pars.set(d52_amplitude = {"min": 1e-3})
        pars.set(d32_amplitude = {"min": 1e-6})

        ref_fit_results = gmodel.fit(y,pars,x=x)

        best_pars = ref_fit_results.best_values
        main_center = best_pars['d52_center']
        subs_center = best_pars['d32_center']

        sec_to_mhz = self.known_peak_separation_mhz / np.abs(main_center-subs_center)

        out_calibrations_dict = {"Seconds to MHz": sec_to_mhz,
                                 "Main peak (sec)": main_center,
                                 "Sub peak (sec)": subs_center,
                                 "Peaks indices": peak_indices,
                                 "lmfit result": ref_fit_results,
                                 "Background samples": background
                                 }

        return out_calibrations_dict


    def plot_calibration_probing(self):
        """
        NEEDS REFACTORING
        """
        x,y  = self.raw_reference_trace
        bkg = self.calibrations_dict["Background samples"]
        pidx = self.calibrations_dict["Peaks indices"]
        fit_result = self.calibrations_dict["lmfit result"]
        
        fig, ax = plt.subplots(3,1,
                               figsize=(4,6),
                               height_ratios=(3,3,1),
                               sharex=True
                            )
        
        ax[0].plot(x, y,'C0',label='bg corr.')
        ax[0].plot(x[pidx],y[pidx],'o',color='red')
        ax[0].set_ylabel('Corr. signal (V)')
        ax[0].legend(loc='upper left')
        ax2 = ax[0].twinx()
        ax2.plot(x,y,'C1',label='raw signal')
        ax2.plot(x,bkg,'r--',label='baseline')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Raw signal (V)')
        
        ax[1].plot(x, y,label='bg corrected')
        ax[1].plot(x, fit_result.init_fit,'--',label='init guess')
        ax[1].plot(x, fit_result.best_fit,'r-',label='best fit',linewidth=1.5)
        ax[1].set_ylabel('Corr. signal (V)')
        ax[1].legend()
        ax[2].plot(x, fit_result.residual,'g')
        ax[2].set_ylabel('Resid. (V)')
        ax[2].set_xlabel('Time (sec)')

        return fig
    
    def plot_calibration(self,
                         figsize: tuple = (4, 6),
                         height_ratios: tuple = (3, 3, 1),
                         main_title: str = "EIT Calibration Plot",
                         y_label_corrected_signal: str = 'Corr. signal (V)',
                         y_label_raw_signal: str = 'Raw signal (V)',
                         y_label_residual: str = 'Resid. (V)',
                         x_label_time: str = 'Time (sec)'
                         ) -> plt.Figure:
        """
        Generates a multi-panel plot visualizing the EIT reference spectrum calibration.

        This plot shows:
        1. The raw EIT signal, its calculated baseline, and the baseline-corrected signal
           with detected peaks.
        2. The baseline-corrected signal, the initial guess from the fit, and the best fit curve.
        3. The residuals of the fit.

        It relies on the 'raw_reference_trace' and data stored in 'calibrations_dict'
        attributes of the class instance.

        Args:
            figsize (tuple, optional): Dimensions of the figure (width, height) in inches. Defaults to (4, 6).
            height_ratios (tuple, optional): Ratios of the heights of the subplots. Defaults to (3, 3, 1).
            main_title (str, optional): The overall title for the figure. Defaults to "EIT Calibration Plot".
            y_label_corrected_signal (str, optional): Y-axis label for corrected signal plots.
            y_label_raw_signal (str, optional): Y-axis label for raw signal plot.
            y_label_residual (str, optional): Y-axis label for residual plot.
            x_label_time (str, optional): X-axis label for the time axis.

        Returns:
            plt.Figure: The generated Matplotlib figure object.

        Raises:
            KeyError: If required keys are missing from `self.calibrations_dict`.
            AttributeError: If `self.raw_reference_trace` is not properly set.
            TypeError: If `self.raw_reference_trace` is not a 2D array or similar iterable.
        """
        # --- 1. Data Retrieval and Validation ---
        try:
            # Assuming raw_reference_trace is structured as [[x_data], [y_data]]
            x_data, y_data_raw = self.raw_reference_trace
        except (AttributeError, TypeError) as e:
            raise AttributeError(
                f"Class attribute 'raw_reference_trace' not found or malformed: {e}. "
                "Ensure it's a 2D array (e.g., [[x_data], [y_data]]) before plotting."
            )

        try:
            bkg = self.calibrations_dict["Background samples"]
            pidx = self.calibrations_dict["Peaks indices"]
            fit_result = self.calibrations_dict["lmfit result"]
            y_data_corrected = y_data_raw - bkg # Re-calculate for consistency or assume it's stored
            # If y_corrected is stored directly, use it:
            # y_data_corrected = self.calibrations_dict.get("Corrected signal", y_data_raw - bkg)

        except KeyError as e:
            raise KeyError(
                f"Missing required data in 'calibrations_dict': {e}. "
                "Ensure 'Background samples', 'Peaks indices', and 'lmfit result' are present."
            )
        except AttributeError as e:
            raise AttributeError(
                f"'calibrations_dict' attribute not found: {e}. "
                "Ensure the calibration process has been run successfully."
            )


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

        return fig
    
    def _bin_raw_image_probing(self):
        try:
            image_bin = np.array([c.reshape(-1, 2**self.bin_power).mean(axis=1) for c in self.raw_image.T])
        except ValueError:
            print('Error: Couldn\'t generate binned image!')

        return image_bin
    
    def _bin_med_raw_image_probing(self):
        try:
            image_bin = np.array([np.media(c.reshape(-1, 2**self.bin_power),axis=1) for c in self.raw_image.T])
        except ValueError:
            print('Error: Couldn\'t generate binned image!')

        return image_bin


    def _bin_raw_image(self) -> np.ndarray:
        """
        Performs binning on the raw EIT image data along the spectral/time dimension.

        This method reduces the resolution of the `self.raw_image` by averaging
        blocks of `2**self.bin_power` samples along the second dimension (columns),
        effectively creating a binned image. The operation is applied column-wise
        after transposing the image.

        Expects `self.raw_image` to be a 2D NumPy array, where rows typically
        represent spatial positions and columns represent spectral/time points.
        The binning occurs along the column (spectral/time) axis.

        Returns:
            np.ndarray: A new NumPy array representing the binned image data.

        Raises:
            AttributeError: If `self.raw_image` or `self.bin_power` are not set.
            ValueError: If `self.bin_power` is negative, or if the spectral/time
                        dimension of `self.raw_image` is not evenly divisible
                        by the bin size (2**self.bin_power), leading to an
                        invalid reshape operation.
            RuntimeError: For other unexpected errors during the binning process.
        """
        # 1. Validate input attributes
        if not hasattr(self, 'raw_image') or not isinstance(self.raw_image, np.ndarray):
            raise AttributeError("`self.raw_image` must be a NumPy array attribute.")
        if not hasattr(self, 'bin_power') or not isinstance(self.bin_power, int):
            raise AttributeError("`self.bin_power` must be an integer attribute.")
        if self.bin_power < 0:
            raise ValueError(f"`bin_power` cannot be negative, got {self.bin_power}. Must be a non-negative integer.")
        if self.raw_image.ndim != 2:
            raise ValueError(f"Expected `self.raw_image` to be a 2D array, but got {self.raw_image.ndim} dimensions.")

        bin_size = 2**self.bin_power
        if bin_size == 0: # This can happen if bin_power is negative, but handled above
            raise ValueError("Bin size cannot be zero. Check `bin_power`.")

        num_spectral_points = self.raw_image.shape[1]
        if num_spectral_points % bin_size != 0:
            raise ValueError(
                f"The number of spectral/time points ({num_spectral_points}) "
                f"is not evenly divisible by the bin size ({bin_size}). "
                "Adjust `bin_power` or handle truncation/padding of the image."
            )

        try:
            # Transpose the image so that the spectral/time dimension becomes the first dimension
            # for iteration, then reshape and mean along that dimension.
            # A more direct way without explicit transpose in list comprehension:
            # image_bin = self.raw_image.reshape(
            #     self.raw_image.shape[0],
            #     num_spectral_points // bin_size,
            #     bin_size
            # ).mean(axis=2)

            binned_rows = []
            for row_data in self.raw_image.T: # Iterate through each row along spatial axis
                # Reshape each row to create blocks of 'bin_size' and take the mean
                binned_row = row_data.reshape(-1, bin_size).mean(axis=1)
                binned_rows.append(binned_row)

            image_bin = np.array(binned_rows).T # Binned image trasposed to restore axes order

        except ValueError as e:
            # Catch specific ValueError from numpy.reshape if the shape is incompatible
            raise ValueError(
                f"Failed to reshape and bin image data. "
                f"Original error: {e}. "
                f"Check if image dimensions ({self.raw_image.shape[1]}) "
                f"are compatible with bin size ({bin_size})."
            )
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred during image binning: {e}")

        return image_bin
    
    def plot_stark_map(self,
                       plot_raw_image: bool = False, # Renamed 'binned' for clarity
                       figsize: tuple = (7, 5),
                       cmap: str = 'jet',
                       cbar_label: str = 'EIT Signal (a.u.)', # Changed from '%' to 'a.u.' (arbitrary units) as % might imply normalized.
                       x_label: str = 'Distance (mm)',
                       y_label: str = 'Blue Detuning (MHz)',
                       title: str = 'Stark Map (Binned Data)'
                       ) -> plt.Figure:
        """
        Generates a 2D colormesh plot of the EIT Stark map.

        This function visualizes the EIT signal strength as a function of
        spatial position (x-axis) and spectral detuning (y-axis). It can
        display either the raw or the binned image data.

        Relies on 'self.x_axis_mm', 'self.spectral_axis_mhz', and either
        'self.raw_image' or 'self.image_binned' attributes being correctly set.

        Args:
            plot_raw_image (bool, optional): If True, plots `self.raw_image`.
                                             If False, plots `self.image_binned`.
                                             Defaults to False (plots binned data).
            figsize (tuple, optional): Dimensions of the figure (width, height) in inches. Defaults to (7, 5).
            cmap (str, optional): Colormap to use for the pcolormesh. Defaults to 'jet'.
            cbar_label (str, optional): Label for the color bar. Defaults to 'EIT Signal (a.u.)'.
            x_label (str, optional): Label for the X-axis. Defaults to 'Distance (mm)'.
            y_label (str, optional): Label for the Y-axis. Defaults to 'Blue Detuning (MHz)'.
            title (str, optional): Title of the plot. Defaults to 'Stark Map (Binned Data)'.

        Returns:
            plt.Figure: The generated Matplotlib figure object.

        Raises:
            AttributeError: If required axis or image data attributes are not found.
            KeyError: If required keys are missing from `self.calibrations_dict` (indirectly, if axes depend on it).
            ValueError: If image data has incorrect dimensions for plotting.
        """
        # --- 1. Data Retrieval and Validation ---
        try:
            x_coords = self.x_axis_mm
            y_coords = self.spectral_axis_mhz
            # The original code retrieved main_pos and sec_to_mhz but didn't use them directly
            # in plot_stark_map; assuming self.spectral_axis_mhz is already derived.
            # If these were crucial for *this* method's calculations, they'd be used here.

            if plot_raw_image:
                plot_data = self.raw_image
                current_title = title if title != 'Stark Map (Binned Data)' else 'Stark Map (Raw Data)'
            else:
                x_coords = self.x_axis_mm_bin
                y_coords = self.spectral_axis_mhz_bin
                plot_data = self.image_binned
                current_title = title # Use default or user-provided title

        except AttributeError as e:
            raise AttributeError(
                f"Required attribute missing: {e}. "
                "Ensure 'x_axis_mm', 'spectral_axis_mhz', and 'raw_image'/'binned_image' "
                "are set before calling plot_stark_map."
            )
        except Exception as e: # Catch any other unexpected errors during data retrieval
            raise RuntimeError(f"An unexpected error occurred during data retrieval for plotting: {e}")

        # Basic check for data dimensions consistency
        if plot_data.shape[0] != len(x_coords) or plot_data.shape[1] != len(y_coords):
             raise ValueError(
                 f"Image data shape {plot_data.shape} does not match axis lengths "
                 f"({len(x_coords)} for x, {len(y_coords)} for y). "
                 "Ensure axes and image dimensions are consistent."
             )
        # --- 2. Figure and Axes Setup ---
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(current_title)

        # --- 3. Plotting with pcolormesh ---
        # Note on pcolormesh: x and y should typically be the coordinates of the *edges*
        # of the cells. If your x_coords and y_coords are centers, pcolormesh might
        # implicitly handle it, or you might need to adjust them (e.g., using np.meshgrid)
        # or use imshow. Assuming they work correctly as is for your data structure.
        mesh = ax.pcolormesh(x_coords, y_coords, plot_data.T, cmap=cmap) # .T is common if image is (spatial, spectral)

        # --- 4. Colorbar and Labels ---
        plt.colorbar(mesh, ax=ax, label=cbar_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect('auto') # Ensures aspect ratio doesn't distort data unless intended

        # --- 5. Final Touches ---
        fig.tight_layout()

        return fig
    
    def plot_corrected_stark_map(self,
                       figsize: tuple = (7, 5),
                       cmap: str = 'jet',
                       cbar_label: str = 'EIT Signal (a.u.)', # Changed from '%' to 'a.u.' (arbitrary units) as % might imply normalized.
                       x_label: str = 'Distance (mm)',
                       y_label: str = 'Blue Detuning (MHz)',
                       title: str = 'Stark Map (Corrected Data)'
                       ) -> plt.Figure:
        """
        Generates a 2D colormesh plot of the EIT Stark map.

        This function visualizes the EIT signal strength as a function of
        spatial position (x-axis) and spectral detuning (y-axis). It can
        display either the raw or the binned image data.

        Relies on 'self.x_axis_mm', 'self.spectral_axis_mhz', and either
        'self.raw_image' or 'self.image_binned' attributes being correctly set.

        Args:
            plot_raw_image (bool, optional): If True, plots `self.raw_image`.
                                             If False, plots `self.image_binned`.
                                             Defaults to False (plots binned data).
            figsize (tuple, optional): Dimensions of the figure (width, height) in inches. Defaults to (7, 5).
            cmap (str, optional): Colormap to use for the pcolormesh. Defaults to 'jet'.
            cbar_label (str, optional): Label for the color bar. Defaults to 'EIT Signal (a.u.)'.
            x_label (str, optional): Label for the X-axis. Defaults to 'Distance (mm)'.
            y_label (str, optional): Label for the Y-axis. Defaults to 'Blue Detuning (MHz)'.
            title (str, optional): Title of the plot. Defaults to 'Stark Map (Binned Data)'.

        Returns:
            plt.Figure: The generated Matplotlib figure object.

        Raises:
            AttributeError: If required axis or image data attributes are not found.
            KeyError: If required keys are missing from `self.calibrations_dict` (indirectly, if axes depend on it).
            ValueError: If image data has incorrect dimensions for plotting.
        """
        # --- 1. Data Retrieval and Validation ---
        try:
            x_coords = self.x_axis_mm_bin
            y_coords = self.spectral_axis_mhz

            x_coords = self.x_axis_mm_bin
            y_coords = self.spectral_axis_mhz_bin
            plot_data = self.image_corrected
            current_title = title # Use default or user-provided title

        except AttributeError as e:
            raise AttributeError(
                f"Required attribute missing: {e}. "
                "Ensure 'x_axis_mm', 'spectral_axis_mhz', and 'raw_image'/'binned_image' "
                "are set before calling plot_stark_map."
            )
        except Exception as e: # Catch any other unexpected errors during data retrieval
            raise RuntimeError(f"An unexpected error occurred during data retrieval for plotting: {e}")

        # Basic check for data dimensions consistency
        if plot_data.shape[0] != len(x_coords) or plot_data.shape[1] != len(y_coords):
             raise ValueError(
                 f"Image data shape {plot_data.shape} does not match axis lengths "
                 f"({len(x_coords)} for x, {len(y_coords)} for y). "
                 "Ensure axes and image dimensions are consistent."
             )
        # --- 2. Figure and Axes Setup ---
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(current_title)

        # --- 3. Plotting with pcolormesh ---
        mesh = ax.pcolormesh(x_coords, y_coords, plot_data.T, cmap=cmap)

        # --- 4. Colorbar and Labels ---
        plt.colorbar(mesh, ax=ax, label=cbar_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect('auto') # Ensures aspect ratio doesn't distort data unless intended

        # --- 5. Final Touches ---
        fig.tight_layout()

        return fig
    
    def _generate_gmodel(self,
                         gaussians_number: int = 2
                         ) -> CompositeModel:
        """
        Generates a composite lmfit Gaussian model with a specified number of Gaussian components.

        Each Gaussian component is given a unique prefix (e.g., 'g0_', 'g1_')
        to avoid parameter name collisions when fitting.

        Args:
            gaussians_number (int, optional): The number of Gaussian components
                                              to include in the composite model.
                                              Defaults to 2.

        Returns:
            lmfit.models.CompositeModel: An lmfit model object representing
                                         the sum of the specified number of Gaussians.

        Raises:
            ValueError: If `gaussians_number` is less than 1.
        """
        if not isinstance(gaussians_number, int) or gaussians_number < 1:
            raise ValueError("`gaussians_number` must be an integer greater than or equal to 1.")

        prefixes = [f'g{i}_' for i in range(gaussians_number)]

        gmodel =ConstantModel(prefix='c_')
        for p in prefixes:
            gmodel += GaussianModel(prefix=p)
        return gmodel
    
    def _nearest_index(self, 
                 f:float, 
                 arr: np.ndarray
                 ):
        return np.array(np.abs(arr-f)).argmin()
    
    def _background_covariance(self, image):
        im = self._baseline_image(image)
        nx, ny = im.shape
        pairwise_matrix = np.zeros((ny, ny))
        for r in im:
            dr = r - r.mean()
            pairwise_matrix += np.outer(dr, dr)
        
        covariance = pairwise_matrix/(nx)
        #covariance = np.identity(ny)
        #print(covariance.shape)
        return covariance
    
    def set_background_covariance(self,
                                  image):
        self.covariance_matrix = self._background_covariance(image)
        return None
    
    def plot_covariance(self):
        x = self.spectral_axis_mhz
        fig, ax = plt.subplots()
        ax.pcolormesh(x, x, self.covariance_matrix, cmap='jet')
        ax.set_aspect(1)
        ax.set_title('Background covariance')
        ax.set_xlabel('Blue detuning (MHz)')
        return fig, ax
    
    def _initialize_peak_parameters(self, x: np.ndarray, y: np.ndarray, num_peaks: int,
                                    peak_height: float, peak_distance: float,
                                    peak_prominence: float, peak_width: list):
        """
        Finds initial peak parameters for the Gaussian model using user-defined scipy.signal.find_peaks parameters.

        Args:
            x (np.ndarray): The x-axis data (spectral axis).
            y (np.ndarray): The y-axis data (spectrum trace).
            num_peaks (int): The number of peaks expected for the current model.
            peak_height (float): Required height of peaks.
            peak_distance (float): Required minimal horizontal distance (in samples) between neighboring peaks.
            peak_prominence (float): Required prominence of peaks.
            peak_width (list): Required width of peaks in samples as `[min_width, max_width]`.

        Returns:
            lmfit.Parameters: lmfit.Parameters object with initial values, or None if not enough peaks found.
        """
        params = lmfit.Parameters()
        params.add('c_c', value=0.0) # Add constant offset parameter

        # Find initial peak candidates using exposed parameters
        peak_indices, _ = find_peaks(y, height=peak_height, distance=peak_distance,
                                     prominence=peak_prominence, width=peak_width)

        if len(peak_indices) < num_peaks:
            # print(f'Warning: Hopes for {num_peaks} peaks but found only {len(peak_indices)}. Cannot initialize model.')
            return None

        # Select the 'num_peaks' strongest peaks
        sorted_peak_indices = peak_indices[np.argsort(y[peak_indices])][-num_peaks:]
        sorted_peak_indices.sort() # Sort by x-value for consistent parameter naming

        # Initialize parameters for each Gaussian component
        for i, p_idx in enumerate(sorted_peak_indices):
            params.add(f'g{i}_center', value=x[p_idx], min=x[p_idx]-10, max=x[p_idx]+10)
            params.add(f'g{i}_amplitude', value=y[p_idx], min=0.01)
            params.add(f'g{i}_sigma', value=7.6, min=1, max=20)

        return params

    def test_gmodels(self,
                     spectrum_trace: np.ndarray,
                     peaks_numbers: list = [1, 2],
                     significance: float = 0.05,
                     peak_height: float = 0.5,
                     peak_distance: float = 1,
                     peak_prominence: float = 0.7,
                     peak_width: list = [2, 20],
                     verbose: bool = False # Added verbose flag
                     ) -> lmfit.model.ModelResult | None:
        """
        Evaluates different Gaussian models based on the number of peaks and
        returns the best fit result based on a chi-squared significance test.

        Args:
            spectrum_trace (np.ndarray): The 1D array of spectral intensity data.
            peaks_numbers (list): A list of integers, each representing a number of
                                   Gaussian peaks to test in the model.
            significance (float): The significance level for the chi-squared test.
            peak_height (float): Passed to scipy.signal.find_peaks. Required height of peaks.
            peak_distance (float): Passed to scipy.signal.find_peaks. Required minimal horizontal distance (in samples) between neighboring peaks.
            peak_prominence (float): Passed to scipy.signal.find_peaks. Required prominence of peaks.
            peak_width (list): Passed to scipy.signal.find_peaks. Required width of peaks in samples as `[min_width, max_width]`.
            verbose (bool): If True, prints detailed fit information during the process.

        Returns:
            lmfit.model.ModelResult: The lmfit result object for the best-fitting
                                     model that meets the significance criterion,
                                     or None if no model meets the criterion.
        """
        if not isinstance(spectrum_trace, np.ndarray) or spectrum_trace.size == 0:
            raise ValueError("spectrum_trace must be a non-empty numpy array.")
        if not all(isinstance(p, int) and p > 0 for p in peaks_numbers):
            raise ValueError("peaks_numbers must be a list of positive integers.")
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1.")

        x = self.spectral_axis_mhz
        y = spectrum_trace

        best_fit_result = None
        best_chi2_score = float('inf')

        for n_peaks in sorted(peaks_numbers):
            if verbose:
                print(f"\n--- Testing model with {n_peaks} peaks ---")
            gmodel = self._generate_gmodel(n_peaks)
            initial_params = self._initialize_peak_parameters(
                x, y, n_peaks, peak_height, peak_distance, peak_prominence, peak_width
            )

            if initial_params is None:
                continue

            try:
                fit_result = gmodel.fit(y, initial_params, x=x)
            except ValueError as e:
                if verbose:
                    print(f'Error fitting {n_peaks} peaks model: {e}')
                continue

            if not fit_result.success:
                if verbose:
                    print(f"Model {n_peaks}: The fit did not converge.")
                continue

            if verbose:
                print(f"Model {n_peaks}: The fit converged successfully.")
                # print(fit_result.fit_report()) # Uncomment for detailed fit report

            chi2_statistic = fit_result.chisqr
            num_fitted_params = len(fit_result.params)
            dof = len(x) - num_fitted_params
            if dof <= 0:
                if verbose:
                    print(f"Warning: Degrees of freedom for {n_peaks} peaks is {dof}. Cannot perform chi-squared test.")
                continue

           # Calculate probability of the result based on the chi2 statistic
            chi2_prob = chi2.cdf(chi2_statistic, df=dof)

            if verbose:
                print(f'Model {n_peaks}: Chi-squared statistic = {chi2_statistic:.2f}, DOF = {dof}, Chi-squared probability = {chi2_prob:.3f}')

            # Save current model's chi2 statistic value
            current_model_score = chi2_statistic

            # If the chi2 statistic is smaller than the best previous chi2 statistic
            # then current statistic becomes the best and the corresponding best fit_result is saved 
            if current_model_score < best_chi2_score:
                best_chi2_score = current_model_score
                best_fit_result = fit_result
                if verbose:
                    print(f"Model {n_peaks}: Currently the best fit (lowest chi-squared).")


        if best_fit_result and verbose:
            print("\n--- Best Fit Result ---")
            print(f"Selected model has {len(best_fit_result.components) - 1} peaks (excluding constant).")
            best_fit_result.plot_fit()
            plt.title("Best Fit Result")
            plt.show() # Ensure plot is displayed
        elif not best_fit_result and verbose:
            print("\nNo successful fit found for any tested model.")

        return best_fit_result
    
    def _loss_function_covar(self, params, x, y_exp, C, model_func):
        """
        Objective function for fitting with correlated errors.

        params: lmfit Parameters object
        x: independent data
        y_exp: experimental data
        model_func: function that calculates the model prediction
        L: whitening matrix (from C^-1 = LL^T)
        """
        C_inv = np.linalg.inv(C)
        L = cholesky(C_inv, lower=True)

        y_model = model_func.eval(params, x=x) # Calculate model prediction
        residuals = y_exp - y_model     # Calculate raw residuals
        transformed_residuals = L.T @ residuals # Apply whitening transformation
        
        return transformed_residuals

    def test_gmodels_covar(self,
                     spectrum_trace: np.ndarray,
                     peaks_numbers: list = [1, 2],
                     significance: float = 0.05,
                     peak_height: float = 0.5,
                     peak_distance: float = 1,
                     peak_prominence: float = 0.7,
                     peak_width: list = [2, 20],
                     verbose: bool = False # Added verbose flag
                     ) -> lmfit.model.ModelResult | None:
        """
        Evaluates different Gaussian models based on the number of peaks and
        returns the best fit result based on a chi-squared significance test.

        Args:
            spectrum_trace (np.ndarray): The 1D array of spectral intensity data.
            peaks_numbers (list): A list of integers, each representing a number of
                                   Gaussian peaks to test in the model.
            significance (float): The significance level for the chi-squared test.
            peak_height (float): Passed to scipy.signal.find_peaks. Required height of peaks.
            peak_distance (float): Passed to scipy.signal.find_peaks. Required minimal horizontal distance (in samples) between neighboring peaks.
            peak_prominence (float): Passed to scipy.signal.find_peaks. Required prominence of peaks.
            peak_width (list): Passed to scipy.signal.find_peaks. Required width of peaks in samples as `[min_width, max_width]`.
            verbose (bool): If True, prints detailed fit information during the process.

        Returns:
            lmfit.model.ModelResult: The lmfit result object for the best-fitting
                                     model that meets the significance criterion,
                                     or None if no model meets the criterion.
        """
        if not isinstance(spectrum_trace, np.ndarray) or spectrum_trace.size == 0:
            raise ValueError("spectrum_trace must be a non-empty numpy array.")
        if not all(isinstance(p, int) and p > 0 for p in peaks_numbers):
            raise ValueError("peaks_numbers must be a list of positive integers.")
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1.")

        x = self.spectral_axis_mhz
        y = spectrum_trace

        best_fit_result = None
        best_chi2_score = float('inf')

        for n_peaks in sorted(peaks_numbers):
            if verbose:
                print(f"\n--- Testing model with {n_peaks} peaks ---")
            gmodel = self._generate_gmodel(n_peaks)
            initial_params = self._initialize_peak_parameters(
                x, y, n_peaks, peak_height, peak_distance, peak_prominence, peak_width
            )

            if initial_params is None:
                continue

            try:
                fit_result = lmfit.minimize(self._loss_function_covar,
                                            initial_params,
                                            args=(x, y, self.covariance_matrix, gmodel))
            except ValueError as e:
                if verbose:
                    print(f'Error fitting {n_peaks} peaks model: {e}')
                continue

            if not fit_result.success:
                if verbose:
                    print(f"Model {n_peaks}: The fit did not converge.")
                continue

            if verbose:
                print(f"Model {n_peaks}: The fit converged successfully.")
                # print(fit_result.fit_report()) # Uncomment for detailed fit report

            chi2_statistic = fit_result.chisqr
            num_fitted_params = len(fit_result.params)
            dof = len(x) - num_fitted_params
            if dof <= 0:
                if verbose:
                    print(f"Warning: Degrees of freedom for {n_peaks} peaks is {dof}. Cannot perform chi-squared test.")
                continue

            # Original multiplication retained for consistency, consider if truly intended.
            chi2_prob = chi2.cdf(chi2_statistic * 5, df=dof)

            if verbose:
                print(f'Model {n_peaks}: Chi-squared statistic = {chi2_statistic:.2f}, DOF = {dof}, Chi-squared probability = {chi2_prob:.3f}')

            current_model_score = chi2_statistic

            if current_model_score < best_chi2_score:
                best_chi2_score = current_model_score
                best_fit_result = fit_result
                if verbose:
                    print(f"Model {n_peaks}: Currently the best fit (lowest chi-squared).")


        if best_fit_result and verbose:
            print("\n--- Best Fit Result ---")
            p = fit_result.params
            plt.plot(x, y, 'o')
            plt.plot(x, gmodel.eval(p, x=x), label='best fit')
            plt.title("Best Fit Result")
            plt.show() # Ensure plot is displayed
        elif not best_fit_result and verbose:
            print("\nNo successful fit found for any tested model.")

        return best_fit_result

    def test_custom_models(self,
                     spectrum_trace: np.ndarray,
                     models_list: list,
                     peaks_numbers: list = [1, 2],
                     significance: float = 0.05,
                     peak_height: float = 0.5,
                     peak_distance: float = 1,
                     peak_prominence: float = 0.7,
                     peak_width: list = [2, 20],
                     verbose: bool = False # Added verbose flag
                     ) -> lmfit.model.ModelResult | None:
        """
        Evaluates different Gaussian models based on the number of peaks and
        returns the best fit result based on a chi-squared significance test.

        Args:
            spectrum_trace (np.ndarray): The 1D array of spectral intensity data.
            peaks_numbers (list): A list of integers, each representing a number of
                                   Gaussian peaks to test in the model.
            significance (float): The significance level for the chi-squared test.
            peak_height (float): Passed to scipy.signal.find_peaks. Required height of peaks.
            peak_distance (float): Passed to scipy.signal.find_peaks. Required minimal horizontal distance (in samples) between neighboring peaks.
            peak_prominence (float): Passed to scipy.signal.find_peaks. Required prominence of peaks.
            peak_width (list): Passed to scipy.signal.find_peaks. Required width of peaks in samples as `[min_width, max_width]`.
            verbose (bool): If True, prints detailed fit information during the process.

        Returns:
            lmfit.model.ModelResult: The lmfit result object for the best-fitting
                                     model that meets the significance criterion,
                                     or None if no model meets the criterion.
        """
        if not isinstance(spectrum_trace, np.ndarray) or spectrum_trace.size == 0:
            raise ValueError("spectrum_trace must be a non-empty numpy array.")
        if not all(isinstance(p, int) and p > 0 for p in peaks_numbers):
            raise ValueError("peaks_numbers must be a list of positive integers.")
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1.")

        x = self.spectral_axis_mhz
        y = spectrum_trace

        best_fit_result = None
        best_chi2_score = float('inf')

        for model in models_list:
            if verbose:
                print(f"\n--- Testing model with {n_peaks} peaks ---")
            gmodel = m
            initial_params = self._initialize_peak_parameters(
                x, y, n_peaks, peak_height, peak_distance, peak_prominence, peak_width
            )

            if initial_params is None:
                continue

            try:
                fit_result = gmodel.fit(y, initial_params, x=x)
            except ValueError as e:
                if verbose:
                    print(f'Error fitting {n_peaks} peaks model: {e}')
                continue

            if not fit_result.success:
                if verbose:
                    print(f"Model {n_peaks}: The fit did not converge.")
                continue

            if verbose:
                print(f"Model {n_peaks}: The fit converged successfully.")
                # print(fit_result.fit_report()) # Uncomment for detailed fit report

            chi2_statistic = fit_result.chisqr
            num_fitted_params = len(fit_result.params)
            dof = len(x) - num_fitted_params
            if dof <= 0:
                if verbose:
                    print(f"Warning: Degrees of freedom for {n_peaks} peaks is {dof}. Cannot perform chi-squared test.")
                continue

           # Calculate probability of the result based on the chi2 statistic
            chi2_prob = chi2.cdf(chi2_statistic, df=dof)

            if verbose:
                print(f'Model {n_peaks}: Chi-squared statistic = {chi2_statistic:.2f}, DOF = {dof}, Chi-squared probability = {chi2_prob:.3f}')

            # Save current model's chi2 statistic value
            current_model_score = chi2_statistic

            # If the chi2 statistic is smaller than the best previous chi2 statistic
            # then current statistic becomes the best and the corresponding best fit_result is saved 
            if current_model_score < best_chi2_score:
                best_chi2_score = current_model_score
                best_fit_result = fit_result
                if verbose:
                    print(f"Model {n_peaks}: Currently the best fit (lowest chi-squared).")


        if best_fit_result and verbose:
            print("\n--- Best Fit Result ---")
            print(f"Selected model has {len(best_fit_result.components) - 1} peaks (excluding constant).")
            best_fit_result.plot_fit()
            plt.title("Best Fit Result")
            plt.show() # Ensure plot is displayed
        elif not best_fit_result and verbose:
            print("\nNo successful fit found for any tested model.")

        return best_fit_result


    def fit_stark_map(self,
                      peaks_numbers: list = [1, 2], # Changed from np.ndarray to list for type hint clarity
                      verbose: bool = False,
                      # Expose find_peaks parameters from test_gmodels
                      peak_height: float = 0.5,
                      peak_distance: float = 1,
                      peak_prominence: float = 0.7,
                      peak_width: list = [2, 20]
                      ) -> list[lmfit.model.ModelResult | None]: # Added return type hint
        """
        Fits Gaussian models to each trace in the corrected image (Stark map).
        Iterates through each row of the image and applies the test_gmodels
        function to find the best-fitting Gaussian model for that trace.

        Args:
            peaks_numbers (list): A list of integers, each representing a number of
                                   Gaussian peaks to test for each trace.
            verbose (bool): If True, enables verbose output from test_gmodels
                            for each trace fit.
            peak_height (float): Passed to test_gmodels and scipy.signal.find_peaks.
            peak_distance (float): Passed to test_gmodels and scipy.signal.find_peaks.
            peak_prominence (float): Passed to test_gmodels and scipy.signal.find_peaks.
            peak_width (list): Passed to test_gmodels and scipy.signal.find_peaks.

        Returns:
            list[lmfit.model.ModelResult | None]: A list of lmfit.model.ModelResult objects
                                                for each fitted trace, or None if a fit failed.
                                                The results are also stored in self.fitted_models.
        Raises:
            ValueError: If `self.image_corrected` is not available or empty.
        """
        if self.image_corrected is None or self.image_corrected.size == 0:
            raise ValueError("No image data available in self.image_corrected to fit.")

        image = self.image_corrected
        fitted_models = []

        for i, y_trace in enumerate(image):
            if verbose:
                print(f'\nProcessing Row {i+1}/{len(image)}...') # Improved progress message
            if not(self.covariance_matrix.shape[0] > 0):
                fit = self.test_gmodels(
                    y_trace,
                    peaks_numbers=peaks_numbers,
                    verbose=verbose,
                    peak_height=peak_height,
                    peak_distance=peak_distance,
                    peak_prominence=peak_prominence,
                    peak_width=peak_width
                )
                fitted_models.append(fit)
            else:
                fit = self.test_gmodels_covar(
                    y_trace,
                    peaks_numbers=peaks_numbers,
                    verbose=verbose,
                    peak_height=peak_height,
                    peak_distance=peak_distance,
                    peak_prominence=peak_prominence,
                    peak_width=peak_width
                )
                fitted_models.append(fit)

        self.fitted_models = fitted_models
        return fitted_models # Return the list of fitted models

    def _extract_fit_parameters(self, fitted_models: list[lmfit.model.ModelResult | None]):
        """
        Extracts parameters and their errors from a list of lmfit ModelResult objects.

        Args:
            fitted_models (list): A list of lmfit.model.ModelResult objects or None.

        Returns:
            tuple: A tuple containing:
                   - x_data (np.ndarray): The x-axis data corresponding to valid models.
                   - results_dict (dict): A dictionary containing extracted 'centers',
                                          'amplitudes', 'fwhm' and their 'errors'.
                   - n_peaks_max (int): The maximum number of peaks found across all valid models.
        """
        # Filter out None models and get corresponding x-axis data
        none_mask = np.array([f is None for f in fitted_models])
        valid_models = np.array(fitted_models)[~none_mask]
        x_data = self.x_axis_mm_bin[~none_mask]

        if not valid_models.size > 0:
            print("No valid fitted models to extract parameters from.")
            return np.array([]), {}, 0

        # Determine maximum number of peaks across all valid models
        # A Gaussian model adds 3 parameters (center, amplitude, sigma) + 1 for constant background
        # So, num_peaks = (len(params) - 1) / 3
        # Ensure conversion to int for peak count
        n_peaks_per_model = [(len(fm.params) - 1) // 3 for fm in valid_models]
        n_peaks_max = int(np.max(n_peaks_per_model)) if n_peaks_per_model else 0

        # Initialize dictionary to store results
        results_dict = {
            'centers': np.full((n_peaks_max, len(valid_models)), np.nan),
            'amplitudes': np.full((n_peaks_max, len(valid_models)), np.nan),
            'fwhm': np.full((n_peaks_max, len(valid_models)), np.nan),
            'centers_err': np.full((n_peaks_max, len(valid_models)), np.nan),
            'amplitudes_err': np.full((n_peaks_max, len(valid_models)), np.nan),
            'fwhm_err': np.full((n_peaks_max, len(valid_models)), np.nan),
        }

        # Populate the results dictionary
        # FWHM conversion factor is 2*sqrt(2*ln(2))
        fwhm_coef = 2 * np.sqrt(2 * np.log(2)) # Assuming sigma from lmfit is standard deviation
                                             # The original code had 2.3538, which is approximately 2*sqrt(2*ln(2))

        for j, vm in enumerate(valid_models):
            current_n_peaks = (len(vm.params) - 1) // 3
            for i in range(current_n_peaks):
                # Extract values
                results_dict['centers'][i, j] = vm.params[f'g{i}_center'].value
                results_dict['amplitudes'][i, j] = vm.params[f'g{i}_amplitude'].value
                results_dict['fwhm'][i, j] = vm.params[f'g{i}_sigma'].value * fwhm_coef # Convert sigma to FWHM

                # Extract errors (stderr will be None if not estimated)
                results_dict['centers_err'][i, j] = vm.params[f'g{i}_center'].stderr
                results_dict['amplitudes_err'][i, j] = vm.params[f'g{i}_amplitude'].stderr
                results_dict['fwhm_err'][i, j] = vm.params[f'g{i}_sigma'].stderr * fwhm_coef if vm.params[f'g{i}_sigma'].stderr is not None else np.nan

        return x_data, results_dict, n_peaks_max


    
    def plot_gmodel_results(self,
                            transitions_labels: list[str] | str | None = None, # Allow list, string, or None
                            marker_style: list[str] = ['o', 'v', 's', '^', '*'],
                            marker_size: int = 3,
                            figsize: tuple[int, int] = (4, 7),
                            y_lim_centers: tuple[int, int] = (-700, 200),
                            y_lim_amp: tuple[int, int] = (0, 150),
                            y_lim_fwhm: tuple[int, int] = (0, 100),
                            ) -> tuple[plt.Figure, plt.Axes] | None: # Return type hint for clarity
        """
        Plots the extracted parameters (centers, amplitudes, FWHM) from fitted Gaussian models.

        Args:
            transitions_labels (list[str] | str | None): A list of strings for legend labels,
                                                           or a comma-separated string, or None.
            marker_style (list[str]): List of matplotlib marker styles to cycle through for peaks.
            marker_size (int): Size of the markers in the plots.
            figsize (tuple[int, int]): Figure size for the plot (width, height).
            y_lim_centers (tuple[int, int]): Y-axis limits for the 'centers' plot.

        Returns:
            tuple[plt.Figure, plt.Axes] | None: A tuple containing the matplotlib Figure and Axes objects,
                                                 or None if no valid models were found.
        Raises:
            ValueError: If `self.fitted_models` is not set or empty.
            ValueError: If `self.x_axis_mm_bin` is not set or does not match `fitted_models` length.
        """
        if not hasattr(self, 'fitted_models') or not self.fitted_models:
            raise ValueError("No fitted models found. Run fit_stark_map first.")
        if not hasattr(self, 'x_axis_mm_bin') or self.x_axis_mm_bin is None:
             raise ValueError("x_axis_mm_bin (distance axis) is not set. Cannot plot results.")
        if len(self.x_axis_mm_bin) != len(self.fitted_models):
            raise ValueError("Length of x_axis_mm_bin does not match the number of fitted models.")

        # Handle transitions_labels input
        if isinstance(transitions_labels, str):
            transitions_labels = transitions_labels.split(',')
        elif transitions_labels is None:
            transitions_labels = [] # Default to empty list if None

        # --- Data Extraction ---
        x_data, results_dict, n_peaks_max = self._extract_fit_parameters(self.fitted_models)

        if not x_data.size > 0:
            print("No valid data to plot after extraction.")
            return None

        # --- Plotting ---
        fig1, ax1 = plt.subplots(3, 1, sharex=True, figsize=figsize)

        for i in range(n_peaks_max):
            # Determine marker and label for the current peak
            current_marker = marker_style[i % len(marker_style)]
            label = transitions_labels[i] if i < len(transitions_labels) else f'Peak {i+1}'

            # Plot Centers
            ax1[0].errorbar(x=x_data, y=results_dict['centers'][i], yerr=results_dict['centers_err'][i],
                             color=f'C{i}', linestyle='None', label=label, capsize=2)
            ax1[0].plot(x_data, results_dict['centers'][i], marker=current_marker, color=f'C{i}',
                        markersize=marker_size, linestyle='None')
            ax1[0].set_ylabel('Blue Detuning (MHz)')
            ax1[0].set_ylim(y_lim_centers)
            ax1[0].grid(True, linestyle=':', alpha=0.6)


            # Plot Amplitudes
            ax1[1].errorbar(x=x_data, y=results_dict['amplitudes'][i], yerr=results_dict['amplitudes_err'][i],
                             linestyle='None', color=f'C{i}', alpha=0.6, capsize=2)
            ax1[1].plot(x_data, results_dict['amplitudes'][i], marker=current_marker, color=f'C{i}',
                        alpha=0.5, markersize=marker_size, linestyle='None')
            ax1[1].set_ylabel('Amplitude (rel. units)')
            ax1[1].set_ylim(y_lim_amp)
            ax1[1].grid(True, linestyle=':', alpha=0.6)


            # Plot FWHM
            ax1[2].errorbar(x=x_data, y=results_dict['fwhm'][i], yerr=results_dict['fwhm_err'][i],
                             linestyle='None', color=f'C{i}', alpha=0.6, capsize=2)
            ax1[2].plot(x_data, results_dict['fwhm'][i], marker=current_marker, color=f'C{i}',
                        alpha=0.5, markersize=marker_size, linestyle='None')
            ax1[2].set_xlabel('Distance (mm)')
            ax1[2].set_ylabel('FWHM (MHz)')
            ax1[2].set_ylim(y_lim_fwhm)
            ax1[2].grid(True, linestyle=':', alpha=0.6)

        # Add legends
        ax1[0].legend(loc='best', title='Transitions')
        # You might want separate legends for amplitude/FWHM if labels apply uniquely
        # For simplicity, I'm just adding one legend.

        fig1.tight_layout()
        plt.show()

        return fig1, ax1
     
    def plot_image_slice(self, 
                         distance = 0 # (mm) distance along the beam
                         ):
        '''
        TBD
        NEEDS REFACTORING
        '''
        nearest_index = np.abs(self.x_axis_mm_bin - distance).argmin()
        nn = self.image_corrected.shape[0]
        slice_bin = self.image_binned[nearest_index]
        slice_corrected = self.image_corrected[nearest_index]
        x_coord_bin = self.spectral_axis_mhz_bin
        _, ax = plt.subplots(figsize=(5,4))
        ax.plot(x_coord_bin, slice_bin, label=f'Corrected #{nearest_index}/{nn}')
        #ax.plot(x_coord, background, label='bg ')
        ax.plot(x_coord_bin, slice_corrected, label='corrected')
        ax.set_xlabel('Blue detuning (MHz)')
        ax.set_ylabel('EIT signal (V)')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend()
        
    
    def _baseline_image(self, image):
        im_out = np.copy(image)
        for i,c in enumerate(image):
            baseline_fitter = Baseline(x_data=np.arange(c.shape[0]))
            bkg, _ = baseline_fitter.asls(c, lam=1e5, p=0.1)
            im_out[i] = c - bkg

        return im_out
    
    def _detect_outliers_zscore(self, data, threshold=1.9):
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

    def get_raw_image(self):
        return self.raw_image

    def get_binned_image(self):
        return self.image_binned
    
    def get_corrected_image(self):
        return self.image_corrected
    
    def get_xcoord(self, binned=True):
        return self.x_axis_mm_bin
    
    def get_ycoord(self):
        return self.spectral_axis_mhz
    
    def get_covariance(self):
        return self.covariance_matrix
    
    def get_fit_results(self):
        x_data, results_dict, n_peaks_max = self._extract_fit_parameters(self.fitted_models)
        return x_data, results_dict, n_peaks_max


    
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
    



#%%
if __name__ == "__main__":

    #ommit_files = [1,2,3,4,5,6,7,10,16,18,19,20,23,28,30,48]
    ommit_files = []

    dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-06-26\\processed'
    #dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-07-01\\processed'

    fl_data, fl_files = read_files(dirpath, key='spec', files_idx_ommit=ommit_files)
    ref_data, ref_files = read_files(dirpath, key='sync', files_idx_ommit=ommit_files)

#%%
    fps = 12.8 # Hz
    sweep = 0.05 # Hz
    fov = 10.8 # mm
    image_number = 37#
    img = np.copy(fl_data[image_number])
    ref = np.copy(ref_data[image_number]).T
    print(ref.shape)
    print(img.shape)
#%%
    fsm = FLStarkMapProcessor(img,
                              ref,
                              fps,
                              sweep,
                              fov,
                              bin_power=7)
# %%
    fig = fsm.plot_calibration(figsize=(6,6))

# %%
    fig = fsm.plot_stark_map(False)
    fig = fsm.plot_corrected_stark_map(figsize=(6,5))
# %%
    fsm.plot_image_slice(2)
# %%
    # fsm.set_background_covariance(fl_data[-5])
    # fsm.plot_covariance()
    # fsm.get_covariance()

# %%
    fsm.fit_stark_map([3,4,5], 
                      verbose=False, 
                      peak_height=0.7, 
                      peak_prominence=0.5, 
                      peak_distance=1)

#%%
    #plot_labels = '$44D_{5/2}$: $|m_J|=5/2$,$44D_{5/2}$: $|m_J|=1/2$,$44D_{5/2}$: $|m_J|=3/2$,$44D_{3/2}$: $|m_J|=1/2$,$44D_{3/2}$: $|m_J|=1/2$'
    plot_labels = None
    fsm.plot_gmodel_results(plot_labels,
                            y_lim_centers=(-700,100),
                            y_lim_fwhm=(0,60))

# %%
    x_data, results_dict, _ = fsm.get_fit_results()

    print(results_dict['centers'].T[0])
    print(results_dict['centers_err'].T[0])
    print(results_dict['amplitudes'].T[0])
    print(results_dict['amplitudes_err'].T[0])
    print(results_dict['fwhm'].T[0])
    print(results_dict['fwhm_err'].T[0])

# %%
# --------------- Test custom models ---------------------


# model = fsm._generate_gmodel(3)

# params1 = lmfit.Parameters()
# params1.add('g0_center', expr='g1_center')
# params1.add('g2_center', expr='g1_center')
# params1.add('g0_amplitude', expr='g1_amplitude')
# params1.add('g2_amplitude', expr='g1_amplitude')


# params2 = lmfit.Parameters()
# params1.add('g0_center', expr='g1_center')
# params1.add('g0_amplitude', expr='g1_amplitude')

# params2 = lmfit.Parameters()

# models_dict = {'model_1': }
# %%
