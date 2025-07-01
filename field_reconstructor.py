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
from lmfit import CompositeModel
from lmfit.models import GaussianModel # Ensure CompositeModel is imported
from scipy.stats import chi2

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
                                              np.linspace(0, half_period_sec, self.image_binned.shape[1])
                                              ) * sec_to_mhz
                self.image_corrected = self._baseline_image()
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
                       min_peak_width: int = 10,
                       max_peak_width: int = 300,
                       min_peak_distance: int = 15,
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

            # Your original approach, rephrased slightly for clarity:
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

        gmodel = GaussianModel(prefix=prefixes[0])
        for i in range(1, len(prefixes)):
            gmodel = gmodel + GaussianModel(prefix=prefixes[i])

        return gmodel
    
    def test_gmodels(self,
                     spectrum_trace: np.ndarray = [],
                     peaks_numbers: list = [1,2],
                     significance: float = 0.05
                     ):
        '''
        NEEDS REFACTORING
        '''
        tested_gmodels = [self._generate_gmodel(p) for p in peaks_numbers]

        x = self.spectral_axis_mhz
        y = spectrum_trace
        #y = self.image_corrected[1] # Line for debugging purposes

        gresults = []
        chi2_scores = []
        for n, gmodel in zip(peaks_numbers, tested_gmodels):
            # Find peaks
            pidx, pprop= find_peaks(y, height=0.05, distance=2, prominence=0.4, width=[2, 40])
            pmax =[]
            if len(pidx) >= n:
                s = y[pidx].argsort()
                pmax = np.flip(pidx[s][-n:])
            else:
                print(f'Hoped for {n} peaks but found {len(pidx)}')
            
            params = lmfit.Parameters()
            for i in range(n):
                params.add(f'g{i}_center', value=x[pmax[i]])
                params.add(f'g{i}_amplitude', value=30, min=1)
                params.add(f'g{i}_sigma', value=10, max=30)
            
            fit_result = gmodel.fit(y,params,x=x)
            gresults.append(fit_result)

            if fit_result.success:
                print(f"Model {n}: The fit converged successfully.")
            else:
                print(f"Model {n}: The fit did not converge.")
                continue

            chi2_out = fit_result.chisqr
            
            
            dof = len(x) - 3*n
            chi2_score = chi2.cdf(chi2_out*13, df=dof)
            chi2_scores.append(chi2_score)
            print(f'Chi squared for model {n} is: score {chi2_score}; statistic {chi2_out}')
            if chi2_score < 1-significance:
                # for gr in gresults:
                    # plt.figure()
                    # gr.plot_fit()
                    # dely = gr.eval_uncertainty(sigma=3)   
                    # plt.plot(x,gr.init_fit,'--', label='Initi guess')
                    # plt.plot(x[pmax], y[pmax], 'o', color='red')
                    # plt.fill_between(x, gr.best_fit-dely, gr.best_fit+dely, color="#ABABAB", label=r'3-$\sigma$ uncertainty band')
                return fit_result
            
    def test_custom_models(self,
                           models: np.ndarray[CompositeModel]
                           ):
        raise NotImplementedError
    
    def fit_stark_map(self,
                      peaks_numbers: np.ndarray = [1,2]
                      ):
        '''
        NEEDS REFACTORING
        '''
        #x = self.spectral_axis_mhz_bin
        image = self.image_corrected

        fitted_models = []
        for i,y in enumerate(image):
            print(f'Row {i}')
            fit = self.test_gmodels(y, peaks_numbers=peaks_numbers)
            fitted_models.append(fit)
        self.fitted_models = fitted_models
        return None
    
    def plot_gmodel_results(self,
                         transitions_str: str = []
                         ):
        '''
        NEEDS REFACTORING
        '''
        transitions_labels = transitions_str.split(sep=',')
        x = self.x_axis_mm_bin

        fitted_models = np.array(self.fitted_models)
        none_mask = np.array([f is None for f in fitted_models])

        m = ['o', 'v', 's', '^']

        fig1, ax1 = plt.subplots(3,1, sharex=True, figsize=(4,7))
        marker_size = 3
        for i in range(4):
            peaks_pos = np.array([r.params[f'g{i}_center'].value for r in fitted_models[~none_mask]], dtype=float)
            pos_err = np.array([r.params[f'g{i}_center'].stderr for r in fitted_models[~none_mask]], dtype=float)
            ax1[0].errorbar(x=x[~none_mask], y=peaks_pos, yerr=pos_err, markersize=marker_size)
            ax1[0].set_ylabel('Blue Detuning (MHz)')

            
            heights = np.array([r.params[f'g{i}_amplitude'].value for r in fitted_models[~none_mask]], dtype=float)
            heights_err = np.array([r.params[f'g{i}_amplitude'].stderr for r in fitted_models[~none_mask]], dtype=float)
            ax1[1].plot(x[~none_mask], heights, marker=m[i], color=f'C{i}', alpha=0.5, markersize=marker_size, linestyle='None')
            ax1[1].errorbar(x[~none_mask],
                            heights,
                            yerr=heights_err,
                            linestyle='None',
                            color=f'C{i}',
                            alpha = 0.4)
            ax1[1].set_ylabel('Amplitude (rel. units)')

            fhwhm_coef = 2.3538
            w = fhwhm_coef*np.array([r.params[f'g{i}_sigma'].value for r in fitted_models[~none_mask]], dtype=float)
            w_err = fhwhm_coef*np.array([r.params[f'g{i}_sigma'].stderr for r in fitted_models[~none_mask]], dtype=float)
            ax1[2].plot(x[~none_mask], w, marker=m[i], color=f'C{i}', alpha = 0.3, markersize=marker_size, linestyle='None')
            ax1[2].errorbar(x = x[~none_mask],
                            y = w,
                            yerr = w_err,
                            linestyle='None',
                            color=f'C{i}',
                            alpha = 0.3)
            ax1[2].set_xlabel('Distance (mm)')
            ax1[2].set_ylabel('FWHM (MHz)')


        # fig2, ax2 = plt.subplots(2,2, sharex=True)
        # axs2 = ax2.flatten()
        # for i in range(4):
        #     peaks_pos = np.array([r.params[f'g{i+1}_center'].value for r in fitted_models], dtype=float)
        #     pos_err = np.array([r.params[f'g{i+1}_center'].stderr for r in fitted_models], dtype=float)

        #     db_mask, _ = _detect_outliers_zscore(peaks_pos, threshold=1.7)

        #     axs2[i].plot(X[~db_mask],
        #                 peaks_pos[~db_mask],
        #                 marker='.',
        #                 alpha = 0.5,
        #                 color = f'C{i}'
        #                 )
        #     axs2[i].errorbar(x = x[~db_mask],
        #                     y = peaks_pos[~db_mask],
        #                     yerr = pos_err[~db_mask],
        #                     color = f'C{i}',
        #                     linestyle='None',
        #                     alpha = 0.3)
        #     axs2[i].set_title(transitions_labels[i])
        # fig2.supxlabel('Distance (mm)', fontsize=14)
        # fig2.supylabel('Blue detuning (MHz)', fontsize=14)
        # fig2.suptitle('Peak positions', fontsize=18)

     
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
        
    
    def _baseline_image(self):
        im_out = np.copy(self.image_binned)
        for i,c in enumerate(self.image_binned):
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
    ommit_files = [1,2,3,4,5,6,7,10,16,18,19,20,23,28,30,48]
    fl_data, fl_files = read_files('./processed/', key='spec', files_idx_ommit=ommit_files)
    ref_data, ref_files = read_files('./processed/', key='sync', files_idx_ommit=ommit_files)

#%%
    fps = 12.8 # Hz
    sweep = 0.05 # Hz
    fov = 10.8 # mm
    image_number = 6
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
                              bin_power=3)
    
# %%
    fig = fsm.plot_calibration(figsize=(6,6))

# %%
    fig = fsm.plot_stark_map(False)
    fig = fsm.plot_corrected_stark_map()
# %%
    fsm.plot_image_slice(6)
# %%
    fsm.fit_stark_map([4])

#%%
    plot_labels = '$44D_{5/2}$: $|m_J|=5/2$,$44D_{5/2}$: $|m_J|=1/2$,$44D_{5/2}$: $|m_J|=3/2$,$44D_{3/2}$: $|m_J|=1/2$,$44D_{3/2}$: $|m_J|=1/2$'
    fsm.plot_gmodel_results(plot_labels)
# %%
