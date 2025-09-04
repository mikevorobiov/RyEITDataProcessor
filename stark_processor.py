'''
Mykhailo Vorobiov

Created: 2025-08-19
Modified: 2025-08-19
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os

from stark_map import StarkMap
from pybaselines import Baseline
import copy
#%%
class StarkProcessor():
    """
    Handles processing tasks for a StarkMap data object.
    
    This class uses composition, holding a raw StarkMap instance and
    generating a new StarkMap instance for the processed result.
    """
    def __init__(self, raw_map: StarkMap):
        """
        Initializes the processor with a raw StarkMap instance.
        
        Args:
            raw_map (StarkMap): The raw, unprocessed StarkMap instance.
        """
        if not isinstance(raw_map, StarkMap):
            raise TypeError("Input 'raw_map' must be an instance of StarkMap.")
        
        self.processed_map = copy.deepcopy(raw_map)

    def _apply_baseline_to_trace(self, trace: np.ndarray, mode='snip', **kwargs) -> np.ndarray:
        """Applies a baseline correction to a single spectral trace."""
        baseline_fitter = Baseline(x_data=np.arange(len(trace)))
        if mode=='snip':
            bkg, _ = baseline_fitter.snip(trace, **kwargs)
        elif mode=='arpls':
            bkg, _ = baseline_fitter.arpls(trace, **kwargs)
        return trace - bkg

    def baseline_map(self, mode='snip', **kwargs) -> StarkMap:
        """
        Applies baseline correction to each trace and stores the result
        in a new StarkMap instance.
        
        Returns:
            StarkMap: A new StarkMap instance with the baseline-corrected data.
        """
        # Determine the map to use for processing
        map_to_process = self.processed_map

        map_data, freq_mhz, dist_mm = map_to_process.get_map_data()
        
        if map_data is None:
            raise ValueError("Raw map data is not set. Cannot perform baseline correction.")

        im_out = np.copy(map_data)
        for i in range(im_out.shape[1]):
            im_out[:, i] = self._apply_baseline_to_trace(im_out[:, i], mode=mode, **kwargs)
        
        # Add the new comment to the processed map
        # map_to_process.add_comment(f"[{dt.date.today()}] Baseline correction applied with lambda={lam}, p={p}.")
        # Get the file_id from the source map
        file_id = map_to_process.get_file_id()
        comments = map_to_process.get_comments()
        self.processed_map = StarkMap(
            file_id=file_id,
            map_data=im_out,
            frequency_mhz=freq_mhz,
            distance_mm=dist_mm,
            comments=comments
        )
        return self.processed_map
    
    def median_baseline(self):

        # Determine the map to use for processing
        map_to_process = self.processed_map

        map_data, freq_mhz, dist_mm = map_to_process.get_map_data()
        median = np.median(map_data, axis=0)
        median_map = np.repeat(median[np.newaxis,:], len(freq_mhz), axis=0)
        print(median_map.shape)
        result_map = map_data - median_map

        # Add the new comment to the processed map
        map_to_process.add_comment(f'[{dt.date.today()}] Baselined by subtracting median of each spectrum.')
        
        # Get the file_id from the source map
        file_id = map_to_process.get_file_id()
        comments = map_to_process.get_comments()
        
        # Create a new StarkMap instance with the new list of comments
        self.processed_map = StarkMap(
            file_id=file_id,
            map_data=result_map,
            frequency_mhz=freq_mhz,
            distance_mm=dist_mm,
            comments=comments
        )

        return self.processed_map


    def _apply_binning_to_trace(self, signal, bin_power):
        '''
        Bin single array of size integer multiple of 2
        '''
        bin_size = 2**bin_power
        binned_signal = signal.reshape(-1, bin_size).mean(axis=1)
        return binned_signal
    
    def vertical_binning(self, bin_power=1):
        bin_size = 2**bin_power
        
        # Determine the map to use for processing
        map_to_process = self.processed_map
        
        map_data, freq_mhz, dist_mm = map_to_process.get_map_data()
        
        freq_mhz_new = freq_mhz[::bin_size]
        map_binned = np.apply_along_axis(self._apply_binning_to_trace, 0, map_data, bin_power=bin_power)

        # Add the new comment to the processed map
        map_to_process.add_comment(f'[{dt.date.today()}] Vertical binning: `2**{bin_power}`')
        
        # Get the file_id from the source map
        file_id = map_to_process.get_file_id()
        comments = map_to_process.get_comments()
        
        # Create a new StarkMap instance with the new list of comments
        self.processed_map = StarkMap(
            file_id=file_id,
            map_data=map_binned,
            frequency_mhz=freq_mhz_new,
            distance_mm=dist_mm,
            comments=comments
        )
        return self.processed_map
    

    def horizontal_binning(self, bin_power=1):
        bin_size = 2**bin_power
        
        # Determine the map to use for processing
        map_to_process = self.processed_map
        
        map_data, freq_mhz, dist_mm = map_to_process.get_map_data()
        
        dist_mm_new = dist_mm[::bin_size]
        map_binned = np.apply_along_axis(self._apply_binning_to_trace, 1, map_data, bin_power=bin_power)

        # Add the new comment to the processed map
        map_to_process.add_comment(f'[{dt.date.today()}] Horizontal binning: `2**{bin_power}`')
        
        # Get the file_id from the source map
        file_id = map_to_process.get_file_id()
        comments = map_to_process.get_comments()
        
        # Create a new StarkMap instance with the new list of comments
        self.processed_map = StarkMap(
            file_id=file_id,
            map_data=map_binned,
            frequency_mhz=freq_mhz,
            distance_mm=dist_mm_new,
            comments=comments
        )
        return self.processed_map
    
    def get_processed_map(self) -> StarkMap | None:
        """Returns the processed StarkMap instance, if it exists."""
        if self.processed_map is None:
            print("Warning: Map has not been processed yet.")
        return self.processed_map
    

# %%
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from stark_map import StarkMap

    # --- Setup: Create a dummy raw data file ---
    dummy_file_name = 'raw_stark_map.npz'
    
    def create_dummy_data(file_name: str) -> None:
        freq_axis = np.linspace(-500, 500, 128)
        dist_axis = np.linspace(0, 10, 512)
        background = np.linspace(20, 50, len(freq_axis))
        signal = 10 * np.exp(-((freq_axis - 100)**2 / 1000)) + 5 * np.exp(-((freq_axis + 150)**2 / 1200))
        dummy_data = (background[:, np.newaxis] + signal[:, np.newaxis] * (1 + 0.1 * dist_axis[np.newaxis, :]))
        sm = StarkMap(file_name, dummy_data, freq_axis, dist_axis)
        sm.plot()
        sm.save(file_name)

    create_dummy_data(dummy_file_name)
    
    # --- Step 1: Load a raw StarkMap instance ---
    print("--- 1. Loading a raw StarkMap instance from file ---")
    try:
        raw_map = StarkMap.load(dummy_file_name)
        raw_map.add_comment("Original raw data from an IR camera.")
    except Exception as e:
        print(f"Failed to load raw map: {e}")
        raw_map = None

    if raw_map:
        # --- Step 2: Initialize the processor with the raw map ---
        print("\n--- 2. Initializing the StarkMapProcessor with the raw map ---")
        processor = StarkMapProcessor(raw_map)
        
        # --- Step 3: Use the processor to generate a new processed map ---
        print("\n--- 3. Applying baseline correction and creating a new map ---")
        processor.horizontal_binning(3)
        processor.baseline_map(lam=1e6, p=0.05)
        processed_map = processor.vertical_binning(1)
        processed_map.print_info()

        plt.plot(*raw_map.get_spectrum(target_distance=8))
        plt.plot(*processed_map.get_spectrum(target_distance=8))
        
        # --- Step 4: Work with the new processed map instance ---
        if processed_map:
            print("\n--- 4. The new processed map instance is ready for use ---")
            processed_map.plot()
            
            # Save the new map
            processed_file_name = 'processed_stark_map.npz'
            processed_map.save(processed_file_name)
            
        # --- Verification: The raw map is unchanged ---
        print("\n--- Verification: The raw map remains untouched ---")
        raw_map.print_info()
        
        # --- Cleanup ---
        os.remove(dummy_file_name)
        if processed_map and os.path.exists(processed_file_name):
            os.remove(processed_file_name)
        
        print("\nTemporary files have been cleaned up.")
# %%
