'''
Mykhailo Vorobiov

Created: 2025-08-19
Modified: 2025-08-19
'''
# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import h5py
#%%
class StarkMap:
    """
    Manages Stark map data, metadata, and related plotting and saving functionalities.
    
    A Stark map represents a 2D dataset where the signal (e.g., EIT)
    is measured as a function of frequency and spatial distance.
    """

    def __init__(self, 
                 file_id: str | None = None,
                 map_data: np.ndarray | None = None,
                 frequency_mhz: np.ndarray | None = None,
                 distance_mm: np.ndarray | None = None,
                 comments: list[str] | None = None):
        """
        Initializes a StarkMap instance, optionally with data.

        Args:
            file_id (str | None): A unique identifier for the map.
            map_data (np.ndarray | None): 2D array of EIT signal.
            frequency_mhz (np.ndarray | None): 1D array of blue laser frequencies (MHz).
            distance_mm (np.ndarray | None): 1D array of distances (mm).
        """
        self.file_id = file_id
        self.comments = comments if comments else []
        self.parameters = {}
        
        # Initialize data fields, checking for provided values
        if map_data is not None and frequency_mhz is not None and distance_mm is not None:
            # If all data arrays are provided, call the set_map method to validate and assign.
            try:
                self.set_map(map_data, frequency_mhz, distance_mm)
            except ValueError as e:
                # Catch the dimension mismatch error and raise it from the constructor
                raise ValueError(f"Error initializing StarkMap with provided data: {e}") from e
        else:
            # If any data array is missing, initialize to None
            self.map = None
            self.frequency_mhz = None
            self.distance_mm = None

    def get_file_id(self) -> str | None:
        """Returns the original file identifier."""
        return self.file_id

    def set_map(self, 
                map_data: np.ndarray,
                frequency_mhz: np.ndarray,
                distance_mm: np.ndarray
                ) -> None:
        """
        Sets the Stark map data and its corresponding axes.

        Args:
            map_data (np.ndarray): 2D array of EIT signal (e.g., in % of IR power).
            frequency_mhz (np.ndarray): 1D array of blue laser frequencies (MHz).
            distance_mm (np.ndarray): 1D array of distances along laser beams (mm).
        
        Raises:
            ValueError: If array dimensions are not compatible.
        """
        if map_data.shape != (len(frequency_mhz), len(distance_mm)):
            raise ValueError(
                f"Shape mismatch: map_data shape {map_data.shape} is not compatible with "
                f"frequency_mhz length {len(frequency_mhz)} and distance_mm length {len(distance_mm)}."
            )
        self.map = map_data
        self.frequency_mhz = frequency_mhz
        self.distance_mm = distance_mm

    def get_map_data(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Gets the Stark map data and its corresponding frequency and distance axes.

        Returns:
            A tuple containing map data, frequency axis, and distance axis, or (None, None, None)
            if the map has not been set.
        """
        return self.map, self.frequency_mhz, self.distance_mm

    def get_spectrum(self, target_distance: float = 0.0) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Gets the Stark spectrum at the closest distance to the target.

        Args:
            target_distance (float): The desired distance to get the spectrum from (in mm).

        Returns:
            A tuple containing the frequency axis and the 1D spectrum array,
            or (None, None) if the map has not been set.
        """
        if self.map is None:
            print("Warning: No Stark map data assigned to this instance.")
            return None, None
            
        closest_dist_idx = np.argmin(np.abs(self.distance_mm - target_distance))
        spectrum = self.map.T[closest_dist_idx]
        
        print(f"Retrieved spectrum at distance {self.distance_mm[closest_dist_idx]:.2f} mm.")
        
        return self.frequency_mhz, spectrum

    def add_comment(self, comment: str) -> None:
        """Adds a comment to the history of the map."""
        if not isinstance(comment, str):
            raise TypeError("Comment must be a string.")
        self.comments.append(comment)
        print(f"Added comment: {comment}")

    def get_comments(self) -> list[str]:
        """
        Returns the list of comments stored for the map.

        Returns:
            list[str]: A list of comments. Returns an empty list if no comments exist.
        """
        if not self.comments:
            print('This Stark map has no comments.')
        return self.comments

    def print_info(self) -> None:
        """
        Prints all available information about the map.
        """
        print("-" * 30)
        print(f"Stark Map Info:")
        print(f"  File ID: {self.file_id}")
        if self.map is not None:
            print(f"  Map Dimensions: {self.map.shape}")
            print(f"  Frequency Range: [{self.frequency_mhz.min():.1f}, {self.frequency_mhz.max():.1f}] MHz")
            print(f"  Distance Range: [{self.distance_mm.min():.2f}, {self.distance_mm.max():.2f}] mm")
        else:
            print("  Map data not yet set.")
        
        print("  Comments:")
        if not self.comments:
            print("    (None)")
        else:
            for i, c in enumerate(self.comments):
                print(f"    {i+1}: {c}")
        print("-" * 30)

    def plot(self, color_map: str = 'jet') -> tuple[plt.Figure, plt.Axes] | None:
        """
        Plots the Stark map using a pcolormesh.

        Args:
            color_map (str): The matplotlib colormap to use.

        Returns:
            A tuple of the Figure and Axes objects, or None if no map data is available.
        """
        if self.map is None:
            print("Warning: No Stark map data assigned. Cannot plot.")
            return None
        
        fig, ax = plt.subplots()
        cmesh = ax.pcolormesh(self.distance_mm, self.frequency_mhz, self.map, cmap=color_map)
        
        cbar = fig.colorbar(cmesh, ax=ax)
        cbar.set_label("EIT Signal (%)")
        
        ax.set_ylabel('Blue Detuning (MHz)')
        ax.set_xlabel('Distance (mm)')
        ax.set_title(f'Stark Map: {self.file_id}')

        fig.tight_layout()
        plt.show()
        
        return fig, ax
    
    def save(self, file_path: str) -> bool:
        """
        Saves the Stark map and its metadata to a compressed NumPy file (.npz).

        Args:
            file_path (str): The path to save the file to.

        Returns:
            bool: True if save was successful, False otherwise.
        """

        if self.map is None:
            print("Warning: No Stark map data assigned. Cannot save.")
            return False
            
        try:
            data_dict = {
                "file_id": self.file_id,
                "comments": self.comments,
                "map_data": self.map,
                "frequency_mhz": self.frequency_mhz,
                "distance_mm": self.distance_mm
            }
            np.savez_compressed(file_path, **data_dict)
            print(f"Stark map saved successfully to '{file_path}'.")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
        
    def save_hdf5(self, 
                  file_path: str, 
                  file_mode='w',
                  compression='gzip') -> bool:
        """
        Saves the Stark map and its metadata to an HDF5 file (.hdf).

        Args:
            file_path (str): The path to save the file to.
            file_access (str): The file mode (can be 'r', 'r+', 'w', 'x' or 'a'; see h5py docs for details)

        Returns:
            bool: True if save was successful, False otherwise.
        """
        if self.map is None:
            print("Warning: No Stark map data assigned. Cannot save.")
            return False
            
        try:
            # Open file for writing
            with h5py.File(file_path, file_mode) as f:
                # 1 Create dataset for Stark map
                smap_set = f.create_dataset("stark_map", 
                                            self.map.shape, 
                                            dtype='f',
                                            compression=compression)
                # Name of the image stack file 
                # from which the map had been created
                smap_set.attrs['image_stack'] = self.file_id

                # 2 Create dataset for frequency detuning samples
                freq_set = f.create_dataset("frequency_mhz", 
                                            self.frequency_mhz.shape, 
                                            dtype='f',
                                            compression=compression)
                freq_set.make_scale('Detuning') #Make the dataset a scale
                freq_set.attrs['units'] = 'MHz' #Associate units (MHz)

                # 3 Create dataset for position along laser beam samples
                dist_set = f.create_dataset("distance_mm", 
                                            self.distance_mm.shape, 
                                            dtype='f',
                                            compression=compression)
                dist_set.make_scale('Position') #Make the dataset a scale
                dist_set.attrs['units'] = 'mm' #Associate units (millimeters)
                
                # 4 Associate scale datasets with their 
                #   respective Stark map dimensions
                f['stark_map'].dims[0].attach_scale(freq_set)
                f['stark_map'].dims[1].attach_scale(dist_set)

                # 5 Populate datasets in the file with the 
                #   current class fields data
                smap_set[...] = self.map
                freq_set[...] = self.frequency_mhz
                dist_set[...] = self.distance_mm


            print(f"Stark map saved successfully to '{file_path}'.")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    @classmethod
    def from_hdf5(cls, file_path: str):
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'.")
            return None
        try:
            with h5py.File(file_path, 'r') as f:
                 print("Keys in the file:", list(f.keys()))
                 print(f['/stark_map'].shape)
                 map_data = np.array(f['/stark_map'])
                 freq_mhz = np.array(f['/frequency_mhz'])
                 dist_mm = np.array(f['/distance_mm'])
                 file_id = f['/stark_map'].attrs['image_stack']
            
            new_map = cls(file_id=file_id,
                          map_data=map_data,
                          frequency_mhz=freq_mhz, 
                          distance_mm=dist_mm
                          )
            return new_map
        except Exception as e:
            print(f"Error initializing map from file: {e}")
            return None

    @classmethod
    def load(cls, file_path: str):
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'.")
            return None
        try:
            data = np.load(file_path, allow_pickle=True)
            print(f"File contents: {', '.join(data.files)}")
            file_id = data.get('file_id', 'Unknown ID').item()
            map_data = data.get('map_data')
            freq_mhz = data.get('frequency_mhz')
            dist_mm = data.get('distance_mm')
            
            # The constructor handles the optional initialization
            new_map = cls(file_id=file_id, map_data=map_data, frequency_mhz=freq_mhz, distance_mm=dist_mm)
            
            comments = data.get('comments')
            if comments is not None and comments.size > 0:
                if isinstance(comments, np.ndarray):
                    new_map.comments = [comments.tolist()]
                else:
                    new_map.comments = [comments.item()]
            
            return new_map
        except Exception as e:
            print(f"Error loading file: {e}")
            return None


# Example
if __name__ == "__main__":
    import datetime as dt
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    from stark_map import StarkMap

    # --- Section 1: Create a dummy map for demonstration ---
    # This part sets up a mock dataset to ensure the example is self-contained.
    print("--- 1. Creating a Dummy Stark Map and Saving It ---")
    
    # Define some dummy data
    freq_axis = np.linspace(-500, 500, 64)
    dist_axis = np.linspace(0, 10, 128)
    dummy_data = np.random.rand(64, 128) + np.exp(-((freq_axis[:, None])**2 / 1e4))
    dummy_file_name = 'test_stark_map.npz'

    # Initialize a new StarkMap object with data
    try:
        sm_new = StarkMap(
            file_id='dummy-data-2025-08-20',
            map_data=dummy_data,
            frequency_mhz=freq_axis,
            distance_mm=dist_axis
        )
        sm_new.add_comment(f'[{dt.date.today()}] Generated dummy data for example.')
        sm_new.add_comment(f'[{dt.date.today()}] Modified.')
        sm_new.save(dummy_file_name)
    except ValueError as e:
        print(f"Error creating dummy map: {e}")
        sm_new = None

    # --- Section 2: Load an existing map from a file ---
    print("\n--- 2. Loading the Saved Stark Map ---")
    
    loaded_map = StarkMap.load(dummy_file_name)
    if loaded_map:
        loaded_map.print_info()
        loaded_map.plot()
        
    # --- Section 3: Extract and plot a single spectrum ---
    print("\n--- 3. Extracting and Plotting a Single Spectrum ---")
    if loaded_map:
        # Get the spectrum at a specific distance
        freq, spectrum = loaded_map.get_spectrum(target_distance=4.5)
        
        if freq is not None and spectrum is not None:
            plt.figure()
            plt.plot(freq, spectrum)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Signal (%)')
            plt.title(f'Spectrum at {loaded_map.distance_mm[np.argmin(np.abs(loaded_map.distance_mm - 4.5))]:.2f} mm')
            plt.grid(True)
            plt.show()

    # --- Section 4: Clean up dummy files ---
    print("\n--- 4. Cleaning Up ---")
    if os.path.exists(dummy_file_name):
        os.remove(dummy_file_name)
        print(f"Removed temporary file: '{dummy_file_name}'.")

# %%
