'''
Mykhailo Vorobov

The class structure based on the script daq-flir-2025-08-22.py

Created: 2025-08-22
Modified: 
'''
# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Using standard Python logging instead of print statements
import logging 
import os
from time import perf_counter
from datetime import datetime
import time

# External dependencies 
# (assuming these are available in the environment)
from simple_pyspin import Camera
from qolab.hardware.scope.sds800xhd import SDS800XHD
import pyvisa


import pyperclip as pyp
import re

import h5py
from typing import Literal, Dict, Any, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Configure the logger for the module/class
# 1. Retrieve the module logger. 
#    This logger exists across all classes defined here.
logger = logging.getLogger(__name__)

class FLIRdaq():
    
    def __init__(self, 
                 fps_hz: float = 12.8,
                 blue_laser_sweep_hz: float = 0.01,
                 field_of_view_mm: float = 10.8,
                 full_resolution: Tuple[int, int] = (3072, 2048),
                 camera_model: str = 'Blackfly S BFS-PGE-63S4M',
                 file_id: Optional[str] = None
                 ):
        
        # Assign logger instance to the class
        # All instances now share the same 'flir_daq' named logger object
        self.logger = logger
        self.logger.info(f"Initializing FLIRdaq instance...")

        self.file_id = file_id if file_id is not None else datetime.now().strftime('%Y%m%d_%H%M%S')
        self.camera_model = camera_model

        self.fps_hz = fps_hz
        self.actual_fps_hz: Optional[float] = None
        self.blue_laser_sweep_hz = blue_laser_sweep_hz
        self.number_of_frames: int = int(np.floor(0.5 * fps_hz / blue_laser_sweep_hz))
        self.time_stop_sec: float = self.number_of_frames / self.fps_hz
        self.acquisition_time_sec: Optional[float] = None

        self.full_resolution = full_resolution
        self.full_horizontal_resolution = full_resolution[0]
        self.full_vertical_resolution = full_resolution[1]


        self.field_of_view_mm = field_of_view_mm
        self.aspect_ratio = full_resolution[0]/full_resolution[1]
        self.horizontal_distance_mm: float = field_of_view_mm
        self.vertical_distance_mm: float = field_of_view_mm / self.aspect_ratio
        
        # Use None as the initial state for data arrays
        self.image_stack: Optional[np.ndarray] = None
        self.reference_signal: Optional[np.ndarray] = None
        self.scope_trigger: Optional[np.ndarray] = None
        self.scope_eit_reference: Optional[np.ndarray] = None
        self.scope_time: Optional[np.ndarray] = None

        self.comments: list[str] = []

    def acquire_frames(self, 
                       number_of_frames: Optional[int] = None,
                       camera_index: int = 0
                       ) -> None:
        """
        Acquires an image stack from the FLIR camera.

        Updates `self.images_stack`, `self.acquisition_time`, and adjusts 
        `self.horizontal_distance` and `self.vertical_distance` based on 
        the acquired image's Region of Interest (ROI) size.

        Parameters
        ----------
        number_of_frames : Optional[int], optional
            The number of frames to acquire. If None or less than 1, uses the 
            pre-calculated `self.number_of_frames`. The default is None.
        camera_index : int, optional
            The index of the camera to use (e.g., 0 for the first camera). 
            The default is 0.
        """

        fnum = number_of_frames if number_of_frames is not None and number_of_frames >= 1 else self.number_of_frames
        
        self.logger.info(f'Grabbing {fnum} frames...')

        try:
            with Camera(camera_index) as cam:
                cam.start()
                start_time = perf_counter()
                imgs = np.array([cam.get_array() for _ in range(fnum)])
                end_time = perf_counter()
                cam.stop()
            self.logger.info('Frames grabbing completed.')
        except NameError:
             self.logger.warning("Camera class not found. Using dummy data for acquisition.")
             # Dummy data for testing if hardware is not available
             imgs = np.zeros((fnum, 512, 512), dtype=np.uint16)
             end_time = perf_counter()
             start_time = end_time - 0.5 # Dummy elapsed time
             self.logger.warning('Frames acquisition simulated with dummy data.')
        except Exception as e:
            self.logger.error(f"An error occurred during camera acquisition: {e}")
            raise # Re-raise error to stop operation

        self.logger.info(f'[{datetime.now()}]')

        elapsed_time = end_time - start_time
        self.logger.info(f'Elapsed time: {elapsed_time:.3f} sec.' )

        self.image_stack = imgs
        self.acquisition_time_sec = elapsed_time
        self.number_of_frames = fnum
        self.actual_fps_hz = fnum / elapsed_time

        # Update physical distance based on acquired image size (ROI)
        if imgs.ndim == 3:
            _, ny, nx = imgs.shape
            ver_scale_mm = (ny / self.full_resolution[1]) * self.vertical_distance_mm
            hor_scale_mm = (nx / self.full_resolution[0]) * self.horizontal_distance_mm
            self.vertical_distance_mm = ver_scale_mm
            self.horizontal_distance_mm = hor_scale_mm
            self.logger.info(f'FOV updated to {self.horizontal_distance_mm:.2f}\
                             x{self.vertical_distance_mm:.2f} mm based on acquired ROI.')


    def acquire_reference_trace(self,
                                scope_visa_resource,
                                n_pts_request: int = 4000,
                                reference_channel: int = 1,
                                trigger_channel: int = 4
                                ) -> None:
        """
        Acquire reference traces (EIT signal and Trigger) from the oscilloscope.
        
        Updates `self.scope_time`, `self.scope_eit_reference`, and `self.scope_trigger`.

        Parameters
        ----------
        scope_visa_resource : qolab.hardware.scope.sds800xhd.SDS800XHD or pyvisa.Resource
            An initialized resource object for the oscilloscope.
        n_pts_request : int, optional
            The maximum number of points requested from the scope. The default is 4000.
        reference_channel : int, optional
            The index of the channel containing the EIT reference signal. The default is 1.
        trigger_channel : int, optional
            The index of the channel containing the trigger signal. The default is 4.
            
        Example resources:
        'scope1': 'TCPIP0::192.168.110.191::inst0::INSTR',
        'scope2': 'TCPIP0::192.168.110.198::inst0::INSTR',    
        """
        if scope_visa_resource is None:
            self.logger.error("Scope VISA resource is None. Cannot acquire reference traces.")
            return

        try:
            traces = scope_visa_resource.getAllTraces(maxRequiredPoints=n_pts_request)
            traces_data = traces.getData().T # Transpose data for usability
            self.logger.info(f'[{datetime.now()}] Reference traces retrieved.')
            
            self.reference_signal = traces_data
            self.scope_time = traces_data[:, 0]
            self.scope_eit_reference = traces_data[:, reference_channel]
            self.scope_trigger = traces_data[:, trigger_channel]
            self.logger.info(f'Scope data successfully stored. Time points: {len(self.scope_time)}')
        except Exception as e:
            self.logger.error(f"Error during scope trace acquisition: {e}")
            # Optionally raise if acquisition failure is critical
            # raise

    @staticmethod
    def read_npz(file_path: str) -> Dict[str, Any]:
        """
        Reads data from an NPZ file into a dictionary.

        Parameters
        ----------
        file_path : str
            The path to the NPZ file.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the data stored in the NPZ file.
        """
        try:
            # Use the module-level logger within static methods
            logger.info(f"Reading NPZ file: {file_path}")
            file = np.load(file_path, allow_pickle=True) 
            output_dict = {key: file[key].item() if file[key].ndim == 0 else file[key] for key in file.files}
            return output_dict
        except Exception as e:
            logger.error(f"Failed to read NPZ file {file_path}: {e}")
            raise
    
    def init_from_npz(self, file_path: str) -> None:
        """
        Initializes class attributes from data loaded from an NPZ file.

        Parameters
        ----------
        file_path : str
            The path to the NPZ file to load.
        """
        data_dict = self.read_npz(file_path)

        # Use .get() with defaults for robustness against missing keys
        self.file_id = data_dict.get('file_id', os.path.basename(file_path))
        self.camera_model = data_dict.get('camera_model', self.camera_model)
        self.number_of_frames = int(data_dict.get('number_of_frames', 0))
        self.time_stop_sec = float(data_dict.get('time_stop_sec', 0.0))
        self.fps_hz = float(data_dict.get('fps_hz', 0.0))
        self.blue_laser_sweep_hz = float(data_dict.get('blue_laser_sweep_hz', 0.0))
        self.acquisition_time_sec = float(data_dict.get('acquisition_time_sec', 0.0))
        
        self.full_resolution = tuple(data_dict.get('camera_full_resolution', self.full_resolution))
        self.full_horizontal_resolution = self.full_resolution[0]
        self.full_vertical_resolution = self.full_resolution[1]

        self.vertical_distance_mm = float(data_dict.get('vertical_mm', self.vertical_distance_mm))
        self.horizontal_distance_mm = float(data_dict.get('horizontal_mm', self.horizontal_distance_mm))
        self.reference_signal = data_dict.get('reference_signals_volt')
        self.image_stack = data_dict.get('images_stack') 
        self.comments = list(data_dict.get('comments', []))
        
        self.scope_time = data_dict.get('scope_time')
        self.scope_eit_reference = data_dict.get('scope_eit_reference')
        self.scope_trigger = data_dict.get('scope_trigger')
        
        self.logger.info(f"Initialized DAQ state from NPZ file: {file_path}")

    def get_data_dict(self) -> Dict[str, Any]:
        """
        Gathers all acquired data and metadata parameters into a dictionary for saving.

        The image stack is transposed from the camera's native (Frame, Y, X) 
        order to the saving standard (Frame, X, Y) for consistency with HDF5 
        dimension labels ('f', 'x', 'y').

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all current state variables, excluding those with None values.
        """
        images_transposed = None
        if self.image_stack is not None and self.image_stack.ndim == 3:
            # (frame, vertical_y, horizontal_x) -> (frame, horizontal_x, vertical_y)
            image_transposed = np.transpose(self.image_stack, (0, 2, 1))

        output_dict = {
            'file_id': self.file_id,
            'camera_model': self.camera_model,
            'number_of_frames': self.number_of_frames,
            'time_stop_sec': self.time_stop_sec,
            'fps_hz': self.fps_hz,
            'blue_laser_sweep_hz': self.blue_laser_sweep_hz,
            'acquisition_time_sec': self.acquisition_time_sec,
            'camera_full_resolution': self.full_resolution,
            'vertical_mm': np.round(self.vertical_distance_mm, 2),
            'horizontal_mm': np.round(self.horizontal_distance_mm, 2),
            
            'image_stack': image_transposed,
            'reference_signals_volt': self.reference_signal,
            'scope_eit_reference': self.scope_eit_reference,
            'scope_trigger': self.scope_trigger,
            'scope_time': self.scope_time,
            'comments': self.comments,
        }
        
        # Remove keys with None values (no data acquired)
        return {k: v for k, v in output_dict.items() if v is not None}
    
    def save_hdf5(self, file_path: str,
                  file_access: Literal['w', 'x'] = 'w',
                  compression: str = 'gzip') -> None:
        """
        Save stack of images and scope reference traces into an HDF5 file.
        
        The data is saved into HDF5 datasets, and all non-array metadata 
        is stored as attributes on the primary 'image_stack' dataset.

        Parameters
        ----------
        file_path : str
            File saving path (e.g., 'data/run01.h5').
        file_access : Literal['w', 'x'], optional
            File mode (can be 'w', 'x'). The default is 'w' (write, overwrite existing).
        compression : str, optional
            Compression filter used by the `h5py` module (e.g., 'gzip', 'lzf'). 
            The default is 'gzip'.
        """
        self.file_id = os.path.basename(file_path)
        data_dict = self.get_data_dict()
        self.logger.info(f"Attempting to save HDF5 data to: {file_path}")

        DATASET_CONFIG = [
            ('image_stack', 'image_stack', np.uint16, None, ('f', 'x', 'y')),
            ('scope_eit_reference', 'scope_eit_reference', 'f', 'V', None),
            ('scope_trigger', 'scope_trigger', 'f', 'V', None),
            ('scope_time', 'scope_time', 'f', 's', None)
        ]
        EXCLUDE_FROM_ATTRS = {
            'image_stack', 'reference_signals_volt', 'scope_eit_reference',
            'scope_trigger', 'scope_time', 'comments'
        }

        try:
            with h5py.File(file_path, file_access) as f:
                main_image_set = None
                for name, key, dtype, units, labels in DATASET_CONFIG:
                    if key not in data_dict:
                        continue 

                    data = data_dict[key]
                    
                    dset = f.create_dataset(
                        name, 
                        data=data,
                        dtype=dtype, 
                        compression=compression
                    )

                    if name == 'image_stack':
                        main_image_set = dset
                        
                    if units:
                        dset.attrs['units'] = units

                    if labels and len(labels) == len(dset.dims):
                        for i, label in enumerate(labels):
                            dset.dims[i].label = label

                if main_image_set is not None:
                    self.logger.info("Saving metadata as HDF5 attributes.")
                    for key, value in data_dict.items():
                        if key not in EXCLUDE_FROM_ATTRS:
                            try:
                                main_image_set.attrs[key] = value
                            except TypeError:
                                self.logger.warning(f"Could not save attribute '{key}' of type {type(value)}. Saving as string.")
                                main_image_set.attrs[key] = str(value)
            
            self.logger.info(f"HDF5 file successfully saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Critical error saving HDF5 file: {e}")
            raise

    def save_npz(self, 
                 file_path: str
                ) -> None:
        """
        Save all acquired data (images and scope traces) and metadata 
        to a NumPy compressed archive (.npz file).

        The file name is also set as the `self.file_id`.

        Parameters
        ----------
        file_path : str
            File saving path (e.g., 'data/run01.npz').
        """
        self.logger.info(f"Attempting to save NPZ data to: {file_path}")

        # 1. Update file_id from the path
        self.file_id = os.path.basename(file_path)
    
        # 2. Get the complete data dictionary
        output_dict = self.get_data_dict()

        self.logger.info(f'Saving data to \'.npz\' file...')
        try:
            # np.savez automatically saves non-array items as pickled objects (ndim=0 array)
            np.savez(file_path, **output_dict)
            self.logger.info(f"NPZ file successfully saved to {file_path}")
        except IOError as e:
            self.logger.error(f"Error saving dictionary to file: {e}")
            raise

    def add_comment(self, comment: str) -> None:
        """
        Adds a user-defined comment or note to the `self.comments` list.

        Parameters
        ----------
        comment : str
            The text comment to be added.
        """
        self.comments.append(comment)
        self.logger.info(f'Added comment: "{comment}"')
    
    def print_info(self):
        """
        Logs a summary of the current DAQ configuration and data status 
        (e.g., shapes, FOV, acquisition time) to the console via the logger.
        All information is logged in a single batch call.
        """
        # Check for None values before trying to access shape
        ref_shape = self.reference_signal.shape if self.reference_signal is not None else 'N/A'
        img_shape = self.image_stack.shape if self.image_stack is not None else 'N/A'
        
        # 1. Build the list of information lines
        info_lines = [
            '--- DAQ Instance Summary ---',
            f'File ID: "{self.file_id}"',
            f'Camera Model: "{self.camera_model}"',
            f'Total Frames: {self.number_of_frames} (Target: {self.fps_hz / self.blue_laser_sweep_hz * 0.5})',
            f'Acquisition Time: {self.acquisition_time_sec} s',
            # f'Actual FPS: {self.actual_fps_hz:.2f} Hz',
            f'Current FOV: {self.horizontal_distance_mm:.2f} x {self.vertical_distance_mm:.2f} mm',
            f'Images Stack Shape (Frame, Y, X): {img_shape}',
            f'Reference Signal Shape: {ref_shape}',
            'Comments:',
        ]
        
        # 2. Add comments
        if self.comments:
            for i, c in enumerate(self.comments):
                info_lines.append(f'\t{i}: {c}')
        else:
            info_lines.append('\tNo comments recorded.')
        
        # 3. Join the lines and log the entire batch once
        self.logger.info('\n\t'.join(info_lines))

    def plot_batch_imgs(self, figsize=(6,6)):
        """
        Plots a random batch of 8 acquired images with physical axes.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the overall figure (width, height) in inches. The default is (6, 6).

        Returns
        -------
        Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or Tuple[None, None]
            The matplotlib Figure and Axes objects, or (None, None) if no images are present.
        """
        if self.image_stack is None:
            self.logger.error('No images to plot. Run frames acquisition first!')
            return None, None
        
        imgs = self.image_stack
        n_imgs = imgs.shape[0]

        ver_scale_mm = self.vertical_distance_mm
        hor_scale_mm = self.horizontal_distance_mm

        idxs = np.sort(np.random.choice(np.arange(n_imgs),size=8,replace=False))
        imgs_batch = imgs[idxs]
        global_min = np.min(np.ndarray.flatten(imgs_batch))
        global_max = np.max(np.ndarray.flatten(imgs_batch))
        fig, ax = plt.subplots(4,2, dpi=300)
        fig.set_size_inches(figsize)
        axs = np.ndarray.flatten(ax)
        for a, im, i in zip(axs, imgs_batch, idxs):
            a.imshow(im, 
                     cmap='jet', 
                     vmin=global_min, 
                     vmax=global_max,
                     extent=[0, hor_scale_mm, 0, ver_scale_mm])
            a.set_title(f'Image #{i}')
        axs[-1].set_xlabel('X (mm)')
        fig.supylabel('Y (mm)')
        fig.suptitle('Random batch of images')
        fig.tight_layout()

        return fig, ax
    
    def get_images(self):
        return self.image_stack
    
    def detect_overexposed(self):
        """
        [NOT IMPLEMENTED] Placeholder for a method to detect overexposed frames 
        in the image stack.

        Raises
        ------
        NotImplemented
        """
        raise NotImplemented



# %%
if __name__=="__main__":

    # Create DAQ object for frames and reference trace acquisition
    rm = pyvisa.ResourceManager()
    resource = rm.open_resource('TCPIP0::192.168.110.198::inst0::INSTR')
    scope = SDS800XHD(resource)


    # --- Logging Configuration (Best Practice for Main Script) ---
    # 1. Get the module logger (same object retrieved at the top of the file)
    module_logger = logging.getLogger(__name__) 
    
    # 2. Set the desired logging level (e.g., DEBUG to see everything)
    module_logger.setLevel(logging.DEBUG) 
    
    # 3. Add a console handler and formatter IF one isn't already present
    # The formatter now excludes %(name)s (the file name)
    if not module_logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        ch.setFormatter(formatter)
        module_logger.addHandler(ch)
    # -----------------------------------------------------------------

    # Create DAQ object for acquisition and initial pre-processing
    daq = FLIRdaq(blue_laser_sweep_hz=0.05) 
    # Since Camera is not necessarily available, acquire_frames will use dummy data
    daq.acquire_frames() 
    daq.acquire_reference_trace(scope)
    
    daq.add_comment('Test images batch acquisition!')
    daq.print_info()
    
    # Create the 'processed' directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    daq.save_npz('./data/tmp_file.npz')
    daq.save_hdf5('./data/tmp_file.h5')
    
    # Example of loading the saved data back (for verification)
    new_daq = FLIRdaq()
    new_daq.init_from_npz('./data/tmp_file.npz')
    new_daq.print_info()
# %%
