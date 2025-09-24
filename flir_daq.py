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

from simple_pyspin import Camera

from qolab.hardware.scope.sds800xhd import SDS800XHD
import pyvisa

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from time import perf_counter
from datetime import datetime
import time

import pyperclip as pyp
import re

import h5py


class FLIRdaq():
    
    def __init__(self, 
                 fps_hz=12.8,
                 blue_laser_sweep_hz=0.01,
                 field_of_view_mm=10.8,
                 aspect_ratio=1.5,
                 full_resolution = (3072, 2048),
                 camera_model = 'Blackfly S BFS-PGE-63S4M',
                 file_id: str | None = None
                 ):
        self.file_id = file_id
        self.camera_model = camera_model

        self.fps_hz = fps_hz
        self.actual_fps_hz = None
        self.blue_laser_sweep_hz = blue_laser_sweep_hz
        self.number_of_frames = int(np.floor(0.5*fps_hz/blue_laser_sweep_hz))
        self.time_stop = self.number_of_frames/fps_hz
        self.acquisition_time = None

        self.full_resolution = full_resolution
        self.full_horizontal_resolution = full_resolution[0]
        self.full_vertical_resolution = full_resolution[1]


        self.field_of_view_mm = field_of_view_mm
        self.aspect_ratio = aspect_ratio
        self.horizontal_distance = field_of_view_mm
        self.vertical_distance = field_of_view_mm / aspect_ratio
        

        self.images_stack = None
        self.reference_signal = None
        self.reference_time = None

        self.comments = []

    
    def acquire_frames(self, 
                       number_of_frames: int | None = None,
                       camera_index:int = 0
                       ):

        fnum = number_of_frames
        if (number_of_frames == None):
            fnum = self.number_of_frames
        elif (number_of_frames < 1):
            fnum = self.number_of_frames
        print('-'*20)
        print(f'Grabbing {fnum} frames...')
        print(f'[{datetime.now()}]')
        with Camera(camera_index) as cam: # Initialize Camera
            cam.start() # Start recording
            start_time = perf_counter()
            imgs = np.array([cam.get_array() for n in range(fnum)]) # Get 'fnum' frames
            end_time = perf_counter()
            cam.stop() # Stop recording
        print('Frames grabbing completed.')
        print(f'[{datetime.now()}]')

        elapsed_time = end_time - start_time
        print(f'Elapsed time: {elapsed_time} sec.' )
        print('-'*20)

        self.images_stack = imgs
        self.acquisition_time = elapsed_time
        self.number_of_frames = fnum
        self.actual_fps_hz = fnum / elapsed_time

        _, ny, nx = imgs.shape
        ver_scale_mm = ny/self.full_resolution[1] * self.vertical_distance
        hor_scale_mm = nx/self.full_resolution[0] * self.horizontal_distance

        self.vertical_distance = ver_scale_mm
        self.horizontal_distance = hor_scale_mm

    def acquire_reference_trace(self,
                                scope_visa_resource,
                                n_pts_request = 4000
                                ):
        '''
        'scope1': 'TCPIP0::192.168.110.191::inst0::INSTR',
        'scope2': 'TCPIP0::192.168.110.198::inst0::INSTR',    
        '''

        traces = scope_visa_resource.getAllTraces(maxRequiredPoints=n_pts_request)
        traces_data = traces.getData().T # Transpose data for usability
        print(f'[{datetime.now()}] Reference traces retrieved.')
        self.reference_signal = traces_data

    @classmethod
    def read_npz(self, file_path):
        file = np.load(file_path)
        output_dict = {key: file[key] for key in file.files}
        return output_dict
    
    def init_from_npz(self, file_path: str) -> None:
        data_dict = self.read_npz(file_path)

        self.file_id = data_dict['file_id']
        self.camera_model = data_dict['camera_model']
        self.number_of_frames = data_dict['number_of_frames']
        self.time_stop = data_dict['time_stop']
        self.fps_hz = data_dict['fps_hz']
        self.blue_laser_sweep_hz = data_dict['blue_laser_sweep_hz']
        self.acquisition_time = data_dict['acquisition_time_sec']
        self.full_resolution = data_dict['camera_full_resolution']
        self.vertical_distance = data_dict['vertical_mm']
        self.horizontal_distance = data_dict['horizontal_mm']
        self.reference_signal = data_dict['reference_signals_volt']
        self.images_stack = data_dict['images_stack'].T
        self.comments = data_dict['comments']

        self.full_horizontal_resolution = self.full_resolution[0]
        self.full_vertical_resolution = self.full_resolution[1]

    def get_data_dict(self):
        # Before saving we would like to make the array such 
        # that 
        # - axis 0 iterates over frames
        # - axis 1 is for the horizontal pixels
        # - axis 2 is for the vertical pixels
        images_transposed = np.transpose(self.images_stack, (0, 2, 1))

        output_dict = {
            'file_id': self.file_id,
            'camera_model': self.camera_model,
            'number_of_frames': self.number_of_frames,
            'time_stop': self.time_stop,
            'fps_hz': self.fps_hz,
            'blue_laser_sweep_hz': self.blue_laser_sweep_hz,
            'acquisition_time_sec': self.acquisition_time,
            'camera_full_resolution': self.full_resolution,
            'vertical_mm': np.round(self.vertical_distance,2),
            'horizontal_mm': np.round(self.horizontal_distance,2),
            'reference_signals_volt': self.reference_signal,
            'images_stack': images_transposed,
            'comments': self.comments
        }
        return output_dict
    
    def save_hdf5(self, file_path: str, file_access='w') -> None:
        data_dict = self.get_data_dict()
        data_keys = ['images_stack','reference_signals_volt', 'camera_full_resolution', 'comments', 'file_id', 'camera_model']
        with h5py.File(file_path, file_access) as f:

            imgs_set = f.create_dataset('/raw_data/images_stack', data_dict['images_stack'].shape, dtype='f', compression="gzip")
            ref_set = f.create_dataset('/raw_data/reference_signals_volt', data_dict['reference_signals_volt'].shape, dtype='f', compression="gzip")
            
            imgs_set[...] = data_dict['images_stack']
            ref_set[...] = data_dict['reference_signals_volt']

            for key in data_dict:
                print(key)
                if key not in data_keys:
                    imgs_set.attrs[key] = data_dict[key]


    def save_images(self, 
                    file_path:str
                    ):
        '''
        Save images, reference trace and other information for further 
        processing and calibration.
        '''
        print("Saving images to: %s" % file_path)

        file_name = re.split(r'[\\/]', file_path)[-1]
        self.file_id = file_name
    
        output_dict = self.get_data_dict()['file_id'] = self.file_id

        print('Saving imags to \'.npz\' file...')
        try:
            np.savez(file_path, **output_dict)
        except IOError as e:
                print(f"Error saving dictionary to file: {e}")
        print(f"Dictionary successfully saved to {file_path}")

    def add_comment(self, comment):
        self.comments.append(comment)
        print(f'Added comment: {comment}')
    
    def print_info(self):
        print('-'*20)
        # Below the `image_stack` entry has been shuffled 
        # to be consistent with what is actually saved
        # using `save_images()` method
        ni, ny, nx = self.images_stack.shape
        print(
            f'file_id \"{self.file_id}\"',
            f'camera_model \"{self.camera_model}\"',
            f'time_stop: {self.time_stop} sec',
            f'number_of_frames: {self.number_of_frames}',
            f'fps_hz: {self.fps_hz} Hz',
            f'blue_laser_sweep_hz: {self.blue_laser_sweep_hz} Hz',
            f'acquisition_time_sec: {self.acquisition_time} s',
            f'camera_full_resolution: {self.full_resolution}',
            f'vertical_mm: {self.vertical_distance:.2f} mm',
            f'horizontal_mm: {self.horizontal_distance:.2f} mm',
            f'reference_signals_volt: {self.reference_signal.shape}',
            f'images_stack: ({ni}, {nx}, {ny})',
            'comments:',
            sep='\n'
        )
        for i, c in enumerate(self.comments):
            print(f'\t{i}: {c}')
        print('-'*20)
    
    def plot_batch_imgs(self, figsize=(6,6)):
        if self.images_stack is None:
            print('ERROR: No images to plot. Run frames acquisition first!')
            return None
        imgs = self.images_stack
        n_imgs = imgs.shape[0]

        ver_scale_mm = self.vertical_distance
        hor_scale_mm = self.horizontal_distance

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
        return self.images_stack
    
    def detect_overexposed(self):
        raise NotImplementedError



# %%
if __name__=="__main__":
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Create DAQ object for acquisition and initial pre-processing
    daq = FLIRdaq(blue_laser_sweep_hz=0.05) 
    daq.acquire_frames()
    daq.acquire_reference_trace()
    # daq.plot_batch_imgs()
    daq.add_comment('Test images batch!')
# %%
    daq.save_images('./processed/tmp_file.npz')
# %%
    daq.print_info()


# %%
