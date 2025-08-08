# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os


#%%
class StarkMapGenerator():
    def __init__(self, images, 
                 fps_hz=13.68, 
                 fov_mm=(10.8,7.2), 
                 camera_resolution=(1536,1024),
                 bin_power=2
                 ):
        
        self.fps = fps_hz # FPS in Hz
        self.fov = fov_mm # field of view (horiz, vert) in mm
        self.resolution_full = camera_resolution # Full camera resolution
        if isinstance(images, str):
            self.raw_images = self._read_images_batch(images)
            self.bin_images = self.vertical_bin_all_images(bin_power)
        else:
            self.raw_images = None
            print(f'ERROR: "{images}" is not a path. Provide valid path to npz file with images.')
    
    def _read_images_batch(self, file_path):
        if os.path.exists(file_path) and os.path.isfile(file_path):
            extension = os.path.splitext(file_path)[1]
            if extension == ".npz":
                try:
                    with np.load(path) as data:
                        images = data['arr_0']
                        return images
                except ValueError:
                    ...
            else:
                print(f"The file '{file_path}' exists but is not .npz: {extension}")
                return None
        else:
            print(f"The file '{file_path}' does not exist or is not a file.")
            return None
        
    def plot_batch_imgs(self, figsize=(6,6), binned=False):
        if not self.raw_images is None:
            if binned:
                images = self.bin_images
            else:
                images = self.raw_images
            n_imgs = images.shape[0]
            idxs = np.sort(np.random.choice(np.arange(n_imgs),size=8,replace=False))
            imgs_batch = images[idxs]
            global_min = np.min(np.ndarray.flatten(imgs_batch))
            global_max = np.max(np.ndarray.flatten(imgs_batch))
            fig, ax = plt.subplots(4,2, dpi=300)
            fig.set_size_inches(figsize)
            axs = np.ndarray.flatten(ax)
            for a, im, i in zip(axs, imgs_batch, idxs):
                a.pcolormesh(im, cmap='jet', vmin=global_min, vmax=global_max)
                a.set_title(f'Image #{i}')
            fig.tight_layout()
            fig.suptitle('Random batch of images')
            return fig, axs
        else:
            print('ERROR: No images batch to plot.')

    def _vertical_binning(self, image, bin_power) -> np.ndarray:

        # 1. Validate input attributes
        if not isinstance(image, np.ndarray):
            raise AttributeError("`image` must be a NumPy array.")
        if not isinstance(bin_power, int):
            raise AttributeError("`bin_power` must be an integer attribute.")
        if bin_power < 0:
            raise ValueError(f"`bin_power` cannot be negative, got {bin_power}. Must be a non-negative integer.")
        if image.ndim != 2:
            raise ValueError(f"Expected `image` to be a 2D array, but got {image.ndim} dimensions.")

        bin_size = 2**bin_power
        if bin_size == 0: # This can happen if bin_power is negative, but handled above
            raise ValueError("Bin size cannot be zero. Check `bin_power`.")

        num_vertical_points = image.shape[1]
        if num_vertical_points % bin_size != 0:
            raise ValueError(
                f"The number of vertical points ({num_vertical_points}) "
                f"is not evenly divisible by the bin size ({bin_size}). "
                "Adjust `bin_power` or handle truncation/padding of the image."
            )

        try:
            # Transpose the image so that the vertical dimension becomes the first dimension
            # for iteration, then reshape and mean along that dimension.
            # A more direct way without explicit transpose in list comprehension:
            # image_bin = self.raw_image.reshape(
            #     self.raw_image.shape[0],
            #     num_spectral_points // bin_size,
            #     bin_size
            # ).mean(axis=2)

            binned_rows = []
            for row_data in image.T: # Iterate through each row along spatial axis
                # Reshape each row to create blocks of 'bin_size' and take the mean
                binned_row = row_data.reshape(-1, bin_size).mean(axis=1)
                binned_rows.append(binned_row)

            image_bin = np.array(binned_rows).T # Binned image trasposed to restore axes order

        except ValueError as e:
            # Catch specific ValueError from numpy.reshape if the shape is incompatible
            raise ValueError(
                f"Failed to reshape and bin image data. "
                f"Original error: {e}. "
                f"Check if image dimensions ({image.shape[1]}) "
                f"are compatible with bin size ({bin_size})."
            )
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred during image binning: {e}")

        return image_bin
    
    def vertical_bin_all_images(self, bin_power):
        n_imgs, ny, nx = self.raw_images.shape
        ny_binned = int(ny / 2**bin_power)
        print(self.raw_images.shape)
        binned_images = np.zeros((n_imgs, ny_binned, nx))
        for i, im in enumerate(self.raw_images):
            binned_images[i] = self._vertical_binning(im, bin_power)
        return binned_images
    
    def plot_bin(self):
        im = self._vertical_binning(self.bin_images[0], 5)
        fig, ax = plt.subplots()
        ax.pcolormesh(im, cmap='jet')

    def generate_stark_maps(self):
        stark_maps_raw = self.bin_images.transpose(1, 0, 2)
        print(stark_maps_raw.shape)
        n_maps, n_freq, n_spatial = stark_maps_raw.shape
        baseline_intensity = np.repeat(stark_maps_raw[:,0][:,:,np.newaxis], 
                                       repeats=n_freq, 
                                       axis=2).transpose(0,2,1)
        print(baseline_intensity.shape)
        intensity_ratio = np.divide(stark_maps_raw, baseline_intensity)
        stark_maps = (np.ones(stark_maps_raw.shape) - intensity_ratio)*100

        return stark_maps
    

# %%
if __name__ == '__main__':
    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-07\\data\\data-imgs-2-2025-08-07.npz'
    sm = StarkMapGenerator(path, bin_power=6)
    sm.plot_batch_imgs(binned=True)
    stark_maps = sm.generate_stark_maps()
# %%
    nspec, ny, nx = stark_maps.shape
    fig, ax = plt.subplots(nrows=nspec, figsize=(3,7))
    flat_stark_maps = stark_maps.flatten()
    glob_max = np.max(flat_stark_maps)
    glob_min = np.min(flat_stark_maps)
    for a,s in zip(ax, stark_maps):
        a.pcolormesh(s, cmap='inferno', 
                     vmin=glob_min,
                     vmax=glob_max)
# %%
