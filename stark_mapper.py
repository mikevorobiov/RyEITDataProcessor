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
                 camera_resolution=(1536,1024)
                 ):
        
        self.fps = fps_hz # FPS in Hz
        self.fov = fov_mm # field of view (horiz, vert) in mm
        self.resolution_full = camera_resolution # Full camera resolution
        if isinstance(images, str):
            self.raw_images = self._read_images_batch(images)
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
        
    def plot_batch_imgs(self, figsize=(6,6)):
        if not self.raw_images is None:
            n_imgs = self.raw_images.shape[0]
            idxs = np.sort(np.random.choice(np.arange(n_imgs),size=8,replace=False))
            imgs_batch = self.raw_images[idxs]
            global_min = np.min(np.ndarray.flatten(imgs_batch))
            global_max = np.max(np.ndarray.flatten(imgs_batch))
            fig, ax = plt.subplots(4,2, dpi=300)
            fig.set_size_inches(figsize)
            axs = np.ndarray.flatten(ax)
            for a, im, i in zip(axs, imgs_batch, idxs):
                a.imshow(im, cmap='jet', vmin=global_min, vmax=global_max)
                a.set_title(f'Image #{i}')
            fig.tight_layout()
            fig.suptitle('Random batch of images')
            return fig, axs
        else:
            print('ERROR: No images batch to plot.')

    def _bin_image(self, image, bin_power) -> np.ndarray:

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
    
    def plot_bin(self):
        im = self._bin_image(self.raw_images[0], 5)
        fig, ax = plt.subplots()
        ax.pcolormesh(im, cmap='jet')
    

# %%
if __name__ == '__main__':
    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-07\\data\\data-imgs-3-2025-08-07.npz'
    sm = StarkMapGenerator(path)
    sm.plot_batch_imgs()
    plt.close()
    sm.plot_bin()

# %%
