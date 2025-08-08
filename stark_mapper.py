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


# %%
if __name__ == '__main__':
    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-07\\data\\data-imgs-3-2025-08-07.npz'
    sm = StarkMapGenerator(path)
    sm.plot_batch_imgs()

# %%
