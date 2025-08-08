# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pybaselines import Baseline

import os


#%%
class StarkMapsGenerator():
    def __init__(self, images, 
                 fps_hz=13.68, # Hz
                 laser_sweep=0.1, # Hz
                 fov_mm=(10.8,7.2), 
                 camera_resolution=(1536,1024),
                 bin_power=2
                 ):
        
        self.fps = fps_hz # FPS in Hz
        self.laser_sweep = laser_sweep # Hz
        self.fov = fov_mm # field of view (horiz, vert) in mm
        self.resolution_full = camera_resolution # Full camera resolution
        if isinstance(images, str):
            self.raw_images = self._read_images_batch(images)
            self.bin_images = self.vertical_bin_all_images(bin_power)
            self.stark_maps = self.generate_stark_maps()
            self.stark_maps_baselined = self.baseline_stark_maps()
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
    
    def get_time_distance(self):
        n_frames, ny, nx = self.bin_images.shape
        time_stop = n_frames/self.fps
        time = np.linspace(0,time_stop,n_frames)
        x = np.linspace(0,self.fov[0],nx)
        y = np.linspace(0,self.fov[1],ny)
        return time, x, y
    
    def _baseline_stark_map(self, image):
        im_out = np.copy(image.T)
        print(im_out.shape)
        for i,c in enumerate(image.T):
            baseline_fitter = Baseline(x_data=np.arange(c.shape[0]))
            bkg, _ = baseline_fitter.asls(c, lam=1e5, p=0.1)
            im_out[i] = c - bkg
        return im_out.T
    
    def baseline_stark_maps(self):
        baselined_batch = np.copy(self.stark_maps)
        for i,smap in enumerate(self.stark_maps):
            baselined_batch[i] = self._baseline_stark_map(smap)
        return baselined_batch 

    def save_sark_map_batch(self, path, option='baselined'):
        time, x, y, = self.get_time_distance()
        save_dict = {'time(s)': time,
                     'HorzDistance(mm)': x,
                     'VertDistance(mm)': y,
                     'SMaps(EITperc)': self.stark_maps_baselined}
        if option == 'baselined':
            np.savez(path, **save_dict)
        else:
            print('ERROR: Wrong option for save!')
    

# %%
if __name__ == '__main__':
    # Custom matpltotlib style 
    # can be deleted with no harm 
    mpl.style.use('custom-style')

    path = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-07\\data\\data-imgs-3-2025-08-07.npz'
    sm = StarkMapsGenerator(path, bin_power=5)
    sm.plot_batch_imgs(binned=True)
# %%
    stark_maps = sm.baseline_stark_maps()[4:]
# %%
    nspec, ny, nx = stark_maps.shape
    fig, ax = plt.subplots(nrows=nspec, 
                           figsize=(3,8),
                           sharex=True)
    flat_stark_maps = stark_maps.flatten()
    glob_max = np.max(flat_stark_maps)
    glob_min = np.min(flat_stark_maps)
    time, x, y = sm.get_time_distance()
    for a,s in zip(ax, stark_maps):
        a.pcolormesh(x, time, s, 
                     cmap='inferno', 
                     vmin=glob_min,
                     vmax=glob_max)
    fig.supxlabel('Distance (mm)',
                  y=0.05)
    fig.supylabel('Frequency (seconds)',
                  x=-0.03)
# %%
    from skimage.restoration import denoise_wavelet, denoise_bilateral, denoise_nl_means, denoise_tv_bregman,denoise_tv_chambolle
    smap_select = stark_maps[9]

    methods_str = ['Original',
                   'Wavelet',
                   'Bilateral']
    denoise_wavelet = denoise_wavelet(smap_select, 
                                      method='BayesShrink',
                                      mode='hard',
                                      sigma=0.5)
    denoise_bilat = denoise_bilateral(smap_select,
                                      win_size=3,
                                      mode='wrap')
    denoise_nlmeans = denoise_nl_means(smap_select, patch_size=50)
    denoise_tv_bregman = denoise_tv_bregman(smap_select,isotropic=False)
    denoise_tv_chambolle = denoise_tv_chambolle(smap_select)

    fig, ax = plt.subplots(2,3,figsize=(10,3))
    ax[0,0].pcolormesh(smap_select, cmap='inferno')
    ax[0,0].set_title(methods_str[0])
    ax[0,1].pcolormesh(denoise_wavelet, cmap='inferno')
    ax[0,1].set_title(methods_str[1])
    ax[0,2].pcolormesh(denoise_bilat, cmap='inferno')
    ax[0,2].set_title(methods_str[2])
    ax[1,0].pcolormesh(denoise_nlmeans, cmap='inferno')
    # ax[1,0].set_title(methods_str[0])
    ax[1,1].pcolormesh(denoise_tv_bregman, cmap='inferno')
    # ax[1,1].set_title(methods_str[1])
    ax[1,2].pcolormesh(denoise_tv_chambolle, cmap='inferno')
    # ax[1,2].set_title(methods_str[2])

    fig, ax = plt.subplots(3,1, figsize=(4,10))
    n_slices = 350
    for i in range(0,nx,n_slices):
        ax[0].plot(smap_select[:,i])
        ax[1].plot(denoise_wavelet[:,i])
        ax[2].plot(denoise_bilat[:,i], 'o-', alpha=0.4)

# %%
    sm.save_sark_map_batch('./maps.npz')

# %%
    df = np.load('./maps.npz')
    print(df.files)
# %%
