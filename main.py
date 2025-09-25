# %%

from field_reconstructor import FLStarkMapProcessor, read_files
import numpy as np


#%%
if __name__ == "__main__":
    #ommit_files = [1,2,3,4,5,6,7,10,16,18,19,20,23,28,30,48]
    ommit_files = []

    #dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-06-26\\processed'
    dirpath = 'G:\\My Drive\\Vaults\\WnM-AMO\\__Data\\2025-08-20\\raw_stark_maps'

    fl_data, fl_files = read_files(dirpath, key='spec', files_idx_ommit=ommit_files)
    ref_data, ref_files = read_files(dirpath, key='sync', files_idx_ommit=ommit_files)


    

#%%
    fps = 18.54 # Hz
    sweep = 0.05 # Hz
    fov = 10.8 # mm
    image_number = 0
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
                              bin_power=1)
# %%
    fig = fsm.plot_calibration(figsize=(6,6))

# %%
    fig = fsm.plot_stark_map(False)
    fig = fsm.plot_corrected_stark_map()
# %%
    fsm.plot_image_slice(8)
# %%
    fsm.fit_stark_map([2,3,4], verbose=True)

#%%
    #plot_labels = '$44D_{5/2}$: $|m_J|=5/2$,$44D_{5/2}$: $|m_J|=1/2$,$44D_{5/2}$: $|m_J|=3/2$,$44D_{3/2}$: $|m_J|=1/2$,$44D_{3/2}$: $|m_J|=1/2$'
    plot_labels = str([])
    fsm.plot_gmodel_results(plot_labels)
# %%