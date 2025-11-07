import numpy as np


data_1 = np.load('data_all_100.npz')
data_2 = np.load('data_all_200.npz')

total_spatiotemporal_fields = np.concatenate([data_1['spatiotemporal_fields'], data_2['spatiotemporal_fields']], axis=0)
total_spatial_intensities = np.concatenate([data_1['spatial_intensities'], data_2['spatial_intensities']], axis=0)
total_spatial_intensities_sequential = np.concatenate([data_1['spatial_intensities_sequential'], data_2['spatial_intensities_sequential']], axis=0)


# save three different npy files
np.save('spatiotemporal_fields.npy', total_spatiotemporal_fields)
np.save('spatial_intensities.npy', total_spatial_intensities)
np.save('spatial_intensities_sequential.npy', total_spatial_intensities_sequential)