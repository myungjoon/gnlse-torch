import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data_all.npz')
print(data['spatiotemporal_fields'].shape)
print(data['spatial_intensities'].shape)
print(data['spatial_intensities_sequential'].shape)

spatial_intensities_sequential = data['spatial_intensities_sequential']
spatiotemporal_fields = data['spatiotemporal_fields']
spatial_intensities = data['spatial_intensities']

plt.subplots(1, 2, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(spatial_intensities_sequential[0, 0, :, :], cmap='turbo')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(spatial_intensities_sequential[0, -1, :, :], cmap='turbo')
plt.colorbar()
plt.show()