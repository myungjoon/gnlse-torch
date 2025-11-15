import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = np.load('spatiotemporal_fields_10cm_10nJ_total.npy')
print(f'The shape of data is {data.shape}', flush=True)

data = data[:, :, 16:112, 16:112, 32:224]

print(f'The shape of data (after reduction) is {data.shape}', flush=True)
np.save('spatiotemporal_fields_10cm_10nJ.npy', data)
