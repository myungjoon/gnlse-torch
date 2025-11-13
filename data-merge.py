import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_1 = np.load('spatiotemporal_fields_10cm_10nJ.npy')
print(f'The shape of data_1 is {data_1.shape}', flush=True)
data_2 = np.load('spatiotemporal_fields_10cm_10nJ_2.npy')
print(f'The shape of data_2 is {data_2.shape}', flush=True)

total_spatiotemporal_fields = np.concatenate([data_1, data_2], axis=0)
print(f'The total number of spatiotemporal fields is {total_spatiotemporal_fields.shape[0]}')

np.save('spatiotemporal_fields_10cm_10nJ_total.npy', total_spatiotemporal_fields)
