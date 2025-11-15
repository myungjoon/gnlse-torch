import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_1 = np.load('spatiotemporal_fields_1cm_40nJ.npy')
print(f'The shape of data_1 is {data_1.shape}', flush=True)
data_2 = np.load('spatiotemporal_fields_1cm_40nJ_2.npy')
print(f'The shape of data_2 is {data_2.shape}', flush=True)

total_spatiotemporal_fields = np.concatenate([data_1, data_2], axis=0)

data_1 = np.load('spatiotemporal_fields_1cm_40nJ_3.npy')
print(f'The shape of data_3 is {data_1.shape}', flush=True)
total_spatiotemporal_fields = np.concatenate([total_spatiotemporal_fields, data_1,], axis=0)
data_1 = np.load('spatiotemporal_fields_1cm_40nJ_4.npy')
print(f'The shape of data_4 is {data_1.shape}', flush=True)

total_spatiotemporal_fields = np.concatenate([total_spatiotemporal_fields, data_1,], axis=0)
print(f'The total number of spatiotemporal fields is {total_spatiotemporal_fields.shape[0]}')
np.save('spatiotemporal_fields_1cm_40nJ_total.npy', total_spatiotemporal_fields)
print('successful', flush=True)
