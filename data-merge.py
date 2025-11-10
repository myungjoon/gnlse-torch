import numpy as np
import os


# current directory change to the directory of the file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


data_1 = np.load('spatiotemporal_fields_2.npy')
print(f'The shape of data_1 is {data_1.shape}')
data_2 = np.load('spatiotemporal_fields_3.npy')
print(f'The shape of data_2 is {data_2.shape}')
data_3 = np.load('spatiotemporal_fields_4.npy')
print(f'The shape of data_3 is {data_3.shape}')

total_spatiotemporal_fields = np.concatenate([data_1, data_2, data_3], axis=0)
print(f'The total number of spatiotemporal fields is {total_spatiotemporal_fields.shape[0]}')

np.save('spatiotemporal_fields_1cm.npy', total_spatiotemporal_fields)