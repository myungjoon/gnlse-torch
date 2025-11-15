import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_1 = np.load('spatiotemporal_fields_1cm_40nJ_total.npy')
print(f'The shape of data_1 is {data_1.shape}', flush=True)

total_spatiotemporal_fields = data_1[-10:]

np.save('spatiotemporal_fields_1cm_40nJ_test.npy', total_spatiotemporal_fields)
