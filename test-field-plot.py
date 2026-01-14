import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 15

fields = np.load('example_total_fields_0.1nJ.npy')
print(fields.shape)
print(fields.dtype)


initial_fieild = fields[0, 0, 