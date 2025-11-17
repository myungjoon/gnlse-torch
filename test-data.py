

import numpy as np


import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    data = np.load('spatiotemporal_fields_10cm_10nJ_2.npy',)
    data2 = np.load('spatiotemporal_fields_10cm_10nJ_2_t10.npy',)

    ind = 1
    input_data = data[ind, 0, :, :, 128]
    output_data = data[ind, 1, :, :, 128]

    print(f'input_data.shape: {input_data.shape}, output_data.shape: {output_data.shape}', flush=True)
    # data type
    print(f'input_data.dtype: {input_data.dtype}, output_data.dtype: {output_data.dtype}', flush=True)

    input_intensity = np.abs(input_data)**2
    output_intensity = np.abs(output_data)**2


    # check the integral (pulse energy)
    for ind in range(data.shape[0]):
        input_energy = np.sum(np.abs(data[ind, 0])**2)
        output_energy = np.sum(np.abs(data[ind, 1])**2)
        print(f'input_energy: {input_energy}, output_energy: {output_energy}', flush=True)
    #subplots for input and output intensity
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(input_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(output_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()

    ind = 1
    input_data = data2[ind, 0, :, :, 128]
    output_data = data2[ind, 1, :, :, 128]

    input_intensity = np.abs(input_data)**2
    output_intensity = np.abs(output_data)**2

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(input_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(output_intensity, cmap='turbo', vmin=0,)
    plt.colorbar()



    plt.show()