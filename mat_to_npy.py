import numpy as np
import scipy.io as sio



num_modes = 6
filepath = './data/GRIN_1030/'
fields = np.zeros((num_modes, 800, 800))

for i in range(num_modes):
    name = f'radius20boundary0000fieldscalarmode{i+1}wavelength1030.mat'
    filename = filepath + name
    data = sio.loadmat(filename)
    output = data['phi']
    fields[i, :, :] = output

np.save('modes.npy', fields)


filepath2 = './betas.mat'

data = sio.loadmat(filepath2)
betas = data['betas'].squeeze()
np.save('betas.npy', betas)

print(f'betas shape: {betas.shape}')