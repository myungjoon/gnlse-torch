import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 15

modes = np.load('mode_coeffs_5.0nJ.npy') 
modes = modes[:, ::10, :]
modes = np.abs(modes)**2
modes = np.sum(modes, axis=2)
print(modes.shape)
print(modes.dtype)

modes2 = np.load('mode_coeffs_5.0nJ_double.npy')
modes2 = modes2[:, ::10, :]
modes2 = np.abs(modes2)**2
modes2 = np.sum(modes2, axis=2)
print(modes2.shape)
print(modes2.dtype)
colors = plt.cm.turbo(np.linspace(0, 1, modes.shape[0]))
plt.figure(figsize=(10, 5))
for i in range(modes.shape[0]):
    plt.plot(modes[i], color=colors[i], label=f'Mode {i}')
    plt.plot(modes2[i], color=colors[i], linestyle='--',)
plt.legend()
plt.show()