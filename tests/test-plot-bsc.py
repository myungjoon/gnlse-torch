import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
output_fields = np.load('output_fields_38000nJ.npy')
output_fields2 = np.load('output_fields_38nJ.npy')
fields_temporal = output_fields.sum(axis=(0,1))
intensities_temporal = np.abs(fields_temporal)**2
fields_temporal2 = output_fields2.sum(axis=(0,1))
intensities_temporal2 = np.abs(fields_temporal2)**2

C0 = 3e8
wvl0 = 1030e-9
Nt = 2**10
time_window = 5 # ps
dt = time_window / Nt
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)
freq = np.fft.fftfreq(Nt, dt)
f0 = C0 / wvl0
freq_abs = f0 + freq

output_spectrum = np.fft.fft(np.abs(np.fft.fft(np.fft.ifftshift(fields_temporal, axes=0)))**2)
output_spectrum2 = np.fft.fft(np.abs(np.fft.fft(np.fft.ifftshift(fields_temporal2, axes=0)))**2)

#normalize output_spectrum and output_spectrum2 to the same total energy
total_energy = np.sum(intensities_temporal)
total_energy2 = np.sum(intensities_temporal2)
output_spectrum = output_spectrum / total_energy
output_spectrum2 = output_spectrum2 / total_energy2

plt.figure()
plt.plot(freq_abs/1e12, output_spectrum)
plt.plot(freq_abs/1e12, output_spectrum2)
plt.show()

