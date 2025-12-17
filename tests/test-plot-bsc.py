import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
output_fields = np.load('../fields_0cm_5nJ.npy')
output_fields = output_fields[0]
fields_temporal = output_fields.sum(axis=(0,1))
intensities_temporal = np.abs(fields_temporal)**2

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

#normalize output_spectrum and output_spectrum2 to the same total energy
total_energy = np.sum(intensities_temporal)
output_spectrum = output_spectrum / total_energy

plt.figure()
plt.plot(freq_abs/1e12, output_spectrum)
plt.show()

