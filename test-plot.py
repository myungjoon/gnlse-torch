import numpy as np
import matplotlib.pyplot as plt
from gnlse import plot_intensity, plot_pulse
plt.rcParams['font.size'] = 15

core_radius = 8e-6
extent = (-16e-6, 16e-6, -16e-6, 16e-6)


linear_fields = np.load('example_total_fields_256_10.0nJ_mode5_double.npy') 
print(linear_fields.shape)
print(linear_fields.dtype)
linear_field = linear_fields[0]
initial_field = linear_field[0]
final_field = linear_field[-1]
np.save('final_field.npy', final_field)
# plot_intensity(initial_field, radius=core_radius, extent=extent, title='Input Field')
# plot_intensity(final_field, radius=core_radius, extent=extent, title='Final Field')
# plot_pulse(initial_field, title='Input Pulse')
# plot_pulse(final_field, title='Final Pulse')

# nonlinear_fields = np.load('example_total_fields_5.0nJ.npy') 
# nonlinear_field = nonlinear_fields[0]
# initial_field = nonlinear_field[0]
# final_field = nonlinear_field[-1]

# plot_intensity(initial_field, radius=core_radius, extent=extent, title='Input Field')
# plot_intensity(final_field, radius=core_radius, extent=extent, title='Final Field')
# plot_pulse(initial_field, title='Input Pulse')
# plot_pulse(final_field, title='Final Pulse')

for i in range(linear_field.shape[0]):
    plot_intensity(linear_field[i], radius=core_radius, extent=extent, title=f'Field_{i}')
    plt.colorbar()

# for i in range(1, linear_field.shape[0]):
#     # diffrence between two adjacent fields
#     diff = np.abs(linear_field[i])**2 - np.abs(linear_field[i-1])**2
#     # print norm of diff
#     print(np.linalg.norm(diff))
plt.show()