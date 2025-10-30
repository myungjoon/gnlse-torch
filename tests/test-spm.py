import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 100
np.random.seed(seed)

device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'double'

num_save = 10

wvl0 = 1030e-9
total_energy = 5e-9 # nJ
L0 = 1.0

# Beam
beam_radius = 10e-6

# Pulse
Nt = 2**10
time_window = 10e-12 # s
dt = time_window / Nt
tfwhm = 0.5e-12 # s
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 25e-6
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 3.2e-20
beta2 = 1.655e-26 

# Simulation domain parameters
Lx, Ly = 3 * core_radius, 3 * core_radius
unit = 1e-6
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
dz = 1e-4
Nz = int(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, n2=n2, radius=core_radius,)
input = Fields(domain, beam_radius, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=False, kerr=True, raman=False, self_steeping=False, num_save=num_save)
sim = Simulation(domain, fiber, input, boundary, config)

# plot fiber index profile
# fiber_indices = fiber.n.cpu().numpy()
# plot_index_profile(fiber_indices)

input_fields = input.fields.cpu().numpy()
plot_fields(input_fields, domain, wvl0=wvl0)
plt.show()

# Simulation
print(f'The simulation starts.')
start_time = time.time()
sim.run()
print(f'Computation time for forward calculation : {time.time() - start_time}')

output_fields = sim.fields.fields
output_fields = output_fields.cpu().numpy()
output_intensity = np.abs(output_fields)**2
# plot spatial profile, temporal profile, and spectrum

plot_fields(output_fields, domain, wvl0=wvl0)
plt.show()