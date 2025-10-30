import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile

# Units of the simulation
# Length : m
# Time : ps
# Wavelength : m
# Energy : nJ
# Power : W
# Intensity : W/m^2
# Field : V/m

# Parameters
# Pulse energy : 38 nJ
# Pulse duration : 60 fs
# Wavelength : 1030 nm
# Propagation distance : 0.2 m
# beta2 : 
# Radius : 62.5 um
# NA : 0.25
# n2 : 2.3 * 1e-20


DISPERSION = True
KERR = False
RAMAN = False
SELF_STEEPING = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'single'


def calculate_self_imaging_distance(core_radius, n_core, n_clad):
    delta = (n_core - n_clad) / n_core
    self_imaging_distance = np.pi * core_radius / np.sqrt(2 * delta)
    return self_imaging_distance

num_save = -1 # no save for field
wvl0 = 1030e-9
total_energy = 1 # energy is not important for self-imaging
L0 = 0.01

# Pulse
Nt = 2**11
time_window = 5 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.1 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

beam_radius = 25e-6

# Fiber parameters
core_radius = 62.5e-6
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
beta2 = 1.655e-26 * (1e12**2)

# Simulation domain parameters
Lx, Ly = 3 * core_radius, 3 * core_radius
unit = 1e-6
Nx, Ny = 512, 512
print(f'The grid size is {Nx}x{Ny}')
dz = 2e-5
Nz = int(L0 / dz)

# Boundary condition
boundary_type = 'periodic'


# Before the simulation, we calculate self-imaging distance
self_imaging_distance = calculate_self_imaging_distance(core_radius, n_core, n_clad)
print(f'The self-imaging distance is {self_imaging_distance} m')


domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, n2=n2, radius=core_radius,)
input = Fields(domain, beam_radius, tfwhm=tfwhm, total_energy=total_energy, t_center=0,)
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=num_save)
sim = Simulation(domain, fiber, input, boundary, config)

# plot fiber index profile
# fiber_indices = fiber.n.cpu().numpy()
# plot_index_profile(fiber_indices)

input_fields = input.fields.cpu().numpy()
input_intensity = np.abs(input_fields)**2

# Check the total energy of the input field
# dx = Lx / Nx  # [m]
# dy = Ly / Ny  # [m]
# dt = time_window / Nt  # [s], since time_window is in ps
# input_total_energy = np.sum(input_intensity) * dx * dy * dt  # [J]
# print(f'Total energy: {input_total_energy}')



# Simulation
print(f'The simulation starts.')
start_time = time.time()
sim.run()
print(f'Computation time for forward calculation : {time.time() - start_time}')


fields_xz = sim.fields_xz.cpu().numpy()
fields_xz_intensity = np.abs(fields_xz)**2
plt.figure()
# aspect ratio is 1:1
plt.imshow(fields_xz_intensity, cmap='turbo', aspect='equal')


fields_zt = sim.fields_zt.cpu().numpy()
# fields_zt_temporal = np.abs(fields_zt)**2
# fields_zt_temporal = fields_zt_temporal.sum(axis=1)
plt.figure()
for i in range(fields_zt.shape[0]):
    plt.plot(t, fields_zt[i], label=f'z = {L0 / fields_zt.shape[0] * i:.2f} m')
plt.legend()
plt.xlabel('Time (ps)')
plt.ylabel('Intensity (a.u.')
plt.show()

output_fields = sim.fields.fields
output_fields = output_fields.cpu().numpy()
output_intensity = np.abs(output_fields)**2


plot_fields(input_fields, domain, wvl0=wvl0, core_radius=core_radius)
plot_fields(output_fields, domain, wvl0=wvl0, core_radius=core_radius)
plt.show()

print(f'The simulation ends.')