import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile

# Parameters
# Pulse energy : 38 nJ
# Pulse duration : 60 fs
# Wavelength : 1064 nm
# Propagation distance : 5 m
# beta2 : 
# Diameter : 62.5 um
# NA : 0.275
# n2 : 2.3 * 1e-20

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False


os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'single'

num_save = 100

wvl0 = 1030e-9
L0 = 0.01 # 1 cm


total_energy = 38 # nJ
# total energy from 0.1 to 50 nJ, randmo
# total_energy = np.random.uniform(0.1, 50) # nJ


# Beam customization
# First 10 modes

# Pulse
Nt = 2**10
time_window = 3 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.06 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 62.5e-6 / 2
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
beta2 = 1.655e-26 * (1e12**2)
beta3 = 23.3e-42

# Simulation domain parameters
Lx, Ly = 3 * core_radius, 3 * core_radius
unit = 1e-6
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
dz = 1e-5
Nz = round(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

# custom mode input fields
modes = np.load('modes_256.npy')
modes = torch.tensor(modes, dtype=torch.complex64, device=device)
num_mode = 10

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, beta3=beta3, n2=n2, radius=core_radius,)
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=num_save)

# Preallocate arrays to store results from all simulations
all_spatiotemporal_fields = []
all_spatial_intensities = []
all_spatial_intensities_sequential = []

for n in range(5):
    coefficients = torch.randn(10, dtype=torch.complex64)
    coefficients = coefficients / torch.linalg.norm(coefficients)
    coefficients = coefficients.reshape((10, 1, 1)).to(device)
    mode_fields = torch.sum(coefficients * modes[:num_mode], dim=0)

    input = Fields(domain, input_type='custom', fields=mode_fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
    sim = Simulation(domain, fiber, input, boundary, config)

    print(f'The simulation {n} starts.')
    sim.run()

    spatiotemporal_fields = sim.spatiotemporal_fields.cpu().numpy()
    spatial_intensities = sim.spatial_intensities.cpu().numpy()
    spatial_intensities_sequential = sim.spatial_intensities_sequential.cpu().numpy()

    all_spatiotemporal_fields.append(spatiotemporal_fields)
    all_spatial_intensities.append(spatial_intensities)
    all_spatial_intensities_sequential.append(spatial_intensities_sequential)

# After all iterations, stack and save a single file
all_spatiotemporal_fields = np.stack(all_spatiotemporal_fields, axis=0)  # shape: (10, ...)
all_spatial_intensities = np.stack(all_spatial_intensities, axis=0)
all_spatial_intensities_sequential = np.stack(all_spatial_intensities_sequential, axis=0)

np.savez('data_all.npz', 
         spatiotemporal_fields=all_spatiotemporal_fields, 
         spatial_intensities=all_spatial_intensities, 
         spatial_intensities_sequential=all_spatial_intensities_sequential)

