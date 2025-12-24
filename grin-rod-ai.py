import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

DS_X = 2
DS_Y = 2
DS_T = 2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}', flush=True)
precision = 'single'

num_save = -1
wvl0 = 775e-9
L0 = 3.0 # 10 cm

# Pulse
total_energy = 5 # nJ
Nt = 2**10
time_window = 20 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.5 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 900e-6 / 2
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 3.2e-20 * 2
beta2 = 1.655e-26 * (1e12**2)
beta3 = 23.3e-42 * (1e12**3)

print(f'beta2: {beta2}, beta3: {beta3}', flush=True)

# Simulation domain parameters
Lx, Ly = 4 * core_radius, 4 * core_radius
unit = 1e-6
Nx, Ny = 2**9, 2**9
print(f'The grid size is {Nx}x{Ny}x{Nt}', flush=True)
dz = 1e-5
Nz = round(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

# custom mode input fields
modes = np.load('modes_2048.npy')
modes = modes[:,::4,::4]
modes = torch.tensor(modes, dtype=torch.complex64, device=device)
num_mode = 6

np.random.seed(45)
input_type = 'custom'
coefficients = torch.tensor([0.3, 0.3, 0.3, 0.2, 0.1, 0.1]) # data1
coefficients = coefficients.reshape((num_mode,1,1)) * np.exp(1j * np.random.uniform(0, 1.0 * np.pi, (num_mode, 1, 1)))
coefficients = coefficients.to(device)

fields = torch.sum(coefficients * modes[:num_mode], dim=0)
fields = fields.to(dtype=torch.complex64)

fields = fields.unsqueeze(0)

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, beta3=beta3, n2=n2, radius=core_radius,)
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T)
input = Fields(domain, input_type='custom', fields=fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
sim = Simulation(domain, fiber, input, boundary, config)

sim.run()

output_fields = sim.output_fields
np.save(f'output_fields_{Nx}_{Ny}_{Nt}_{total_energy}.npy', output_fields.cpu().numpy().squeeze())
print('End of simulation')
