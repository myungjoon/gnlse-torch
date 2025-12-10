import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile

# seed for random number generation
np.random.seed(42)
torch.manual_seed(42)

# Parameters
# Pulse energy : 50 nJ
# Pulse duration : 100 fs
# Wavelength : 1030 nm
# Propagation distance : 10 cm
# beta2 : 1.655e-26 s^2/m
# beta3 : 23.3e-42 s^3/m
# Fiber diameter : 62.5 um
# Fiber NA : 0.25
# n2 : 2.3e-20 m^2/W

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'double'

num_save = 100
wvl0 = 1550e-9
L0 = 3.0

# Pulse
total_energy = 5 # nJ
Nt = 2**12
time_window = 30 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.250 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 16.0e-6 / 2
NA = 0.14
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
beta2 = 1.655e-26 * (1e12**2)
beta3 = 23.3e-42 * (1e12**3)

print(f'beta2: {beta2}, beta3: {beta3}', flush=True)

# Simulation domain parameters
Lx, Ly = 4 * core_radius, 4 * core_radius
unit = 1e-6
Nx, Ny = 64, 64
print(f'The grid size is {Nx}x{Ny}')
dz = 1e-5
Nz = round(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

# custom mode input fields
modes = np.load('modes.npy')
modes = torch.tensor(modes, dtype=torch.complex64, device=device)
num_mode = 3

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, beta3=beta3, n2=n2, radius=core_radius,)
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING,
                 batch_num=BATCH_NUM, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T)

# Preallocate arrays to store results from all simulations
all_spatiotemporal_fields = []
all_spatial_intensities = []
all_spatial_intensities_sequential = []

num_data = 31

num_iters = (num_data + BATCH_NUM - 1) // BATCH_NUM  # Ceiling division to handle all data points
print(f'batch size: {BATCH_NUM}')
print(f'number of iterations: {num_iters}')
modes = modes.unsqueeze(0)

# coefficients = torch.randn((num_data, num_mode), dtype=torch.complex64)
coefficients = torch.ones((num_data, num_mode), dtype=torch.complex64)
# coefficients[0,0] = 1
# coefficients[1] = coefficients[1] * torch.tensor(np.exp(1j * np.pi / 4), dtype=torch.complex64)
coefficients = coefficients.to(device)

all_spatiotemporal_fields = np.zeros((num_data, 2, Nx//DS_X//2, Ny//DS_Y//2, Nt//DS_T//2), dtype=np.complex64)
start_time = time.time()


for n in range(num_iters):
    # coefficients = torch.randn((BATCH_NUM, num_mode), dtype=torch.complex64)
    start_idx = n * BATCH_NUM
    end_idx = min((n + 1) * BATCH_NUM, num_data)
    coeffs = coefficients[start_idx:end_idx] 
    # Ensure modes are properly expanded for the batch size
    batch_size = coeffs.shape[0]
    modes_batch = modes.expand(batch_size, -1, -1, -1)  # Expand to (batch_size, num_mode, Nx, Ny)
    input_fields = torch.sum(coeffs[:,:,None, None] * modes_batch[:, :num_mode], dim=1)

    # Create a config with the correct batch size for this iteration
    config_iter = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING,
                           batch_num=batch_size, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T)
    input = Fields(domain, input_type='custom', fields=input_fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
    sim = Simulation(domain, fiber, input, boundary, config_iter)

    print(f'The simulation {n} starts.', flush=True)
    sim.run()

    all_spatiotemporal_fields[start_idx:end_idx] = sim.spatiotemporal_fields.cpu().numpy()
    # spatiotemporal_fields = sim.spatiotemporal_fields.cpu().numpy()
    # spatiotemporal_fields = spatiotemporal_fields[:, 16:112, 16:112, 32:224]
    # spatial_intensities = sim.spatial_intensities.cpu().numpy()
    # spatial_intensities_sequential = sim.spatial_intensities_sequential.cpu().numpy()
    
print(f'Total calculation time : {time.time() - start_time}', flush=True)

np.save(f'spatiotemporal_fields_{int(L0*100)}cm_{total_energy}nJ_{num_data}_{BATCH_NUM}.npy', all_spatiotemporal_fields)

