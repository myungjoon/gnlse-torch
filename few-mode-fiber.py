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

BATCH_NUM = 2
DS_X = 1
DS_Y = 1
DS_T = 1

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

precision = 'double'

num_save = 100
wvl0 = 1550e-9
L0 = 1.0

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
dz = 5e-5
Nz = round(L0 / dz)

ts = np.linspace(0, time_window, Nt)
t1 = 12.2e-3
t2 = 32e-3

def get_hrw(ts, t1=12.2e-3, t2=32e-3):
    hr = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hrw = np.fft.ifft(hr) * Nt
    return hrw

hrw = get_hrw(ts)

# plt.plot(ts, hrw)
# plt.show()

hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)


# Boundary condition
boundary_type = 'periodic'




# custom mode input fields
modes = np.load('modes_FMF.npy')
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

print(f'batch size: {BATCH_NUM}')
modes = modes.unsqueeze(0)

# coefficients = torch.randn((BATCH_NUM, num_mode), dtype=torch.complex64)
coefficients = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.complex64)
coefficients = coefficients.to(device)
coefficients = coefficients[:,:, None, None]
fields = torch.sum(coefficients * modes, dim=1)

input_fields = Fields(domain, input_type='custom', fields=fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
sim = Simulation(domain, fiber, input_fields, boundary, config)
print(f'The simulation starts.', flush=True)
start_time = time.time()
sim.run()
print(f'Total calculation time : {time.time() - start_time}', flush=True)
output_fields = sim.fields.fields.cpu().numpy()
input_fields = input_fields.fields.cpu().numpy()
input_fields1 = input_fields[0]
input_fields2 = input_fields[1]
input_fields3 = input_fields[2]
output_fields1 = output_fields[0]
output_fields2 = output_fields[1]
output_fields3 = output_fields[2]
plot_fields(input_fields1, domain, wvl0=wvl0, core_radius=core_radius)
plot_fields(input_fields2, domain, wvl0=wvl0, core_radius=core_radius)
plot_fields(input_fields3, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields1, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields2, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields3, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields2, domain, wvl0=wvl0, core_radius=core_radius)
plt.show()
np.save(f'fields_{int(L0*100)}cm_{total_energy}nJ.npy', output_fields)