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

num_save = -1

wvl0 = 1030e-9
total_energy = 0.38 # nJ
L0 = 0.1

# Beam customization
# First 10 modes


# Pulse
Nt = 2**11
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

# Simulation domain parameters
Lx, Ly = 4 * core_radius, 4 * core_radius
unit = 1e-6
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
dz = 5e-5
Nz = int(L0 / dz)

# Boundary condition
boundary_type = 'periodic'


# beam radius
beam_radius = 25e-6

# phase map of 32x32 random from 0 to 2*pi
sx, sy = 32, 32
phase_map = np.random.uniform(0, 2*np.pi, (sx, sy))
phase_map = torch.tensor(phase_map, dtype=torch.float32, device=device)
px, py = (Nx * (beam_radius * 2) / Lx) // sx, (Ny * (beam_radius * 2) / Ly) // sy

Px, Py = int(sx*px), int(sy*py)


# Create total_phase_map by expanding each pixel into (px, py) block
superpixel_phase_map = torch.zeros(sx * int(px), sy * int(py), dtype=torch.float32, device=device)
for i in range(sx):
    for j in range(sy):
        start_x = i * int(px)
        end_x = (i + 1) * int(px)
        start_y = j * int(py)
        end_y = (j + 1) * int(py)
        superpixel_phase_map[start_x:end_x, start_y:end_y] = phase_map[i, j]

total_phase_map = torch.zeros((Nx, Ny), dtype=torch.float32, device=device)
# superpixel in the center of the total_phase_map
total_phase_map[Nx//2-Px//2:Nx//2+Px//2, Ny//2-Py//2:Ny//2+Py//2] = superpixel_phase_map
total_phase_map_np = total_phase_map.cpu().numpy()
# plt.figure()
# plt.imshow(total_phase_map_np, cmap='turbo')
# plt.colorbar()

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, n2=n2, radius=core_radius,)
input = Fields(domain, input_type='gaussian', beam_radius=beam_radius, tfwhm=tfwhm, total_energy=total_energy, phase_map=total_phase_map,) # spatially gaussian and gaussian pulse
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=num_save)
sim = Simulation(domain, fiber, input, boundary, config)

plot_fields(total_phase_map_np, domain, wvl0=wvl0, core_radius=core_radius)


# plot fiber index profile
# fiber_indices = fiber.n.cpu().numpy()
# plot_index_profile(fiber_indices)

input_fields = input.fields.cpu().numpy()
input_intensity = np.abs(input_fields)**2

plot_fields(input_fields[:,:,1024], domain, wvl0=wvl0, core_radius=core_radius)


# plt.figure()
# plt.imshow(input_intensity[:, :, 1024], cmap='turbo', vmax=5e7)
# plt.colorbar()
# plt.show()


# Check the total energy of the input field
dx = Lx / Nx  # [m]
dy = Ly / Ny  # [m]
dt = time_window / Nt  # [s], since time_window is in ps
input_total_energy = np.sum(input_intensity) * dx * dy * dt  # [J]
print(f'Total energy: {input_total_energy}')


# Simulation
print(f'The simulation starts.')
start_time = time.time()
sim.run()
print(f'Computation time for forward calculation : {time.time() - start_time}')

output_fields = sim.fields.fields
output_fields = output_fields.cpu().numpy()
np.save(f'output_fields_{total_energy}nJ.npy', output_fields)
output_intensity = np.abs(output_fields)**2


fields_zt = sim.fields_zt.cpu().numpy()
# fields_zt_temporal = np.abs(fields_zt)**2
# fields_zt_temporal = fields_zt_temporal.sum(axis=1)
plt.figure()
for i in range(fields_zt.shape[0]):
    plt.plot(t, fields_zt[i], label=f'z = {L0 / fields_zt.shape[0] * i:.2f} m')
plt.legend()
plt.xlabel('Time (ps)')
plt.ylabel('Intensity (a.u.)')

plot_fields(input_fields, domain, wvl0=wvl0, core_radius=core_radius)
plot_fields(output_fields, domain, wvl0=wvl0, core_radius=core_radius)
plt.show()

print(f'The simulation ends.')