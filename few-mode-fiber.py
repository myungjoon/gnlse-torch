import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile


def pulse_width_rms_diff(I: torch.Tensor, t: torch.Tensor, eps: float = 1e-12):
    """
    Differentiable RMS (2nd-moment) pulse width.
    I: (Nt,) intensity tensor (nonnegative preferred)
    t: (Nt,) time grid tensor (seconds)
    returns: width = 2*sigma_t
    """
    # optional: baseline removal (also differentiable)
    I = I - I.min()

    w = I + eps  # avoid divide-by-zero
    Z = torch.sum(w)
    t_mean = torch.sum(t * w) / Z
    var = torch.sum(((t - t_mean) ** 2) * w) / Z
    sigma = torch.sqrt(var + eps)
    return 2.0 * sigma

def spectral_shift_centroid_from_time(E_t: torch.Tensor,
                                      Omega: torch.Tensor,
                                      dim_t: int = -1,
                                      Omega_ref: float | torch.Tensor = 0.0,
                                      eps: float = 1e-12):
    """
    Differentiable spectral centroid shift.
    E_t: complex tensor (..., Nt)  (time-domain field)
    Omega: tensor (Nt,) angular frequency grid aligned with FFT bins [rad/s]
    Omega_ref: reference angular frequency (scalar or tensor broadcastable)
    returns: dOmega (...,)
    """
    # FFT along time axis
    E_w = torch.fft.fft(E_t, dim=dim_t)
    S = (E_w.real**2 + E_w.imag**2)  # |E_w|^2, differentiable

    # reshape Omega for broadcasting
    shape = [1] * E_t.ndim
    shape[dim_t] = -1
    Omega_b = Omega.view(*shape).to(S.device, S.dtype)

    num = torch.sum(Omega_b * S, dim=dim_t)
    den = torch.sum(S, dim=dim_t) + eps
    Omega_c = num / den

    if not torch.is_tensor(Omega_ref):
        Omega_ref = torch.tensor(Omega_ref, device=Omega_c.device, dtype=Omega_c.dtype)
    return Omega_c - Omega_ref


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
RAMAN = True
SELF_STEEPING = False

BATCH_NUM = 3
DS_X = 1
DS_Y = 1
DS_T = 1

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

precision = 'double'

num_save = 100
wvl0 = 1550e-9
L0 = 3.0

# Pulse
total_energy = 1 # nJ
Nt = 2**12
time_window = 30 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # ss
tfwhm = 0.250 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 16.0e-6 / 2
NA = 0.14
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
beta2 = -2.55e-26 * (1e12**2)
beta3 = 2.3e-40 * (1e12**3)


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

def get_hrw(ts, dt, t1=12.2e-3, t2=32e-3):
    """
    ts: (Nt,) time array in ps, starting at 0 (0..(Nt-1)dt)
    dt: time step in ps
    t1,t2: Raman parameters in ps (silica: 12.2 fs, 32 fs -> 12.2e-3 ps, 32e-3 ps)

    Returns:
      hrw = FFT(h_discrete), where h_discrete = h(ts)*dt and sum(h_discrete)=1
    """
    hr = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hr[ts < 0] = 0.0

    # discrete convolution kernel: multiply by dt then normalize so sum=1
    h_discrete = hr * dt
    h_discrete = h_discrete / (np.sum(h_discrete) + 1e-30)
    hrw = np.fft.fft(h_discrete)
    return hrw

# plt.plot(ts, hrw)
# plt.show()
ts = np.arange(Nt) * dt   # 0..(Nt-1)dt in ps (endpoint 문제 없음)
hrw = get_hrw(ts, dt, t1=12.2e-3, t2=32e-3)
hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)


# Boundary condition
boundary_type = 'periodic'

# custom mode input fields
modes = np.load('modes_FMF.npy')
modes = torch.tensor(modes, dtype=torch.complex64, device=device)
num_mode = 3

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, beta3=beta3, n2=n2, radius=core_radius, hrw=hrw)
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
coefficients = torch.tensor([[0.333, 0.333, 0.333], [1, 0, 0], [0.2, 0.4, 0.4]], dtype=torch.complex64)
coefficients = coefficients.to(device)
coefficients = coefficients[:,:, None, None]
fields = torch.sum(coefficients * modes, dim=1)

input_fields = Fields(domain, input_type='custom', fields=fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
sim = Simulation(domain, fiber, input_fields, boundary, config)
print(f'The simulation starts.', flush=True)
start_time = time.time()
sim.run()
print(f'Total calculation time : {time.time() - start_time}', flush=True)


input_fields = input_fields.fields.cpu().numpy()
input_fields1 = input_fields[0]
input_fields2 = input_fields[1]
input_fields3 = input_fields[2]
output_fields = sim.fields.fields.cpu().numpy()
output_fields1 = output_fields[0]
output_fields2 = output_fields[1]
output_fields3 = output_fields[2]
saved_temporal_fields = sim.saved_temporal_fields.cpu().numpy()
saved_spectrum = sim.saved_spectrum.cpu().numpy()


np.save(f'fields_{int(L0*100)}cm_{total_energy}nJ.npy', output_fields)
np.save(f'temporal_fields_{int(L0*100)}cm_{total_energy}nJ.npy', saved_temporal_fields)
np.save(f'spectrum_{int(L0*100)}cm_{total_energy}nJ.npy', saved_spectrum)

plt.figure()
plt.imshow(saved_temporal_fields[0].real, aspect='auto', cmap='turbo', origin='lower', extent=[-0.5 * time_window, 0.5 * time_window, 0, L0])
plt.xlim(-1, 5)
plt.xlabel('Time (ps)')
plt.ylabel('Distance (m)')
plt.savefig(f'temporal_fields_{int(L0*100)}cm_{total_energy}nJ.png', dpi=300)

plt.figure()
plt.imshow(saved_spectrum[0].real, aspect='auto', cmap='turbo', origin='lower')
plt.savefig(f'spectrum_{int(L0*100)}cm_{total_energy}nJ.png', dpi=300)

plt.figure()
plt.imshow(saved_temporal_fields[1].real, aspect='auto', cmap='turbo', origin='lower', extent=[-0.5 * time_window, 0.5 * time_window, 0, L0])
plt.xlabel('Time (ps)')
plt.ylabel('Distance (m)')
plt.xlim(-1, 5)
plt.savefig(f'temporal_fields_{int(L0*100)}cm_{total_energy}nJ_1.png', dpi=300)

plt.figure()
plt.imshow(saved_spectrum[1].real, aspect='auto', cmap='turbo', origin='lower')
plt.savefig(f'spectrum_{int(L0*100)}cm_{total_energy}nJ_1.png', dpi=300)

plt.figure()
plt.imshow(saved_temporal_fields[2].real, aspect='auto', cmap='turbo', origin='lower', extent=[-0.5 * time_window, 0.5 * time_window, 0, L0])
plt.xlabel('Time (ps)')
plt.ylabel('Distance (m)')
plt.xlim(-1, 5)
plt.savefig(f'temporal_fields_{int(L0*100)}cm_{total_energy}nJ_2.png', dpi=300)

plt.figure()
plt.imshow(saved_spectrum[2].real, aspect='auto', cmap='turbo', origin='lower')
plt.savefig(f'spectrum_{int(L0*100)}cm_{total_energy}nJ_2.png', dpi=300)

# plt.show()


# fobj = pulse_width_rms_diff(output_fields)
# print(f'Pulse width: {fobj}')

# Gradient caclulation using finite difference method
# d_coefficients = torch.zeros((BATCH_NUM, num_mode), dtype=torch.complex64)
# for i in range(num_mode):
#     pass

# input_fields1 = input_fields[0]
# input_fields2 = input_fields[1]
# input_fields3 = input_fields[2]
# output_fields1 = output_fields[0]
# output_fields2 = output_fields[1]
# output_fields3 = output_fields[2]
# plot_fields(input_fields1, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(input_fields2, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(input_fields3, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields1, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields2, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields3, domain, wvl0=wvl0, core_radius=core_radius)
# plot_fields(output_fields2, domain, wvl0=wvl0, core_radius=core_radius)
# plt.show()
