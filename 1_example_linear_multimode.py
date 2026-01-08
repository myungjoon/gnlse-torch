import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, Fiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile, plot_intensity


DISPERSION = True
KERR = False
RAMAN = False
SELF_STEEPING = False

BATCH_NUM = 1
DS_X = 1
DS_Y = 1
DS_T = 1

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

precision = 'single'

if precision == 'double':
    real_type = torch.float64
    complex_type = torch.complex128
elif precision == 'single':
    real_type = torch.float32
    complex_type = torch.complex64
else:
    raise ValueError(f'Invalid precision: {precision}')

def get_n_core(wvl, B, C):
     
    w2 = (wvl*1e6)**2
    terms = (B * w2[..., None]) / (w2[..., None] - C**2)
    
    n_core = np.sqrt(1 + np.sum(terms, axis=-1))
    return n_core

if __name__ == '__main__':

    num_save = 50
    wvl0 = 1030e-9
    L0 = 0.005

    # Pulse
    total_energy = 10.0 # nJ
    Nt = 2**11
    time_window = 10 # ps
    dt = time_window / Nt
    tfwhm = 1.0 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 5.0e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # Fiber parameters
    core_radius = 16.0e-6 / 2
    NA = 0.14
    n_core0 = 1.45
    n_clad0 = np.sqrt(n_core0**2 - NA**2)

    # Simulation domain parameters
    Lx, Ly = 4 * core_radius, 4 * core_radius
    unit = 1e-6
    Nx, Ny = 128, 128
    print(f'The grid size is {Nx}x{Ny}')
    
    Nf = 20
    c = 299.792458
    freq_range = 100  
    freq_min = c/wvl0*1e-6 + freq_range/2
    freq_max = c/wvl0*1e-6 - freq_range/2
    
    f = np.linspace(freq_min,freq_max, Nf)
    wvl = c/f *1e-6

    # Sellmeier equation
    B = np.array([0.6962, 0.4079, 0.8975])
    C = np.array([0.0684, 0.1162, 9.8961])

    n_core = get_n_core(wvl, B, C)
    n_clad = np.sqrt(n_core**2 - NA**2)
    
    w0 = 2 * np.pi * (c / wvl0 * 1e-6) # 중심 각주파수 (rad/ps)
    omega = 2 * np.pi * f # 주파수 배열 (rad/ps)

    # Beta(omega) calculation (unit: 1/mm)
    beta_raw = (omega / c) * n_core 

    # Polynomial fitting (no centering and scaling)
    poly_order = 7 
    coeffs = np.polyfit(omega, beta_raw, poly_order)
    p = np.poly1d(coeffs) # 다항식 객체 생성

    # Extraction of coefficients by differentiation (beta_n)
    betas = []
    current_poly = p
    for i in range(4): # beta0, beta1, beta2, beta3
        # Calculate the value at the center frequency w0
        val = current_poly(w0)
        betas.append(val)
        
        # Calculate the next order for differentiation
        current_poly = np.polyder(current_poly)

    beta2 = betas[2]*1e6
    beta3 = betas[3]*1e6
    print(f"beta2: {beta2}, beta3: {beta3}", flush=True)

    # Boundary condition
    boundary_type = 'periodic'

    # custom mode input fields
    num_modes = 6
    mode_fields = np.load('modes_128x128.npy')
    mode_fields = torch.tensor(mode_fields, dtype=complex_type, device=device)

    domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
    fiber = Fiber(domain, n_core0, n_clad0, beta2=beta2, beta3=beta3, radius=core_radius,)
    boundary = Boundary(domain, boundary_type=boundary_type)
    config = SimConfig(wvl0=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING,
                            batch_num=BATCH_NUM, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T)

    print(f'batch size: {BATCH_NUM}')
    mode_fields = mode_fields.unsqueeze(0)

    coeffs = torch.ones(num_modes, dtype=complex_type)
    coeffs = coeffs.to(device)
    coeffs = np.reshape(coeffs, (-1, num_modes, 1, 1))
    input_fields = torch.sum(coeffs * mode_fields, dim=1)
    initial_fields = Fields(domain, input_type='custom', fields=input_fields, tfwhm=tfwhm, total_energy=total_energy, t_center=0,)

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    input_fields = initial_fields.fields[0].cpu().numpy()
    plot_intensity(input_fields, radius=core_radius, extent=extent, title='Input Field')
    plt.show()
    sim = Simulation(domain, fiber, initial_fields, boundary, config, mode_fields=mode_fields)
    sim.run()
    
    saved_temporal_fields = sim.saved_temporal_fields.cpu().numpy()
    saved_spectrum = sim.saved_spectrum.cpu().numpy()
    output_fields = sim.fields.fields[0].detach().cpu().numpy()
            
    plot_intensity(output_fields, radius=core_radius, extent=extent, title='Output Field')

    # plot_mode_energy_evolution(saved_fields, mode_fields, dz=L0/num_save*1e3)

    plt.figure()
    plt.imshow(saved_temporal_fields[0].real, aspect='auto', cmap='turbo', origin='lower', extent=[-0.5 * time_window, 0.5 * time_window, 0, L0])
    plt.xlim(-1, 5)
    plt.xlabel('Time (ps)')
    plt.ylabel('Distance (m)')
    plt.savefig(f'temporal_fields_{int(L0*100)}cm_{total_energy}nJ.png', dpi=300)

    plt.figure()
    plt.imshow(saved_spectrum[0].real, aspect='auto', cmap='turbo', origin='lower')
    plt.savefig(f'spectrum_{int(L0*100)}cm_{total_energy}nJ.png', dpi=300)

    plt.show()