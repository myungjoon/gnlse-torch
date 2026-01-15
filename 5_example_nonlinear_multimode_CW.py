import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time

from gnlse import Domain, Fiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile, plot_intensity

def calculate_total_energy(fields, dx=1e-6, dy=1e-6, dt=1e-12):
    return np.sum(np.abs(fields)**2) * dx * dy * dt

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

BATCH_NUM = 1
DS_X = 1
DS_Y = 1
DS_T = 1

MODE_DECOMP_STEP = 100

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

precision = 'double'

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

    num_save = 20
    wvl0 = 1030e-9
    L0 = 0.1

    # Pulse
    total_energy = 10.0 # nJ
    peak_power = 50000.0 # W
    Nt = 1
    time_window = 4 # ps
    dt = time_window / Nt
    tfwhm = 0.1 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 1.0e-6
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # Fiber parameters
    core_radius = 16.0e-6 / 2
    NA = 0.14
    n_core0 = 1.45
    n_clad0 = np.sqrt(n_core0**2 - NA**2)
    n2 = 2.3e-20
    n = np.load('n.npy')
    n = torch.tensor(n, dtype=complex_type, device=device)
    # Simulation domain parameters
    Lx, Ly = 8 * core_radius, 8 * core_radius
    unit = 1e-6
    Nx, Ny = 256, 256
    print(f'The grid size is {Nx}x{Ny}')
    dx, dy = Lx / Nx, Ly / Ny


    Nf = 20
    c = 299.792458
    freq_range = 100
    freq_min = c/wvl0*1e-6 + freq_range/2
    freq_max = c/wvl0*1e-6 - freq_range/2
    
    f = np.linspace(freq_min, freq_max, Nf)
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

    A_eff = 5.0e-11
    P0 = 0.94 * total_energy * 1e-9 / (tfwhm * 1e-12)

    w0_ = 2 * np.pi * (3e8 / wvl0)
    gamma = n2 * w0_ / (3e8 * A_eff)
    

    L_d = (tfwhm/1.665)**2 / abs(beta2)
    L_nl = 1 / (gamma * P0) 
    print(f'P0 = {P0} W')
    print(f'A_eff = {A_eff} m')
    
    
    print(f'L_d = {L_d}')
    print(f'L_nl = {L_nl}')

    print(f'L_d/L_nl = {L_d/L_nl}')

    # Boundary condition
    boundary_type = 'absorbing'

    # custom mode input fields
    num_modes = 6
    mode_fields = np.load(f'modes_{Nx}x{Ny}.npy')
    mode_fields = torch.tensor(mode_fields, dtype=complex_type, device=device)

    # normalize mode_fields, each mode should have the same energy
    mode_fields = mode_fields / torch.sqrt(torch.sum(torch.abs(mode_fields)**2, dim=(1,2), keepdim=True))

    # new_mode_fields = np.load('final_field.npy')
    # new_mode_fields = torch.tensor(new_mode_fields, dtype=complex_type, device=device)
    # new_mode_fields = new_mode_fields.unsqueeze(0).unsqueeze(0).squeeze(-1)
    domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
    fiber = Fiber(domain, n_core0, n_clad0, custom_n=n, n2=n2, beta2=beta2, beta3=beta3, radius=core_radius,)
    boundary = Boundary(domain, boundary_type=boundary_type)
    config = SimConfig(wvl0=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING,
                            batch_num=BATCH_NUM, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T, mode_decomp_step=MODE_DECOMP_STEP)

    print(f'batch size: {BATCH_NUM}')
    # mode_fields = mode_fields.unsqueeze(0)

    # coeffs = torch.zeros(num_modes, dtype=complex_type)
    coeffs = torch.ones(num_modes, dtype=complex_type)
    # coeffs[0] = 5.0
    # coeffs[:4] = 0
    coeffs[3] = 1.5
    print(f'coeffs: {coeffs}')
    coeffs = coeffs.to(device)
    coeffs = torch.reshape(coeffs, (-1, num_modes, 1, 1))
    input_fields = torch.sum(coeffs * mode_fields, dim=1)
    initial_fields = Fields(domain, input_type='custom', fields=input_fields, tfwhm=tfwhm, total_energy=total_energy, peak_power=peak_power, t_center=0,)

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    input_fields = initial_fields.fields[0].cpu().numpy()
    
    plot_intensity(input_fields, radius=core_radius, extent=extent, title='Input Field')
    plt.colorbar()
    plt.show()
    sim = Simulation(domain, fiber, initial_fields, boundary, config, mode_fields=mode_fields)
    sim.run()
    output_fields = sim.fields.fields[0].cpu().numpy()



    plot_intensity(output_fields, radius=core_radius, extent=extent, title='Output Field')
    input_energy = calculate_total_energy(input_fields, dx=dx, dy=dy, dt=dt*1e-12)
    output_energy = calculate_total_energy(output_fields, dx=dx, dy=dy, dt=dt*1e-12)

    print(f'input energy : {input_energy}')
    print(f'output energy : {output_energy}')

    saved_total_fields = sim.saved_total_fields.detach().cpu().numpy()
    saved_total_fields = saved_total_fields[:, :, 64:-64, 64:-64, :]
    np.save(f'example_total_fields_{Nx}_{total_energy}nJ_mode5_{precision}.npy', saved_total_fields)

    if MODE_DECOMP_STEP > 0:
        mode_coeffs = sim.mode_coeffs[0].detach().cpu().numpy()
        np.save(f'mode_coeffs_{total_energy}nJ_mode5_{precision}.npy', mode_coeffs)

    plt.show()