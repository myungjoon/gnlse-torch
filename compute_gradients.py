import numpy as np
import torch
import os, time

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig

# seed for random number generation
np.random.seed(42)
torch.manual_seed(42)

# Parameters (matching data-generation.py)
DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

DS_X = 2
DS_Y = 2
DS_T = 2
BATCH_NUM = 1  # Use batch size of 1 for gradient computation

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'single'
num_save = 100

wvl0 = 1030e-9
L0 = 0.05  # 5 cm

# Pulse
total_energy = 20  # nJ
Nt = 2**10
time_window = 2  # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.06  # ps

# Fiber parameters
core_radius = 62.5e-6 / 2
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 2.3e-20
beta2 = 1.655e-26 * (1e12**2)
beta3 = 23.3e-42 * (1e12**3)

# Simulation domain parameters
Lx, Ly = 4 * core_radius, 4 * core_radius
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
dz = 1e-5
Nz = round(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

# Load modes
modes = np.load('modes.npy')
modes = torch.tensor(modes, dtype=torch.complex64, device=device)
num_mode = 30
modes = modes.unsqueeze(0)  # Add batch dimension: (1, num_mode, Nx, Ny)

domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, beta3=beta3, n2=n2, radius=core_radius,)
boundary = Boundary(domain, boundary_type)

# Baseline coefficients
baseline_coeffs = torch.ones((1, num_mode), dtype=torch.complex64, device=device)
# baseline_coeffs[0, 0] = 1.0  # You can customize this

# Finite difference step size
h = 1e-6  # Small perturbation for finite difference

# Intensity calculation mode: 'max', 'center', or 'integrated'
intensity_mode = 'integrated'  # Change this to use different intensity measures
# 'max' - maximum intensity at spatial center over time (default)
# 'center' - intensity at temporal center
# 'integrated' - time-integrated intensity at spatial center

# Calculate center indices for downsampled output
# Output shape: (batch, 2, Nx//DS_X//2, Ny//DS_Y//2, Nt//DS_T//2)
output_Nx = Nx // DS_X // 2
output_Ny = Ny // DS_Y // 2
output_Nt = Nt // DS_T // 2
center_x = output_Nx // 2
center_y = output_Ny // 2
center_t = output_Nt // 2  # Temporal center

print(f'Output dimensions: ({output_Nx}, {output_Ny}, {output_Nt})')
print(f'Center indices: ({center_x}, {center_y}, {center_t})')

def compute_intensity_at_center(spatiotemporal_fields, mode='max'):
    """
    Compute intensity at the spatial center.
    
    Args:
        spatiotemporal_fields: Output from simulation, shape (batch, 2, Nx, Ny, Nt)
        mode: 'max' - maximum intensity at spatial center over time (default)
              'center' - intensity at temporal center
              'integrated' - time-integrated intensity at spatial center
    
    Returns:
        Intensity value (scalar)
    """
    # spatiotemporal_fields shape: (batch, 2, Nx, Ny, Nt)
    output_fields = spatiotemporal_fields[0, 1, :, :, :]  # Get output (index 1) from first batch
    center_field = output_fields[center_x, center_y, :]  # (Nt,)
    intensity = np.abs(center_field)**2
    
    if mode == 'max':
        return np.max(intensity)  # Maximum intensity at center over time
    elif mode == 'center':
        return intensity[center_t]  # Intensity at temporal center
    elif mode == 'integrated':
        dt_output = time_window / output_Nt
        return np.sum(intensity) * dt_output  # Time-integrated intensity
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_simulation(coeffs):
    """Run simulation with given coefficients and return intensity at center."""
    input_fields = torch.sum(coeffs[:,:,None, None] * modes[:, :num_mode], dim=1)
    
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, 
                      raman=RAMAN, self_steeping=SELF_STEEPING,
                      batch_num=1, num_save=num_save, ds_x=DS_X, ds_y=DS_Y, ds_t=DS_T)
    input_fields_obj = Fields(domain, input_type='custom', fields=input_fields, 
                             tfwhm=tfwhm, total_energy=total_energy, t_center=0)
    sim = Simulation(domain, fiber, input_fields_obj, boundary, config)
    sim.run()
    
    spatiotemporal_fields = sim.spatiotemporal_fields.cpu().numpy()
    intensity = compute_intensity_at_center(spatiotemporal_fields, mode=intensity_mode)
    return intensity

print("Running baseline simulation...")
baseline_intensity = run_simulation(baseline_coeffs)
print(f"Baseline intensity at center (mode: {intensity_mode}): {baseline_intensity:.6e}")

# Initialize gradient arrays
# Gradient with respect to real and imaginary parts of each coefficient
# These are real gradients (intensity is real-valued)
gradients_real = np.zeros(num_mode, dtype=np.float64)  # dI/d(Re(c_i))
gradients_imag = np.zeros(num_mode, dtype=np.float64)  # dI/d(Im(c_i))

print(f"\nComputing gradients for {num_mode} coefficients...")
print("This may take a while as we need to run {} simulations...".format(2 * num_mode + 1))

start_time = time.time()

for i in range(num_mode):
    # Perturb real part
    coeffs_pert_real = baseline_coeffs.clone()
    coeffs_pert_real[0, i] = coeffs_pert_real[0, i] + h
    intensity_pert_real = run_simulation(coeffs_pert_real)
    gradients_real[i] = (intensity_pert_real - baseline_intensity) / h
    
    # Perturb imaginary part
    coeffs_pert_imag = baseline_coeffs.clone()
    coeffs_pert_imag[0, i] = coeffs_pert_imag[0, i] + 1j * h
    intensity_pert_imag = run_simulation(coeffs_pert_imag)
    gradients_imag[i] = (intensity_pert_imag - baseline_intensity) / h
    
    if (i + 1) % 5 == 0:
        print(f"  Completed {i+1}/{num_mode} coefficients", flush=True)

total_time = time.time() - start_time
print(f"\nTotal computation time: {total_time:.2f} seconds")
print(f"Average time per simulation: {total_time / (2 * num_mode):.2f} seconds")

# Combine real and imaginary gradients into complex gradient representation
# dI/dc = dI/d(Re(c)) + j * dI/d(Im(c))
# This represents the gradient of real-valued intensity w.r.t. complex coefficients
gradients_complex = gradients_real.astype(np.complex128) + 1j * gradients_imag.astype(np.complex128)

print("\nGradients computed!")
print(f"Baseline intensity: {baseline_intensity:.6e}")
print(f"\nGradient magnitudes:")
for i in range(num_mode):
    print(f"  Mode {i}: |grad| = {np.abs(gradients_complex[i]):.6e}, "
          f"Re(grad) = {np.real(gradients_complex[i]):.6e}, "
          f"Im(grad) = {np.imag(gradients_complex[i]):.6e}")

# Save results
results = {
    'baseline_coefficients': baseline_coeffs.cpu().numpy(),
    'baseline_intensity': baseline_intensity,
    'gradients_real': gradients_real,
    'gradients_imag': gradients_imag,
    'gradients_complex': gradients_complex,
    'h': h,
    'intensity_mode': intensity_mode,
    'center_indices': (center_x, center_y, center_t),
    'output_dims': (output_Nx, output_Ny, output_Nt),
}

output_filename = f'gradients_intensity_center_{intensity_mode}.npz'
np.savez(output_filename, **results)
print(f"\nResults saved to '{output_filename}'")

