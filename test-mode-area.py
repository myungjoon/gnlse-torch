import numpy as np

modes = np.load('modes_64x64.npy')
print(f"Modes shape: {modes.shape}")

# Mode area formula: A_eff = (∫|E|² dxdy)² / (∫|E|⁴ dxdy)

def calculate_mode_area(E, dx=1.0, dy=1.0):
    """
    Calculate the effective mode area.

    Args:
        E: 2D array of the electric field (can be complex)
        dx, dy: spatial grid spacing

    Returns:
        A_eff: effective mode area
    """
    intensity = np.abs(E)**2

    # Integrate |E|^2 over the domain using trapezoidal rule
    int_E2 = np.trapz(np.trapz(intensity, dx=dx, axis=0), dx=dy)

    # Integrate |E|^4 over the domain
    int_E4 = np.trapz(np.trapz(intensity**2, dx=dx, axis=0), dx=dy)

    # Calculate mode area
    A_eff = int_E2**2 / int_E4

    return A_eff

# Calculate mode area for the first mode
Lx, Ly = 32e-6, 32e-6
dx, dy = Lx / 64, Ly / 64
E = modes[0]
A_eff = calculate_mode_area(E, dx, dy)
print(f"Mode area of the first mode: {A_eff}")
