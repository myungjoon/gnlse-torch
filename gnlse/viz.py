import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

plt.rcParams['font.size'] = 15

C0 = 299792458  # m/s

def n_to_lm(n):
    """
    Convert mode index n to (l,m) pair according to GRIN fiber LP mode indexing.
    n starts from 1, with n=1 corresponding to (l,m)=(0,1)
    
    The ordering follows:
    1. Smaller l+m values come first
    2. For equal l+m values, pairs with smaller l come first
    
    Examples:
    n=1 -> (0,1)
    n=2 -> (0,2)
    n=3 -> (1,1)
    n=4 -> (0,3)
    n=5 -> (1,2)
    n=6 -> (2,1)
    n=7 -> (0,4)
    n=8 -> (1,3)
    n=9 -> (2,2)
    n=10 -> (3,1)
    etc.
    """
    if n < 1:
        raise ValueError("Mode index n must be a positive integer")
    
    # Start from group 1 (which has l+m=1)
    group_sum = 1
    count = 0
    
    while True:
        # Number of (l,m) pairs in current group where l+m=group_sum
        group_size = group_sum
        
        # If n is in the current group
        if count + group_size >= n:
            # Calculate position within group (0-indexed)
            pos_in_group = n - count - 1
            
            # For group with sum=group_sum, l goes from 0 to group_sum-1
            l = pos_in_group
            m = group_sum - l
            
            return (l, m)
        
        # Move to next group
        count += group_size
        group_sum += 1

def plot_fields(fields, domain, wvl0=1030e-9, core_radius=None, extent=None, interpolation=None):
    Nt = domain.Nt
    time_window = domain.time_window
    dt = time_window / Nt
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)
    freq = np.fft.fftfreq(Nt, dt * 1e-12)
    f0 = C0 / wvl0

    freq_abs = f0 + freq
    wavelength = C0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)
    
    extent = [-0.5 * domain.Lx, 0.5 * domain.Lx, -0.5 * domain.Ly, 0.5 * domain.Ly]

    # subplot 0 : spatial profile
    E_spatial = np.abs(fields)**2
    E_spatial = E_spatial.sum(axis=2)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    im = ax[0].imshow(E_spatial, cmap='turbo', interpolation=interpolation, extent=extent)

    ax[0].set_xlabel('X ($\mu m$)')
    ax[0].set_ylabel('Y ($\mu m$)')

    if core_radius is not None:
        fiber_core = Circle((0, 0), core_radius, fill=False, linestyle='-', edgecolor='white', linewidth=2.0)
        ax[0].add_patch(fiber_core)
    plt.tight_layout()

    # subplot 1 : temporal profile, only length of time window is larger than 1
    if domain.Nt > 1:
        E_temporal = np.abs(fields)**2
        E_temporal = E_temporal.sum(axis=(0,1))
        ax[1].plot(t, E_temporal)
        ax[1].set_xlabel('Time (ps)')
        ax[1].set_ylabel('Intensity (a.u.)')
        plt.tight_layout()

    # subplot 2 : spectrum
    if domain.Nt > 1:
        E_spectrum = np.fft.ifftshift(fields, axes=2)
        E_spectrum = np.sum(np.abs(np.fft.fft(E_spectrum, axis=2))**2, axis=(0,1))
        E_spectrum = np.fft.fftshift(E_spectrum)
        ax[2].plot(wavelength_nm, E_spectrum)
        ax[2].set_xlabel('Wavelength (nm)')
        ax[2].set_ylabel('Intensity (a.u.)')
        ax[2].set_xlim(1400, 1800)
        plt.tight_layout()


def plot_mode_evolution(modes, dz, num_modes=10):
    plt.figure()
    z = np.arange(0, len(modes)*dz, dz)
    for i in range(num_modes):
        l, m = n_to_lm(i+1)
        plt.plot(z, np.sum(np.abs(modes[:, i])**2, axis=1), label=f'LP{l}{m}')
    plt.xlabel('z (m)')
    plt.ylabel('Amplitude')

    plt.legend()

def plot_index_profile(n):
    fig, ax = plt.subplots()
    im = ax.imshow(n, cmap='Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

def plot_beam_intensity(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots()
    
    # extent = [-150/2, 150/2, -150/2, 150/2]
    xtick = np.linspace(0, field.shape[1], 11)
    ytick = np.linspace(0, field.shape[0], 11)
    xlabel = np.linspace(extent[0], extent[1], 11)
    ylabel = np.linspace(extent[2], extent[3], 11)

    if interpolation is not None:
        im = ax.imshow(np.abs(field)**2, cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(np.abs(field)**2, cmap='turbo',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x:.0f}' for x in xlabel])
    ax.set_yticklabels([f'{y:.0f}' for y in ylabel])

    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_beam_phase(field, indices=None, extent=None, interpolation=None):
    fig, ax = plt.subplots()
    
    eps = 1e-5
    extent = [-75, 75, -75, 75]
    xtick = np.linspace(0, field.shape[1]+eps, 5)
    ytick = np.linspace(0, field.shape[0]+eps, 5)
    xlabel = np.linspace(extent[0], extent[1], 5)
    ylabel = np.linspace(extent[2], extent[3], 5)

    if interpolation is not None:
        im = ax.imshow(np.angle(field), cmap='turbo', interpolation=interpolation)
    else:
        im = ax.imshow(np.angle(field), cmap='turbo',)
    ax.set_xticks(xtick)
    ax.set_yticks(ytick)
    ax.set_xticklabels([f'{x}' for x in xlabel])
    ax.set_yticklabels([f'{y}' for y in ylabel])

    ax.set_xlabel(r'x ($\mu m$)')
    ax.set_ylabel(r'y ($\mu m$)')
    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    if indices is not None:
        ax.contour(indices, levels=[np.min(indices)], colors='white', linewidths=2)

def plot_energy_evolution(energies, dz):
    plt.figure()
    plt.plot(np.arange(0, len(energies)*dz, dz), energies)
    plt.xlabel('z (m)')
    plt.ylabel('Energy')
    plt.title('Energy Evolution')

def plot_input_and_output_beam(input_field, output_field, radius=10, extent=None, indices=None, interpolation=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(np.abs(input_field)**2, cmap='turbo', interpolation=interpolation, extent=extent)
    ax[0].set_title('Input Field')

    ax[0].set_xlim([-75, 75])
    ax[0].set_ylim([-75, 75])
    ax[0].set_xticks([-75, 0, 75])
    ax[0].set_yticks([-75, 0, 75])
    ax[0].set_xticklabels([-75, 0, 75])
    ax[0].set_yticklabels([-75, 0, 75])
    ax[0].set_xlabel(r'x ($\mu m$)')
    ax[0].set_ylabel(r'y ($\mu m$)')
    
    fiber0 = Circle((0, 0), radius, fill=False, linestyle='--', edgecolor='white', linewidth=2.0)
    ax[0].add_patch(fiber0)

    ax[1].imshow(np.abs(output_field)**2, cmap='turbo', interpolation=interpolation, extent=extent)
    ax[1].set_title('Output Field')
    fiber1 = Circle((0, 0), radius, fill=False, linestyle='--', edgecolor='white', linewidth=2.0)
    ax[1].add_patch(fiber1)

    ax[1].set_xlim([-75, 75])
    ax[1].set_ylim([-75, 75])
    ax[1].set_xticks([-75, 0, 75])
    ax[1].set_yticks([-75, 0, 75])
    ax[1].set_xticklabels([-75, 0, 75])
    ax[1].set_yticklabels([-75, 0, 75])
    ax[1].set_xlabel(r'x ($\mu m$)')
    ax[1].set_ylabel(r'y ($\mu m$)')

    plt.tight_layout()

def plot_intensity(field, radius=None, extent=None, title=None):
    total_intensity = np.sum(np.abs(field)**2, axis=-1)
    plt.figure(figsize=(5,5.5), layout='constrained')
    plt.imshow(total_intensity, aspect='auto', origin='lower', cmap='turbo', extent=extent)
    if radius is not None:
        circle = Circle((0, 0), radius, color='white', fill=False, linewidth=2.0)
        plt.gca().add_patch(circle)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=18)