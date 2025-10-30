import numpy as np
import torch
import os
from scipy import stats

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


def correlation(field_ref, field_test):
    """
    Calculate the Pearson correlation coefficient between two complex fields
    based on their intensity distributions.

    Parameters
    ----------
    field_ref : ndarray
        Reference complex field (Nx, Ny, ...).
    field_test : ndarray
        Test or simulated complex field of the same shape as field_ref.

    Returns
    -------
    corr : float
        Pearson correlation coefficient between intensity patterns.
    p_value : float
        Two-tailed p-value for testing non-correlation.
    """
    

    # Convert to intensity
    I_ref = np.abs(field_ref)**2
    I_test = np.abs(field_test)**2

    # Normalize
    I_ref = (I_ref - np.mean(I_ref)) / np.std(I_ref)
    I_test = (I_test - np.mean(I_test)) / np.std(I_test)

    # Flatten and compute Pearson correlation
    corr, p_value = stats.pearsonr(I_ref.flatten(), I_test.flatten())

    return corr, p_value

def print_total_energy(domain, field):
    """
    Print the total energy of the field. Now, consider the pulse profile.
    """
    # Calculate the total energy
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny  
    total_energy = np.sum(np.abs(field)**2) * dx * dy
    print(f'Total energy: {total_energy:.4f} J')

def pulse_total_energy(domain, field, return_power_time=False):
    """
    Compute total pulse energy: ∭ |E(x,y,t)|^2 dx dy dt
    field: (Nx, Ny, Nt) complex; torch.Tensor or np.ndarray
    return_power_time=True이면 P(t)=∬|E|^2 dx dy (크기 Nt)도 반환
    """
    dx = domain.Lx / domain.Nx
    dy = domain.Ly / domain.Ny
    dt = domain.time_window / domain.Nt

    # Torch ↔ NumPy 분기
    is_torch = hasattr(field, "dtype") and "torch" in str(type(field))
    if is_torch:
        import torch
        intensity = (field.real**2 + field.imag**2)
        if return_power_time:
            P_t = intensity.sum(dim=(0,1)) * (dx * dy)          # (Nt,)
            E_total = P_t.sum() * dt                            # scalar (tensor)
            return E_total, P_t
        else:
            E_total = intensity.sum() * (dx * dy * dt)
            return E_total
    else:
        intensity = (field.real**2 + field.imag**2)
        if return_power_time:
            P_t = intensity.sum(axis=(0,1)) * (dx * dy)         # (Nt,)
            E_total = P_t.sum() * dt
            return E_total, P_t
        else:
            E_total = intensity.sum() * (dx * dy * dt)
            return E_total

def print_total_energy(domain, field):
    E_total = pulse_total_energy(domain, field)

    try:
        E_value = E_total.item()
    except AttributeError:
        E_value = float(E_total)
    print(f"Total energy: {E_value:.6g} (units per your normalization)")

def correlation(simulation, reference, dx=1e-6):
    """
    Calculate the correlation between two fields.
    """
    reference = np.abs(reference)**2
    simulation = np.abs(simulation)**2

    reference = (reference - np.mean(reference)) / np.std(reference)
    simulation = (simulation - np.mean(simulation)) / np.std(simulation)

    corr, p_value = stats.pearsonr(reference.flatten(), simulation.flatten())
    print(f"Pearson Correlation: {corr:.4f}, p-value: {p_value:.4e}")
    return corr, p_value

def save_results(results_dict, base_dir="results"):
    """
    Saves each item in a data dictionary as a separate .npy file.
    
    Args:
        results_dict (dict): A dictionary in the format {'filename': data, ...}.
        base_dir (str): The base directory to save the result files.
    """

    # 1. Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
       
    # 2. Iterate through each item in the dictionary and save it
    for filename, data in results_dict.items():
        # Combine the path
        filepath = os.path.join(base_dir, filename)
        
        # Process data based on its type
        if isinstance(data, torch.Tensor):
            processed_data = data.detach().cpu().numpy()
        elif isinstance(data, (np.ndarray, list, tuple)):
            processed_data = np.asarray(data)
        else:
            print(f"Warning: Cannot save '{filename}' because its type ({type(data)}) is not supported.")
            continue # Skip to the next item

        # Add .npy extension if not present and save
        if not filepath.endswith('.npy'):
            filepath += '.npy'
        
        # if there's already a file with the same name, append a number to avoid overwriting
        counter = 1
        while os.path.exists(filepath):

            filepath = filepath.replace('.npy', f'_{counter}.npy')
            counter += 1

        np.save(filepath, processed_data)

    print(f"Finished saving {list(results_dict.keys())} results.")