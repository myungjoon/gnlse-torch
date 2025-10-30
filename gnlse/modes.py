import numpy as np
import torch

from .mmfsim.modes import GrinLPMode
from scipy.special import genlaguerre, factorial

class Mode:
    def __init__(self, domain, fiber, input_beam, l=0, m=0, device='cpu'):
        self.Lx = domain.Lx
        self.Ly = domain.Ly
        self.Nx = domain.Nx
        self.Ny = domain.Ny

        self.radius = fiber.radius
        self.n_core = fiber.n_core
        self.n_clad = fiber.n_clad

        self.wvl0 = input_beam.wvl0
        self.device = device
        self.l = l
        self.m = m

        # self.field = self.calculate_field()
       
    def calculate_field(self):       
        grid = grids.Grid(pixel_size=self.Lx/self.Nx, pixel_numbers=(self.Nx, self.Ny))
        grin_fiber = GrinFiber(radius=self.radius, wavelength=self.wvl0, n1=self.n_core, n2=self.n_clad)
        mode = GrinLPMode(self.l, self.m)
        mode.compute(grin_fiber, grid)
        
        field = torch.tensor(mode._fields)
        # summation = torch.sum(torch.abs(field)**2)
        field = field # / torch.sqrt(summation)
        return field.to(self.device)

def calculate_mode_field(grid, grin_fiber, n, device='cpu'):
    l, m, k = n_to_lm(n)

    mode = GrinLPMode(l, m)
    mode.compute(grin_fiber, grid)
    mode_field = torch.tensor(mode._fields[:, :, k])

    return mode_field.to(device)


def calculate_effective_mode_area(mode_field, dx, dy):
    dA = dx * dy  # Area of a pixel
    integral_E2 = np.sum(np.abs(mode_field)**2) * dA
    integral_E4 = np.sum(np.abs(mode_field)**4) * dA

    A_eff = (integral_E2**2) / integral_E4 
    return A_eff


def n_to_lm(n):
    """
    Convert mode index n to (l,m) pair according to GRIN fiber LP mode indexing.
    n starts from 1, with n=1 corresponding to (l,m)=(0,1)
    
    The ordering follows:
    1. Smaller l+m values come first
    2. For equal l+m values, pairs with smaller l come first
    
    Examples:
    n=1 -> (0,1)
    n=2 -> (1,1)a
    n=3 -> (1,1)b
    n=4 -> (0,2)
    n=5 -> (1,2)
    n=6 -> (2,1)
    n=7 -> (0,4)
    n=8 -> (1,3)
    n=9 -> (2,2)
    n=10 -> (3,1)
    etc.
    """
    if n < 0:
        raise ValueError("Mode index n must be a positive integer")
    
    manual_lm_pairs = [
        # Mode Group 1 (N=1): 1개
        (0,1,0),  # LP01
        
        # Mode Group 2 (N=2): 2개  
        (1,1,0),  # LP11a
        (1,1,1),  # LP11b
        
        # Mode Group 3 (N=3): 3개
        (0,2,0),  # LP02
        (2,1,0),  # LP21a
        (2,1,1),  # LP21b
        
        # Mode Group 4 (N=4): 4개
        (1,2,0),  # LP12a
        (1,2,1),  # LP12b
        (3,1,0),  # LP31a
        (3,1,1),  # LP31b
        
        # Mode Group 5 (N=5): 5개
        (0,3,0),  # LP03
        (2,2,0),  # LP22a
        (2,2,1),  # LP22b
        (4,1,0),  # LP41a
        (4,1,1),  # LP41b
        
        # Mode Group 6 (N=6): 6개
        (1,3,0),  # LP13a
        (1,3,1),  # LP13b
        (3,2,0),  # LP32a
        (3,2,1),  # LP32b
        (5,1,0),  # LP51a
        (5,1,1),  # LP51b
        
        # Mode Group 7 (N=7): 7개
        (0,4,0),  # LP04
        (2,3,0),  # LP23a
        (2,3,1),  # LP23b
        (4,2,0),  # LP42a
        (4,2,1),  # LP42b
        (6,1,0),  # LP61a
        (6,1,1),  # LP61b
        
        # Mode Group 8 (N=8): 8개
        (1,4,0),  # LP14a
        (1,4,1),  # LP14b
        (3,3,0),  # LP33a
        (3,3,1),  # LP33b
        (5,2,0),  # LP52a
        (5,2,1),  # LP52b
        (7,1,0),  # LP71a
        (7,1,1),  # LP71b
    ]

    return manual_lm_pairs[n]  # n starts from 1, so we use n-1 for 0-indexing

def calculate_modes(domain, fiber, input, num_modes=10, device='cpu'):
    grid = grids.Grid(pixel_size=domain.Lx/domain.Nx, pixel_numbers=(domain.Nx, domain.Ny))
    grin_fiber = GrinFiber(radius=fiber.radius, wavelength=input.wvl0, n1=fiber.nc, n2=fiber.n0)

    modes = torch.zeros((num_modes, domain.Nx, domain.Ny), device=device)
    for n in range(num_modes):
        l, m, k = n_to_lm(n)
        
        mode = GrinLPMode(l, m)
        mode.compute(grin_fiber, grid)
        field = torch.tensor(mode._fields[:, :, k]).to(device)
        modes[n] = field
    return modes

def decompose_modes(field, modes, num_modes=10,):
    # Decompose the input beam into the modes of the fiber
    coefficients = torch.zeros(num_modes, dtype=torch.complex64)

    for n in range(num_modes):
        overlap = torch.sum(modes[n] * field.conj())    
        coefficients[n] = overlap

    # normalize coefficients of tensor coefficients
    # coefficients = coefficients / torch.norm(coefficients)

    return coefficients


if __name__ == "__main__":
    import mmfsim.grid as grids
    from mmfsim.fiber import GrinFiber
    grid = grids.Grid(pixel_size=0.5e-6, pixel_numbers=(128,128))
    fiber = GrinFiber(radius=26e-6, wavelength=1064e-9, n1=1.465, n2=1.45)
    mode1 = GrinLPMode(2, 2)
    mode1.compute(fiber, grid)
    field = mode1._fields
    print(f"Mode field shape: {field.shape}")
