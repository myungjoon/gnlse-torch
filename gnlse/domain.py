import torch

class Domain:
    def __init__(self, Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision='single', device='cpu'):
        self.Lx = Lx    # domain length in x direction
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.time_window = time_window
        self.Nz = Nz
        self.dz = dz

        self.rdtype = torch.float32 if precision == 'single' else torch.float64
        self.cdtype = torch.complex64 if precision == 'single' else torch.complex128
        self.device = device


        self.X, self.Y, self.t = self.generate_grids(Lx, Ly, Nx, Ny, Nt, time_window)
        self.KX, self.KY, self.W = self.generate_freqs(Lx, Ly, Nx, Ny, Nt, time_window)
        
    def generate_grids(self, Lx, Ly, Nx, Ny, Nt, time_window):
        x = torch.linspace(-Lx/2, Lx/2, Nx, dtype=self.rdtype, device=self.device)
        y = torch.linspace(-Ly/2, Ly/2, Ny, dtype=self.rdtype, device=self.device)
        t = torch.linspace(-0.5 * time_window, 0.5 * time_window, Nt, dtype=self.rdtype, device=self.device)
        X, Y, t = torch.meshgrid(x, y, t, indexing='ij')
        return X, Y, t

    def generate_freqs(self, Lx, Ly, Nx, Ny, Nt, time_window):
        kx = torch.fft.fftfreq(Nx, d=Lx/Nx, dtype=self.rdtype, device=self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, d=Ly/Ny, dtype=self.rdtype, device=self.device) * 2 * torch.pi
        w = torch.fft.fftfreq(Nt, d=time_window/Nt, dtype=self.rdtype, device=self.device) * 2 * torch.pi
        KX, KY, W = torch.meshgrid(kx, ky, w, indexing='ij')
        return KX, KY, W    

