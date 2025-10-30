import torch

class Boundary:
    def __init__(self, domain, boundary_type='periodic',):
        
        self.domain = domain
        self.boundary = self.create_boundary(boundary_type)

    def create_boundary(self, boundary_type):
        if boundary_type == "periodic":
            boundary = 1.0
        elif boundary_type == "absorbing":
            radius = self.domain.Lx * 0.9
            boundary = torch.exp(-2*((torch.sqrt(self.domain.X[:,:,0:1]**2+self.domain.Y[:,:,0:1]**2)/radius)**10))
        return boundary