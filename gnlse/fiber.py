import torch

class GRINFiber:
    def __init__(self, domain, n_core, n_clad, beta2=0, beta3=0, n2=2e-20, radius=25e-6, ):
        self.domain = domain
        self.n_core = n_core
        self.n_clad = n_clad
        self.n2 = n2
        self.beta2 = beta2
        self.beta3 = beta3
        self.radius = radius

        self.n = self.create_fiber()
        

    def create_fiber(self,):
        if self.domain.X.shape[0] > 2:
            n_shape = self.domain.X.shape[:2]
        elif self.domain.X.shape[0] == 2:
            n_shape = self.domain.X.shape
        else:
            raise ValueError(f"Domain shape {self.domain.X.shape} is not supported")
        delta = (self.n_core**2 - self.n_clad**2) / (2 * self.n_core**2)
        n = torch.zeros(n_shape, dtype=self.domain.rdtype, device=self.domain.device)
        # only first two dimensions are used for fiber
        R = torch.sqrt(self.domain.X[:,:,0] **2 + self.domain.Y[:,:,0]**2)
        
        
        n[torch.where(R > self.radius)] = self.n_clad
        n[torch.where(R <= self.radius)] = self.n_core * torch.sqrt(1 - 2 * delta * (R[torch.where(R <= self.radius)]/self.radius)**2)
        
        return n