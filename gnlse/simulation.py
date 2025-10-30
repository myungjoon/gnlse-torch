import numpy as np
import torch
import os
from tqdm import tqdm

is_slurm_job = 'SLURM_JOB_ID' in os.environ

from dataclasses import dataclass

@dataclass
class SimConfig:
    center_wavelength: float
    num_save: int = -1
    dispersion: bool = True
    kerr: bool = True
    raman: bool = False
    self_steeping: bool = False


class Simulation:
    def __init__(self, domain, fiber, fields, boundary, config):
        self.domain = domain
        self.fiber = fiber
        self.fields = fields
        self.boundary = boundary
        if config is None:
            config = SimConfig()
        else:
            self.config = config

        self.num_save_xz = 500
        self.num_save_zt = 10
        self.cnt = 0
        self.cnt_xz = 0
        self.cnt_zt = 0

        self.device = domain.device

        self.calculate_K()
        self.calculate_Dt()
        self.D = self.Dt + self.KZ

    def calculate_K(self):
        self.k0 = 2 * torch.pi / self.config.center_wavelength
        self.KZ = -(self.domain.KX[:,:,0]**2 + self.domain.KY[:,:,0]**2) / (2 * self.k0 * self.fiber.n_clad)
        self.KZ = torch.unsqueeze(self.KZ, 2)
        self.Kin = self.k0 * (self.fiber.n - self.fiber.n_clad)

    def calculate_Dt(self):
        omega = self.domain.W[0, 0, :]                     # shape: (Nt,)
        self.Dt = (self.fiber.beta2 * omega**2) / 2.0 + (self.fiber.beta3 * omega**3) / 6.0
        self.Dt = self.Dt.view(1, 1, -1)

    def _propagate_one_step(self, fields,):

        # Linear propagation calculation (Half-step)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = torch.fft.ifftn(fields,)

        # Nonlinear calculation
        # if self.config.kerr:
        #     Knl = self.fiber.n2 * self.k0 * torch.abs(fields)**2
        # else:
        #     Knl = 0
        fields = fields * torch.exp(1j * (torch.unsqueeze(self.Kin, 2) + self.fiber.n2 * self.k0 * torch.abs(fields)**2) * self.domain.dz)

        # Linear propagation calculation (Half-step)
        fields = torch.fft.fftn(fields,)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = fields * self.boundary.boundary

        return fields

    def _propagate_one_step_no_dispersion(self, fields,):
        if self.config.kerr:
            Knl = self.fiber.n2 * self.k0 * torch.abs(fields)**2
        else:
            Knl = 0
        fields = fields * torch.exp(1j * (torch.unsqueeze(self.Kin, 2) + Knl) * self.domain.dz)
        fields = fields * self.boundary.boundary
        return fields

    def run(self,):
        fields = self.fields.fields
        if self.config.num_save > 0:
            save_step = self.domain.Nz // self.config.num_save

        self.spatial_intensities = torch.zeros((2, self.domain.Nx // 2, self.domain.Ny // 2), device=self.device, dtype=torch.float32) # input and output
        self.spatiotemporal_fields = torch.zeros((2, self.domain.Nx // 2, self.domain.Ny // 2, self.domain.Nt // 2), device=self.device, dtype=fields.dtype) # input and output
        self.spatial_intensities_sequential = torch.zeros((self.config.num_save+1, self.domain.Nx // 2, self.domain.Ny // 2), device=self.device, dtype=torch.float32) # input + num_save

        self.spatiotemporal_fields[0, :, :, :] = fields[::2, ::2, ::2]
        self.spatial_intensities[0, :, :] = torch.sum(torch.abs(fields[::2, ::2, ::2])**2, axis=2)
        fields = torch.fft.fftn(fields)

        for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
            if self.config.num_save > 0 and i % save_step == 0:
                spatial_fields = torch.fft.ifftn(fields)
                spatial_fields = torch.sum(torch.abs(spatial_fields)**2, axis=2)
                spatial_fields = spatial_fields[::2, ::2]
                self.spatial_intensities_sequential[self.cnt, :, :] = spatial_fields
                self.cnt += 1
            # if i % save_step_xz == 0:
            #     self.fields_xz[self.cnt_xz] = torch.fft.ifftn(fields,)[:, fields.shape[1]//2, fields.shape[2]//2]
            #     self.cnt_xz += 1
            # if i % save_step_zt == 0:
            #     E_temporal = torch.sum(torch.abs(torch.fft.ifftn(fields))**2, axis=(0,1))
            #     # self.fields_zt[self.cnt_zt] = torch.fft.ifftn(fields[fields.shape[0]//2, fields.shape[1]//2, :])
            #     self.fields_zt[self.cnt_zt] = E_temporal
            #     self.cnt_zt += 1
            
            fields = self._propagate_one_step(fields,)
        
        fields = torch.fft.ifftn(fields)


        self.spatiotemporal_fields[1, :, :, :] = fields[::2, ::2, ::2]
        self.spatial_intensities[1, :, :] = torch.sum(torch.abs(fields[::2, ::2, ::2])**2, axis=2)
        self.spatial_intensities_sequential[self.cnt, :, :] = self.spatial_intensities[1, :, :]