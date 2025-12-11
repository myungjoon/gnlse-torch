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
    batch_num: int = 1
    ds_x: int = 1
    ds_y: int = 1
    ds_t: int = 1


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
        
        self.cnt = 0
        self.cnt_xz = 0
        self.cnt_zt = 0


        # currently not used
        # self.num_save_xz = 500
        # self.num_save_zt = 10

        self.device = domain.device

        self.calculate_K()
        self.calculate_Dt()
        self.D = self.Dt + self.KZ

    def calculate_K(self):
        self.k0 = 2 * torch.pi / self.config.center_wavelength
        self.KZ = -(self.domain.KX[:,:,0]**2 + self.domain.KY[:,:,0]**2) / (2 * self.k0 * self.fiber.n_clad)
        self.Kin = self.k0 * (self.fiber.n - self.fiber.n_clad)

        self.KZ = self.KZ[None, :, :, None]
        self.Kin = self.Kin[None, :, :, None]

    def calculate_Dt(self):
        omega = self.domain.W[0, 0, :]                     # shape: (Nt,)
        self.Dt = (self.fiber.beta2 * omega**2) / 2.0 + (self.fiber.beta3 * omega**3) / 6.0
        self.Dt = self.Dt.view(1, 1, 1, -1)

    def _propagate_one_step(self, fields,):

        # Linear propagation calculation (Half-step)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))

        # Nonlinear propagation calculation
        fields = fields * torch.exp(1j * (self.Kin + self.fiber.n2 * self.k0 * torch.abs(fields)**2) * self.domain.dz)

        fields = torch.fft.fftn(fields, dim=(1, 2, 3))
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = fields * self.boundary.boundary

        return fields

    def run(self,):
        fields = self.fields.fields
        # if self.config.num_save > 0:
        #     save_step = self.domain.Nz // self.config.num_save

        # self.spatial_intensities = torch.zeros((2, self.domain.Nx // 2, self.domain.Ny // 2), device=self.device, dtype=torch.float32) # input and output        
        # self.spatial_intensities_sequential = torch.zeros((self.config.num_save+1, self.domain.Nx // 2, self.domain.Ny // 2), device=self.device, dtype=torch.float32) # input + num_save

        # self.spatiotemporal_fields = torch.zeros((self.config.batch_num, 2, self.domain.Nx // self.config.ds_x // 2, self.domain.Ny // self.config.ds_y // 2, self.domain.Nt // self.config.ds_t // 2), device=self.device, dtype=fields.dtype) # input and output
        # save_fields cut both quarter of the fields before downsampling
        # field_shape = fields.shape

        # save_fields = fields[:, field_shape[1]//2-field_shape[1]//4:field_shape[1]//2+field_shape[1]//4, field_shape[2]//2-field_shape[2]//4:field_shape[2]//2+field_shape[2]//4, field_shape[3]//2-field_shape[3]//4:field_shape[3]//2+field_shape[3]//4]
        # self.spatiotemporal_fields[:, 0, :, :, :] = save_fields[:, ::self.config.ds_x, ::self.config.ds_y, ::self.config.ds_t]
        
        fields = torch.fft.fftn(fields, dim=(1, 2, 3))

        for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):            
            fields = self._propagate_one_step(fields,)
        
        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))
        self.output_fields = fields
        # save_fields = fields[:, field_shape[1]//2-field_shape[1]//4:field_shape[1]//2+field_shape[1]//4, field_shape[2]//2-field_shape[2]//4:field_shape[2]//2+field_shape[2]//4, field_shape[3]//2-field_shape[3]//4:field_shape[3]//2+field_shape[3]//4]
        # self.spatiotemporal_fields[:, 1, :, :, :] = save_fields[:, ::self.config.ds_x, ::self.config.ds_y, ::self.config.ds_t]