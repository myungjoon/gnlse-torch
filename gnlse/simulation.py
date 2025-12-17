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
    save_spatial: bool = False
    save_temporal: bool = True
    dispersion: bool = True
    kerr: bool = True
    raman: bool = False
    fr: float = 0.18
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

    def _propagate_one_step(self, fields, is_save_fields=False):

        # Linear propagation calculation (Half-step)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))

        # Nonlinear propagation calculation
        intensity = torch.abs(fields)**2
        if self.config.raman:
            I_w = torch.fft.fft(intensity.to(torch.complex64), dim=3)
            R_t = torch.fft.ifft(I_w * self.H_R_w, dim=3).real
            NL = (1.0 - self.config.fr) * intensity + self.config.fr * R_t

        else:
            NL = intensity

        fields = fields * torch.exp(1j * (self.Kin + self.fiber.n2 * self.k0 * NL) * self.domain.dz)

        fields = torch.fft.fftn(fields, dim=(1, 2, 3))
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = fields * self.boundary.boundary

        if is_save_fields:
            fields = torch.fft.ifftn(fields, dim=(1, 2, 3))
            intensity = torch.abs(fields)**2
            if self.config.save_spatial:
                self.saved_spatial_fields[self.cnt, :, :] = torch.sum(intensity, axis=2)
            if self.config.save_temporal:
                self.saved_temporal_fields[self.cnt, :] = torch.sum(intensity, axis=(-3,-2))
            self.cnt += 1
            fields = torch.fft.fftn(fields, dim=(1, 2, 3))

        return fields

    def run(self,):
        fields = self.fields.fields
        if self.config.num_save > 0:
            save_step = self.domain.Nz // self.config.num_save
            if self.config.save_spatial:
                self.saved_spatial_fields = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nx, self.domain.Ny), device=self.device, dtype=fields.dtype)

            if self.config.save_temporal:
                self.saved_temporal_fields = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nt), device=self.device, dtype=fields.dtype)
                intensity = torch.abs(fields)**2
                self.saved_temporal_fields[self.cnt, :] = torch.sum(intensity, axis=(-3,-2))
        fields = torch.fft.fftn(fields, dim=(1, 2, 3))

        for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):   
            is_save_fields = True if save_step > 0 and i % save_step == 0 else False
            fields = self._propagate_one_step(fields, is_save_fields)

            # if self.config.num_save > 0 and i % save_step == 0:
            #     spatial_fields = torch.fft.ifftn(fields)
            #     spatial_fields = torch.sum(torch.abs(spatial_fields)**2, axis=2)
            #     spatial_fields = spatial_fields[::2, ::2]
            #     self.spatial_intensities_sequential[self.cnt, :, :] = spatial_fields
            #     self.cnt += 1
            # if i % save_step_xz == 0:
            #     self.fields_xz[self.cnt_xz] = torch.fft.ifftn(fields,)[:, fields.shape[1]//2, fields.shape[2]//2]
            #     self.cnt_xz += 1
            # if i % save_step_zt == 0:
            #     E_temporal = torch.sum(torch.abs(torch.fft.ifftn(fields))**2, axis=(0,1))
            #     # self.fields_zt[self.cnt_zt] = torch.fft.ifftn(fields[fields.shape[0]//2, fields.shape[1]//2, :])
            #     self.fields_zt[self.cnt_zt] = E_temporal
            #     self.cnt_zt += 1

        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))
        self.fields.fields = fields
        # save_fields = fields[:, field_shape[1]//2-field_shape[1]//4:field_shape[1]//2+field_shape[1]//4, field_shape[2]//2-field_shape[2]//4:field_shape[2]//2+field_shape[2]//4, field_shape[3]//2-field_shape[3]//4:field_shape[3]//2+field_shape[3]//4]
        # self.spatiotemporal_fields[:, 1, :, :, :] = save_fields[:, ::self.config.ds_x, ::self.config.ds_y, ::self.config.ds_t]