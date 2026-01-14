import numpy as np
import torch
import os
from tqdm import tqdm

is_slurm_job = 'SLURM_JOB_ID' in os.environ

from dataclasses import dataclass

@dataclass
class SimConfig:
    wvl0: float
    num_save: int = -1
    save_spatial: bool = False
    save_temporal: bool = False
    save_spectrum: bool = False
    save_total: bool = True
    dispersion: bool = True
    kerr: bool = True
    raman: bool = False
    fr: float = 0.18
    self_steeping: bool = False
    batch_num: int = 1
    ds_x: int = 1
    ds_y: int = 1
    ds_t: int = 1
    mode_decomp_step: int = 0  # Modal decomposition every N steps (0 = disabled)

class Simulation:
    def __init__(self, domain, fiber, fields, boundary, config, mode_fields=None):
        self.domain = domain
        self.fiber = fiber
        self.fields = fields
        self.boundary = boundary
        if config is None:
            config = SimConfig()
        else:
            self.config = config
        self.mode_fields = mode_fields
        self.cnt = 0
        
        self.device = domain.device

        self.calculate_K()
        self.calculate_Dt()
        self.D = self.Dt + self.KZ

        # FFT dimensions: 2D spatial for CW (Nt=1), 3D for pulsed
        self.fft_dims = (1, 2) if domain.Nt == 1 else (1, 2, 3)

    def calculate_K(self):
        self.k0 = 2 * torch.pi / self.config.wvl0
        self.KZ = -(self.domain.KX[:,:,0]**2 + self.domain.KY[:,:,0]**2) / (2 * self.k0 * self.fiber.n_clad)
        self.Kin = self.k0 * (self.fiber.n - self.fiber.n_clad)

        self.KZ = self.KZ[None, :, :, None]
        self.Kin = self.Kin[None, :, :, None]

    def calculate_Dt(self):
        if self.domain.Nt == 1:
            # CW mode: no dispersion
            self.Dt = torch.zeros(1, 1, 1, 1, dtype=self.domain.cdtype, device=self.device)
        else:
            omega = self.domain.W[0, 0, :]                     # shape: (Nt,)
            self.Dt = (self.fiber.beta2 * omega**2) / 2.0 + (self.fiber.beta3 * omega**3) / 6.0
            self.Dt = self.Dt.view(1, 1, 1, -1)

    def _propagate_one_step(self, fields, is_save_fields=False):

        # Linear propagation calculation (Half-step)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        fields = torch.fft.ifftn(fields, dim=self.fft_dims)

        # Nonlinear propagation calculation
        intensity = torch.abs(fields)**2
        if self.config.raman:
            I0  = torch.fft.ifftshift(intensity, dim=3)
            I_w = torch.fft.fft(I0.to(torch.complex64), dim=3)

            R0  = torch.fft.ifft(I_w * self.fiber.hrw, dim=3).real  # conv result in t=0@idx0 convention
            R_t = torch.fft.fftshift(R0, dim=3)                     # back to centered convention

            NL = (1.0 - self.config.fr) * intensity + self.config.fr * R_t

        else:
            NL = intensity

        fields = fields * torch.exp(1j * (self.Kin + self.fiber.n2 * self.k0 * NL) * self.domain.dz)
        fields = fields * self.boundary.boundary
        fields = torch.fft.fftn(fields, dim=self.fft_dims)
        fields = fields * torch.exp(1j  * self.D * self.domain.dz / 2)
        

        if is_save_fields and self.cnt < self.config.num_save:

            fields = torch.fft.ifftn(fields, dim=self.fft_dims)
            intensity = torch.abs(fields)**2

            if self.config.save_spatial:
                self.saved_spatial_fields[self.cnt, :, :] = torch.sum(intensity, axis=2)
            if self.config.save_temporal:
                self.saved_temporal_fields[:, self.cnt, :] = torch.sum(intensity, axis=(-3,-2))
            if self.config.save_spectrum:
                self.saved_spectrum[:, self.cnt, :] = torch.fft.fftshift(torch.sum(torch.abs(torch.fft.fft(torch.fft.ifftshift(fields, axis=-1), axis=-1))**2, axis=(-3,-2)), axis=-1)
            if self.config.save_total:
                self.saved_total_fields[:, self.cnt, ...] = fields
            self.cnt += 1
            fields = torch.fft.fftn(fields, dim=self.fft_dims)

        return fields


    def mode_decomposition(self, fields):
        # Complex inner product: c_n = ∫ E(x,y,t) * mode_n^*(x,y) dxdy
        # fields: (batch, Nx, Ny, Nt), mode_fields: (num_modes, Nx, Ny)
        # Result: (batch, num_modes, Nt)
        mode_conj = torch.conj(self.mode_fields)  # (num_modes, Nx, Ny)
        overlap = torch.einsum('bxyt,bmxy->bmt', fields, mode_conj)
        return overlap

    def run(self,):
        if self.fiber.hrw is not None:
            self.fiber.hrw = self.fiber.hrw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # self.calculate_raman_response()

        fields = self.fields.fields

        # Initialize modal decomposition storage
        if self.config.mode_decomp_step > 0 and self.mode_fields is not None:
            num_modes = self.mode_fields.shape[1]
            num_decomp_saves = self.domain.Nz // self.config.mode_decomp_step + 1
            self.mode_coeffs = torch.zeros(
                (self.config.batch_num, num_modes, num_decomp_saves, self.domain.Nt),
                dtype=fields.dtype, device=self.device
            )
            self.mode_decomp_cnt = 0
            # Save initial mode decomposition
            self.mode_coeffs[:, :, self.mode_decomp_cnt, :] = self.mode_decomposition(fields)
            self.mode_decomp_cnt += 1

        if self.config.num_save > 0:
            save_step = self.domain.Nz // self.config.num_save
            self.modes = torch.zeros((self.config.batch_num, self.config.num_save+1), dtype=fields.dtype, device=self.device)
            if self.config.save_spatial:
                self.saved_spatial_fields = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nx, self.domain.Ny), device=self.device, dtype=fields.dtype)
            if self.config.save_temporal:
                self.saved_temporal_fields = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nt), device=self.device, dtype=fields.dtype)
                intensity = torch.abs(fields)**2
                self.saved_temporal_fields[:, self.cnt, :] = torch.sum(intensity, axis=(-3,-2))
            if self.config.save_spectrum:
                self.saved_spectrum = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nt), device=self.device, dtype=fields.dtype)
                self.saved_spectrum[:, self.cnt, :] = torch.fft.fftshift(torch.sum(torch.abs(torch.fft.fft(torch.fft.ifftshift(fields, axis=-1), axis=-1))**2, axis=(-3,-2)), axis=-1)
            if self.config.save_total:
                self.saved_total_fields = torch.zeros((self.config.batch_num, self.config.num_save+1, self.domain.Nx, self.domain.Ny, self.domain.Nt), device=self.device, dtype=fields.dtype)
                self.saved_total_fields[:, self.cnt, ...] = fields
            # self.modes[:, self.cnt] = self.mode_decomposition(fields)
            self.cnt += 1
        else:
            save_step = -1
        fields = torch.fft.fftn(fields, dim=self.fft_dims)

        # for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
        for i in tqdm(range(self.domain.Nz)):
            is_save_fields = True if save_step > 0 and i % save_step == 0 else False

            fields = self._propagate_one_step(fields, is_save_fields)

            # Modal decomposition every N steps
            if (self.config.mode_decomp_step > 0 and
                self.mode_fields is not None and
                (i + 1) % self.config.mode_decomp_step == 0):
                fields_spatial = torch.fft.ifftn(fields, dim=self.fft_dims)
                self.mode_coeffs[:, :, self.mode_decomp_cnt, :] = self.mode_decomposition(fields_spatial)
                self.mode_decomp_cnt += 1

        fields = torch.fft.ifftn(fields, dim=self.fft_dims)
        self.fields.fields = fields
        self.saved_total_fields[:, -1, ...] = fields
        # save_fields = fields[:, field_shape[1]//2-field_shape[1]//4:field_shape[1]//2+field_shape[1]//4, field_shape[2]//2-field_shape[2]//4:field_shape[2]//2+field_shape[2]//4, field_shape[3]//2-field_shape[3]//4:field_shape[3]//2+field_shape[3]//4]
        # self.spatiotemporal_fields[:, 1, :, :, :] = save_fields[:, ::self.config.ds_x, ::self.config.ds_y, ::self.config.ds_t]
