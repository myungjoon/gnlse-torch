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
    def __init__(self, domain, fiber, fields, boundary, config=None):
        self.domain = domain
        self.fiber = fiber
        self.fields = fields
        self.boundary = boundary
        # config 처리 로직 유지
        if config is None:
            # config = SimConfig() # (사용자 환경에 맞게)
            pass 
        else:
            self.config = config
        
        self.device = domain.device

        # 계산 함수 호출
        self.calculate_K()
        self.calculate_Dt()
        
        # [핵심 수정 1] self.D = self.Dt + self.KZ 삭제! (메모리 폭발 원인)
        # 대신 Propagator를 미리 작게 쪼개서 계산해둡니다.
        
        # 공간 Propagator (Size: 1 x Nx x Ny x 1) - 매우 작음
        self.prop_spatial = torch.exp(1j * self.KZ * self.domain.dz / 2)
        
        # 시간 Propagator (Size: 1 x 1 x 1 x Nt) - 매우 작음
        self.prop_temporal = torch.exp(1j * self.Dt * self.domain.dz / 2)

    def calculate_K(self):
        self.k0 = 2 * torch.pi / self.config.center_wavelength
        # KZ 계산
        self.KZ = -(self.domain.KX[:,:,0]**2 + self.domain.KY[:,:,0]**2) / (2 * self.k0 * self.fiber.n_clad)
        # Kin 계산
        self.Kin = self.k0 * (self.fiber.n - self.fiber.n_clad)

        # 차원 맞추기 (Broadcasting 준비)
        self.KZ = self.KZ[None, :, :, None]   # (1, Nx, Ny, 1)
        self.Kin = self.Kin[None, :, :, None] # (1, Nx, Ny, 1)

    def calculate_Dt(self):
        omega = self.domain.W[0, 0, :]
        # Dt 계산
        self.Dt = (self.fiber.beta2 * omega**2) / 2.0 + (self.fiber.beta3 * omega**3) / 6.0
        # 차원 맞추기
        self.Dt = self.Dt.view(1, 1, 1, -1)   # (1, 1, 1, Nt)



    def _propagate_one_step(self, fields):
        
        # ----------------------------------------------------
        # 1. Linear Half-step
        # ----------------------------------------------------
        # 기존: fields = fields * torch.exp(1j * self.D * dz / 2)  <-- (X) 메모리 터짐
        
        # [수정] In-place 연산으로 순차 적용
        fields.mul_(self.prop_spatial)   # 공간 효과 적용 (Broadcasting)
        fields.mul_(self.prop_temporal)  # 시간 효과 적용 (Broadcasting)
        
        # IFFT (메모리 할당 발생하지만 불가피함)
        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))

        # ----------------------------------------------------
        # 2. Nonlinear Step (메모리 최적화 적용)
        # ----------------------------------------------------
        # 기존: fields = fields * torch.exp(1j * (Kin + n2*k0*|A|^2)*dz) <-- (X) 메모리 터짐
        
        # 미리 계산해둘 상수들 (루프 안에서 반복 계산 방지)
        # gamma_dz = n2 * k0 * dz
        gamma_dz = self.fiber.n2 * self.k0 * self.domain.dz
        
        # Kin term도 dz를 미리 곱해둡니다. (Kin * dz)
        # self.Kin은 (1, Nx, Ny, 1) 형태이므로 broadcasting 됩니다.
        kin_dz = self.Kin * self.domain.dz

        # 시간축(Nt) 길이
        Nt = fields.shape[-1]
        
        # 청크 사이즈: 메모리 상태에 따라 조절 (예: 10 ~ 50)
        # 작을수록 메모리는 안전하지만 속도는 아주 조금 느려질 수 있음
        chunk_size = 2**8 

        for i in range(0, Nt, chunk_size):
            # 1) 시간축 슬라이싱 (View만 생성하므로 메모리 거의 안 씀)
            end = min(i + chunk_size, Nt)
            f_slice = fields[..., i:end] 
            
            # 2) 조각에 대해서만 |E|^2 및 위상 계산 (메모리 아주 조금 사용)
            # phase_slice는 (Batch, Nx, Ny, chunk_size) 크기라 매우 작음
            phase_slice = f_slice.abs()
            phase_slice.square_()        # |A|^2
            
            # 3) 비선형 위상 계산 (In-place)
            phase_slice.mul_(gamma_dz)   # n2*k0*dz*|A|^2
            phase_slice.add_(kin_dz)     # + Kin*dz (Broadcasting)
            
            # 4) 연산자 생성 및 적용
            # exp(1j * phase) 대신 polar 사용
            nonlinear_op = torch.polar(torch.ones_like(phase_slice), phase_slice)
            
            # 5) 원본 fields에 덮어쓰기 (In-place)
            # f_slice는 fields의 View이므로, 여기에 곱하면 원본 fields가 바뀝니다.
            f_slice.mul_(nonlinear_op)
            
            # 루프 돌 때마다 임시 변수 삭제 (안전장치)
            del phase_slice
            del nonlinear_op

        # ----------------------------------------------------
        # 3. Linear Half-step (Back to spectral domain)
        # ----------------------------------------------------
        fields = torch.fft.fftn(fields, dim=(1, 2, 3))
        
        # 순차 적용 (In-place)
        fields.mul_(self.prop_spatial)
        fields.mul_(self.prop_temporal)
        
        # 경계 조건 적용
        fields.mul_(self.boundary.boundary)

        return fields

    def run(self,):
        fields = self.fields.fields

        
        fields = torch.fft.fftn(fields, dim=(1, 2, 3))

        for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):            
            fields = self._propagate_one_step(fields,)
            if i % 100 == 0:
                print(f'iteration {i}', flush=True)
        
        fields = torch.fft.ifftn(fields, dim=(1, 2, 3))
        self.output_fields = fields
