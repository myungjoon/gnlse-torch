import torch
import math

class Fields:
    def __init__(self, domain,
                 input_type='gaussian',
                 fields=None,
                 beam_radius=25e-6,        # spatial beam radius
                 tfwhm=0.5,                # temporal FWHM (same unit as domain.t)
                 total_energy=150.0,
                 t_center=0.0,
                 cx=0.0, cy=0.0,
                 phase_map=None):

        self.domain = domain
        self.total_energy = float(total_energy)
        self.t_center = float(t_center)

        self.device = domain.device
        self.rdtype = domain.rdtype
        self.cdtype = domain.cdtype
        

        # ---- 공간 가우시안 ----
        if input_type == 'gaussian':
            spatial_profile = self._gaussian_beam(domain, beam_radius, cx, cy,
                                    device=self.device, rdtype=self.rdtype, cdtype=self.cdtype)  # (Nx,Ny)
            if phase_map is not None:
                
                spatial_profile = spatial_profile * torch.exp(1j * phase_map)
                # FFT of spatial_profile for launching beam
                spatial_profile = torch.fft.fftshift(torch.fft.fftn(spatial_profile))
                
        elif input_type == 'custom':
            spatial_profile = fields

        # ---- 시간 가우시안 ----
        temporal_profile = self._temporal_pulse(domain, tfwhm,
                                 t_center=self.t_center,
                                 device=self.device, rdtype=self.rdtype, cdtype=self.cdtype)  # (Nt,)

        # ---- 결합: (Nx,Ny,Nt) ----
        fields = spatial_profile.unsqueeze(-1) * temporal_profile.view(1, 1, -1)  # complex

        # ---- 에너지 정규화 ----
        dx = domain.Lx / domain.Nx
        dy = domain.Ly / domain.Ny
        dt = domain.time_window / domain.Nt
        fields = self._normalize_to_energy(fields, dx, dy, dt, self.total_energy)
        self.fields = fields

    @staticmethod
    def _gaussian_beam(domain, w, cx, cy, device, rdtype, cdtype):
        X = domain.X[:, :, 0].to(device=device, dtype=rdtype)
        Y = domain.Y[:, :, 0].to(device=device, dtype=rdtype)
        R2 = (X - cx)**2 + (Y - cy)**2
        spatial_profile = torch.exp(-R2 / (w**2))
        return spatial_profile.to(dtype=cdtype)

    @staticmethod
    def _temporal_pulse(domain, tfwhm, t_center=0.0,
                        device=None, rdtype=torch.float32, cdtype=torch.complex64):
        # domain.t은 (Nx,Ny,Nt)이므로 한 점(x0,y0)에서 t축만 가져옴
        t = domain.t[0, 0, :].to(device=device, dtype=rdtype)  # (Nt,)
        
        # if length of the t is 1, then return 1
        if len(t) == 1:
            return torch.ones(1, device=device, dtype=rdtype)

        two = torch.as_tensor(2.0, dtype=rdtype, device=device)
        t0 = torch.as_tensor(tfwhm, dtype=rdtype, device=device) / (two * torch.sqrt(torch.log(two)))

        tt = t - torch.as_tensor(t_center, dtype=rdtype, device=device)
        phase = -(tt**2) / (2 * t0**2)
        temporal_profile = torch.exp(phase).to(cdtype)  # (Nt,), complex
        return temporal_profile

    @staticmethod
    def _normalize_to_energy(fields, dx, dy, dt, E_target):
        # dx : meter
        # dy : meter
        # dt : picosecond
        # E_target : nanojoule

        intensity = torch.abs(fields)**2
        E_current = intensity.sum() * (dx * dy * dt)
        scale = torch.sqrt(torch.as_tensor(E_target, dtype=intensity.dtype, device=fields.device) / E_current) * math.sqrt(1e12 / 1e9)
        return fields * scale