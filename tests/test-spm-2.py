import numpy as np
import matplotlib.pyplot as plt
import torch
import os, time
import numpy as np
from scipy import special

from gnlse import Domain, GRINFiber, Fields, Boundary, Simulation, SimConfig
from gnlse import plot_fields, plot_index_profile

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1,p=40,w=26.5e-6):
    return np.exp(-2*((np.sqrt(x**2+y**2)/w)**p))

def modes(m,l):
    # ref: Arash Mafi, "Bandwidth Improvement in Multimode Optical Fibers Via Scattering From Core Inclusions," J. Lightwave Technol. 28, 1547-1555 (2010)
    # mode numbers:
    p=l-1   # l-1 of LP_ml
    m=m   # m of LP_ml 

    Apm=np.sqrt(np.math.factorial(p)/np.pi/np.math.factorial(p+np.abs(m)))

    c = 299792458               # [m/s]
    n0 = 1.45                   # Refractive index of medium (1.44 for 1550 nm, 1.45 for 1030 nm)
    lambda_c = 1030e-9          # Central wavelength of the input pulse in [m]
    R = 25e-6                   # fiber radius
    w=2*np.pi*c/lambda_c        # [Hz]
    k0 = w*n0/c
    delta = 0.01                #

    N_2 = 0.5*(R**2)*(k0**2)*(n0**2)*delta
    ro_0= R/(4*N_2)**0.25

    Epm=Apm*(np.sqrt(x1**2+y1**2)**np.abs(m))/(ro_0**(1+np.abs(m)))*np.exp(-(x1**2+y1**2)/2/ro_0**2)*special.eval_genlaguerre(p,np.abs(m),(x1**2+y1**2)/ro_0**2,out=None)

    Epm_=np.multiply(Epm,(np.cos(m*np.arctan2(y,x))+np.sin(m*np.arctan2(y,x))))
    return cp.asarray(Epm_/np.max(np.abs(Epm_))) #cp_super_gauss2d =cp.asarray(z)
    #cp_Epm=cp.asarray(Epm_/np.max(np.abs(Epm_)))
    #return 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 100
np.random.seed(seed)

device_id = 1
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

precision = 'double'


C0 = 299792458 # [m/s]
num_save = 10

wvl0 = 1030e-9
total_energy = 5e-9 # nJ
L0 = 1.0

# Beam
beam_radius = 10e-6

# Pulse
Nt = 2**10
time_window = 2e-12 # s
dt = time_window / Nt
tfwhm = 0.5e-12 # s
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

# Fiber parameters
core_radius = 25e-6
NA = 0.25
n_clad = 1.45
n_core = np.sqrt(NA**2 + n_clad**2)
n2 = 3.2e-20
beta2 = 1.655e-26 

# Simulation domain parameters
Lx, Ly = 3 * core_radius, 3 * core_radius
unit = 1e-6
Nx, Ny = 256, 256
print(f'The grid size is {Nx}x{Ny}')
dz = 1e-4
Nz = int(L0 / dz)

# Boundary condition
boundary_type = 'periodic'

timesteps=len(t)
spacewidth=3 * core_radius
xres = spacewidth/((2**6))
x = y = np.arange(-spacewidth*0.5, (spacewidth*0.5),xres)
[X,Y,T] = np.meshgrid(x,y,t)

## FOURIER DOMAIN
fs = 1/time_window
freq = C0 / wvl0 + fs*t
w=2*np.pi*C0/wvl0 # [Hz]
omegas=2*np.pi*freq
wt = omegas - w

#CHECK KX
a = np.pi/xres  # grid points in "frequency" domain--> {2*pi*(points/mm)}
N = len(x)
zbam = np.arange(-a,(a-2*a/N)+(2*a/N),2*a/N)
kx = np.transpose(zbam) 
ky = kx
[KX,KY,WT] = np.meshgrid(kx,ky,wt)

## OPERATORS
k0 = w * n_clad / C0
beta2 = 24.8e-27
beta3 = 23.3e-42
gamma = (2 * np.pi * n2 / wvl0)
delta = 0.01
NL1 = -1j*((k0*delta)/(core_radius**2))*((X**2)+(Y**2))

D1 = (0.5*1j/k0)*((-1j*(KX))**2+(-1j*(KY))**2)
D2 = ((-0.5*1j*beta2)*(-1j*(WT))**2)+((beta3/6)*(-1j*(WT))**3)
D = D1 + D2
s_imgper = (np.pi*core_radius)/np.sqrt(2*delta)
dz = s_imgper/48
DFR = np.exp(D*dz/2)

## INPUT 
flength = s_imgper * 10
fstep = flength/dz

p_don=20
t_fwhm = 100e-15
Ppeak = 1e9 #270*50e3 # W 180

data_s=np.zeros((480, 64, 64))
data_t=np.zeros((480, 1024))
fwhm=20e-6

print(f'The simulation starts.')
start_time = time.time()

for ulas2 in range(1000):
  coefs=np.random.rand(6)
  coefs=coefs/np.sum(coefs)
  A_transverse=cp.abs(modes(0,1,fwhm))*cp.exp(1j*(cp.angle(modes(0,2,fwhm))*coefs[1]+cp.angle(modes(0,3,fwhm))*coefs[2]+cp.angle(modes(1,1,fwhm))*coefs[3]+cp.angle(modes(1,2,fwhm))*coefs[4]+cp.angle(modes(2,1,fwhm))*coefs[5]+cp.angle(modes(0,1,fwhm))*coefs[0] ))
  pulse_time=cp.exp(-(T**2)/(2*(t_fwhm/2.35482)**2))
  A=( pulse_time.transpose() * A_transverse.transpose() ).transpose()
  A_tr_max =cp.max(cp.squeeze(cp.sum(cp.square(cp.abs(A)),axis=2)))
  A=A/cp.sqrt(A_tr_max)*cp.sqrt(Ppeak/(cp.pi*(fwhm**2)))
  Ain = A

  for ugur in range(int(fstep)):
      Einf=cp.fft.fftshift(cp.fft.fftn(Ain));
      Ein2=cp.fft.ifftn(cp.fft.ifftshift(Einf*DFR));
      Eout = Ein2;
      
      NL2 = 1j*gamma*cp.abs(Eout)**2;
      NL = NL1+NL2;
      Eout = Eout*cp.exp(NL*dz);
      
      Einf=cp.fft.fftshift(cp.fft.fftn(Eout));
      Ein2=cp.fft.ifftn(cp.fft.ifftshift(Einf*DFR));
      Ain =cp.multiply(cp_super_gauss2d,Ein2);
      Ain_cpu=Ain

      Ain_cpu=cp.square(cp.abs(Ain_cpu))

      ss =cp.squeeze(cp.sum(Ain_cpu,axis=2))
      tt =cp.sum(cp.squeeze(cp.sum(Ain_cpu,axis=0)),axis=0)

      data_s[ugur,:,:]=ss.get()
      data_t[ugur,:]=tt.get()

A_dis=A.get()
A_disp=np.squeeze(np.sum(A_dis,axis=2))


print(f'Computation time for forward calculation : {time.time() - start_time}')


domain = Domain(Lx, Ly, time_window, Nx, Ny, Nt, Nz, dz, precision=precision, device=device)
fiber = GRINFiber(domain, n_core, n_clad, beta2=beta2, n2=n2, radius=core_radius,)
input = Fields(domain, beam_radius, tfwhm=tfwhm, total_energy=total_energy, t_center=0,) # spatially gaussian and gaussian pulse
boundary = Boundary(domain, boundary_type)
config = SimConfig(center_wavelength=wvl0, dispersion=False, kerr=True, raman=False, self_steeping=False, num_save=num_save)
sim = Simulation(domain, fiber, input, boundary, config)

# plot fiber index profile
# fiber_indices = fiber.n.cpu().numpy()
# plot_index_profile(fiber_indices)

input_fields = input.fields.cpu().numpy()
plot_fields(input_fields, domain, wvl0=wvl0)
plt.show()

# Simulation
print(f'The simulation starts.')
start_time = time.time()
sim.run()
print(f'Computation time for forward calculation : {time.time() - start_time}')

output_fields = sim.fields.fields
output_fields = output_fields.cpu().numpy()
output_intensity = np.abs(output_fields)**2
# plot spatial profile, temporal profile, and spectrum

plot_fields(output_fields, domain, wvl0=wvl0)
plt.show()