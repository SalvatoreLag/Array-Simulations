import array_functions as af
import numpy as np
import healpy as hp
import scipy as sp
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

# Array and stations
nStations = 10
psi_max = 60
dpsi = psi_max/nStations
stationRots = np.radians(np.arange(0,psi_max,dpsi))

N = 8
p = af.hex_positions(N,1.6)
J = p.shape[0]

# Define pixels
l = np.arange(-1,1.01,0.01)
m = np.arange(-1,1.01,0.01)
L,M = np.meshgrid(l,m)
mask = np.where(L**2+M**2>1)
ll = L.flatten()
mm  = M.flatten()

# Compute voltage beam patterns
Av = np.zeros((nStations,len(ll)))
for idx,psi in enumerate(stationRots):
    R = np.array([[np.cos(psi),-np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]]).T
    pRot = p@R
    a = af.array_factor_lm(ll,mm,0.2,0.2,pRot)
    Av[idx,:] = np.abs(a)**2/J**2

# Compute average power pattern
avg_beam = np.mean(sp.linalg.khatri_rao(Av,np.conj(Av)),0)
avg_beam = 10*np.log10(avg_beam)
avg_beam = avg_beam.reshape((len(l),-1))

avg_beam[mask] = np.inf

# Plot
plt.pcolor(L,M,avg_beam,cmap='turbo')
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar()
plt.savefig('./Outputs/StationRotations.png')
