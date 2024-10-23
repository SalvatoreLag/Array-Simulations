import array_functions as af
import numpy as np
import healpy as hp
import scipy.linalg as sp
import matplotlib.pyplot as plt

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

# Compute voltage beam pattern
Av = np.zeros((nStations,len(l),len(m)))
for idx,psi in enumerate(stationRots):
    R = np.array([[np.cos(psi),-np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]]).T
    pRot = p@R
    a = af.array_factor_lmgrid(l,m,0.2,0.2,pRot)
    Av[idx,:,:] = np.abs(a)**2/J

# Compute average power pattern
baselines = np.zeros((nStations**2,len(l),len(m)))
for idx1 in range(nStations):
    for idx2 in range(nStations):
        if idx1 >= idx2:
            baselines[(idx1*nStations)+idx2,:,:] = Av[idx1,:,:]*np.conj(Av[idx2,:,:])
        else:
            baselines[(idx1*nStations)+idx2,:,:] = np.conj(Av[idx1,:,:])*Av[idx2,:,:]

average_beam = np.mean(baselines,0)/J**2

average_beam = 10*np.log10(average_beam)

average_beam[mask] = np.inf

# Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.pcolor(L,M,average_beam,cmap='turbo')
ax.set_xlabel('l [-]')
ax.set_ylabel('m [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/StationRotations.png')
