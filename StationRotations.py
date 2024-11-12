'''Plot the average station pattern resulting from rotations
of identical phased-array stations.'''

#%% Imports
import array_functions as af
import numpy as np
import healpy as hp
import scipy as sp
import matplotlib.pyplot as plt
import map_functions as mf
import scienceplots

plt.style.use(['science','ieee'])

#%% Define pixels
l = np.arange(-1,1.01,0.01)
m = np.arange(-1,1.01,0.01)
L,M = np.meshgrid(l,m)
mask = np.where(L**2+M**2>1)
ll = L.flatten()
mm  = M.flatten()

t0 = np.radians(20)
p0 = np.radians(125)
l0 = np.sin(t0)*np.cos(p0)
m0 = np.sin(t0)*np.sin(p0)

wl = sp.constants.c/5e9

#%% Element
filename = './ElementPatterns/Farfield60_5GHz.txt'
E,theta,phi = mf.import_pattern(filename,1)

Del = af.numerical_directivity(E**2,theta,phi)
Ael = Del/4/np.pi*wl**2
print(Ael)
Nel = 100/Ael
print(Nel)

#%% Stations
N = 10
d = 1.6
p = af.hex_positions(N,d)
J = p.shape[0]
print(J)

#%% Interferometer
nStations = 20
print(nStations)
psi_max = 60
dpsi = psi_max/nStations
stationRots = np.radians(np.arange(0,psi_max,dpsi))

#%% Compute voltage beam patterns
Av = np.zeros((int(nStations),len(ll)))
for idx,psi in enumerate(stationRots):
    R = np.array([[np.cos(psi),-np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]]).T
    pRot = p@R
    a = af.array_factor_lm(ll,mm,l0,m0,pRot)
    Av[idx,:] = np.abs(a)**2/J**2

# Compute average power pattern
avg_beam = np.mean(sp.linalg.khatri_rao(Av,np.conj(Av)),0)
avg_beam = 10*np.log10(avg_beam)
avg_beam = avg_beam.reshape((len(l),-1))

avg_beam[mask] = np.inf

#%% Plot
plt.pcolor(L,M,avg_beam,cmap='turbo',vmin=-40)
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar(label='[dB]')
plt.savefig('./Outputs/StationRotationsHex20_30.png')
# plt.show()
