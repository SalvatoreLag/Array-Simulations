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

#%% Set figures of merit
Nel = 14000
As = 1.92

#%% Stations
N = 10
# Ny = np.round(N/np.sqrt(3),decimals=0).astype(int)
# d = np.sqrt(As)/wl/(N-1)
# d = np.sqrt(As/(N-1)/(Ny-1))/wl
d = 1.6
p = af.hex_positions(N,d)
J = p.shape[0]
print(J)

# Aps = ((N-1)*d*wl)**2
# Aps = (N-1)*(Ny-1)*(d*wl)**2
# print(Aps)

#%% Interferometer
# nStations = np.round(Nel/J)
nStations = 20
print(nStations)
psi_max = 30
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

# %%
plt.pcolor(L,M,10*np.log10(Av[20,:].reshape((len(l),-1))),cmap='turbo',vmin=-40)
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar(label='[dB]')
# plt.show()

# %%
d = 1.6
N = 10
p = af.hex_positions(N,d)
# p = af.hex_positions(N,d)
J = p.shape[0]

alpha = np.radians(60)

#%%
ns1 = int(alpha*N)
# ns1 = 20

for ns in range(ns1,ns1+1):
    dalpha = alpha/ns
    stationRots = np.arange(0,alpha,dalpha)

    Av = np.zeros((ns,len(ll)))
    for idx,psi in enumerate(stationRots):
        R = np.array([[np.cos(psi),-np.sin(psi)],
                    [np.sin(psi), np.cos(psi)]]).T
        pRot = p@R
        a = af.array_factor_lm(ll,mm,l0,m0,pRot)
        Av[idx,:] = np.abs(a)**2

    # Compute average power pattern
    avg_beam = np.mean(sp.linalg.khatri_rao(Av,np.conj(Av)),0)
    avg_beam = 10*np.log10(avg_beam/np.max(avg_beam))
    avg_beam = avg_beam.reshape((len(l),-1))

    avg_beam[mask] = np.inf

    plt.pcolor(L,M,avg_beam,cmap='turbo',vmin=-30,vmax=-10)
    plt.xlabel('l [-]')
    plt.ylabel('m [-]')
    plt.colorbar(label='[dB]')
    plt.show()

# %%
