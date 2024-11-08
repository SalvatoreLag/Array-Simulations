#%%
import array_functions as af
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

#%% Array and stations
N = 10
diameter = 1.6

# Define pixels
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

# Frequency averaging array beam
f0 = 5e9
BW_norm = 1e9/f0
nf = 21
fs = np.linspace(1-BW_norm/2,1+BW_norm/2,nf)

A = np.zeros((nf,len(ll)))
for idx, f in enumerate(fs):
    p = af.hex_positions(N,f*diameter)
    A[idx,:] = np.abs(af.array_factor_lm(ll,mm,l0,m0,p))**2
Beam = np.mean(A,0)
Beam = Beam.reshape((len(l),-1))

#%% Plot
Beam_plot = 10*np.log10(Beam)-10*np.log10(np.max(Beam))
Beam_plot[mask] = np.inf
plt.pcolor(L,M,Beam_plot,cmap='turbo',vmin=-20)
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar(label='[dB]')
# plt.savefig('./Outputs/FrequencyAveraging.png')
