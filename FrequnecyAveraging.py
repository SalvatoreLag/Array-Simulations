import array_functions as af
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

# Array and stations
N = 10
diameter = 1.6

# Define pixels
l = np.arange(-1,1.01,0.01)
m = np.arange(-1,1.01,0.01)
L,M = np.meshgrid(l,m)
mask = np.where(L**2+M**2>1)
ll = L.flatten()
mm  = M.flatten()

l0 = 0.2
m0 = 0.2

# Frequency averaging array beam
f0 = 5e9
BW_norm = 1e9/f0
nf = 21
fs = np.linspace(1-BW_norm/2,1+BW_norm/2,nf)

A = np.zeros((nf,len(ll)))
for idx, f in enumerate(fs):
    p = af.upa_positions(N,f*diameter)
    A[idx,:] = np.abs(af.array_factor_lm(ll,mm,l0,m0,p))**2
Beam = np.mean(A,0)
Beam = Beam.reshape((len(l),-1))

# Plot
Beam_plot = 10*np.log10(Beam)-10*np.log10(np.max(Beam))
Beam_plot[mask] = np.inf
plt.pcolor(L,M,Beam_plot,cmap='turbo')
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar()
plt.savefig('./Outputs/FrequencyAveraging.png')
