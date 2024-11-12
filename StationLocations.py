'''Compute the baselines and PSF for an interferometer whose stations
form a uniform circular array.'''

#%% Imports
import array_functions as af
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

#%% Define stations locations
rp = 10
phip = np.linspace(0,2*np.pi,25)
px = rp*np.cos(phip)
py = rp*np.sin(phip)
p1 = np.stack((px,py)).T

plt.figure()
plt.scatter(px,py,np.ones_like(px)*4)
plt.axis('equal')
plt.xlabel('x [-]')
plt.ylabel('y [-]')
plt.savefig('./Outputs/UniformCircular.png')

#%% Define baselines in uv space
u0 = (np.tile(np.atleast_2d(px).T,(1,len(px)))-np.tile(px,(len(px),1))).flatten()
v0 = (np.tile(np.atleast_2d(py).T,(1,len(py)))-np.tile(py,(len(py),1))).flatten()
u = np.stack((u0,v0)).T

plt.figure()
plt.scatter(u0,v0,np.ones_like(u0)*2)
plt.axis('equal')
plt.xlabel('u [-]')
plt.ylabel('v [-]')
plt.savefig('./Outputs/Baselines.png')

#%% Plot PSF
l = np.arange(-1,1.01,0.01)
m = np.arange(-1,1.01,0.01)
L,M = np.meshgrid(l,m)
mask = np.where(L**2+M**2>1)
ll = L.flatten()
mm  = M.flatten()

a1 = np.real(af.array_factor_lm(ll,mm,0,0,u)).reshape((len(l),-1))
a1 = 10*np.log10(a1/np.max(a1))
a1[mask] = np.inf

plt.figure()
plt.pcolor(L,M,a1,cmap='turbo',vmin=-20)
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.colorbar(label='[dB]')
plt.savefig('./Outputs/PSF.png')
