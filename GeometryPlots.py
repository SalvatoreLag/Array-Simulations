#%%
import numpy as np
import scipy as sp
import array_functions as af
import map_functions as mf
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

#%% Array grids
c0 = sp.constants.c
f0 = 5e9
l0 = c0/f0

dn = 1.1
N = 10

p1 = af.hex_positions(N,dn)*l0
p2 = af.upa_positions(N,N,dn,dn)*l0
d = dn*l0

#%% Plots
fig1 = plt.figure()
ax1 = fig1.subplots()
fig2 = plt.figure()
ax2 = fig2.subplots()

ax1.scatter(p1[:,0],p1[:,1],s=0.2)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.axis('square')
ax1.set_ylim(-0.05,0.64)
ax1.set_xlim(-0.05,0.64)
ax2.scatter(p2[:,0],p2[:,1],s=0.2)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.axis('square')
ax2.set_ylim(-0.05,0.64)
ax2.set_xlim(-0.05,0.64)

for i in range(p1.shape[0]):
    circle = plt.Circle((p1[i,0],p1[i,1]),d/2,color='k',fill=False)
    ax1.add_patch(circle)

for i in range(p2.shape[0]):
    circle = plt.Circle((p2[i,0],p2[i,1]),d/2,color='k',fill=False)
    ax2.add_patch(circle)

fig1.tight_layout()
fig1.savefig('./Outputs/StationLocsHex.png')
fig2.tight_layout()
fig2.savefig('./Outputs/StationLocsRec.png')
