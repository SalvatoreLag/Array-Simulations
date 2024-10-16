import numpy as np
import array_functions as af
import healpy as hp
import matplotlib.pyplot as plt

# Define array
Nx = 10
Ny = 1
dx = 0.5
dy = 0
p = af.upa_positions(Nx,Ny,dx,dy)

a = np.ones(Nx)

# Analytical directivity
D1 = af.linear_directivity(a,dx)

# Numerical directivity
theta = np.radians(np.arange(91))
phi = np.radians(np.arange(361))
T,P = np.meshgrid(theta,phi)
U = np.sin(T)*np.cos(P)
V = np.sin(T)*np.sin(P)

A = af.array_pattern_grid(theta,phi,0,0,p)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,A_steered,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/Asteered.png')

# With element pattern

