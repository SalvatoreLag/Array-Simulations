import array_functions as af
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Array
N = 5
d = 0.9
f0 = 5e9
l0 = 3e8/f0

p = af.hex_positions(N,d)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.scatter(p[:,0]*l0,p[:,1]*l0)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.grid()
plt.savefig('./Outputs/positions.png')

# Define visible space
nside = 64
sky_center = [0,0,1]    
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
u,v,_ = hp.pix2vec(nside,sky_pixels)

# Define scanning space
scan_fov = 30
scan_pixels = np.atleast_1d(6000) #np.atleast_1d(hp.query_disc(nside,sky_center,scan_fov))

# Get matrix of steered beams
A_matrix = af.array_pattern_matrix(nside,sky_pixels,scan_pixels,p)

# Plot normalized steered beam
A_steered = 10*np.log10(A_matrix[0,:])-10*np.log10(max(A_matrix[0,:]))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,A_steered,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/AsteeredHP.png')

# Theta-phi grid approach
theta = np.radians(np.arange(91))
phi = np.radians(np.arange(361))
T,P = np.meshgrid(theta,phi)
U = np.sin(T)*np.cos(P)
V = np.sin(T)*np.sin(P)
theta0,phi0 = hp.pix2ang(nside,scan_pixels)

A_grid = af.array_pattern_grid(theta,phi,theta0,phi0,p)
A_grid = 10*np.log10(A_grid/np.max(A_grid))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.pcolor(U,V,A_grid,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/AsteeredGrid.png')
