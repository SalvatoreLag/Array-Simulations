#%%
import array_functions as af
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import map_functions as mf
import scienceplots

plt.style.use(['science','ieee'])

#%% Arrays
N = 10
d = 1.1
p2 = af.hex_positions(N,d)
p1 = af.upa_positions(N,N,d,d)

#%% Element pattern
filename = './ElementPatterns/Farfield90_5GHz.txt'
E,theta,phi = mf.import_pattern(filename,1)
half = 91
theta_half = theta[:half]
Ehalf = E[:,:half]
T,P = np.meshgrid(theta_half,phi)
L = np.sin(T)*np.cos(P)
M = np.sin(T)*np.sin(P)

#%% Array pattern - Theta-phi grid approach
theta0 = np.radians(20)
phi0 = np.radians(125)

tt = T.reshape(-1)
pp = P.reshape(-1)
Arec = np.abs(af.array_factor_tp(tt,pp,theta0,phi0,p1))**2
Arec = Arec.reshape((len(phi),-1))
Ahex = np.abs(af.array_factor_tp(tt,pp,theta0,phi0,p2))**2
Ahex = Ahex.reshape((len(phi),-1))

fig = plt.figure(figsize=(7.16,2.9))
ax1,ax2 = fig.subplots(1,2)

img1 = ax1.pcolor(L,M,10*np.log10(Arec/np.max(Arec)),vmin=-30,cmap='turbo')
ax1.set_xlabel('l [-]')
ax1.set_ylabel('m [-]')
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)

img2 = ax2.pcolor(L,M,10*np.log10(Ahex/np.max(Ahex)),vmin=-30,cmap='turbo')
ax2.set_xlabel('l [-]')
ax2.set_ylabel('m [-]')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)

fig.colorbar(img1,ax=ax1,label='[dB]')   
fig.colorbar(img2,ax=ax2,label='[dB]')
fig.tight_layout()
fig.savefig('./Outputs/ArrayPatterns.png')

#%% With real elements
Arec = (Ehalf**2)*Arec
Ahex = (Ehalf**2)*Ahex

fig = plt.figure(figsize=(7.16,2.9))
ax1,ax2 = fig.subplots(1,2)

img1 = ax1.pcolor(L,M,10*np.log10(Arec/np.max(Arec)),vmin=-30,cmap='turbo')
ax1.set_xlabel('l [-]')
ax1.set_ylabel('m [-]')
ax1.set_xlim(-1,1)
ax1.set_ylim(-1,1)

img2 = ax2.pcolor(L,M,10*np.log10(Ahex/np.max(Ahex)),vmin=-30,cmap='turbo')
ax2.set_xlabel('l [-]')
ax2.set_ylabel('m [-]')
ax2.set_xlim(-1,1)
ax2.set_ylim(-1,1)

fig.colorbar(img1,ax=ax1,label='[dB]')   
fig.colorbar(img2,ax=ax2,label='[dB]')
fig.tight_layout()
fig.savefig('./Outputs/ArrayPatternsElements.png')

plt.figure()
plt.pcolor(L,M,10*np.log10(Arec/np.max(Arec)),vmin=-30,cmap='turbo')
plt.xlabel('l [-]')
plt.ylabel('m [-]')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.colorbar(label='[dB]')
plt.savefig('./Outputs/ArrayPatternsElementsRec.png')

#%% Array pattern - Healpix approach
# Define visible space
nside = 64
sky_center = [0,0,1]    
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
u,v,_ = hp.pix2vec(nside,sky_pixels)

# Define scanning space
scan_pixels = np.atleast_1d(hp.ang2pix(nside,theta0,phi0))

# Get matrix of steered beams
A_matrix = np.abs(af.array_factor_hpmatrix(nside,sky_pixels,scan_pixels,p1))**2

# Plot normalized steered beam
A_plot = 10*np.log10(A_matrix[0,:])-10*np.log10(max(A_matrix[0,:]))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,A_plot,vmin=-30,cmap='turbo')
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/AsteeredHP.png')

# With real elements
filename = './HealpixPatterns/Farfield120_5GHz.txt'
E = np.loadtxt(filename)
E = E[sky_pixels]
A_elem = (E**2)*A_matrix[0,:]

A_plot = 10*np.log10(A_elem/np.max(A_elem))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,A_plot,vmin=-30,cmap='turbo')
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.colorbar(im,ax=ax)
plt.savefig('./Outputs/AElementHP.png')
