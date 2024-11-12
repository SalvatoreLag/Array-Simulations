import array_functions as af
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

#%% Array
N = 10
d = 0.9
p = af.hex_positions(N,d)
J = p.shape[0]

#%% Get clean image
filename = './CleanImages/LowRes3.txt'
clean_img = np.loadtxt(filename)
nside = hp.npix2nside(len(clean_img))

#%% Define visible space
sky_center = [0,0,1]    
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
l,m,_ = hp.pix2vec(nside,sky_pixels)

#%% Plot clean image
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(l,m,clean_img[sky_pixels])
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/clean_uv.png')

#%% Get element pattern
filename = './HealpixPatterns/Farfield120_5GHz.txt'
E = np.loadtxt(filename)
E = E[sky_pixels]
apparent_sky = E*clean_img[sky_pixels]

#%% Plot apparent sky
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(l,m,apparent_sky)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/apparentSky_uv.png')

#%% Define scanning space
scan_fov = np.radians()
scan_pixels = hp.query_disc(nside,sky_center,scan_fov)
u0,v0,_ = hp.pix2vec(nside,scan_pixels)

#%% Imaging (slow)
dirty_img = np.zeros(len(scan_pixels))
for idx,pix in enumerate(scan_pixels):
    A = np.abs(af.array_factor_hp(nside,sky_pixels,np.atleast_1d(pix),p))**2
    dirty_img[np.atleast_1d(idx)] = A@apparent_sky

# Plot dirty image
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u0,v0,dirty_img)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('l [-]')
ax.set_ylabel('m [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/dirty2_lm.png')

#%% Imaging (fast)
A = af.array_manifold(l,m,p)
x = A@apparent_sky
R = np.outer(x,np.conj(x))

# Matched beamformer
Id = np.zeros(len(scan_pixels))
for i in range(len(scan_pixels)):
    w = A[:,i]/J
    Id[i] = np.real(np.conj(w)@R@w)
plt.figure()
plt.tripcolor(l,m,Id)
plt.colorbar()
plt.savefig('./Outputs/dirty3_lm.png')
    