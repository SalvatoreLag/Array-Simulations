import array_beam as ab
import array_positions as ap
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Array and stations
N = 5
diameter = 0.9

# Define visible space
nside = 64
sky_center = [0,0,1]    
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
nPixels = len(sky_pixels)
u,v,_ = hp.pix2vec(nside,sky_pixels)

# Define scan angle
theta0 = np.radians(45)
phi0 = np.radians(45)
scan_pixel = np.atleast_1d(hp.ang2pix(nside,theta0,phi0))

# Frequency averaging array beam
f0 = 5e9
BW_norm = 0.5e9/f0
nf = 21
fs = np.linspace(1-BW_norm/2,1+BW_norm/2,nf)

A = np.zeros((nf,nPixels))
for idx, f in enumerate(fs):
    p = ap.hex_positions(N,f)
    A[idx,:] = ab.array_pattern_loop(nside,sky_pixels,scan_pixel,p)
Beam = np.mean(A,0)

# Plot
Beam_plot = 10*np.log10(Beam)-10*np.log10(np.max(Beam))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,Beam_plot,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/FrequencyAveraging.png')
