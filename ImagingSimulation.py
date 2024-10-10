import array_positions as ap
import array_beam as ab
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import time

# Array
N = 5
diameter = 0.9
f0 = 5e9
l0 = 3e8/f0
p = ap.tpa_positionsV2(N,diameter)

# Get clean image
filename = './CleanImages/LowRes3.txt'
clean_img = np.loadtxt(filename)
nside = hp.npix2nside(len(clean_img))

# Define visible space
sky_center = [0,0,1]    
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
seen_img = clean_img[sky_pixels]
u,v,_ = hp.pix2vec(nside,sky_pixels)

# Plot clean image
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.tripcolor(u,v,seen_img)
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.savefig('./Outputs/clean_uv.png')

# Get element pattern
filename = './HealpixPatterns/Farfield120_5GHz.txt'
E = np.loadtxt(filename)
E = E[sky_pixels]

# Define scanning space and imaging
scan_fov = np.radians(30)
scan_pixels = hp.query_disc(nside,sky_center,scan_fov)
u0,v0,_ = hp.pix2vec(nside,scan_pixels)

dirty_img = np.zeros(len(scan_pixels))
for idx,pix in enumerate(scan_pixels):
    A = ab.array_pattern_loop(nside,sky_pixels,np.atleast_1d(pix),p)
    dirty_img[np.atleast_1d(idx)] = (E*A)@seen_img

# Plot dirty image
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.tripcolor(u0,v0,dirty_img)
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
plt.savefig('./Outputs/dirty_uv.png')