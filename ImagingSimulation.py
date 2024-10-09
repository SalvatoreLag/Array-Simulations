import array_positions as ap
import array_beam as ab
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Array
N = 5
diameter = 0.9

p = ap.tpa_positionsV2(N,diameter)
fig,ax = plt.subplots()
ax.scatter(p[:,0],p[:,1])
plt.savefig('./Outputs/positions.png')

# Clean image
filename = './CleanImages/LowRes2.txt'
clean_img = np.loadtxt(filename)

nside = hp.npix2nside(len(clean_img))
sky_center = [0,0,1]
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
u,v,_ = hp.pix2vec(nside,sky_pixels)

fig,ax = plt.subplots()
ax.tricontourf(u,v,clean_img[sky_pixels])
plt.savefig('./Outputs/clean_uv.png')

hp.orthview(clean_img,rot=[-90,90,0])
plt.savefig('./Outputs/clean_orth.png')

# Element pattern
filename = './HealpixPatterns/Farfield120_5GHz.txt'
E = np.loadtxt(filename)
E = E
E_plot = 20*np.log10(E/np.max(E))

fig,ax = plt.subplots()
ax.tricontourf(u,v,E_plot[sky_pixels])
plt.savefig('./Outputs/pattern_uv.png')

hp.orthview(E,rot=[0,90,0])
plt.savefig('./Outputs/pattern_orth.png')

# Array pattern and imaging
# scan_fov = 30
# scan_pixels = hp.query_disc(nside,sky_center,scan_fov)
# A = ab.array_pattern(nside,sky_pixels,scan_pixels,p)
# dirty_img = A@clean_img[sky_pixels]

dirty_img = np.ones_like(clean_img)*np.inf
for pix in sky_pixels:
    pix = np.atleast_1d(pix)
    A = ab.array_pattern(nside,sky_pixels,pix,p)
    dirty_img[pix] = A@clean_img[sky_pixels]

fig,ax = plt.subplots()
ax.tricontourf(u,v,dirty_img[sky_pixels])
plt.savefig('./Outputs/dirty_uv.png')

hp.orthview(dirty_img,rot=[0,90,0])
plt.savefig('./Outputs/dirty_orth.png')



