import array_positions as ap
import array_beam as ab
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Array
N = 5
diameter = 0.9

p = ap.tpa_positionsV2(N,diameter)

# Clean image
filename = './CleanImages/LowRes.txt'
clean_img = np.loadtxt(filename)

nside = hp.npix2nside(len(clean_img))
sky_center = [0,0,1]
sky_fov = np.radians(90)
sky_pixels = hp.query_disc(nside,sky_center,sky_fov)
u,v,_ = hp.pix2vec(nside,sky_pixels)

fig,ax = plt.subplots()
ax.tricontourf(u,v,clean_img[sky_pixels])
plt.show()

hp.orthview(clean_img,rot=[-90,90,0])
plt.show()

# Element pattern
filename = './HealpixPatterns/Farfield120_5GHz.txt'
E = np.loadtxt(filename)
E = E
E_plot = 20*np.log10(E/np.max(E))

fig,ax = plt.subplots()
ax.tricontourf(u,v,E_plot[sky_pixels])
plt.show()

hp.orthview(E,rot=[0,90,0])
plt.show()


