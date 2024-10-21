import array_functions as af
import numpy as np
import healpy as hp
import scipy.linalg as sp
import matplotlib.pyplot as plt

# Array and stations
N = 5
diameter = 0.9
p = af.hex_positions(N,diameter)
nStations = 36
stationRots = np.radians(np.linspace(0,180,nStations))

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

# Station rotations array beam
a = np.zeros((nStations,nPixels))
for idx,psi in enumerate(stationRots):
    R = np.array([[np.cos(psi),-np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]]).T
    pRot = p@R
    a[idx,:] = af.array_factor(nside,sky_pixels,scan_pixel,pRot)

# Correlation between stations and PSF
station_beams = sp.khatri_rao(a,np.conj(a))
average_beam = np.mean(station_beams,0)

# Plot PSF
beam_plot = 10*np.log10(average_beam)-10*np.log10(np.max(average_beam))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,beam_plot,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/StationRotations.png')
