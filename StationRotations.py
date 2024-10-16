import array_functions as af
import numpy as np
import healpy as hp
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
A = np.zeros((nStations,nPixels))
for idx,psi in enumerate(stationRots):
    R = np.array([[np.cos(psi),-np.sin(psi)],
                  [np.sin(psi), np.cos(psi)]]).T
    pRot = p@R
    A[idx,:] = af.array_pattern_loop(nside,sky_pixels,scan_pixel,pRot)

# Correlation between stations and PSF
Beams = np.zeros((nStations**2,nPixels))
Aconj = np.conj(A)
for idx1 in range(nStations):
    for idx2 in range(nStations):
        if idx1>=idx2:
            Beams[idx1+idx2,:] = A[idx1,:]*Aconj[idx2,:]
        else:
            Beams[idx1+idx2,:] = Aconj[idx1,:]*A[idx2,:]
PSF = np.mean(Beams,0)

# Plot PSF
PSF_plot = 10*np.log10(PSF)-10*np.log10(np.max(PSF))
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
im = ax.tripcolor(u,v,PSF_plot,vmin=-40)
ax.axis('equal')
ax.set(xlim=(-1,1),ylim=(-1,1))
ax.set_xlabel('u [-]')
ax.set_ylabel('v [-]')
fig.colorbar(im,ax=ax)
plt.savefig('./Outputs/StationRotations.png')
