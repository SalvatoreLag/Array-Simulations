import numpy as np
import array_functions as af
import healpy as hp
import matplotlib.pyplot as plt
import map_functions as mf

# Define array
Nx = 10
Ny = 1
dx = 0.5
dy = 0
p = af.upa_positions(Nx,Ny,dx,dy)

a = np.ones(Nx)

# Analytical directivity
D1 = af.linear_directivity(a,dx)
print(10*np.log10(D1))

# Element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_pattern(filename,1)
Ehalf = E[:,:91]
theta_half = theta[:91]

# Numerical directivity
A = af.array_pattern_grid(theta,phi,0,0,p)
D2 = af.numerical_directivity(A,theta,phi)
print(10*np.log10(D2))


