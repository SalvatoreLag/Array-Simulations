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
print(D1)

# Numerical directivity
theta = np.radians(np.arange(0,91))
phi = np.radians(np.arange(0,361))
T,P = np.meshgrid(theta,phi)
U = np.sin(T)*np.cos(P)
V = np.sin(T)*np.sin(P)

A = af.array_pattern_grid(theta,phi,0,0,p)
dt = np.radians(1)
dp = np.radians(1)
D2 = af.numerical_directivity(A,theta,phi,dt,dp)
print(D2)

# With element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_halfPattern(filename,1)


