import numpy as np
import array_functions as af
import matplotlib.pyplot as plt
import map_functions as mf
import scienceplots

plt.style.use(['science','ieee'])

# Define array
Nx = 10
dx = 0.5
p = af.upa_positions(Nx,1,dx,0)
a = np.ones(Nx)

# Analytical directivity
D1 = af.linear_directivity(a,dx)
print(10*np.log10(D1))

# Element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_pattern(filename,1)
T,P = np.meshgrid(theta,phi)
tt = T.flatten()
pp = P.flatten()

# Numerical directivity
A2 = np.abs(af.array_factor_tp(tt,pp,0,0,p))**2
A2 = A2.reshape((len(phi),-1))
D2 = af.numerical_directivity(A2,theta,phi)
print(10*np.log10(D2))

# Element directivity
D3 = af.numerical_directivity(E,theta,phi)
print(10*np.log10(D3))

# Array with real elements directivity
A4 = (E**2)*A2
D4 = af.numerical_directivity(A4,theta,phi)
print(10*np.log10(D4))

# Different spacings plot
nspacings = 100
spacings = np.linspace(0.3,8,nspacings)

directivities1 = np.zeros(nspacings)
directivities2 = np.zeros(nspacings)
for idx,dx in enumerate(spacings):
    p = af.upa_positions(Nx,1,dx,0)
    A = np.abs(af.array_factor_tp(tt,pp,0,0,p))**2
    A = A.reshape((len(phi),-1))
    Aelem = (E**2)*A
    
    directivities1[idx] = 10*np.log10(af.numerical_directivity(A,theta,phi))
    directivities2[idx] = 10*np.log10(af.numerical_directivity(Aelem,theta,phi))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.plot(spacings,directivities1,label='Isotropic elements')
ax.plot(spacings,directivities2,label='Real elements')
ax.grid()
ax.legend()
ax.set_xlabel('normalized spacing [-]')
ax.set_ylabel('Directivity [dB]')
plt.savefig('./Outputs/linearDirectivityVsSpacing.png')
