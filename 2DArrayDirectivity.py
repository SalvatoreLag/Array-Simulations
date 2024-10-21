import numpy as np
import array_functions as af
import matplotlib.pyplot as plt
import map_functions as mf

# Define array
Nx = 15
Ny = 7
N = 10

# Element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_pattern(filename,1)
Delem = af.numerical_directivity(E**2,theta,phi)

# Different spacings plot
nspacings = 60
spacings = np.linspace(0.4,4,nspacings)

directivities1 = np.zeros(nspacings)
directivities2 = np.zeros(nspacings)
for idx,dx in enumerate(spacings):
    p = af.upa_positions(Nx,Ny,dx,dx)
    a = af.array_factor_grid(theta,phi,0,0,p)
    Aelem = (E**2)*(np.abs(a)**2)
    directivities1[idx] = af.numerical_directivity(Aelem,theta,phi)

    p = af.hex_positions(10,dx)
    a = af.array_factor_grid(theta,phi,0,0,p)
    Aelem = (E**2)*(np.abs(a)**2)
    directivities2[idx] = af.numerical_directivity(Aelem,theta,phi)

Aeff1 = directivities1/4/np.pi
Aenc1 = (Nx-1)*spacings*(Ny-1)*spacings
Ael = Nx*Ny*Delem/4/np.pi

Aeff2 = directivities2/4/np.pi
Aenc2 = N*spacings*N*np.sqrt(3)*spacings

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.loglog(spacings,Aeff1,'-',label='Effective area')
ax.loglog(spacings,Aenc1,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='Nelem x Aelem')
ax.grid()
ax.legend()
ax.set_xlabel('normalized spacing [-]')
ax.set_ylabel('normalized effective area [-]')
ax.set_title('Rectangular grid')
plt.savefig('./Outputs/AreaVsSpacingRect.png')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.loglog(spacings,Aeff2,'-',label='Effective area')
ax.loglog(spacings,Aenc2,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='Nelem x Aelem')
ax.grid()
ax.legend()
ax.set_xlabel('normalized spacing [-]')
ax.set_ylabel('normalized effective area [-]')
ax.set_title('Hexagonal grid')
plt.savefig('./Outputs/AreaVsSpacingHex.png')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.loglog(spacings,Aeff1,label='Rectangular grid')
ax.loglog(spacings,Aeff2,label='Hexagonal grid')
ax.grid()
ax.legend()
ax.set_xlabel('normalized spacing [-]')
ax.set_ylabel('normalized effective area [-]')
ax.set_title('Comparison')
plt.savefig('./Outputs/AreaVsSpacingComparison.png')
