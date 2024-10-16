import numpy as np
import array_functions as af
import matplotlib.pyplot as plt
import map_functions as mf

# Define array
Nx = 4
Ny = 2

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
    A = af.array_pattern_grid(theta,phi,0,0,p)
    Aelem = (E**2)*A
    directivities1[idx] = af.numerical_directivity(Aelem,theta,phi)

    p = af.hex_positions(3,dx)
    A = af.array_pattern_grid(theta,phi,0,0,p)
    Aelem = (E**2)*A
    directivities2[idx] = af.numerical_directivity(Aelem,theta,phi)

Aeff1 = directivities1/4/np.pi
Aenc1 = 3*spacings*1*spacings
Ael = 8*Delem/4/np.pi

Aeff2 = directivities2/4/np.pi
Aenc2 = 3*spacings*np.sqrt(3)*spacings

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
