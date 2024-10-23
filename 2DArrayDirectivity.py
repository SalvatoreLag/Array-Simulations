import numpy as np
import scipy as sp
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
    a = af.array_factor_tpgrid(theta,phi,0,0,p)
    Aelem = (E**2)*(np.abs(a)**2)
    directivities1[idx] = af.numerical_directivity(Aelem,theta,phi)

    p = af.hex_positions(10,dx)
    a = af.array_factor_tpgrid(theta,phi,0,0,p)
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
plt.savefig('./Outputs/AreaVsSpacingHex.png')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.loglog(spacings,Aeff1,label='Rectangular grid')
ax.loglog(spacings,Aeff2,label='Hexagonal grid')
ax.grid()
ax.legend()
ax.set_xlabel('normalized spacing [-]')
ax.set_ylabel('normalized effective area [-]')
plt.savefig('./Outputs/AreaVsSpacingComparison.png')

# Area vs Number of elements
f0 = 6
l0 = sp.constants.c/(f0*1e9)

iterations = 20
areas_rec = np.zeros((3,iterations))
areas_hex = np.zeros((3,iterations))
Js_rec = np.zeros(iterations)
Js_hex = np.zeros(iterations)

FoVs = [60,90,120]
ds = [1.6,1.12,0.9]

for i,tup in enumerate(zip(FoVs,ds)):
    filename = f'./ElementPatterns/Farfield{tup[0]}_{f0}GHz.txt'
    E, theta, phi = mf.import_pattern(filename,1)
    E = E**2
    Delem = af.numerical_directivity(E**2,theta,phi)
    print(f'Element directivity: {10*np.log10(Delem)} dB')
    for j in range(iterations):
        p = af.hex_positions(N=j+3,d=tup[1])
        Js_hex[j] = p.shape[0]
        Apattern = E*np.abs(af.array_factor_tpgrid(theta,phi,0,0,p))**2
        directivity = af.numerical_directivity(Apattern,theta,phi)
        areas_hex[i,j] = directivity/4/np.pi*l0**2
        p = af.upa_positions(Nx=j+3,Ny=j+3,dx=tup[1],dy=tup[1])
        Js_rec[j] = p.shape[0]
        Apattern = E*np.abs(af.array_factor_tpgrid(theta,phi,0,0,p))**2
        directivity = af.numerical_directivity(Apattern,theta,phi)
        areas_rec[i,j] = directivity/4/np.pi*l0**2

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(Js_hex,areas_hex[0,:],label=f'd={ds[0]},hexagonal',ls='-',color='blue')
ax.plot(Js_rec,areas_rec[0,:],label=f'd={ds[0]},rectangular',ls='--',color='blue')
ax.plot(Js_hex,areas_hex[1,:],label=f'd={ds[1]},hexagonal',ls='-',color='orange')
ax.plot(Js_rec,areas_rec[1,:],label=f'd={ds[1]},rectangular',ls='--',color='orange')
ax.plot(Js_hex,areas_hex[2,:],label=f'd={ds[2]},hexagonal',ls='-',color='green')
ax.plot(Js_rec,areas_rec[2,:],label=f'd={ds[2]},rectangular',ls='--',color='green')
ax.grid()
ax.set_xlabel('Number of elements [-]')
ax.set_ylabel('Effective area [m^2]')
ax.legend()
plt.savefig(f'./Outputs/AreaVsNumberComparison{f0}.png')
