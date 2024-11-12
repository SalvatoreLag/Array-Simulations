#%%
import numpy as np
import scipy as sp
import array_functions as af
import matplotlib.pyplot as plt
import map_functions as mf
import scienceplots

plt.style.use(['science','ieee'])

#%% Define arrays
Nx = 15
Ny = 7
N = 10

#%% Element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_pattern(filename,1)
T,P = np.meshgrid(theta,phi)
tt = T.flatten()
pp = P.flatten()
l0 = sp.constants.c/5e9

Delem = af.numerical_directivity(E**2,theta,phi)

#%% Different spacings plot
nspacings = 40
spacings = np.linspace(0.4,3,nspacings)

directivities1 = np.zeros(nspacings)
directivities2 = np.zeros(nspacings)
for idx,dx in enumerate(spacings):
    p = af.upa_positions(Nx,Ny,dx,dx)
    a = af.array_factor_tp(tt,pp,0,0,p)
    a = a.reshape((len(phi),-1))
    Aelem = (E**2)*(np.abs(a)**2)
    directivities1[idx] = af.numerical_directivity(Aelem,theta,phi)

    p = af.hex_positions(N,dx)
    a = af.array_factor_tp(tt,pp,0,0,p)
    a = a.reshape((len(phi),-1))
    Aelem = (E**2)*(np.abs(a)**2)
    directivities2[idx] = af.numerical_directivity(Aelem,theta,phi)

Aeff1 = directivities1/4/np.pi*l0**2
Aenc1 = (Nx-1)*spacings*(Ny-1)*spacings*l0**2
Ael = Nx*Ny*Delem/4/np.pi*l0**2

Aeff2 = directivities2/4/np.pi*l0**2
Aenc2 = (N-1)*spacings*(np.round(Nx/np.sqrt(3),decimals=0).astype(int)-1)*spacings*l0**2

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(spacings,Aeff1,'-',label='Effective area')
ax.loglog(spacings,Aenc1,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='N x $A_e$')
ax.grid()
ax.legend()
ax.set_xlabel('Normalized elements spacing [-]')
ax.set_ylabel('Area [$m^2$]')
plt.savefig('./Outputs/AreaVsSpacingRect.png')

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(spacings,Aeff2,'-',label='Effective area')
ax.loglog(spacings,Aenc2,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='N x $A_e$')
ax.grid()
ax.legend()
ax.set_xlabel('Normalized elements spacing [-]')
ax.set_ylabel('Area [$m^2$]')
plt.savefig('./Outputs/AreaVsSpacingHex.png')

#%% Area vs number of elements

l0 = sp.constants.c/(5e9)

iterations = 10
areas_rec = np.zeros((3,iterations))
areas_hex = np.zeros((3,iterations))
Js_rec = np.zeros(iterations)
Js_hex = np.zeros(iterations)

FoVs = [60,90,120]
ds = [1.6,1.12,0.9]

for i,tup in enumerate(zip(FoVs,ds)):
    filename = f'./ElementPatterns/Farfield{tup[0]}_5GHz.txt'
    E, theta, phi = mf.import_pattern(filename,1)
    T,P = np.meshgrid(theta,phi)
    tt = T.flatten()
    pp = P.flatten()
    
    t0 = np.radians(20)
    p0 = np.radians(125)
    
    E = E**2
    for j in range(iterations):
        p = af.hex_positions(N=j+10,d=tup[1])
        Js_hex[j] = p.shape[0]
        Apattern = np.abs(af.array_factor_tp(tt,pp,t0,p0,p))**2
        Apattern = Apattern.reshape((len(phi),-1))
        Apattern = E*Apattern
        directivity = af.numerical_directivity(Apattern,theta,phi)
        areas_hex[i,j] = directivity/4/np.pi*l0**2
        p = af.upa_positions(Nx=j+10,Ny=j+10,dx=tup[1],dy=tup[1])
        Js_rec[j] = p.shape[0]
        Apattern = np.abs(af.array_factor_tp(tt,pp,t0,p0,p))**2
        Apattern = Apattern.reshape((len(phi),-1))
        Apattern = E*Apattern
        directivity = af.numerical_directivity(Apattern,theta,phi)
        areas_rec[i,j] = directivity/4/np.pi*l0**2

plt.figure()
plt.grid()
plt.plot(Js_rec,areas_rec[0,:],linestyle='-',color='b',marker='x',markersize=3,label='A rectangular')
plt.plot(Js_hex,areas_hex[0,:],linestyle='--',color='r',marker='s',markersize=2,label='A triangular')
plt.plot(Js_rec,areas_rec[1,:],linestyle='-.',color='b',marker='^',markersize=2,label='B rectangular')
plt.plot(Js_hex,areas_hex[1,:],linestyle=':',color='r',marker='*',markersize=3,label='B triangular')
plt.plot(Js_rec,areas_rec[2,:],linestyle='-',color='b',marker='o',markersize=2,label='C rectangular')
plt.plot(Js_hex,areas_hex[2,:],linestyle='--',color='r',marker='d',markersize=2,label='C triangular')
plt.xlabel('Number of elements [-]')
plt.ylabel('Effective area [$m^2$]')
plt.legend()
plt.savefig('./Outputs/AreaVsNumberScan.png')
