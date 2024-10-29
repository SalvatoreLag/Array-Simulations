import numpy as np
import scipy as sp
import array_functions as af
import matplotlib.pyplot as plt
import map_functions as mf
import scienceplots

plt.style.use(['science','ieee'])

# Define arrays
Nx = 15
Ny = 7
N = 10

# Element pattern
filename = './ElementPatterns/Farfield120_5GHz.txt'
E, theta, phi = mf.import_pattern(filename,1)
T,P = np.meshgrid(theta,phi)
tt = T.flatten()
pp = P.flatten()

Delem = af.numerical_directivity(E**2,theta,phi)

# Different spacings plot
nspacings = 40
spacings = np.linspace(0.4,2,nspacings)

directivities1 = np.zeros(nspacings)
directivities2 = np.zeros(nspacings)
for idx,dx in enumerate(spacings):
    p = af.upa_positions(Nx,Ny,dx,dx)
    a = af.array_factor_tp(tt,pp,0,0,p)
    a = a.reshape((len(phi),-1))
    Aelem = (E**2)*(np.abs(a)**2)
    directivities1[idx] = af.numerical_directivity(Aelem,theta,phi)

    # fig, (ax1,ax2) = plt.subplots(1,2)
    # ax1.scatter(p[:,0],p[:,1],np.ones(p.shape[0]))
    # for xc,yc in zip(p[:,0],p[:,1]):
    #     circle = plt.Circle((xc,yc),dx/2,color='b',fill=False)
    #     ax1.add_patch(circle)
    # ax1.axis('equal')

    p = af.hex_positions(N,dx)
    a = af.array_factor_tp(tt,pp,0,0,p)
    a = a.reshape((len(phi),-1))
    Aelem = (E**2)*(np.abs(a)**2)
    directivities2[idx] = af.numerical_directivity(Aelem,theta,phi)

    # ax2.scatter(p[:,0],p[:,1],np.ones(p.shape[0]))
    # for xc,yc in zip(p[:,0],p[:,1]):
    #     circle = plt.Circle((xc,yc),dx/2,color='b',fill=False)
    #     ax2.add_patch(circle)
    # ax2.axis('equal')
    # plt.show()

Aeff1 = directivities1/4/np.pi
Aenc1 = (Nx-1)*spacings*(Ny-1)*spacings
Ael = Nx*Ny*Delem/4/np.pi

Aeff2 = directivities2/4/np.pi
Aenc2 = (N-1)*spacings*(np.round(Nx/np.sqrt(3),decimals=0).astype(int)-1)*spacings

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(spacings,Aeff1,'-',label='Effective area')
ax.loglog(spacings,Aenc1,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='Nelem x Aelem')
ax.grid()
ax.legend()
ax.set_xlabel('Normalized spacing [-]')
ax.set_ylabel('Normalized effective area [-]')
plt.savefig('./Outputs/AreaVsSpacingRect.png')

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(spacings,Aeff2,'-',label='Effective area')
ax.loglog(spacings,Aenc2,'--',label='Enclosed area')
ax.loglog(spacings,np.ones(nspacings)*Ael,':',label='Nelem x Aelem')
ax.grid()
ax.legend()
ax.set_xlabel('Normalized spacing [-]')
ax.set_ylabel('Normalized effective area [-]')
plt.savefig('./Outputs/AreaVsSpacingHex.png')

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot()
# ax.loglog(spacings,Aeff1,label='Rectangular grid')
# ax.loglog(spacings,Aeff2,label='Hexagonal grid')
# ax.grid()
# ax.legend()
# ax.set_xlabel('Normalized spacing [-]')
# ax.set_ylabel('Normalized effective area [-]')
# plt.savefig('./Outputs/AreaVsSpacingComparison.png')

# Area vs Number of elements
for f0 in [4,5,6]:
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
        T,P = np.meshgrid(theta,phi)
        tt = T.flatten()
        pp = P.flatten()
        
        E = E**2
        Delem = af.numerical_directivity(E**2,theta,phi)
        print(f'Element directivity: {10*np.log10(Delem)} dB')
        for j in range(iterations):
            p = af.hex_positions(N=j+3,d=tup[1])
            Js_hex[j] = p.shape[0]
            Apattern = np.abs(af.array_factor_tp(tt,pp,0,0,p))**2
            Apattern = Apattern.reshape((len(phi),-1))
            Apattern = E*Apattern
            directivity = af.numerical_directivity(Apattern,theta,phi)
            areas_hex[i,j] = directivity/4/np.pi*l0**2
            p = af.upa_positions(Nx=j+3,Ny=j+3,dx=tup[1],dy=tup[1])
            Js_rec[j] = p.shape[0]
            Apattern = np.abs(af.array_factor_tp(tt,pp,0,0,p))**2
            Apattern = Apattern.reshape((len(phi),-1))
            Apattern = E*Apattern
            directivity = af.numerical_directivity(Apattern,theta,phi)
            areas_rec[i,j] = directivity/4/np.pi*l0**2

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Js_hex,areas_hex[0,:],label=f'hex')
    ax.plot(Js_rec,areas_rec[0,:],label=f'rec')
    ax.set_xlabel('Number of elements [-]')
    ax.set_ylabel('Effective area [$m^2$]')
    ax.legend()
    plt.savefig(f'./Outputs/AreaVsNumberComparison{f0}_{ds[0]}.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Js_hex,areas_hex[1,:],label='hex')
    ax.plot(Js_rec,areas_rec[1,:],label='rec')
    ax.set_xlabel('Number of elements [-]')
    ax.set_ylabel('Effective area [$m^2$]')
    ax.legend()
    plt.savefig(f'./Outputs/AreaVsNumberComparison{f0}_{ds[1]}.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Js_hex,areas_hex[2,:],label='hex')
    ax.plot(Js_rec,areas_rec[2,:],label='rec')
    ax.set_xlabel('Number of elements [-]')
    ax.set_ylabel('Effective area [$m^2$]')
    ax.legend()
    plt.savefig(f'./Outputs/AreaVsNumberComparison{f0}_{ds[2]}.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Js_rec,areas_rec[0,:],label=f'd={ds[0]}')
    ax.plot(Js_rec,areas_rec[1,:],label=f'd={ds[1]}')
    ax.plot(Js_rec,areas_rec[2,:],label=f'd={ds[2]}')
    ax.set_xlabel('Number of elements [-]')
    ax.set_ylabel('Effective area [$m^2$]')
    ax.legend()
    plt.savefig(f'./Outputs/AreaVsNumberRectangular{f0}.png')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Js_hex,areas_hex[0,:],label=f'd={ds[0]}')
    ax.plot(Js_hex,areas_hex[1,:],label=f'd={ds[1]}')
    ax.plot(Js_hex,areas_hex[2,:],label=f'd={ds[2]}')
    ax.set_xlabel('Number of elements [-]')
    ax.set_ylabel('Effective area [$m^2$]')
    ax.legend()
    plt.savefig(f'./Outputs/AreaVsNumberHexagonal{f0}.png')
