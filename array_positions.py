import numpy as np

def upa_positions(Nx,Ny,dx,dy):
    """   
    :param Nx: number of elements on the x axis
    :param Ny: number of elements on the y axis
    :param dx: element spacing in the x direction
    :param dy: element spacring in the y direction
    :returns p: (Nx*Ny,2) matrix of element positions
    """
    px = np.tile(np.arange(Nx),Ny)*dx
    py = np.repeat(np.arange(Ny),Nx)*dy
    p = np.stack((px,py),1)
    return p

def tpa_positions(Nx,Ny,dx,dy):
    p = upa_positions(Nx,Ny,dx,dy)
    for i in range(1,Ny+1,2):
        p[i*Nx:(i+1)*Nx,0] = p[i*Nx:(i+1)*Nx,0]+dx/2
    return p 

def tpa_positionsV2(N,d):
    dx = d
    dy = dx*np.sqrt(3)
    Nx = N
    Ny = np.round(Nx/np.sqrt(3),decimals=0).astype(int)

    p1 = upa_positions(Nx,Ny,dx,dy)
    p2 = upa_positions(Nx-1,Ny-1,dx,dy)
    p2 = p2+np.array([dx/2,dy/2])
    p = np.concatenate((p1,p2),0)
    return p