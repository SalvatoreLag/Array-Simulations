import numpy as np

def upa_positions(Nx,Ny,dx,dy):
    px = np.tile(np.arange(Nx),Ny)*dx
    py = np.kron(np.arange(Nx),np.ones(Ny))*dy
    pz = np.zeros(Nx*Ny)
    p = np.stack([px,py,pz],0)

    return p

def tpa_positions(Nx,Ny,dx,dy):
    p = upa_positions(Nx,Ny,dx,dy)
    for i in range(1,Ny+1,2):
        p[0,i*Nx:(i+1)*Nx] = p[0,i*Nx:(i+1)*Nx]+dx/2
    return p 
