import numpy as np

def upa_positions(Nx,Ny,dx,dy):
    """   
    Compute positions  for elements in a regular rectangular grid.

    Parameters
    ----------
    Nx: int
        number of elements on the x axis.
    Ny: int
        number of elements on the y axis.
    dx: float
        element spacing in the x direction.
    dy: float
        element spacring in the y direction.
    
    Returns 
    -------
    p: array_like 
        (Nx*Ny,2) matrix of element positions.
    """
    
    px = np.tile(np.arange(Nx),Ny)*dx
    py = np.repeat(np.arange(Ny),Nx)*dy
    p = np.stack((px,py),1)
    return p

def hex_positions(N,d):
    """   
    Compute positions  for elements in a regular hexagonal grid.

    Parameters
    ----------
    N: int
        number of elements on the x axis.
    d: float
        element spacing in the x direction.

    Returns 
    -------
    p: array_like 
        matrix of element positions.

    Notes
    -----
    The spacing on the y direction is computed to obtain a regular 
    hexagonal grid, and the number of elements on the y axis is computed 
    to obtain a configuration as close to a square as possible.
    """
    
    dx = d
    dy = dx*np.sqrt(3)
    Nx = N
    Ny = np.round(Nx/np.sqrt(3),decimals=0).astype(int)

    p1 = upa_positions(Nx,Ny,dx,dy)
    p2 = upa_positions(Nx-1,Ny-1,dx,dy)
    p2 = p2+np.array([dx/2,dy/2])
    p = np.concatenate((p1,p2),0)
    return p