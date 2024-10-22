import numpy as np
import healpy as hp


def upa_positions(Nx:int,Ny: int,dx:float=0.5,dy:float=0.5) -> np.ndarray:
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


def hex_positions(N:int,d:float=0.5) -> np.ndarray:
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


def array_factor_matrix(nside:int,source_pixels:np.ndarray,scan_pixels:np.ndarray,positions:np.ndarray) -> np.ndarray:
    """
    Compute the array pattern steered in specific direction(s),
    using the healpix format

    Parameters
    ----------
    nside: int
        healpix resolution parameter.
    source_pixels: array_like
        indices for the healpix pixels corresponding to source directions.
    scan_pixels: array_like
        indices for the healpix pixels corresponding to steering directions.
    positions: array_like
        (Nelem,2) matrix of normalized element positions.

    Returns 
    -------
    A: array_like 
        matrix of complex steered array factors.

    Notes
    -----
    This implementation computes the steered array pattern for all 
    steering directions in one matrix computation. Each row in the output
    array is a healpix map for a different steering direction.
    """
    
    sx,sy,_ = hp.pix2vec(nside,source_pixels)
    S = np.stack((sx,sy))
    s0x,s0y,_ = hp.pix2vec(nside,scan_pixels)
    S0 = np.stack((s0x,s0y))

    nsource = len(source_pixels)
    nscan = len(scan_pixels)
    # Repeat the source matrix for each steering angle
    S = np.tile(S,(1,nscan))
    # Repeat each steering direction to match the dimension of the source matrix
    S0 = np.repeat(S0,nsource,axis=1)

    A = np.sum(np.exp(2j*np.pi*positions@(S-S0)),0)
    A = A.reshape((nscan,nsource))
    
    return A


def array_factor(nside:int,source_pixels:np.ndarray,scan_pixel:np.ndarray,positions:np.ndarray) -> np.ndarray:
    """
    Compute the array pattern steered in specific direction,
    using the healpix format.

    Parameters
    ----------
    nside: int
        healpix resolution parameter.
    source_pixels: array_like
        indices for the pixels corresponding to source directions.
    scan_pixel: array_like
        index for the pixel corresponding to steering direction.
    positions: array_like
        (Nelem,2) matrix of normalized element positions.

    Returns 
    -------
    A: array_like 
        array containing the complex healpix map of array factor steered  
        in direction specified by scan_pixel. 

    Notes
    -----
    This implementation computes the steered array pattern for one
    steering direction. Call this in a loop for multiple directions.
    """
    
    sx,sy,_ = hp.pix2vec(nside,source_pixels)
    S = np.stack((sx,sy))
    s0x,s0y,_ = hp.pix2vec(nside,scan_pixel)
    S0 = np.stack((s0x,s0y))

    a = np.sum(np.exp(2j*np.pi*positions@(S-S0)),0)
    
    return a


def array_factor_grid(theta:np.ndarray,phi:np.ndarray,theta0:float,phi0:float,positions:np.ndarray) -> np.ndarray:
    """
    Compute the array pattern steered in specific direction,
    on a theta-phi grid.

    Parameters
    ----------
    theta: array_like
        Elevation angles in radians where to compute the array pattern.
    phi: array_like
        Azimuthal angles in radians where to compute the array pattern.
    theta0: float
        Elevation steering angle.
    phi0: float
        Azimuthal steering angle.
    positions: array_like
        (Nelem,2) matrix of normalized element positions.

    Returns 
    -------
    A: array_like 
        theta-phi matrix containing the array pattern steered 
        in direction specified by theta0 and phi0. 
    """

    ntheta = len(theta)
    nphi = len(phi)
    T,P = np.meshgrid(theta,phi)
    t = T.reshape(nphi*ntheta)
    p = P.reshape(nphi*ntheta)

    S = hp.ang2vec(t,p)[:,:2].T
    S0 = hp.ang2vec(np.atleast_1d(theta0),np.atleast_1d(phi0))[:,:2].T

    a = np.sum(np.exp(2j*np.pi*positions@(S-S0)),0)
    
    return a.reshape((nphi,ntheta))


def linear_directivity(a:np.ndarray,d:float) -> float:
    """
    Compute the directivity for a uniform linear array with tapering a
    and spacing d.

    Parameters
    ----------
    a: array_like
        array of tapering coefficients.
    d: float
        element spacing.

    Returns 
    -------
    Directivity: float 
        directivity. 
    """

    N = len(a)
    D_num = np.sum(np.abs(a))**2
    coeff_products = np.kron(a,a).reshape((N,N))

    diff_matrix = np.zeros((N,N))
    for i in range(1,N):
        diag = np.ones(N-i)*i
        diff_matrix = diff_matrix + np.diag(-diag,i) + np.diag(diag,-i)
    
    sinc_args = 2*d*diff_matrix
    D_den = np.sum(coeff_products*np.sinc(sinc_args))
    Directivity = D_num/D_den

    return Directivity


def radiated_power(PowerPattern:np.ndarray,theta:np.ndarray,phi:np.ndarray) -> float:
    """
    Evaluate numerically the total radiated power.

    Parameters
    ----------
    PowerPattern: array_like
        2D array of the radiated field with phi dependence on the 0th axis and
        thtea dependence on the 1st axis.
    theta: array_like
        elevation angles in radians where the radiated field is evaluated.
    phi: array_like
        azimuth angles in radians where the radiated field is evaluated.

    Returns 
    -------
    Prad: float 
        total radiated power. 
    """

    dt = theta[1]-theta[0]
    dp = phi[1]-phi[0]

    return np.sum(np.sum(PowerPattern*np.sin(theta),1)*dt,0)*dp


def numerical_directivity(PowerPattern:np.ndarray,theta:np.ndarray,phi:np.ndarray) -> float:
    """
    Evaluate numerically the directivity of a radiation pattern.

    Parameters
    ----------
    PowerPattern: array_like
        2D array of the radiated field with phi dependence on the 0th axis and
        thtea dependence on the 1st axis.
    theta: array_like
        elevation angles in radians where the radiated field is evaluated.
    phi: array_like
        azimuth angles in radians where the radiated field is evaluated.

    Returns 
    -------
    Directivity: float 
        directivity. 
    """

    D_num = 4*np.pi*np.max(PowerPattern)
    D_den = radiated_power(PowerPattern,theta,phi)
    return D_num/D_den
