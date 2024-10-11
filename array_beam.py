import numpy as np
import healpy as hp

def array_pattern_matrix(nside:int,source_pixels:np.ndarray,scan_pixels:np.ndarray,positions:np.ndarray) -> np.ndarray:
    """
    Compute the array pattern steered in specific direction(s).

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
        array of steered array patterns.

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

    A = np.abs(np.sum(np.exp(1j*2*np.pi*positions@(S-S0)),0))**2
    A = A.reshape((nscan,nsource))
    
    return A

def array_pattern_loop(nside:int,source_pixels:np.ndarray,scan_pixel:np.ndarray,positions:np.ndarray) -> np.ndarray:
    """
    Compute the array pattern steered in specific direction.

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
        array containing the healpix map of the array pattern steered 
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

    A = np.abs(np.sum(np.exp(1j*2*np.pi*positions@(S-S0)),0))**2
    
    return A

