import numpy as np
import healpy as hp

def array_pattern(nside,source_pixels,scan_pixels,positions):
    sx,sy,_ = hp.pix2vec(nside,source_pixels)
    S = np.stack((sx,sy))
    s0x,s0y,_ = hp.pix2vec(nside,scan_pixels)
    S0 = np.stack((s0x,s0y))

    nsource = source_pixels.size
    nscan = scan_pixels.size
    if nscan == 1:
        S0 = np.expand_dims(S0,1)
    S = np.tile(S,(1,nscan))
    S0 = np.repeat(S0,nsource,axis=1)

    A = np.abs(np.sum(np.exp(1j*2*np.pi*positions@(S-S0)),0))**2
    A = A.reshape((nscan,nsource))
    
    return A


