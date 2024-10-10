import numpy as np
import healpy as hp

def array_pattern_matrix(nside,source_pixels,scan_pixels,positions):
    sx,sy,_ = hp.pix2vec(nside,source_pixels)
    S = np.stack((sx,sy))
    s0x,s0y,_ = hp.pix2vec(nside,scan_pixels)
    S0 = np.stack((s0x,s0y))

    nsource = len(source_pixels)
    nscan = len(scan_pixels)
    S = np.tile(S,(1,nscan))
    S0 = np.repeat(S0,nsource,axis=1)

    A = np.abs(np.sum(np.exp(1j*2*np.pi*positions@(S-S0)),0))**2
    A = A.reshape((nscan,nsource))
    
    return A

def array_pattern_loop(nside,source_pixels,scan_pixel,positions):
    sx,sy,_ = hp.pix2vec(nside,source_pixels)
    S = np.stack((sx,sy))
    s0x,s0y,_ = hp.pix2vec(nside,scan_pixel)
    S0 = np.stack((s0x,s0y))

    A = np.abs(np.sum(np.exp(1j*2*np.pi*positions@(S-S0)),0))**2
    
    return A
