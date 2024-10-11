import numpy as np
import healpy as hp

def CSTmap2healpix(in_filename:str,nside_out:int) -> np.ndarray:
    """   
    Map a CST theta-phi map into an healpix map

    Parameters
    ----------
    in_filename: str
        name of the txt file containing the CST theta-phi map.
    nside_out: int
        healpix resolution for the output map.

    Returns 
    -------
    healpix_map: array_like 
        transformed healpix map.
    """
    
    table = np.loadtxt(in_filename,skiprows=2)
    theta = np.radians(table[:,0])
    phi = np.radians(table[:,1])
    Eabs = table[:,2]

    nside = hp.pixelfunc.get_min_valid_nside(len(Eabs))
    npix = hp.nside2npix(nside)
    healpix_map = np.ones(npix)*np.inf
    pixels = hp.ang2pix(nside,theta,phi)

    healpix_map[pixels] = Eabs
    for i in range(0,npix):
        if i not in pixels:
            z_interp = 0
            neighbors = hp.get_all_neighbours(nside,i)
            values = healpix_map[neighbors]
            idx = np.where(np.isfinite(values))
            z_interp = np.sum(values[idx[0]])/len(idx[0])
            healpix_map[i] = z_interp
    healpix_map = hp.ud_grade(healpix_map,nside_out)

    return healpix_map


if __name__=='__main__':
    for bw in [60,90,120]:
        for f in [4,5,6]:
            in_name = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
            healpix_map = CSTmap2healpix(in_name,64)
            out_name = f'./HealpixPatterns/Farfield{bw}_{f}GHz.txt'
            np.savetxt(out_name,healpix_map)
