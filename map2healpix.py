import numpy as np
import healpy as hp


for bw in [60,90,120]:
    for f in [4,5,6]:
        in_name = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        print(in_name)
        table = np.loadtxt(in_name,skiprows=2)
        theta = np.radians(table[:,0])
        phi = np.radians(table[:,1])
        Eabs = table[:,2]

        nside = hp.pixelfunc.get_min_valid_nside(len(Eabs))
        npix = hp.nside2npix(nside=nside)
        element_map = np.ones(npix)*np.inf
        pixels = hp.ang2pix(nside,theta,phi)

        element_map[pixels] = Eabs
        for i in range(0,npix):
            if i not in pixels:
                z_interp = 0
                neighbors = hp.get_all_neighbours(nside,i)
                values = element_map[neighbors]
                idx = np.where(np.isfinite(values))
                z_interp = np.sum(values[idx[0]])/len(idx[0])
                element_map[i] = z_interp


        element_map = hp.ud_grade(element_map,64)
        out_name = f'./HealpixPatterns/Farfield{bw}_{f}GHz.txt'
        np.savetxt(out_name,element_map)
