#%%
import numpy as np
import map_functions as mf
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(['science','ieee'])


#%%
thetap = np.arange(-90,90,1)

for bw in [60,90,120]:
    plt.figure()
    for f in [4,5,6]:
        filename = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        E,theta,phi = mf.import_pattern(filename,1)

        Eplane1 = E[0,:90]
        Eplane2 = np.flip(E[180,:90])
        Eplane = 20*np.log10(np.concat((Eplane2,Eplane1)))
        plt.plot(thetap,Eplane,label=f'{f} GHz')
  
    plt.grid()
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('Radiation pattern [dB]')
    plt.savefig(f'./Outputs/ElementPatternEplane{bw}')    

for bw in [60,90,120]:
    plt.figure()
    for f in [4,5,6]:
        filename = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        E,theta,phi = mf.import_pattern(filename,1)

        Hplane1 = E[90,:90]
        Hplane2 = np.flip(E[270,:90])
        Hplane = 20*np.log10(np.concat((Hplane2,Hplane1)))
        plt.plot(thetap,Hplane,label=f'{f} GHz')
  
    plt.grid()
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('Radiation pattern [dB]')
    plt.savefig(f'./Outputs/ElementPatternHplane{bw}')    

# %%
