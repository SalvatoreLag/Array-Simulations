'''Plot the E-plane and H-plane cuts for element patterns
imported from CST.'''

#%% Imports
import numpy as np
import map_functions as mf
import scienceplots
import matplotlib.pyplot as plt
import array_functions as af

plt.style.use(['science','ieee'])


#%% E-plane
thetap = np.arange(-90,90,1)

fig = plt.figure(figsize=(7.16,2.5))
axs = fig.subplots(1,3)
letters = ['A','B','C']
ls = ['--r','-k',':b']

for i,bw in enumerate([60,90,120]):
    ax = axs[i]
    for j,f in enumerate([4,5,6]):
        filename = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        if bw == 120:
            filename = f'./ElementPatterns/Farfield{bw}_{f}GHz_new.txt'
        E,theta,phi = mf.import_pattern(filename,1)
        P = af.radiated_power(E**2,theta,phi)
        D = (4*np.pi*E**2)/P

        Eplane1 = D[0,:90]
        Eplane2 = np.flip(D[180,:90])
        Eplane = 10*np.log10(np.concat((Eplane2,Eplane1)))
        ax.plot(thetap,Eplane,ls[j],label=f'{f} GHz')
  
    ax.grid()
    ax.legend(loc='lower center')
    ax.set_xlabel('Theta [deg]')
    ax.set_ylabel('Directivity [dBi]')
    ax.set_xlim(-90,90)
    ax.set_ylim(-20,20)
    ax.annotate(f'Horn {letters[i]}', (-75,16))

fig.tight_layout()
fig.savefig(f'./Outputs/ElementPatternEplaneSub')    

#%% H-plane
ls = ['--r','-k',':b']

for bw in [60,90,120]:
    plt.figure()
    for j,f in enumerate([4,5,6]):
        filename = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        if bw == 120:
            filename = f'./ElementPatterns/Farfield{bw}_{f}GHz_new.txt'
        E,theta,phi = mf.import_pattern(filename,1)
        P = af.radiated_power(E**2,theta,phi)
        D = (4*np.pi*E**2)/P

        Hplane1 = D[89,:90]
        Hplane2 = np.flip(D[270,:90])
        Hplane = 10*np.log10(np.concat((Hplane2,Hplane1)))
        plt.plot(thetap,Hplane,ls[j],label=f'{f} GHz')
  
    plt.grid()
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('Directivity [dBi]')
    plt.savefig(f'./Outputs/ElementPatternHplane{bw}')    
