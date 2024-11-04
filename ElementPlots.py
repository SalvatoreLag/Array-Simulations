#%%
import numpy as np
import map_functions as mf
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(['science','ieee'])


#%%
thetap = np.arange(-90,90,1)

fig = plt.figure(figsize=(7.16,2.5))
axs = fig.subplots(1,3)
letters = ['A','B','C']

for i,bw in enumerate([60,90,120]):
    ax = axs[i]
    for f in [4,5,6]:
        filename = f'./ElementPatterns/Farfield{bw}_{f}GHz.txt'
        E,theta,phi = mf.import_pattern(filename,1)

        Eplane1 = E[0,:90]
        Eplane2 = np.flip(E[180,:90])
        Eplane = 20*np.log10(np.concat((Eplane2,Eplane1)))
        ax.plot(thetap,Eplane,label=f'{f} GHz')
  
    ax.grid()
    ax.legend(loc='lower center')
    ax.set_xlabel('Theta [deg]')
    ax.set_ylabel('Radiation pattern [dB]')
    ax.set_xlim(-90,90)
    ax.set_ylim(-5,32)
    ax.annotate(f'Horn {letters[i]}', (-75,28))

fig.tight_layout()
fig.savefig(f'./Outputs/ElementPatternEplaneSub')    

#%%

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
