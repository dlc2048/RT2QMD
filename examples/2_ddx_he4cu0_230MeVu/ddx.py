import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator


xs = 1.02506e-22 * 1e22 * 1e3  # barn [mm2] -> b [cm2], from Glauber-Gribov model
sr_arr = [
    0.013743422111471273,
    0.02738224848216898,
    0.05984506274483176,
    0.08748762889068717,
    0.1263199485198028,
    0.19408641856390302,
    0.2567557224614877
]

class DDX:
    def __init__(self, file_name: str, nskip: int=0):
        self.data = None

        data = []
        n    = 0
        with open(file_name) as file:  
            for line in file:
                if n < nskip:
                    n += 1
                    continue
                items = line.split()
                data += [list(map(float, items[:4]))]
        self.data = np.array(data)

theta_target = [5, 10, 20, 30, 40, 60, 80]
norm         = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

g4_qmd_63  = []
g4_qmd_65  = []
rt2_qmd_63 = []
rt2_qmd_65 = []
exfor      = []

number_fraction = (0.69812812, 0.30187188)

# retrive data
for theta, sr in zip(theta_target, sr_arr):
    g4_seg   = DDX('g4qmd/29063/ddx_n{}.txt'.format(theta))
    g4_seg.data[:,2] /= g4_seg.data[:,1] - g4_seg.data[:,0]
    g4_seg.data[:,2] /= 1e8
    g4_seg.data[:,2] /= sr
    g4_seg.data[:,2] *= xs
    g4_qmd_63  += [g4_seg]

for theta, sr in zip(theta_target, sr_arr):
    g4_seg   = DDX('g4qmd/29065/ddx_n{}.txt'.format(theta))
    g4_seg.data[:,2] /= g4_seg.data[:,1] - g4_seg.data[:,0]
    g4_seg.data[:,2] /= 1e8
    g4_seg.data[:,2] /= sr
    g4_seg.data[:,2] *= xs
    g4_qmd_65  += [g4_seg]

for theta in theta_target:
    rt2_seg = DDX('ddx_he4cu63_n{}_0.txt'.format(theta), nskip=6)
    rt2_seg.data[:,2] /= rt2_seg.data[:,1] - rt2_seg.data[:,0]
    rt2_seg.data[:,2] *= xs
    rt2_qmd_63 += [rt2_seg]

for theta in theta_target:
    rt2_seg = DDX('ddx_he4cu65_n{}_0.txt'.format(theta), nskip=6)
    rt2_seg.data[:,2] /= rt2_seg.data[:,1] - rt2_seg.data[:,0]
    rt2_seg.data[:,2] *= xs
    rt2_qmd_65 += [rt2_seg]

for theta in theta_target:
    exfor += [DDX('exfor/{}_1.txt'.format(theta))]


plt.rcParams['font.family'] = 'Arial'

# plot
fig = plt.figure(figsize=(5,7.5))
gs  = GridSpec(1, 1, wspace=0)

ax = []
for i in range(1):
    ax += [fig.add_subplot(gs[i])]

    

for i in range(0,7):
    data  = np.copy(rt2_qmd_63[i].data) * number_fraction[0] + np.copy(rt2_qmd_65[i].data) * number_fraction[1]
    label = '' if i else 'RT2QMD'
    ax[0].step(data[:,1], data[:,2] * norm[i], 'r', label=label)

    data  = np.copy(g4_qmd_63[i].data) * number_fraction[0] + np.copy(g4_qmd_65[i].data) * number_fraction[1]
    label = '' if i else 'G4 QMD'
    ax[0].step(data[:,1], data[:,2] * norm[i], 'g', label=label)

    data  = exfor[i].data
    label = '' if i else 'EXFOR (L. Heilbronn et al.)'
    if not i % 2:
        ax[0].errorbar((data[:,0] + data[:,1]) * 0.5, data[:,2] * norm[i] * 1e3, yerr = data[:,3] * norm[i] * 1e3, fmt='k^', label=label)
    else:
        ax[0].errorbar((data[:,0] + data[:,1]) * 0.5, data[:,2] * norm[i] * 1e3, yerr = data[:,3] * norm[i] * 1e3, fmt='k^', label=label, markerfacecolor='white')

ax[0].set_xlim(1e0, 1e3)
ax[0].set_ylim(1e-8, 1e2)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_ylabel('DDX [mb/sr/MeV]')

ax[0].text(1.5,  40,    r"$5^\circ$")
ax[0].text(1.5,  4,     r"$10^\circ (\times 10^{-1})$")
ax[0].text(1.5,  0.4,   r"$20^\circ (\times 10^{-2})$")
ax[0].text(1.5,  0.04,  r"$30^\circ (\times 10^{-3})$")
ax[0].text(1.5,  4e-3,  r"$40^\circ (\times 10^{-4})$")
ax[0].text(1.5,  4e-4,  r"$60^\circ (\times 10^{-5})$")
ax[0].text(1.5,  4e-5,  r"$80^\circ (\times 10^{-6})$")

ax[0].set_xticks([1e0, 1e1, 1e2, 1e3])
ax[0].yaxis.set_minor_locator(LogLocator(numticks=999, subs="auto"))

xl = ax[0].set_xlabel('Neutron energy [MeV]')
# xl.set_position((1.0, xl.get_position()[1]))

ax[0].legend(loc='lower left')
# ax[1].legend(loc='lower left')

plt.savefig('ddx.svg',  bbox_inches='tight')
plt.show()
