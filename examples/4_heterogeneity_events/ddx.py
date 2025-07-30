import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator


plt.rcParams['font.family'] = 'Arial'


xs     =  0.61865 * 1e3  # averaged mb, from Glauber-Gribov model


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

theta_target = [15, 30, 45, 60]
norm         = [1e0, 1e-1, 1e-2, 1e-3]

rt2_qmd = []

# retrive data
for theta in theta_target:
    rt2_seg = DDX('heterogeneous_a{}_0.txt'.format(theta), nskip=6)
    rt2_seg.data[:,2] /= rt2_seg.data[:,1] - rt2_seg.data[:,0]
    rt2_seg.data[:,2] *= xs
    rt2_qmd += [rt2_seg]

# plot
fig = plt.figure(figsize=(5,7.5))
gs  = GridSpec(1, 1, wspace=0)

ax = []
for i in range(1):
    ax += [fig.add_subplot(gs[i])]

for i in range(4):
    data  = rt2_qmd[i].data
    label = '' if i else 'RT2QMD'
    ax[0].step(data[:,1], data[:,2] * norm[i], 'r', label=label)

ax[0].set_xlim(1e0, 1e3)
ax[0].set_ylim(1e-8, 1e2)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
xl = ax[0].set_xlabel('Neutron energy [MeV]')
# xl.set_position((1.0, xl.get_position()[1]))
ax[0].set_ylabel('DDX [mb/sr/MeV]')
ax[0].yaxis.set_minor_locator(LogLocator(numticks=999, subs="auto"))

ax[0].set_xticks([1e0, 1e1, 1e2, 1e3])

ax[0].text(1.2,   6,    r"$15^\circ$")
ax[0].text(1.2,   0.5,  r"$30^\circ (\times 10^{-1})$")
ax[0].text(1.2,   0.05, r"$45^\circ (\times 10^{-2})$")
ax[0].text(1.2,   4e-4, r"$60^\circ (\times 10^{-3})$")

ax[0].legend(loc='lower left')

plt.savefig('ddx.svg', bbox_inches='tight')
plt.show()
