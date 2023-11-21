# Script for generating figure 1 in the paper
# Author: Yifan Wang
# Figure 1b: schematic 1D potential energy landscape
# Figure 1c: schematic stress-strain curve of metallic glass, with inset showing the yielding point (data from 20K simulation)

import os
import numpy as np
import matplotlib.pyplot as plt

# Parameter settings

workdir = os.path.dirname(os.path.realpath(__file__))

figure_dir = os.path.join(workdir, 'figs')
os.makedirs(figure_dir, exist_ok=True)

rawdata_dir = os.path.join(workdir, 'rawdata')

savefig = True
fs = 18
fstk = 14

# Figure 1b

xv1 = np.linspace(0.1, 0.9, 1001)*np.pi*4
xv2 = xv1/4*33
yv1 = np.cos(xv1) + 0.2*xv1
yv2 = np.sin(xv2)*0.3

xa, ya, wa, ka = 6.5, 3, 2, 0.2
xva = np.linspace(xa-wa, xa+wa, 1001)
yva = ya - ka*(xva-xa)**2

xb, yb = np.array([3.25, 4.05, 4.8]), np.array([0.3, 0.8, 1.7])
wb, kb = 0.3, 5
xvb = np.linspace(xb-wb, xb+wb, 1001)
yvb = yb - kb*(xvb-xb)**2

fig, ax = plt.subplots()
ax.plot(xv1, yv1, 'k')
ax.plot(xv1, yv2 + yv1, 'C0', linewidth=3)
ax.plot(xva, yva, 'C1')
ax.plot(xvb, yvb, 'C1')
ax.set_xlim([xv1.min(), xv1.max()])
ax.set_ylim([yv2.min()-0.5, yv1.max()+1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Average Strain Coordinate', fontsize=fs)
ax.set_ylabel('Potenrial Energy', fontsize=fs)
# ax.set_title('  (a)', loc='left', y=0.8, fontsize=fs)
ax.text(x=xa , y=ya+0.2,s=r'$\alpha$', ha='center', fontsize=fs, color='C1')
for k in range(len(xb)):
    ax.text(x=xb[k], y=yb[k]+0.2,s=r'$\beta$', ha='center', fontsize=fs, color='C1')

ax.set_title('b ', y=0.9, loc='left', va='top', ha='right', fontsize=fs, weight='bold')
fig.tight_layout()
if savefig:
    figname = 'Figure1b'
    fig.savefig(os.path.join(figure_dir, '%s.pdf'%figname))

# Figure 1c-inset

rawdata = np.loadtxt(os.path.join(rawdata_dir, 'Figure1c_full.txt'))

plot_range_x = (7.4, 8.6)
plot_range_y = (4.8, 5.2)

ind = np.logical_and(plot_range_x[0] < rawdata[:, 0], rawdata[:, 0] < plot_range_x[1])
rawdata[:, 0] = np.round(rawdata[:, 0], 4)

fig, ax = plt.subplots()
ax.plot(rawdata[ind, 0], rawdata[ind, 1], linewidth=5)
ax.set_xlim(plot_range_x)
ax.set_ylim(plot_range_y)
plt.axis('off')
if savefig:
    figname = 'Figure1c_inset'
    fig.savefig(os.path.join(figure_dir, '%s.svg'%figname), transparent=True)

plt.show()