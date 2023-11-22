# Script for generating supplementary figure in the paper
# Author: Yifan Wang
# Supplementary Fig. 5a: stress-strain curve of 2K tensile loading
# Supplementary Fig. 5b: MNADM of the 2K tensile loading

import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Parameter settings

workdir = os.path.dirname(os.path.realpath(__file__))

figure_dir = os.path.join(workdir, 'figs')
os.makedirs(figure_dir, exist_ok=True)

rawdata_dir = os.path.join(workdir, 'rawdata')

savefig = True
fs = 18
fstk = 14
lw = 2
figsize=(5, 4)

# Figure S5a
figname = 'FigureS5a'
###################################################################

# Set up matplotlib session
fig = plt.figure(figsize=figsize, tight_layout=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[:, :2])
ax2 = ax.twinx()

######################### Load data ###############################
temp = 2

datafile = os.path.join(rawdata_dir, 'FigureS5a.txt')
rawdata = np.loadtxt(datafile)

xplot = rawdata[:, 0]
x_els = rawdata[:, 1]
yplot = rawdata[:, 2]

indmax = np.argmax(xplot)
x_l_plot = xplot[:indmax+1]
x_u_plot = xplot[indmax+1:]
x_l_elastic = x_els[:indmax+1]
x_u_elastic = x_els[indmax+1:]
y_l_plot = yplot[:indmax+1]
y_u_plot = yplot[indmax+1:]

# visualization
x_offset, color_u, label = 0.0, 'C0', r'$2\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)

dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset - 0.001, y_l_plot.max() - dy
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max()
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

###################################################################
ax.set_ylim([0, 0.5])
s0_interp = interp1d(x_l_elastic, y_l_plot, kind='linear', fill_value='extrapolate')
eps_ticklabels = np.round(np.arange(0, 1.4, 0.2), decimals=5)
eps_ticks = s0_interp(eps_ticklabels)
ax2.set_yticks(eps_ticks)
ax2.set_yticklabels(eps_ticklabels)
ax2.set_ylim(ax.get_ylim())
ax.set_xlim([-0.025, 0.035])
ax.set_xlabel(r'$\epsilon_{\rm pl} = \epsilon - \epsilon_{\rm el}$ (%)', fontsize=fs)
ax.set_ylabel(r'$\sigma_{yy}$ (GPa)', fontsize=fs)
ax2.set_ylabel(r'$\epsilon$ (%)', fontsize=fs)

ax.set_title(r'  $\mathbf{a}$ 2K: $\varepsilon_{\rm lim} = 0.19$%', y=0.9, loc='left', va='top', fontsize=fs)
ax.tick_params(direction='in', labelsize=fstk)
ax2.tick_params(direction='in', labelsize=fstk)

ax.annotate(r'1', xy=(-0.001, 0.1), xytext=(-50, 0), color='k',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            bbox=dict(boxstyle='circle,pad=0.1', fc='w', ec='k'),
            arrowprops=dict(arrowstyle='-|>')
           )
ax.annotate(r'2', xy=(0.004, 0.1), xytext=(50, 0), color='k',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            bbox=dict(boxstyle='circle,pad=0.1', fc='w', ec='k'),
            arrowprops=dict(arrowstyle='-|>')
           )

fig.tight_layout()

if savefig:
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))


# Figure S5b
###################################################################



plt.show()