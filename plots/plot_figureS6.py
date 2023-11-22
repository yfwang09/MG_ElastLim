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

# Figure S6a
figname = 'FigureS6a'
############################################################

# Set up matplotlib session
fig = plt.figure(figsize=figsize, tight_layout=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[:, :2])
ax2 = ax.twinx()

for datafilename, x_offset, color_u, label in zip(
    ['FigureS6a-1.txt', 'FigureS6a-2.txt'],
    [0.03, 0.00],
    ['C1', 'C3'],
    [r'$5\,{\rm K}$', r'$5\,{\rm K}$']
    ):

    temp = 5
    datafile = os.path.join(rawdata_dir, datafilename)
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
    # x_offset, color_u, label = 0.03, 'C1', r'$5\,{\rm K}$'
    ax.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, label=label)
    ax2.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, linewidth=0)
    ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
    dy = 0.05
    dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
    x0, y0 = x_offset, y_l_plot.max() - dy
    ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
                arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
    dy = -0.05
    dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
    x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.005, y_u_plot.max()
    ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
                arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

###################################################################
ax.set_ylim([0, 0.6])
# s0_interp = interp1d(strain_l0 - first_xl0, stress_l0 - first_yl0, kind='linear', fill_value='extrapolate')
s0_interp = interp1d(x_l_elastic, y_l_plot, kind='linear', fill_value='extrapolate')
eps_ticklabels = np.round(np.arange(0, 1.0, 0.1), decimals=5)
eps_ticks = s0_interp(eps_ticklabels)
ax2.set_yticks(eps_ticks)
ax2.set_yticklabels(eps_ticklabels)
ax2.set_ylim(ax.get_ylim())
ax.set_xlim([-0.015, 0.075])
ax.set_xlabel(r'$\epsilon_{\rm pl} = \epsilon - \epsilon_{\rm el}$ (%)', fontsize=fs)
ax.set_ylabel(r'$\sigma_{yy}$ (GPa)', fontsize=fs)
ax2.set_ylabel(r'$\epsilon$ (%)', fontsize=fs)

ax.tick_params(direction='in', labelsize=fstk)
ax2.tick_params(direction='in', labelsize=fstk)
ax.set_title(r'  $\mathbf{a}$ 5K: $\varepsilon_{\rm lim} = 0.56$%', loc='left', y=0.9, va='top', fontsize=fs)

fig.tight_layout()

if savefig:
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))


# Figure S6d
figname = 'FigureS6d'
############################################################
# Set up matplotlib session
fig = plt.figure(figsize=(5, 4), tight_layout=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[:, :2])
ax2 = ax.twinx()

######################### Load data ###############################
temp = 20

for datafilename, x_offset, color_u, label in zip(
    ['FigureS6d-2.txt', 'FigureS6d-1.txt'],
    [0.15, 0.0],
    ['C5', 'C4'],
    [r'$20\,{\rm K}$', r'$20\,{\rm K}$']
    ):

    datafile = os.path.join(rawdata_dir, datafilename)
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
    ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
    ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
    dy = 0.2
    dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
    x0, y0 = x_offset + 0.2, 2.0
    ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
                arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
    dy = -0.2
    dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
    x0, y0 = x_offset + 0.25, 2.0
    ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
                arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))

#################################################

###################################################################
ax.set_ylim([0, 3.0])
# s0_interp = interp1d(strain_l - first_xl0, stress_l - first_yl0, kind='linear', fill_value='extrapolate')
s0_interp = interp1d(x_l_elastic, y_l_plot, kind='linear', fill_value='extrapolate')
eps_ticklabels = np.round(np.arange(0, 4.5, 0.5), decimals=5)
eps_ticks = s0_interp(eps_ticklabels)
ax2.set_yticks(eps_ticks)
ax2.set_yticklabels(eps_ticklabels)
ax2.set_ylim(ax.get_ylim())
ax.set_xlim([-0.02, 0.56])
ax.set_xlabel(r'$\epsilon_{\rm pl} = \epsilon - \epsilon_{\rm el}$ (%)', fontsize=fs)
ax.set_ylabel(r'$\sigma_{yy}$ (GPa)', fontsize=fs)
ax2.set_ylabel(r'$\epsilon$ (%)', fontsize=fs)

ax.tick_params(direction='in', labelsize=fstk)
ax2.tick_params(direction='in', labelsize=fstk)
ax.set_title(r'  $\mathbf{d}$ 20K: $\varepsilon_{\rm lim} = 3.23$%', loc='left', y=0.9, va='top', fontsize=fs)
fig.tight_layout()

if savefig:
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

plt.show()