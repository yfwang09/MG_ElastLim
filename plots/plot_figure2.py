# Script for generating supplementary figure in the paper
# Author: Yifan Wang
# Supplementary Fig. 6a: stress-strain curve of 5K tensile loading
# Supplementary Fig. 6d: stress-strain curve of 20K tensile loading

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
figsize=(12, 4)

# Figure S2
figname = 'Figure2'

# Set up matplotlib session
fig = plt.figure(figsize=figsize, tight_layout=True)
gs = GridSpec(1, 3)
ax = fig.add_subplot(gs[:, :2])
ax1 = fig.add_subplot(gs[:, 2])
ax2 = ax.twinx()

################### 0K cycle ##################
datafilename = 'Figure2-0K.txt'
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
x_offset, color_u, label = 0, 'k', r'$0\,{\rm K}$ limit'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset + 0.01, y_l_plot.max() + 0.01, label, ha='center', va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset - 0.004, y_l_plot.max() - dy
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max()
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 2K cycle ##################
datafilename = 'Figure2-2K.txt'
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
x_offset, color_u, label = 0.02, 'C0', r'$2\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset, y_l_plot.max() + 0.01, label, ha='center', va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset - 0.004, y_l_plot.max() - dy
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max()
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 5K reversible cycle ##################
datafilename = 'Figure2-5K-1.txt'
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
x_offset, color_u, label = 0.06, 'C1', r'$5\,{\rm K}$'
ax.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, label=label)
ax2.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, linewidth=0)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset, y_l_plot.max() + 0.02, label, va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.01, y_l_plot.max() - dy
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() - 0.002, y_u_plot.max()-0.02
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 5K irreversible cycle ##################
datafilename = 'Figure2-5K-2.txt'
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
x_offset, color_u, label = 0.04, 'C3', r'$5\,{\rm K}$'
ax.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, label=label)
ax2.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, linewidth=0)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset, y_l_plot.max() + 0.02, label, va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.000, y_l_plot.max() - dy
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max()
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 10K cycle ##################
datafilename = 'Figure2-10K.txt'
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
x_offset, color_u, label = 0.09 + (x_l_plot - x_l_elastic)[0] - (x_l_plot - x_l_elastic)[3], 'C2', r'$10\,{\rm K}$'
ax.plot((x_l_plot - x_l_elastic + x_offset), y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset, y_l_plot.max() + 0.02, label, va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset - 0.002, y_l_plot.max() - dy
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max() 
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 15K cycle ##################
datafilename = 'Figure2-15K.txt'
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
x_offset, color_u, label = 0.13 + (x_l_plot - x_l_elastic)[0] - (x_l_plot - x_l_elastic)[10], 'C8', r'$15\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text(x_offset, y_l_plot.max() + 0.02, label, va='bottom', color=color_u, fontsize=fstk)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset - 0.002, y_l_plot.max() - dy
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.004, y_u_plot.max()
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
#################################################

################### 20K irreversible cycle ##################
datafilename = 'Figure2-20K-1.txt'
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
x_offset, color_u, label = 0.2, 'C5', r'$20\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.02, 0.45
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.08, 0.45
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))

x_l_final, y_l_final = x_l_plot - x_l_elastic + x_offset, y_l_plot
x_u_final, y_u_final = x_u_plot - x_u_elastic + x_offset, y_u_plot

idx1 = np.abs(y_l_final - 0.5).argmin()
idx2 = np.abs(y_u_final - 0.5).argmin()
ax.annotate('', xy=(x_l_final[idx1], y_l_final[idx1]), xytext=(x_u_final[idx2], y_u_final[idx2]),
            arrowprops=dict(arrowstyle='-', linestyle='--', color=color_u,
                            # connectionstyle='arc,angleA=30,angleB=40,armA=84,armB=63,rad=0')
                            connectionstyle='arc,angleA=30,angleB=40,armA=28,armB=21,rad=0')
           )

ax.text(x_l_final[idx1]-0.02, y_l_final[idx1]-0.02, label, ha='right', va='top', color=color_u, fontsize=fstk)
ax.annotate(r'$\varepsilon_{\rm lim}(20\,{\rm K}) = 3.23\,\%$', 
            xy=(x_l_final[idx1]+0.005, y_l_final[idx1]+0.025), annotation_clip=False,
            xytext=(-20, 0), textcoords='offset points', ha='right', va='center', color=color_u, fontsize=fstk,
            arrowprops=dict(arrowstyle='-|>', linestyle='-', color=color_u)
           )

#################################################

################### 20K reversible cycle ##################
datafilename = 'Figure2-20K-2.txt'
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
x_offset, color_u, label = 0.16, 'C4', r'$20\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)
ax.text((x_l_plot-x_l_elastic+x_offset)[-20], y_l_plot.max() + 0.01, label, ha='right', va='bottom', fontsize=fstk, color=color_u)
dy = 0.06
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.005, y_l_plot.max() - dy - 0.01
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.06
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = (x_u_plot - x_u_elastic + x_offset).max() + 0.015, y_l_plot.max() + dy + 0.015
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))

#################################################

################### 30K cycle ##################
datafilename = 'Figure2-30K.txt'
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
x_offset, color_u, label = 0.28, 'C6', r'$30\,{\rm K}$'
ax.plot(x_l_plot - x_l_elastic + x_offset, y_l_plot, color_u, label=label)
ax.plot(x_u_plot - x_u_elastic + x_offset, y_u_plot, color_u)

dy = 0.05
dx = np.polyfit(y_l_plot, x_l_plot - x_l_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.02, 0.45
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))
dy = -0.05
dx = np.polyfit(y_u_plot, x_u_plot - x_u_elastic + x_offset, 1)[0] * dy
x0, y0 = x_offset + 0.07, 0.45
# ax.arrow(x0, y0, dx, dy)
ax.annotate('', xy=(x0+dx, y0+dy), xycoords='data', xytext=(x0, y0), textcoords='data', 
            arrowprops=dict(arrowstyle='-|>', linewidth=lw, color=color_u))

x_l_final, y_l_final = x_l_plot - x_l_elastic + x_offset, y_l_plot
x_u_final, y_u_final = x_u_plot - x_u_elastic + x_offset, y_u_plot

idx1 = np.abs(y_l_final - 0.5).argmin()
idx2 = np.abs(y_u_final - 0.5).argmin()
ax.annotate('', xy=(x_l_final[idx1], y_l_final[idx1]), xytext=(x_u_final[idx2], y_u_final[idx2]),
            arrowprops=dict(arrowstyle='-', linestyle='--', color=color_u,
                            # connectionstyle='arc,angleA=60,angleB=60,armA=40,armB=40,rad=0')
                            connectionstyle='arc,angleA=60,angleB=60,armA=10,armB=10,rad=0')
           )

ax.text(x_u_final[idx2]+0.01, y_u_final[idx2]-0.02, label, ha='left', va='top', color=color_u, fontsize=fstk)
ax.annotate(r'$3.16\,\%$', xy=(x_u_final[idx2]+0.005, y_u_final[idx2]+0.02), annotation_clip=False,
            xytext=(20, 0), textcoords='offset points', ha='left', va='center', color=color_u, fontsize=fstk,
            arrowprops=dict(arrowstyle='-|>', linestyle='-', color=color_u)
           )

#################################################

###################################################################
# ax.legend(frameon=False)
ax.set_ylim([0, 0.5])
# s0_interp = interp1d(strain_l0 - first_xl0, stress_l0 - first_yl0, kind='linear', fill_value='extrapolate')
s0_interp = interp1d(x_l_elastic, y_l_plot, kind='linear', fill_value='extrapolate')
eps_ticklabels = np.round(np.arange(0, 1.0, 0.1), decimals=5)
eps_ticks = s0_interp(eps_ticklabels)
ax2.set_yticks(eps_ticks)
ax2.set_yticklabels(eps_ticklabels)
ax2.set_ylim(ax.get_ylim())
ax.set_xlim([-0.02, 0.37])
ax.set_xlabel(r'$\epsilon_{\rm pl} = \epsilon - \epsilon_{\rm el}$ (%)', fontsize=fs)
ax.set_ylabel(r'$\sigma_{yy}$ (GPa)', fontsize=fs)
ax2.set_ylabel(r'$\epsilon$ (%)', fontsize=fs)

datafile = os.path.join(rawdata_dir, 'Figure2b.txt')
rawdata = np.loadtxt(datafile)
templist = rawdata[:, 0]
elimlist = rawdata[:, 1]

arr_xy0 = [templist[1], elimlist[1]+0.01]
arr_xy1 = [templist[2], elimlist[2]+0.01]
ax1.plot(templist, elimlist, 'o-k')
ax1.yaxis.set_label_position('right')
ax1.yaxis.tick_right()
ax1.set_ylim([0, 4.0])
ax1.set_xlabel('T (K)', fontsize=fs)
ax1.set_ylabel(r'$\varepsilon_{\rm lim}$ (%)', fontsize=fs)
ax1.set_xticks(templist)
ax1.annotate('', xy=(templist[2], elimlist[2]+0.2), xycoords='data',
                 xytext=(templist[1], elimlist[1]+0.2), textcoords='data',
             arrowprops=dict(arrowstyle='->', linestyle=':', color='C2', linewidth=3),
            )
ax1.annotate('', xy=(templist[5]-1.5, elimlist[5]-0.2), xycoords='data',
                 xytext=(templist[4]-0.2, elimlist[4]+0.5), textcoords='data',
             arrowprops=dict(arrowstyle='->', linestyle=':', color='C2', linewidth=3),
            )
ax1.annotate('', xy=(templist[4]-1, elimlist[4]+0.3), xycoords='data',
                 xytext=(templist[2]+1, elimlist[2]+0.3), textcoords='data',
             arrowprops=dict(arrowstyle='->', linestyle='-', color='C3', linewidth=3),
            )
ax1.annotate('', xy=(templist[1]-0.1, elimlist[1]+0.2), xycoords='data',
                 xytext=(templist[0], elimlist[0]+0.2), textcoords='data',
             arrowprops=dict(arrowstyle='->', linestyle='-', color='C3', linewidth=3),
            )
ax1.annotate('', xy=(templist[6]-1, elimlist[6]+0.15), xycoords='data',
                 xytext=(templist[5]+1, elimlist[5]+0.15), textcoords='data',
             arrowprops=dict(arrowstyle='->', linestyle='-', color='C3', linewidth=3),
            )
ax.tick_params(direction='in', labelsize=fstk)
ax1.tick_params(direction='in', labelsize=fstk)
ax2.tick_params(direction='in', labelsize=fstk)

ax.set_title('a     ', y=0.9, loc='left', va='top', ha='right', fontsize=fs, weight='bold')
ax1.set_title('b  ', y=0.9, loc='left', va='top', ha='right', fontsize=fs, weight='bold')

ax1.annotate('Elastic Limit\nAnomaly', xy=(15, 2), xytext=(13.5, 3), ha='right', va='bottom', color='C2', fontsize=fstk,
             arrowprops=dict(arrowstyle='-|>', linestyle='-', color='C2', linewidth=2))
ax1.annotate('', xy=(3.5, 0.8), xytext=(8.8, 3), ha='right', va='bottom', color='C2', fontsize=fstk,
             arrowprops=dict(arrowstyle='-|>', linestyle='-', color='C2', linewidth=2))

# fig.tight_layout()

if savefig:
    figname = 'Figure2-elastic-limit'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

plt.show()