# Script for generating figure S7 in the paper
# Author: Yifan Wang
# Supplementary Figure 7: Schematic NEB calculation workflow

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

# Supplementary Figure 7

xmin, xmax = 0, 1
emin, emax = 0.1, 0.9
semin, semax = 0.1, 0.9
dsig = 0.05
eij    = 0.6
ei, si = eij, eij*1
ej, sj = eij, si-dsig
ecal   = 0.4
eical, sical = ecal, ecal*1
ejcal, sjcal = ecal, sical-dsig
smin, smax = 0, 1

ss_x, ss_y = [xmin, ei, ej, xmax], [smin, si, sj, smax-dsig]
sigi_x, sigi_y = [xmin, emax, emax], [smin, semax, semax-dsig]
sigj_x, sigj_y = [emin, emin, xmax], [semin, semin-dsig, smax-dsig]

fig, ax = plt.subplots()
ax.plot(ss_x, ss_y, 'k', label='MD(2K) loading')
ax.plot(sigi_x, sigi_y, '--C0', label=r'MS(0K) loading')
ax.plot(sigj_x, sigj_y, '--C1', label=r'MS unloading')
ax.plot(eical, sical, 'oC0')
ax.plot(ejcal, sjcal, 'oC1')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Strain', fontsize=fs)
ax.set_ylabel('Stress', fontsize=fs)

ax.annotate(r'1', xy=(eical, sical), xytext=(-50, 0), color='C0',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            bbox=dict(boxstyle='circle,pad=0.1', fc='w', ec='C0'),
            arrowprops=dict(arrowstyle='-|>', linestyle='--', linewidth=2, color='C0', shrinkB=10)
           )
ax.annotate(r'2', xy=(ejcal, sjcal), xytext=(30, -40), color='C1',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            bbox=dict(boxstyle='circle,pad=0.1', fc='w', ec='C1'),
            arrowprops=dict(arrowstyle='-|>', linestyle='--', linewidth=2, color='C1', shrinkB=10)
           )
offsetx, offsety = 0.015, 0.03
rectx = [eical-offsetx, eical+offsetx, ejcal+offsetx, ejcal-offsetx, eical-offsetx]
recty = [sical+offsety, sical+offsety, sjcal-offsety, sjcal-offsety, sical+offsety]
ax.plot(rectx, recty, '--C2')

ax.annotate(r'$\varepsilon_{\rm max}$', xy=(emax, semax), xytext=(-50, 0), color='C0',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='C0')
           )
ax.annotate(r'$\varepsilon_{\rm min}$', xy=(emin, semin-dsig), xytext=(50, 0), color='C1',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='C1')
           )
ax.annotate(r'$\varepsilon_{\rm ST}$', xy=(ei, si), xytext=(0, 50), color='k',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='k')
           )
ax.annotate(r'NEB(MEP)', xy=(eical, 0.5*(sical+sjcal)), xytext=(100, 0), color='C2',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='<|-', color='C2')
           )
ax.annotate(r'ESL', xy=(0.75, 0.5*(sical+sjcal)), xytext=(50, 0), color='C2',
            textcoords='offset points', ha='center', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='<|-', color='C2')
           )
ax.legend(fontsize=fstk)

ax.set_xlim([xmin, xmax])
ax.set_ylim([smin-dsig, smax])
fig.tight_layout()

if savefig:
    figname = 'FigureS7'
    fig.savefig(os.path.join(figure_dir, '%s.pdf'%figname))

plt.show()