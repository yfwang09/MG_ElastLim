# Script for generating figure in the paper
# Author: Yifan Wang
# Supplementary Figure 9: Free volume analysis

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_func import load_ST_energy_strain, cal_dE_Eb

# Parameter settings

workdir = os.path.dirname(os.path.realpath(__file__))

figure_dir = os.path.join(workdir, 'figs')
os.makedirs(figure_dir, exist_ok=True)

rawdata_dir = os.path.join(workdir, 'rawdata')

savefig = True
fs = 18
fstk = 14

# Supplementary Figure 9
state_list = ['2-1', '8-7', '7-6', '2-4', '2-7', '4-6']
frames_list = [[0, 184, 226, 452, 678, 904],
               [0, 86, 212, 425, 638, 851],
               [949, 1088, 1135, 1228, 1368, 1508],
               [854, 1027, 1094, 1201, 1375, 1549],
               [0, 377, 553, 754, 1131, 1509],
               [949, 1145, 1342, 1539, 1736]]
frames_target = [184, 86, 1135, 1094, 553]             

state_list = ['2-1', '7-6', '2-7']
frames_list = [[0, 184, 226, 452, 678, 904],
               [949, 1088, 1135, 1228, 1368, 1508],
               [0, 377, 553, 754, 1131, 1509],
               [949, 1145, 1342, 1539, 1736]
              ]
frames_target = [184, 1135, 553]
event_list = ['I', 'II', 'III']

dr = 5.0 # max_dr_list = [5.0, ]
num_refs = 9
N = num_refs + 3
input_direct = os.path.join(rawdata_dir, 'FV_data')

fontsize = 14
fs = 18
# markers = ['o', 's', 'D', '^', 'h', 'p', '*']
markers = ['o', ]*7
colors = ['C0', 'C1', 'C2',]

from scipy.interpolate import interp1d

fig = plt.figure(figsize=(8, 12))
gs = GridSpec(4, 1)

ax1 = fig.add_subplot(gs[:2])
ax  = fig.add_subplot(gs[2:3], sharex=ax1)
ax2 = fig.add_subplot(gs[3:4], sharex=ax)

Eeig_list = []# [104.7693, ]
spatial_list = []#[(180, 0)]

for iA, iB, emin, emax, color, event in zip([1, 7, 2], [2, 6, 7], 
                                     [0, 0.36, 0], [0.3616, 0.6, 0.6],
                                     colors, event_list
                                    ):
    xv = np.linspace(emin, emax, 100)
    strain, energy, Efun, emin_data, emax_data, eeig, Eeig, Efwd_fun, Ebwd_fun = load_ST_energy_strain(iA, iB, return_dict=False, return_Efwd_Ebwd=True)
    yv = Efun(xv)
    dEfwd, dEbwd, Ebfwd, Ebbwd = cal_dE_Eb(yv, return_dict=False)
    if iA == 1 and iB == 2:
        ax1.plot(xv, -Ebfwd*1000, 'k', label=r'$E_{\rm b}^{\rm fwd}$')
        ax1.plot(xv, -Ebbwd*1000, 'k--', label=r'$E_{\rm b}^{\rm bwd}$')
        ax1.plot(eeig,-Eeig*1000, 'ko', label=r'$E_{\rm eig}(\varepsilon_{\rm eig})$')
    ax1.plot(xv, Ebfwd*1000, color)
    ax1.plot(xv, Ebbwd*1000, color+'--')
    ax1.plot(eeig, Eeig*1000, color+'o')
    print(iA, iB, emin, emax, Eeig*1000)
    Eeig_list.append(Eeig*1000)
    if iA == 7 and iB == 6:
        xy = (xv[-1], Ebbwd[-1]*1000)
    else:
        xy = (xv[0], Ebfwd[0]*1000)
    ax1.annotate('Event %s'%event, xy=xy, 
                 color=color, fontsize=fs, ha='center', va='bottom')
ax1.legend(fontsize=fstk, frameon=False, ncol=3)

for s in range(len(state_list)):
    state = state_list[s]
    states = [3*s, 3*s+1, 3*s+2]
    input_dir = os.path.join(input_direct, 'state%s'%state, 'code', 'output')

    frames = frames_list[s]
    Vf_strain = np.zeros((len(frames),), dtype=[('x', 'float'), ('stz', 'i4'), ('Vol', 'float'), ('deltaVol', 'float')]) 
    for f in range(len(frames)):
        frame = frames[f]
        strain = frame/2500
        feature_vol_file = os.path.join(input_dir, 'feature_vol_T20K_frame_%d_dr_%.1f.npz' %(frame, dr))
        feature_vol = np.load(feature_vol_file)['arr_0']

        ## Classsify the satom region and Nonsatom region

        Eb = feature_vol['Eb']
        stz_size = feature_vol['stz_size']
        aveVoro = feature_vol['ave_activated_voro'] 
        aveVol = feature_vol['ave_activated_vol']

        num_refs = 1
        for i in range(num_refs):
            stz = stz_size[[i+3, i+3+N, i+3+N*2]]
            Vol = aveVol[[i+3, i+3+N, i+3+N*2]]
            deltaVol = (Vol - Vol[0]) # /Vol[0]*100
            deltaVol1 = Vol - Vol[0]

            if frame in frames_target:
                ax.plot(strain, stz[-1], color=colors[s], marker = markers[s],) 
                ax2.plot(strain, deltaVol[-1], color=colors[s], marker = markers[s], )
                spatial_list.append((stz[-1], deltaVol[-1]))

        Vf_strain[f] = (strain, stz[-1], Vol[-1], deltaVol[-1])

    xv = Vf_strain['x']
    f = interp1d(Vf_strain['x'], Vf_strain['stz'], kind='cubic')
    f2 = interp1d(Vf_strain['x'], Vf_strain['deltaVol'], kind='cubic')
    ax.plot(xv, f(xv), color=colors[s], )
    ax2.plot(xv, f2(xv), color=colors[s], )
    if s == 1:
        xv_plot = xv[-1]
        fv_plot, f2_plot = f(xv[-1]), f2(xv[-1])
        xytext = (0, 0)
    else:
        xv_plot = xv[0] - 0.02
        fv_plot, f2_plot = f(xv[0]), f2(xv[0])
        xytext = (0, -5)
    ax.annotate('Event %s'%event_list[s], xy=(xv_plot, fv_plot),
                xytext=xytext, textcoords='offset points',
                fontsize=fs, color=colors[s], ha='center', va='top')
    ax2.annotate('Event %s'%event_list[s], xy=(xv_plot, f2_plot), 
                 xytext=xytext, textcoords='offset points',
                 fontsize=fs, color=colors[s], ha='center', va='top')
    

ax2.set_xlabel(r'Strain $\varepsilon$ (%)', fontsize=fs)
ax1.set_ylabel(r'$E_{\rm b}$ (meV)', fontsize=fs)
ax.set_ylabel(r'$n_{\rm STZ}$', fontsize=fs)
ax2.set_ylabel(r'$\Delta\overline{V}_{\rm f}\,(\AA^3\,{\rm per\,atom})$', fontsize=fs)

ax1.set_title('a       ', y=0.99, loc='left', va='top', ha='right', fontsize=fontsize+4, weight='bold')
ax.set_title('b       ', y=0.99, loc='left', va='top', ha='right', fontsize=fontsize+4, weight='bold')
ax2.set_title('c       ', y=0.99, loc='left', va='top', ha='right', fontsize=fontsize+4, weight='bold')

ax1.set_ylim(0, 20)
ax.set_ylim(0, 80)
ax2.set_ylim(-.15, .15)
ax2.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
ax2.set_xlim(-0.1, 0.7)

ax1.tick_params(direction='in', labelsize= fstk)
ax.tick_params(direction='in', labelsize= fstk)
ax2.tick_params(direction='in', labelsize= fstk)

plt.tight_layout()

if savefig:
    figname = 'FigureS9'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

######################## Figure S10 ########################
fig, ax = plt.subplots()

xval = Eeig_list
yval = np.array(spatial_list)[:, 0]
xv = np.linspace(0, 10)
pf = np.polyfit(xval, yval, 1)
print(pf)
yv = pf[0]*xv + pf[1]
ax.plot(xval, yval, 'ko')
ax.plot(xv, yv, '--k')

ax.set_ylim(0,80)
ax.set_xlabel(r'$E_{\rm eig}$ (meV)', fontsize=fs)
ax.set_ylabel(r'$n_{\rm STZ}$', fontsize=fs)
ax.tick_params(direction='in', labelsize=fstk)

ax.annotate('Event I', xy=(xval[0], yval[0]), 
            xytext=(0, 40), textcoords='offset points',
            ha='center', va='bottom', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='k', shrinkB=5))
ax.annotate('Event II', xy=(xval[1], yval[1]), 
            xytext=(40, 0), textcoords='offset points',
            ha='left', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='k', shrinkB=5))
ax.annotate('Event III', xy=(xval[2], yval[2]), 
            xytext=(-40, 0), textcoords='offset points',
            ha='right', va='center', fontsize=fs,
            arrowprops=dict(arrowstyle='-|>', color='k', shrinkB=5))
fig.tight_layout()

if savefig:
    figname='FigureS10'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

plt.show()