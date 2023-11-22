# Script for generating figure 1 in the paper
# Author: Yifan Wang
# Figure 4: predict the elastic limit based on the thermal activation model

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

########################### define functions ############################
import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

pfit = [1.28, 0]
ncpu = 16
fs, fstk = 18, 14

rawdata = loadmat(os.path.join(rawdata_dir, 'T20K_states.mat'))
state_dict_array = rawdata['state_dict_array'][0]
ST_dict_array = rawdata['ST_dict_array'][0]

from plot_func import load_ST_energy_strain, cal_dE_Eb, solve_strain_from_Eb

######################### Load data ###############################
from scipy.misc import derivative
from scipy.optimize import fsolve

iA, iB = 1, 2
emin, emax = -0.13, 0.3616
# iA, iB = 18, 19
# emin, emax = -2, 3.5
# iA, iB = 2, 7
# emin, emax = -0.5, 0.8
xv = np.linspace(emin, emax, 100)
strain, energy, Efun, emin_data, emax_data, eeig, Eeig, Efwd, Ebwd = load_ST_energy_strain(iA, iB, return_dict=False, return_Efwd_Ebwd=True)
yv = Efun(xv)
dEfwd, dEbwd, Ebfwd, Ebbwd = cal_dE_Eb(yv, return_dict=False)

EvT = 0.4639/2*1e-3
Eeig = Efun(eeig).max()-Efun(eeig)[0]
Teig = Eeig/EvT

Tlist = [1e-6, 2, 3, 5, 10, 20]
clist = ['k', 'C0', 'C6', 'C1', 'C2', 'C4']
# eT = np.array(Tlist)*EvT
gradEb = derivative(Efwd, eeig, dx=1e-7)
kB = 8.6173324e-5 # eV/K
h  = 4.135667696e-15 # eV.s
erate = 1e7*100 # %/s
nu0= 3.14e11
eT = EvT*np.array(Tlist)
for i in range(len(Tlist)):
    yvals = eT[i]*np.ones(2)*1000

x_offset = np.arange(len(Tlist))*0.4*10
epl = 0.15*10
lw = 2
shrk = 0
seig = eeig/pfit[0]

#################### plot figure ############################
version_arrangement = 'v10'
if version_arrangement == 'v10':
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(3, 3)

    ax4 = fig.add_subplot(gs[0, :2])
    ax0 = fig.add_subplot(gs[1:, :1])
    ax = fig.add_subplot(gs[1:, 1:2])
    ax2 = ax.twinx()
    ax1 = fig.add_subplot(gs[0:, 2:])
elif version_arrangement == 'v11':
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(5, 2)

    ax4 = fig.add_subplot(gs[0, :2])
    ax0 = fig.add_subplot(gs[1:3, :1])
    ax = fig.add_subplot(gs[1:3, 1:2])
    ax2 = ax.twinx()
    ax1 = fig.add_subplot(gs[3:5, :])


xv_curves = np.array([emin, (emin+eeig)/2, eeig, (emax+eeig)/2, emax])
dx_between_curves = 6
Ecurves = Efun(xv_curves)
Ecurves_plot = 1000*(Ecurves - Ecurves.min(axis=1, keepdims=True))

for i in range(xv_curves.size):
    xv_plot = np.arange(ncpu) + i*(ncpu+dx_between_curves)
    ax4.plot(xv_plot, Ecurves_plot[i, :], 'k-')
    ax4.annotate('1',  xy=(xv_plot[0]-1, Ecurves_plot[i, 0]), xycoords='data', ha='right', va='bottom', fontsize=fs,
                bbox=dict(boxstyle='circle,pad=0.1', fc='w'))
    ax4.annotate('2', xy=(xv_plot[-1]+1, Ecurves_plot[i,-1]), xycoords='data', ha='left', va='bottom', fontsize=fs, 
                bbox=dict(boxstyle='circle,pad=0.1', fc='w'))

ylim = ax4.get_ylim()
ax4.set_ylim([ylim[0], ylim[0] + 1.5*(ylim[1]-ylim[0])])
ax4.annotate('', xy=(0, 1.4*ylim[1]), xycoords='data', 
             xytext=((ncpu+dx_between_curves)*xv_curves.size-dx_between_curves-1, 1.4*ylim[1]), textcoords='data',
            arrowprops=dict(arrowstyle='<|-', lw=2))
ax4.annotate('Increasing Strain', xy=(np.mean(ax4.get_xlim()), 1.5*ylim[1]), xycoords='data',
             ha='center', va='bottom', fontsize=fs)

labels = [r'$\varepsilon_{\rm min}$', '', r'$\varepsilon_{\rm eig}$', '', r'$\varepsilon_{\rm max}$']
for i in range(xv_curves.size):
    xv_plot = np.arange(ncpu) + i*(ncpu+dx_between_curves)
    ax4.annotate(r'$\varepsilon=%.2f\,\%%$'%xv_curves[i], xy=(xv_plot[ncpu//2], 1.3*ylim[1]), xycoords='data', ha='center', va='top', fontsize=fstk)
    ax4.annotate(labels[i], xy=(xv_plot[ncpu//2], 0), xycoords='data', ha='center', va='top', fontsize=fs, color='r')

ax4.axis('off')

ax0.plot(xv, Ebfwd*1000, 'k')#'C%d--'%iAB)
ax0.plot(xv, Ebbwd*1000, 'k')#'C%d--'%iAB)
ax0.plot(xv, np.ones_like(xv)*Eeig*1000, 'r:')
ax0.annotate(r'$E_{\rm eig}$', xy=(xv[-1], Eeig*1000), xycoords='data', va='bottom', ha='right', fontsize=fstk, color='r')
ax0.annotate(r'$E_{\rm b}^{\rm fwd}$', xy=( xv[0],  Ebfwd[0]*1000), xycoords='data', va='bottom', ha='left', fontsize=fs)
ax0.annotate(r'$E_{\rm b}^{\rm bwd}$', xy=(xv[-10], Ebbwd[-10]*1000), xycoords='data', va='bottom', ha='right', fontsize=fs)

strain_lims = []
ET_lims = []

for i in range(len(Tlist)):
    ETi = eT[i]*1000
    xvals = solve_strain_from_Eb(eT[i], Efun, emin, emax)
    if Tlist[i] < 1e-3: xvals = (xvals[0], emax)
    print(Tlist[i], xvals)
    yvals = ETi*np.ones(2)

    strain_lims.append(xvals)
    ET_lims.append(yvals)

    color = clist[i]
    ax0.plot(xvals, yvals, '--o', color=color)
    if Tlist[i] == 20:
        ax0.plot(xvals[0]*np.ones(2), [0, ETi], '--', color=color)
        ax0.plot(xvals[1]*np.ones(2), [0, ETi], '--', color=color)

    if Tlist[i] > Teig:
        event_type = 'C'
        xvals = np.flip(xvals)
    elif xvals[0] > 0:
        event_type = 'B'
    else:
        event_type = 'A'
    
    if Tlist[i] == 20:
        xT, yT, ha, va = xvals[0], ETi, 'left', 'top'
    elif Tlist[i] == 10:
        xT, yT, ha, va = xvals[0], ETi, 'left', 'bottom'
    elif Tlist[i] == 5:
        xT, yT, ha, va = xvals[0], ETi+0.3, 'center', 'bottom'
    elif Tlist[i] == 3:
        xT, yT, ha, va = xvals[1], ETi+0.05, 'left', 'bottom'
    elif Tlist[i] == 2:
        xT, yT, ha, va = xvals[1], ETi, 'left', 'bottom'
    else:
        xT, yT, ha, va = xvals[1], ETi+0.05, 'right', 'bottom'
    ax0.annotate(r'$%d\,{\rm K}$ (%s)'%(Tlist[i], event_type), xy=(xT, yT), xycoords='data', ha=ha, va=va, fontsize=fstk, color=clist[i])

    ax.plot(np.array(xvals)+x_offset[i], np.array(xvals)/pfit[0], '.', color=color)
    ax.plot(np.array(xvals)+x_offset[i]+epl, np.array(xvals)/pfit[0], '.', color=color)
    ax2.plot(np.array(xvals)+x_offset[i], np.array(xvals), '.', color=color)
    ax2.plot(np.array(xvals)+x_offset[i]+epl, np.array(xvals), '.', color=color)
    epsl = np.linspace(xvals[0], xvals[1])
    sigl = epsl/pfit[0]
    eps_pl_fwd = np.array([0, epl]) + xvals[1]
    sig_pl_fwd = xvals[1]/pfit[0]*np.ones(2)
    epsu = epsl[::-1] + epl
    sigu = sigl[::-1]
    eps_pl_bwd = np.array([epl, 0]) + xvals[0]
    sig_pl_bwd = xvals[0]/pfit[0]*np.ones(2)

    if event_type == 'C':
        eps = np.concatenate([epsl, eps_pl_fwd, epsu, eps_pl_bwd])
        sig = np.concatenate([sigl, sig_pl_fwd, sigu, sig_pl_bwd])
        ax.fill(eps + x_offset[i], sig, color=color, alpha=0.5)

    ax.annotate('', xy=(xvals[1]+x_offset[i], xvals[1]/pfit[0]), xycoords='data',
                xytext=(0 + x_offset[i], 0), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, 
                                shrinkA=shrk,shrinkB=shrk)
               )
    ax.annotate('', xy=(xvals[1]+x_offset[i]+epl, xvals[1]/pfit[0]), xycoords='data',
                xytext=(xvals[1]+x_offset[i], xvals[1]/pfit[0]),   textcoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, 
                                shrinkA=shrk,shrinkB=shrk)
               )
    if xvals[0] < 0:
        ax.annotate('', xy=(0 + x_offset[i] + epl, 0), xycoords='data',
                    xytext=(xvals[1]+x_offset[i]+epl, xvals[1]/pfit[0]), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, linestyle='-', 
                                    shrinkA=shrk,shrinkB=shrk)
                   )
        ax.annotate('', xy=(0 + x_offset[i], 0), xycoords='data',
                    xytext=(xvals[0]+x_offset[i], xvals[0]/pfit[0]), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, linestyle=':', 
                                    shrinkA=shrk,shrinkB=shrk)
                   )
        ax.annotate('', xy=(xvals[0]+x_offset[i]+epl, xvals[0]/pfit[0]), xycoords='data',
                    xytext=(xvals[1]+x_offset[i]+epl, xvals[1]/pfit[0]), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, linestyle=':',
                                    shrinkA=shrk,shrinkB=shrk)
                   )
        ax.annotate('', xy=(xvals[0]+x_offset[i], xvals[0]/pfit[0]), xycoords='data',
                    xytext=(xvals[0]+x_offset[i]+epl, xvals[0]/pfit[0]), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, linestyle=':',
                                    shrinkA=shrk,shrinkB=shrk)
                   )
        ax.plot(epl+x_offset[i], 0, 'o', color=color)
        if event_type == 'C':
            ax.annotate('', xy=(0 + x_offset[i], 0), xycoords='data',
                    xytext=(0 + x_offset[i] + epl, 0), textcoords='data',
                    arrowprops=dict(arrowstyle='<|-|>', lw=lw, color=color, 
                                    shrinkA=shrk,shrinkB=shrk)
                   )
    else:
        ax.annotate('', xy=(xvals[0]+x_offset[i]+epl, xvals[0]/pfit[0]), xycoords='data',
                    xytext=(xvals[1]+x_offset[i]+epl, xvals[1]/pfit[0]), textcoords='data',
                    arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, 
                                    shrinkA=shrk,shrinkB=shrk)
                   )
        ax.annotate('', xy=(xvals[0]+x_offset[i], xvals[0]/pfit[0]), xycoords='data',
                xytext=(xvals[0]+x_offset[i]+epl, xvals[0]/pfit[0]), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, 
                                shrinkA=shrk,shrinkB=shrk)
               )
    ax.plot(0 + x_offset[i], 0, 'o', color=color)
    if Tlist[i] == 20:
        xT, yT, ha, va = x_offset[i], -0.15, 'left', 'top'
    elif Tlist[i] == 10:
        xT, yT, ha, va = x_offset[i], -0.15, 'center', 'top'
        ax2.annotate(r'$\varepsilon_{\rm c}^{\rm fwd}$', xy=(xvals[0]+x_offset[i], xvals[0]), xycoords='data', ha='left', va='top', color=color, fontsize=fs)
        ax2.annotate(r'$\varepsilon_{\rm c}^{\rm bwd}$', xy=(xvals[1]+x_offset[i], xvals[1]), xycoords='data', ha='left', va='bottom', color=color, fontsize=fs)
    elif Tlist[i] == 5:
        xT, yT, ha, va = x_offset[i], -0.01, 'left', 'top'
    elif Tlist[i] == 3:
        xT, yT, ha, va = x_offset[i], -0.08, 'left', 'top'
    elif Tlist[i] == 2:
        xT, yT, ha, va = x_offset[i], -0.11, 'left', 'top'
        ax2.annotate(r'$\varepsilon_{\rm c}^{\rm bwd}$', xy=(xvals[0]+x_offset[i], xvals[0]), xycoords='data', ha='left', va='top', color=color, fontsize=fs)
        ax2.annotate(r'$\varepsilon_{\rm c}^{\rm fwd}$', xy=(xvals[1]+x_offset[i], xvals[1]), xycoords='data', ha='left', va='bottom', color=color, fontsize=fs)
    else:
        xT, yT, ha, va = x_offset[i], -0.15, 'left', 'top'
    ax2.annotate('%dK (%s)'%(Tlist[i], event_type), xy=(xT, yT), xycoords='data', va=va, ha=ha, fontsize=fstk, color=clist[i])

xlim = ax.get_xlim()
xlim = (xlim[0], xlim[0]+(xlim[1]-xlim[0])*1.05)
ax.plot([xlim[0], x_offset[-1]+epl], np.zeros_like(xlim), '--k')
ax2.plot([0, xlim[1]], np.ones_like(xlim)*eeig, 'r:')
ax.set_xlim(xlim)
ax.set_ylim(-0.15)
ax2.set_ylim((ax.get_ylim()[0]*pfit[0], ax.get_ylim()[1]*pfit[0]))
ax2.tick_params(axis='y', labelsize=fstk)
ax.set_xticks([])
ax.tick_params(axis='y', labelsize=fstk)
ax.set_xlabel(r'$\varepsilon_{\rm pl}$ (%)', fontsize=fs)
ax.set_ylabel(r'$\sigma$ (GPa)', fontsize=fs)
ax2.set_ylabel(r'$\varepsilon$ (%)', fontsize=fs)
ax2.annotate(r'  $\varepsilon_{\rm eig}$  ', xy=(xlim[1], eeig), xycoords='data', 
             ha='right', va='bottom', fontsize=fstk, color='r')

ax0.set_xlabel('Strain (%)', fontsize=fs)
ax0.set_ylabel(r'$E_{\rm b}$ (meV)', fontsize=fs)
ax0.tick_params(labelsize=fstk, direction='in')
ax.tick_params(labelsize=fstk, direction='in')

############################################################
iABlist = [(1, 2), (2, 7), (18, 19)]
labels = ['I', 'III', 'V']
TtypeB = 1
xvlist = []
yvlist = []

for i in range(3):
    iA, iB = iABlist[i]
    print('Event %s: state %d - %d'%(labels[i], iA, iB))
    strain, energy, Efun, emin_data, emax_data, eeig, Eeig, Efwd, Ebwd = load_ST_energy_strain(iA, iB, return_dict=False, return_Efwd_Ebwd=True)
    print('  eigen barrier: %.4f%% -- %.4f meV'%(eeig, 1000*Eeig))
    gradEb = derivative(Efwd, eeig, dx=1e-7)
    print('  gradient at eigen strain: %.4f meV'%(gradEb*1000))
    cEbfwd, cEbbwd = Eeig - gradEb*eeig,  Eeig + gradEb*eeig
    print('  intercept: fwd %.4f meV bwd %.4f meV'%(cEbfwd*1000, cEbbwd*1000))
    etypeB = (cEbbwd - cEbfwd)/gradEb
    print('  strain of A->B transition: %.4f%%'%etypeB)
    emin, emax = eeig + Eeig/gradEb, eeig - Eeig/gradEb
    print('  strain range: (%.4f%%, %.4f%%)'%(emin, emax))
    Efnew = lambda x: gradEb*x + cEbfwd
    Ebnew = lambda x: -gradEb*x+ cEbbwd
    
    xv = np.linspace(emin_data, etypeB, 100)
    ax1.plot(Efwd(xv)/EvT, xv, '--C%d'%i)
    if iB == 7:
        emax_plot = emax_data+0.1
    elif iB == 19:
        emax_plot = emax_data-0.5
    else:
        emax_plot = emax_data
    xv = np.linspace(etypeB, emax_plot, 100)
    kB = 8.6173324e-5 # eV/K
    h  = 4.135667696e-15 # eV.s
    erate = 1e7*100 # %/s
    nu0= 3.14e11
    solveTv = lambda T: Efwd(xv) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
    Tv = fsolve(solveTv, np.ones_like(xv)*Efwd(etypeB)/EvT)
    ax1.plot(Tv, xv, '--C%d'%i)

    solveTAB= lambda T: Efwd(etypeB) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
    TAB= fsolve(solveTAB,Efwd(etypeB)/EvT)
    TAB = Efwd(etypeB)/EvT
    ax1.plot(TAB, etypeB, '^C%d'%i, markersize=12)
    solveTe = lambda T: Efwd(eeig) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
    Te = fsolve(solveTe, Efwd(eeig)/EvT)
    
    EtypeB = kB*TtypeB*np.log(-(kB*TtypeB)*nu0/erate/gradEb)
    print(EtypeB)
    func_e0 = lambda x: Efwd(x) - EtypeB
    e0 = root_scalar(func_e0, method='bisect', bracket=[etypeB, emax_plot]).root
    print(TtypeB, e0, Efwd(e0), EtypeB)
    xv0 = np.linspace(e0, etypeB, 100)
    solveyv = lambda T: Efwd(xv0) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
    yv = fsolve(solveyv, np.ones_like(xv0)*Efwd(etypeB)/EvT)
    xvlist.append(xv0)
    yvlist.append(yv)
    TtypeB = yv[-1] # Efwd(etypeB)
    print(TtypeB, Efwd(etypeB))
    
    if iB == 2:
        yval = emin_data
    elif iB == 7:
        yval = (eeig+etypeB)/2
    elif iB == 19:
        yval = 3.12
    xval = Efwd(yval)/EvT
    ax1.annotate('Event %s'%labels[i], xy=(xval, yval), xycoords='data', ha='left', va='bottom', fontsize=fs)#, fontfamily='serif')

templist = [0, 2, 5, 10, 15, 20, 30]
elimlist = [0.3616, 0.1948, 0.562, 0.4388, 0.422, 3.228, 3.16]
MDmark, = ax1.plot(templist, elimlist, '*r', label='MD data', markersize=12)

xvplot = np.concatenate(xvlist)
yvplot = np.concatenate(yvlist)
ax1.plot(yvplot, xvplot, '-k', label='Model', lw=2)

ax1.tick_params(labelsize=fstk, direction='in')
ax1.set_xlabel('T (K)', fontsize=fs)
ax1.set_ylabel(r'$\varepsilon_{\rm lim}$ (%)', fontsize=fs)
ax1.set_ylim(-0.14)
ax1.set_xlim(-2,52)
ax1.yaxis.set_label_position('right')
ax1.yaxis.tick_right()


ax1.annotate('Eq.(1) Prediction', xy=(18, 1.2), xytext=(40, 0), textcoords='offset points', 
             fontsize=fs, ha='left', va='center',
             arrowprops=dict(arrowstyle='-|>', lw=2))
ax1.annotate('MD Results', xy=(templist[-1], elimlist[-1]), xytext=(0, -40), textcoords='offset points', 
             fontsize=fs, ha='center', va='top', color='r',
             arrowprops=dict(arrowstyle='-|>', lw=2, color='r', shrinkB=5))

############################################################

ax4.set_title(r'${\bf a}$  ', fontsize=fs, loc='left', ha='right')
ax0.set_title(r'${\bf b}$  ', y=0.9, fontsize=fs, loc='left', ha='right')
ax.set_title(r'${\bf c}$      ', y=0.9, fontsize=fs, loc='left', ha='right')
ax1.set_title(r'${\bf d}$  ', fontsize=fs, loc='left', ha='right')

fig.tight_layout()

if savefig:
    figname = 'Figure4'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))


###################################################################
############## Figure S8: strain rate dependence ##################

iA, iB = 1, 2
emin, emax = -0.13, 0.3616
# iA, iB = 18, 19
# emin, emax = -2, 3.5
# iA, iB = 2, 7
# emin, emax = -0.5, 0.8
xv = np.linspace(emin, emax, 100)
strain, energy, Efun, emin_data, emax_data, eeig, Eeig, Efwd, Ebwd = load_ST_energy_strain(iA, iB, return_dict=False, return_Efwd_Ebwd=True)
yv = Efun(xv)
dEfwd, dEbwd, Ebfwd, Ebbwd = cal_dE_Eb(yv, return_dict=False)

Eeig = Efun(eeig).max()-Efun(eeig)[0]
Teig = Eeig/EvT

Tlist = [1e-6, 2, 3, 5, 10, 20]
clist = ['k', 'C0', 'C6', 'C1', 'C2', 'C4']
gradEb = derivative(Efwd, eeig, dx=1e-7)
kB = 8.6173324e-5 # eV/K
h  = 4.135667696e-15 # eV.s
erate = 1e7*100 # %/s
nu0= 3.14e11
eT = EvT*np.array(Tlist)
for i in range(len(Tlist)):
    yvals = eT[i]*np.ones(2)*1000

x_offset = np.arange(len(Tlist))*0.4*10
epl = 0.15*10
lw = 2
shrk = 0
seig = eeig/pfit[0]

fig, ax1 = plt.subplots(figsize=(5,4))

xv_curves = np.array([emin, (emin+eeig)/2, eeig, (emax+eeig)/2, emax])
dx_between_curves = 6
Ecurves = Efun(xv_curves)
Ecurves_plot = 1000*(Ecurves - Ecurves.min(axis=1, keepdims=True))

############################################################
iABlist = [(1, 2), (2, 7), (18, 19)]
labels = ['I', 'III', 'V']

for ierate, erate in enumerate([1e7*100, 1e4*100, 1e1*100, 1e-2*100]):
    TtypeB = 1
    xvlist = []
    yvlist = []
    
    for i in range(len(iABlist)):
        iA, iB = iABlist[i]
        print('Event %s: state %d - %d'%(labels[i], iA, iB))
        strain, energy, Efun, emin_data, emax_data, eeig, Eeig, Efwd, Ebwd = load_ST_energy_strain(iA, iB, return_dict=False, return_Efwd_Ebwd=True)
        print('  eigen barrier: %.4f%% -- %.4f meV'%(eeig, 1000*Eeig))
        gradEb = derivative(Efwd, eeig, dx=1e-7)
        print('  gradient at eigen strain: %.4f meV'%(gradEb*1000))
        cEbfwd, cEbbwd = Eeig - gradEb*eeig,  Eeig + gradEb*eeig
        print('  intercept: fwd %.4f meV bwd %.4f meV'%(cEbfwd*1000, cEbbwd*1000))
        etypeB = (cEbbwd - cEbfwd)/gradEb
        print('  strain of A->B transition: %.4f%%, %.4f meV'%(etypeB, Efwd(etypeB)*1000))
        emin, emax = eeig + Eeig/gradEb, eeig - Eeig/gradEb
        print('  strain range: (%.4f%%, %.4f%%)'%(emin, emax))
        Efnew = lambda x: gradEb*x + cEbfwd
        Ebnew = lambda x: -gradEb*x+ cEbbwd

        if iB == 7:
            emax_plot = emax_data+0.2
        elif iB == 19:
            emax_plot = emax_data-0.1
        else:
            emax_plot = emax_data-0.05
        emax_plot = emax_data+0.2
        emin, emax = emin_data, emax_plot
        xv = np.linspace(etypeB, emax_plot, 100)
        xv = np.linspace(1e-3, emax_plot, 100)
        kB = 8.6173324e-5 # eV/K
        h  = 4.135667696e-15 # eV.s
        nu0= 3.14e11
        solveTv = lambda T: Efwd(xv) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
        Tv = fsolve(solveTv, np.ones_like(xv)*Efwd(etypeB)/EvT)
        ind = np.logical_and(Tv > Tv.min(), Tv > 1e-3)
        xv = xv[ind]
        Tv = Tv[ind]
        # ax1.plot(Tv, xv, '--C%d'%i)

        solveTAB = lambda T: Efwd(etypeB) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
        TAB = fsolve(solveTAB, Efwd(etypeB)/EvT)
        # ax1.plot(TAB, etypeB, '^C%d'%i, markersize=12)
        solveTe = lambda T: Efwd(eeig) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
        Te = fsolve(solveTe, Efwd(eeig)/EvT)

        EtypeB = kB*TtypeB*np.log(-(kB*TtypeB)*nu0/erate/gradEb)
        print(EtypeB)
        func_e0 = lambda x: Efwd(x) - EtypeB
        print(emin, func_e0(emin), emax, func_e0(emax))
        if i == 0:
            e0 = emax
        else:
            e0 = root_scalar(func_e0, method='bisect', bracket=[emin, emax]).root
        xv0 = np.linspace(e0, etypeB, 100)
        solveyv = lambda T: Efwd(xv0) - kB*T*np.log(-(kB*T)*nu0/erate/gradEb)
        yv = fsolve(solveyv, np.ones_like(xv0)*Efwd(etypeB)/EvT)
        
        if i == 0:
            ind = np.logical_and(yv > yv.min(), yv > 1e-3)
            xv0 = xv0[ind]
            yv  = yv[ind]
        
        xvlist.append(xv0)
        yvlist.append(yv)
        TtypeB = yv[-1]
    
    if ierate == 0:
        ax1.plot(np.concatenate(yvlist), np.concatenate(xvlist), '-k') #, label='Model') #, lw=2)
    else:
        ax1.plot(np.concatenate(yvlist), np.concatenate(xvlist), '-C%d'%(ierate-1))

# templist = [0, 2, 5, 10, 15, 20, 30]
# elimlist = [0.3616, 0.1948, 0.562, 0.4388, 0.422, 3.228, 3.16]
rawdata_elim = np.loadtxt(os.path.join(rawdata_dir, 'Figure2b.txt'))
templist = rawdata_elim[:, 0]
elimlist = rawdata_elim[:, 1]
MDmark, = ax1.plot(templist, elimlist, '*r', label='MD data', markersize=6)

ax1.tick_params(labelsize=fstk, direction='in')
ax1.set_xlabel('T (K)', fontsize=fs)
ax1.set_ylabel(r'$\varepsilon_{\rm lim}$ (%)', fontsize=fs)
ax1.set_ylim(0, 4)
ax1.set_xlim(-2, 22)
ax1.yaxis.set_label_position('right')
ax1.yaxis.tick_right()

############################################################

fig.tight_layout()

if savefig:
    figname = 'FigureS8'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

plt.show()