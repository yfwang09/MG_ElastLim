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

from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import minimize

rawdata = loadmat(os.path.join(rawdata_dir, 'T20K_states.mat'))
state_dict_array = rawdata['state_dict_array'][0]
ST_dict_array = rawdata['ST_dict_array'][0]

ref_energy = lambda x: 1.5*x**2 + state_dict_array[0]['energy'][0, 0][0][0]

wstrip = 0.5
ncpu = 16
nfit = 102
arrow_thresh = 0.01

e_contour = np.insert(np.arange(0.2, 1.1, 0.1), 0, 2e-6)
E_offset = 2.5e-4

fs = 18

# Figure 3abc
###########################################################

def plot_states(fig, ax, plot_state_order, state_position, label_y_position=None, savedata=None):
    if label_y_position is None:
        label_y_position = -0.035*np.ones_like(plot_state_order)
    for i in range(len(plot_state_order)):
        state_i = None
        for j in range(1, len(state_dict_array)):
            if state_dict_array[j]['state_id'] == plot_state_order[i]:
                state_i = state_dict_array[j]
                break
        if state_i is None: raise ValueError()
        state_id = state_i['state_id'][0, 0][0][0]
        strain = state_i['strain'][0, 0][0]
        energy_raw = state_i['energy'][0, 0][0]
        energy_ref = ref_energy(strain)
        energy = energy_raw - energy_ref
        emin = state_i['emin'][0, 0][0][0]
        emax = state_i['emax'][0, 0][0][0]
        Efun = interp1d(strain[emin:emax], energy[emin:emax], fill_value='extrapolate')

        # draw energy stripes
        yi = state_position[i]
        collapse_flag = (yi > np.floor(yi))
        yval = np.array([0, wstrip]) + state_position[i]
        zval = energy[emin:emax]
        X = yval[[0,1,1,0]]
        Y = np.array([zval.min(), zval.min(), zval.max(), zval.max()])
        ax.fill(X, Y, color=(0.9, 0.9, 0.9), ec=None)
        
        # draw strain contour lines
        for j in range(len(e_contour)):
            if e_contour[j] > strain[emin] and e_contour[j] < strain[emax]:
                ax.plot(yval, Efun(e_contour[j])*np.ones_like(yval), color=(0.5, 0.5, 0.5))

        # draw state id
        if not collapse_flag:
            ax.annotate('%d'%state_id, [yi+wstrip/2, label_y_position[i]], textcoords='data', fontsize=fs, 
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='circle,pad=0.1', fc='w'))

def plot_MEP(fig, ax, plot_state_order, state_position, plot_MEP_order, savedata=None):
    for i_MEP in plot_MEP_order:
        iA, iB = plot_state_order[i_MEP[0]], plot_state_order[i_MEP[1]]
        for j in range(len(ST_dict_array)):
            ST_i = ST_dict_array[j]
            if ST_i['state_A'][0, 0][0][0] == iA and ST_i['state_B'][0, 0][0][0] == iB:
                energy = ST_i['saved_MEP'][0, 0].squeeze()
                break
            elif ST_i['state_A'][0, 0][0][0] == iB and ST_i['state_B'][0, 0][0][0] == iA:
                energy = ST_i['saved_MEP'][0, 0][:, ::-1].squeeze()
                break
        strain = ST_i['strain'][0, 0][0]
        energy = energy - ref_energy(strain)[:, None]
        fit_range = np.round(np.linspace(0, strain.size-1, nfit)).astype(int)
        Efun = interp1d(strain[fit_range], energy[fit_range, :], fill_value='extrapolate', axis=0)
        energy_fit = Efun(strain)
        emin_ST, emax_ST = ST_i['emin'][0, 0][0][0], ST_i['emax'][0, 0][0][0]
        
        # Draw energy barrier strip
        yi1, yi2 = state_position[i_MEP[0]], state_position[i_MEP[1]]
        state_v = np.linspace(yi1 + wstrip, yi2, ncpu)
        Emin, Emax = energy[0, :], energy[-1, :]
        X = np.concatenate([state_v, state_v[::-1]])
        Y = np.concatenate([Emin, Emax[::-1]])
        ax.fill(X, Y, color=(0.97, 0.97, 0.97), ec=None)
        
        # Draw strain contour lines
        for j in range(len(e_contour)):
            if e_contour[j] > strain[0] and e_contour[j] < strain[-1]:
                ax.plot(state_v, Efun(e_contour[j]) + E_offset, color=(0.5, 0.5, 0.5))
        
        # Draw eigen barrier
        state_iA, state_iB = None, None
        for j in range(len(state_dict_array)):
            state_j = state_dict_array[j]
            if state_j['state_id'][0, 0][0][0] == iA:
                state_iA = state_dict_array[j]
            elif state_j['state_id'][0, 0][0][0] == iB:
                state_iB = state_dict_array[j]
        if state_iA is None or state_iB is None: raise ValueError()
        strain_A, strain_B = state_iA['strain'][0, 0][0], state_iB['strain'][0, 0][0]
        energy_A, energy_B = state_iA['energy'][0, 0][0], state_iB['energy'][0, 0][0]
        emin_A, emax_A = state_iA['emin'][0, 0][0][0], state_iA['emax'][0, 0][0][0]
        emin_B, emax_B = state_iB['emin'][0, 0][0][0], state_iB['emax'][0, 0][0][0]
        Efun_A = interp1d(strain_A[emin_A:emax_A], energy_A[emin_A:emax_A], fill_value='extrapolate')
        Efun_B = interp1d(strain_B[emin_B:emax_B], energy_B[emin_B:emax_B], fill_value='extrapolate')
        dEfunAB= lambda x: np.abs(Efun_A(x) - Efun_B(x))
        x1 = np.maximum(strain_A[emin_A], strain_B[emin_B])
        x2 = np.minimum(strain_A[emax_A], strain_B[emax_B])
        res = minimize(dEfunAB, x0=(x1+x2)/2, method='Powell')
        e_eig = res.x
        if e_eig > x1 + 1e-3 and e_eig < x2 - 1e-3:
            line_eig, = ax.plot(state_v, Efun(e_eig).flatten(), color='r', linewidth=3)
    return line_eig
            
def plot_loading(fig, ax, state_load_array, ST_load_array, pos_offset=0):
    lw = 3
    for i in range(len(state_load_array)):
        state_load_i = state_load_array[i]
        state_range = state_load_i['range']
        collapse_flag = (state_load_i['pos'] > np.floor(state_load_i['pos']))
        state_v = state_load_i['pos'] + state_load_i['dx'] + pos_offset
        for j in range(len(state_dict_array)):
            if state_dict_array[j]['state_id'][0,0][0][0] == state_load_i['IS']:
                state_dict_i = state_dict_array[j]
                break
        strain = state_dict_i['strain'][0, 0][0]
        energy = state_dict_i['energy'][0, 0][0] - ref_energy(strain)
        Efun = interp1d(strain, energy, fill_value='extrapolate')
        yval = Efun(state_range)
        if np.abs(yval[1] - yval[0]) > arrow_thresh and (not collapse_flag):
            ax.annotate('', xy=[state_v, yval[1]], xycoords='data', 
                            xytext=[state_v, yval[0]], textcoords='data', 
                        arrowprops=dict(arrowstyle='-|>', color=state_load_i['color'], 
                                        lw=lw, connectionstyle='angle',
                                        shrinkA=0, shrinkB=0.001
                                       )
                       )
        else:
            ax.plot(state_v*np.ones(2), yval, color=state_load_i['color'], linewidth=lw)

    for i in range(len(ST_load_array)):
        e_ev_i = ST_load_array[i]
        color = e_ev_i['color']
        ST_range = e_ev_i['range']
        iA, iB = e_ev_i['iA'], e_ev_i['iB']
        pos = np.array(e_ev_i['pos']) + pos_offset
        for j in range(len(ST_dict_array)):
            ST_i = ST_dict_array[j]
            if ST_i['state_A'][0, 0][0][0] == iA and ST_i['state_B'][0, 0][0][0] == iB:
                energy = ST_i['saved_MEP'][0, 0].squeeze()
                break
            elif ST_i['state_A'][0, 0][0][0] == iB and ST_i['state_B'][0, 0][0][0] == iA:
                energy = ST_i['saved_MEP'][0, 0][:, ::-1].squeeze()
                break
        
        xl, xl1 = pos[0] + e_ev_i['dx'], pos[0] + wstrip
        xr, xr1 = pos[1] + e_ev_i['dx'], pos[1]
        state_v = np.linspace(xl1, xr1, ncpu)
        strain = ST_i['strain'][0, 0][0]
        energy = energy - ref_energy(strain)[:, None]
        fit_range = np.round(np.linspace(0, energy.shape[0]-1, nfit)).astype(int)
        Efun = interp1d(strain[fit_range], energy[fit_range, :], fill_value='extrapolate', axis=0)
        
        if len(ST_range) == 1:
            Eplot = Efun(ST_range[0]) + E_offset
            line, = ax.plot(state_v, Eplot, color=color, linewidth=lw)
            ax.plot([xl, state_v[0]],  Eplot[0]*np.ones(2),  color=color, linewidth=lw)
            ax.plot([state_v[-1], xr], Eplot[-1]*np.ones(2), color=color, linewidth=lw)
        else:
            Eplot1 = Efun(ST_range[0]) + E_offset
            Eplot2 = Efun(ST_range[1]) + E_offset
            X = np.concatenate([[xl, ], state_v, [xr, xr], state_v[::-1], [xl,],], axis=-1)
            Y = np.concatenate([Eplot1[0:1], Eplot1, [Eplot1[-1], Eplot2[-1]], Eplot2[::-1], Eplot2[0:1]], axis=-1)
            ax.fill(X, Y, color=color, alpha=0.5, linestyle=None)
    return line

def mark_label_grids(fig, ax, label_grids):
    for igrid in range(len(label_grids)):
        state_v, i_IS, ei = tuple(label_grids[igrid])
        for j in range(1, len(state_dict_array)):
            if state_dict_array[j]['state_id'][0, 0][0][0] == i_IS:
                state_j = state_dict_array[j]
                break
        id_j = state_j['state_id'][0, 0][0][0]
        strain_j = state_j['strain'][0, 0][0]
        energy_j = state_j['energy'][0, 0][0] - ref_energy(strain_j)
        fit_range = np.round(np.linspace(0, strain_j.size-1, nfit)).astype(int)
        Efun_j = interp1d(strain_j[fit_range], energy_j[fit_range], fill_value='extrapolate')
        Eplot = Efun_j(ei)
        ax.annotate('$' + r'\varepsilon = %.1f'%(ei,) + '\%$', 
                    xy=(state_v + 1.1* wstrip, Eplot), xycoords='data', 
                    va='center', fontsize=fstk
                   )

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_ylim([-0.02, 0.25])

plot_state_order_list = [[1, 2], 
                         [1, 2, 4, 6, 7, 8], 
                         [1, 2, 7, 6, 8,]
                        ]
state_position_list = [[1, 2], 
                       [3, 4, 5, 6, 7, 8],
                       [9, 10, 11, 12, 12, ]
                      ]
plot_MEP_order_list = [[[0, 1], ],
                       [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
                       [[0, 1], [1, 2], [2, 3], [2, 4], ]
                      ]
label_y_position_list = [[-0.015, -0.015], 
                         [-0.015, -0.015, -0.015, -0.015, -0.015, -0.015],
                         [-0.015, -0.015, -0.015, 0.205, -0.015, ]]

for i in range(len(plot_state_order_list)):
    plot_state_order = plot_state_order_list[i]
    state_position = state_position_list[i]
    plot_MEP_order = plot_MEP_order_list[i]
    
    plot_states(fig, ax, plot_state_order, state_position, label_y_position=label_y_position_list[i], savedata=os.path.join(figure_dir, 'Figure3_state.txt'))
    line_eig = plot_MEP(fig, ax, plot_state_order, state_position, plot_MEP_order, savedata=os.path.join(figure_dir, 'Figure3_Eb.txt'))

label_lines = [line_eig, ]

############ 0K
dx = 1/3*wstrip
color = 'k'
state_load_array = [{'pos': 1, 'IS': 1, 'range': [0, 0.3616], 'dx': dx, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.3616, 0], 'dx': dx, 'color': color},
                   ]
ST_load_array = [{'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.3616, ], 'dx': dx, 'color': color, 
                  'text': r'$\varepsilon_{\rm lim}(0{\rm K})=0.3616\%$'},
                ]

line = plot_loading(fig, ax, state_load_array, ST_load_array)
label_lines.append(line)

############ 2K
dx = 2/3*wstrip
color = 'C0'
state_load_array = [{'pos': 1, 'IS': 1, 'range': [0, 0.1948], 'dx': dx, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.1948, 0], 'dx': dx, 'color': color},
                   ]
ST_load_array = [{'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.1948, ], 'dx': dx, 'color': color,
                  'text': r'$\varepsilon_{\rm lim}(2{\rm K})=0.1948\%$'},
                ]

line = plot_loading(fig, ax, state_load_array, ST_load_array)
label_lines.append(line)

############ 5K
dxl= 1/4*wstrip
dx = 2/4*wstrip
dxu= 3/4*wstrip
color = 'C1'
pos_offset = 2
state_load_array = [{'pos': 1, 'IS': 1, 'range': [0, 0.1578], 'dx': dxu, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.3, 0.0468], 'dx': dxu, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.1578, 0.4600], 'dx': dx, 'color': color},
                    {'pos': 3, 'IS': 4, 'range': [0.3832, 0.5540], 'dx': dx, 'color': color},
                    {'pos': 4, 'IS': 6, 'range': [0.5540, 0.3834], 'dx': dx, 'color': color},
                    {'pos': 5, 'IS': 7, 'range': [0.4780, 0.0000], 'dx': dx, 'color': color},
                    {'pos': 6, 'IS': 8, 'range': [0.1160, 0.0000], 'dx': dx, 'color': color},
                   ]
ST_load_array = [{'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.0468, 0.1578], 'dx': dxu, 'color': color},
                 {'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.1578], 'dx': dxu, 'color': color},
                 {'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.0468], 'dx': dxu, 'color': color},
                 {'pos': [2, 3], 'iA': 2, 'iB': 4, 'range': [0.3832, 0.4600], 'dx': dx, 'color': color},
                 {'pos': [3, 4], 'iA': 4, 'iB': 6, 'range': [0.5540, ], 'dx': dx, 'color': color,
                  'text': r'$\varepsilon_{\rm lim}(5{\rm K})=0.5540\%$'},
                 {'pos': [4, 5], 'iA': 6, 'iB': 7, 'range': [0.3834, 0.4762], 'dx': dx, 'color': color},
                 {'pos': [5, 6], 'iA': 7, 'iB': 8, 'range': [0.0140, 0.1160], 'dx': dx, 'color': color},
                ]

line = plot_loading(fig, ax, state_load_array, ST_load_array, pos_offset=pos_offset)
label_lines.append(line)

############ 10K
dx = 1/4*wstrip
color = 'C2'
pos_offset = 8
state_load_array = [{'pos': 2, 'IS': 2, 'range': [0.0000, 0.4388], 'dx': dx, 'color': color},
                    {'pos': 1, 'IS': 4, 'range': [0.3572, 0.4388], 'dx': dx, 'color': color},
                    {'pos': 3, 'IS': 7, 'range': [0.4388, 0.0000], 'dx': dx, 'color': color},
                    {'pos': 4, 'IS': 6, 'range': [0.4388, 0.3796], 'dx': dx, 'color': color},
                   ]
ST_load_array = [{'pos': [1, 2], 'iA': 4, 'iB': 2, 'range': [0.3572, 0.4388], 'dx': dx, 'color': color},
                 {'pos': [2, 3], 'iA': 2, 'iB': 7, 'range': [0.4388, ], 'dx': dx, 'color': color},
                 {'pos': [3, 4], 'iA': 7, 'iB': 6, 'range': [0.3796, 0.4388], 'dx': dx, 'color': color},
                ]

############ 20K
dx = 2/4*wstrip
dxu = 3/4*wstrip
color = 'C4'
pos_offset = 8
state_load_array = [{'pos': 1, 'IS': 1, 'range': [0.0000, 0.3500], 'dx': dx, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.0000, 0.3500], 'dx': dx, 'color': color},
                    {'pos': 3, 'IS': 7, 'range': [0.3500, 0.5800], 'dx': dx, 'color': color},
                    {'pos': 4, 'IS': 6, 'range': [0.5800, 0.3796], 'dx': dx, 'color': color},
                    {'pos': 4, 'IS': 8, 'range': [0.3404, 0.1840], 'dx': dxu, 'color': color},
                    {'pos': 3, 'IS': 7, 'range': [0.3796, 0.1840], 'dx': dxu, 'color': color},
                    {'pos': 2, 'IS': 2, 'range': [0.1840, 0.0000], 'dx': dxu, 'color': color},
                   ]
ST_load_array = [{'pos': [1, 2], 'iA': 1, 'iB': 2, 'range': [0.0000, 0.3500], 'dx': dx, 'color': color},
                 {'pos': [2, 3], 'iA': 2, 'iB': 7, 'range': [0.3500, ], 'dx': dx, 'color': color},
                 {'pos': [3, 4], 'iA': 7, 'iB': 6, 'range': [0.3796, 0.5800], 'dx': dx, 'color': color},
                 {'pos': [3, 4], 'iA': 7, 'iB': 6, 'range': [0.5800, ], 'dx': dx, 'color': color},
                 {'pos': [3, 4], 'iA': 7, 'iB': 6, 'range': [0.3796, ], 'dx': dx, 'color': color},
                 {'pos': [2, 3], 'iA': 2, 'iB': 7, 'range': [0.1840, ], 'dx': dxu, 'color': color},
                 {'pos': [3, 4], 'iA': 7, 'iB': 8, 'range': [0.1840, 0.3404], 'dx': dxu, 'color': color},
                ]

line = plot_loading(fig, ax, state_load_array, ST_load_array, pos_offset=pos_offset)
label_lines.append(line)

#################
label_grids = [[2, 2, 0.4], 
               [2, 2, 0.5], [2, 2, 0.6],
               [4, 2, 2e-6], [4, 2, 0.2],
               [7, 7, 0.4], [7, 7, 0.5], [7, 7, 0.6], 
               [5, 4, 0.7],
               [12, 8, 2e-6], [12, 8, 0.2], [12, 8, 0.3],
               [12, 6, 0.4], [12, 6, 0.5], [12, 6, 0.6], 
              ]
mark_label_grids(fig, ax, label_grids)

ax.set_xticks([])
ax.set_yticks(np.arange(0, 0.30, 0.05).round(decimals=2))
ax.tick_params(axis='y', labelsize=fstk)
ax.set_xlabel(r'State ID', fontsize=fs)
ax.set_ylabel(r'$E(\varepsilon)-E_{\rm ref}$', fontsize=fs)
ax.set_xlim([0.5, 14])
ax.annotate(r'${\bf a}$ 0K & 2K', xy=(1, 0.20), xycoords='data', ha='left', va='top', fontsize=fs)
ax.annotate(r'${\bf b}$ 5K', xy=(4.8, 0.20), xycoords='data', ha='right', va='top', fontsize=fs)
ax.annotate(r'${\bf c}$ 20K', xy=(10, 0.20), xycoords='data', ha='right', va='top', fontsize=fs)

fig.tight_layout()

if savefig:
    figname = 'Figure3abc'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))


# Figure 3def
###########################################################
figsize = (12, 2)

fig, ax = plt.subplots(figsize=figsize)

node_list = [[1, 2], 
             [1, 2, 4, 6, 7, 8], 
             [1, 2, 4, 7, 6, 8]
            ]

node_position_list = [[[1, 0],  [2, 0]],
                      [[4, 0],  [5, 0],  [5, 1], [6, 1], [6, 0], [7, 0]],
                      [[9, 0], [10, 0], [10, 1], [11, 0], [11, 1], [12, 0]]
                     ]

edge_list = [[dict(idx=(0, 1), edgetype='A', label='I'), 
             ],
             [dict(idx=(0, 1), edgetype='C', label='I'), 
              dict(idx=(1, 2), edgetype='C', label='II'),
              dict(idx=(2, 3), edgetype='A', label='III'),
              dict(idx=(3, 4), edgetype='C', label='II'),
              dict(idx=(4, 5), edgetype='C', label='I'),
             ],
             [dict(idx=(0, 1), edgetype='C', label='I'),
              dict(idx=(1, 3), edgetype='B', label='III'),
              dict(idx=(3, 4), edgetype='C', label='II'),
              dict(idx=(3, 5), edgetype='C', label='I'),
             ]
            ]

for i in range(len(node_list)):
    nodes = node_list[i]
    node_position = node_position_list[i]
    edges = edge_list[i]
    
    for inode in range(len(nodes)):
        ax.annotate('%d'%nodes[inode], xy=tuple(node_position[inode]), xycoords='data', ha='center', va='center',
                    bbox=dict(boxstyle='circle, pad=0.1', fc='w'), fontsize=fs)
    
    for iedge in range(len(edges)):
        idx = edges[iedge]['idx']
        posA, posB = tuple(node_position[idx[0]]), tuple(node_position[idx[1]])
        edgetype = edges[iedge]['edgetype']
        label = edges[iedge]['label']

        if label == 'II':
            ha, va = 'left', 'center'
        else:
            ha, va = 'center', 'bottom'
        if edgetype == 'A':
            dy, shrink = 0.02, 12
            ax.annotate('', xy=posB, xycoords='data', xytext=posA, textcoords='data',
                        arrowprops=dict(arrowstyle='-|>', shrinkA=shrink, shrinkB=shrink, color='C1', lw=2))
            ax.annotate(label, xy=((posB[0]+posA[0])/2, (posB[1]+posA[1])/2), xycoords='data', ha=ha, va=va, fontsize=fs, fontfamily='serif')
        elif edgetype == 'B':
            dy, shrink = 0.08, 12
            ax.annotate('', xy=[posB[0], posB[1]+dy], xycoords='data',
                        xytext=[posA[0], posA[1]+dy], textcoords='data',
                        arrowprops=dict(arrowstyle='-|>', shrinkA=shrink, shrinkB=shrink, color='C1', lw=2))
            ax.annotate('', xy=[posA[0], posA[1]], xycoords='data',
                        xytext=[posB[0], posB[1]], textcoords='data',
                        arrowprops=dict(arrowstyle='-|>', shrinkA=shrink, shrinkB=shrink, color='C1', lw=2))
            ax.annotate(label, xy=((posB[0]+posA[0])/2, (posB[1]+posA[1])/2+dy), xycoords='data', ha=ha, va=va, fontsize=fs, fontfamily='serif')
        elif edgetype == 'C':
            dy, shrink = 0.02, 12

            ax.annotate('', xy=posB, xycoords='data', xytext=posA, textcoords='data',
                        arrowprops=dict(arrowstyle='<|-|>', shrinkA=shrink, shrinkB=shrink, color='C0', lw=2))
            ax.annotate(label, xy=((posB[0]+posA[0])/2, (posB[1]+posA[1])/2), xycoords='data', ha=ha, va=va, fontsize=fs, fontfamily='serif')

ax.set_xlim([-0.5, 14])
ax.set_ylim([0, 2])

# Titles
ax.annotate(r'${\bf d}$ 0K & 2K', xy=(0.8, 1.8), xycoords='data', ha='left', va='top', fontsize=fs)
ax.annotate(r'${\bf e}$ 5K', xy=(3.8, 1.8), xycoords='data', ha='left', va='top', fontsize=fs)
ax.annotate(r'${\bf f}$ 20K', xy=(8.8, 1.8), xycoords='data', ha='left', va='top', fontsize=fs)

# Event types
x1, x2, yA, yB, yC, dy, shrink = 12.5, 13.5, 1.5, 1.1, 0.7, 0.04, 12
ax.annotate('Type A', xy=(x2, yA), xycoords='data', ha='left', va='center', fontsize=fstk)
ax.annotate('', xy=(x1, yA), xycoords='data', xytext=(x2, yA), textcoords='data',
            arrowprops=dict(arrowstyle='<|-', color='C1', lw=2, shrinkA=shrink, shrinkB=shrink),)
ax.annotate('Type B', xy=(x2, yB), xycoords='data', ha='left', va='center', fontsize=fstk)
ax.annotate('', xy=(x1, yB+dy), xycoords='data', xytext=(x2, yB+dy), textcoords='data',
            arrowprops=dict(arrowstyle='-|>', color='C1', lw=2, shrinkA=shrink, shrinkB=shrink),)
ax.annotate('', xy=(x1, yB-dy), xycoords='data', xytext=(x2, yB-dy), textcoords='data',
            arrowprops=dict(arrowstyle='<|-', color='C1', lw=2, shrinkA=shrink, shrinkB=shrink),)
ax.annotate('Type C', xy=(x2, yC), xycoords='data', ha='left', va='center', fontsize=fstk)
ax.annotate('', xy=(x1, yC), xycoords='data', xytext=(x2, yC), textcoords='data',
            arrowprops=dict(arrowstyle='<|-|>', shrinkA=shrink, shrinkB=shrink, color='C0', lw=2))

ax.axis('off')
fig.tight_layout()

if savefig:
    figname = 'Figure3def'
    fig.savefig(os.path.join(figure_dir, figname+'.pdf'))

plt.show()