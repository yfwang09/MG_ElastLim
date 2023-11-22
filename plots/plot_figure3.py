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
figsize = (12, 2)

# Figure 3abc
###########################################################


# Figure 3def
###########################################################
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