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

pfit = [1.28, 0]
ncpu = 16
fs, fstk = 18, 14

rawdata = loadmat(os.path.join(rawdata_dir, 'T20K_states.mat'))
state_dict_array = rawdata['state_dict_array'][0]
ST_dict_array = rawdata['ST_dict_array'][0]

def load_ST_energy_strain(iA, iB, return_dict=True, verbose=False, nfit=102, return_Efwd_Ebwd=False):
    found_flag = False
    for ST_i in ST_dict_array:
        if ST_i['state_A'] == iA and ST_i['state_B'] == iB:
            found_flag = True
            energy = ST_i['saved_MEP'][0, 0].squeeze()
            break
        elif ST_i['state_B'] == iA and ST_i['state_A'] == iB:
            found_flag = True
            energy = ST_i['saved_MEP'][0, 0][:, ::-1, 0]
            break
    if found_flag:
        strain = np.round(ST_i['strain'][0, 0][0], decimals=4)
        fit_range = np.round(np.linspace(0, strain.size - 1, nfit)).astype(int)
        Efun = interp1d(strain[fit_range], energy[fit_range, :], kind='linear', axis=0, fill_value='extrapolate')
        dEfun = lambda x: Efun(x)[..., -1] - Efun(x)[..., 0]
        if dEfun(strain[0])*dEfun(strain[-1]) < 0:
            sol = root_scalar(dEfun, method='bisect', bracket=strain[[0,-1]], xtol=1e-7)
        else:
            class sol:
                root = strain[0]
        if verbose:
            print(sol)
        if return_dict:
            return dict(strain=strain, energy=energy, Efun=Efun, emin=strain[0], emax=strain[-1], e_eig=sol.root)
        else:
            if return_Efwd_Ebwd:
                Efwd = lambda x: Efun(x).max(axis=-1) - Efun(x)[..., 0]
                Ebwd = lambda x: Efun(x).max(axis=-1) - Efun(x)[..., -1]
                Eeig = Efwd(sol.root)
                return (strain, energy, Efun, strain[0], strain[-1], sol.root, Eeig, Efwd, Ebwd)
            else:
                return (strain, energy, Efun, strain[0], strain[-1], sol.root)
    else:
        return None

def cal_dE_Eb(energy, return_dict=True):
    Ebfwd = energy.max(axis=1) - energy[:, 0]
    Ebbwd = energy.max(axis=1) - energy[:, -1]
    dEfwd = energy[:, -1] - energy[:, 0]
    dEbwd = energy[:, 0] - energy[:, -1]
    if return_dict:
        return dict(Ebfwd=Ebfwd,Ebbwd=Ebbwd,dEfwd=dEfwd,dEbwd=dEbwd)
    else:
        return (dEfwd, dEbwd, Ebfwd, Ebbwd)

def solve_strain_from_Eb(Ebval, Efun, emin, emax, verbose=False):
    Ebfwd = lambda x: Efun(x).max() - Efun(x)[...,  0] - Ebval
    Ebbwd = lambda x: Efun(x).max() - Efun(x)[..., -1] - Ebval
    if Ebbwd(emax) < 0:
        root1 = emax
    else:
        sol1 = root_scalar(Ebbwd, method='bisect', bracket=(emin, emax), xtol=1e-5)
        root1 = sol1.root
    if Ebfwd(emin) < 0:
        root2 = emin
    else:
        sol2 = root_scalar(Ebfwd, method='bisect', bracket=(emin, emax), xtol=1e-5)
        root2 = sol2.root
    if verbose:
        print(root1, root2)
    return (root1, root2)