###############################################################################
# This script perform a series of post-processing of free volume groups;
# via this script, we could calculate the evolved free volume groups;
# and trace any specific volume groups
# Jing Liu updated on 2020.11.05

# Import NumPy module.
import numpy as np

import sys, os, argparse, glob, re, shlex, shutil, subprocess, math, queue, time
from collections import namedtuple
from typing import List, Tuple

# Import OVITO modules.
import ovito
from ovito import dataset
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from workflow_volume_analysis_utils import *

########################## Initialization ###########################

debug_output = True              # switch for printing debug info
parser = arguments()
setupfile = parser.parse_args().input

print('setupfile =', setupfile, os.path.exists(setupfile))

if os.path.exists(setupfile):
    parse_lines = read_setup(setupfile)
    info_args = sys.argv[1:] + shlex.split(' '.join(parse_lines))
else:
    info_args = sys.argv[1:]

if debug_output:
    print(info_args)

# Set variables from the arguments (input file)
setup_args = parser.parse_args(info_args)
status = setup_args.nstatus

# set up for the atomic parameters and volume parameters
temp = setup_args.temp
strain = setup_args.strain
frame = setup_args.frame
num_frame = setup_args.num_frame
startstep, endstep, dstep = setup_args.drange

R1 = setup_args.R1
R2 = setup_args.R2
schmitt_radius = setup_args.schmitt_radius
alpha = setup_args.alpha
mesh_size = setup_args.mesh_size
dL = setup_args.dL

# set up for the directories
input_direct = setup_args.input_direct
output_direct = setup_args.output_direct
extend_atom_direct = setup_args.extend_atom_direct
atom_group_direct = setup_args.atom_group_direct
atom_outvol_direct = setup_args.atom_outvol_direct
hist_direct = setup_args.hist_direct

# make new directories for the output file
input_direct = os.path.join(input_direct, 'T%dK-%.2f'%(temp, strain))
output_dir = os.path.join(input_direct, output_direct)
os.makedirs(output_dir, exist_ok=True)
extend_atom_dir = os.path.join(output_dir, extend_atom_direct)
atom_group_dir = os.path.join(output_dir, atom_group_direct)
atom_outvol_dir = os.path.join(output_dir, atom_outvol_direct)
hist_dir = os.path.join(output_dir, hist_direct)
os.makedirs(extend_atom_dir, exist_ok=True)
os.makedirs(atom_group_dir, exist_ok=True)
os.makedirs(atom_outvol_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)


# read the atomic information and volume parameters
input_dir = os.path.join(input_direct, 'frame_%d/'%frame)
keyword = 'T%dK_%.2f_frame_%d'%(temp, strain, frame)
lammps_dump = setup_args.lammps_dump
ref_dump = setup_args.ref_dump
last_dump = setup_args.last_dump
lammps_data = os.path.join(input_dir, lammps_dump)
ref_data = os.path.join(input_dir, ref_dump)
last_data = os.path.join(input_dir, last_dump)

if status in ['3', '4', '5', '6']:
    boundary, dxyz = GetBoxSize(last_data)
    num_meshes_by_axis = buildMesh(boundary, mesh_size)
    print('boundary:', boundary)
    print('dxyz:', dxyz)
    print('num_meshes_by_axis:', num_meshes_by_axis)

if status in ['5', '6']:
    volume_group_direct = setup_args.volume_group_direct
    group_i_direct = setup_args.group_i_direct
    temp_direct = setup_args.temp_direct
    vol_th_hi = setup_args.vol_th_hi
    vol_th_lo = setup_args.vol_th_lo
    trace_th_hi = setup_args.trace_th_hi
    trace_th_lo = setup_args.trace_th_lo

    volume_group_direct = ''.join(volume_group_direct[:-1] + '_n%d_%d/' % (vol_th_lo, vol_th_hi))
    volume_group_dir = os.path.join(output_dir, volume_group_direct)
    group_i_dir = os.path.join(volume_group_dir, group_i_direct)
    temp_dir = os.path.join(output_dir, temp_direct)
    os.makedirs(volume_group_dir, exist_ok=True)
    os.makedirs(group_i_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

if debug_output:
    print('input_dir:', input_dir)
    print('output_dir:', output_dir)

########################## Processes ################################

########################## Main Script ##############################

if status == '0':
    print('no status # input. Please use -n option')
    sys.exit()

elif status == '1':
    ############### Meshing the box and connect the meshes ##########################
    # exports the connected volume groups to the output directory.
    # 'volume_group_%s_%d.npz'%(keyword, step) would be obtained.
    # This data later is being used in the calculation of evolved volume groups.
    # run by: ovitos Workflow_volume_analysis.py -n 1

    print('\n status 1: Meshing the box and connect the meshes.')

    # Get the information of S atom
    print('\nGet the information of S atom')
    tp_0 = time.time()
    satom_coords = SelectMiseAtom(last_data, ref_data)
    boundary, dxyz = GetBoxSize(last_data)
    print('satom_coords:', satom_coords)
    print('boundary:', boundary)
    print('Lx, Ly, Lz:', dxyz)

    tp_1 = time.time()
    print('time getting the S atom:', tp_1 - tp_0)

    # Load a simulation series of a Cu-Zr metallic glass.
    for step in range(startstep, endstep, dstep):
        # load data

        tp_load_data = time.time()

        print('\nprocessing configuration at step:', step)
        data = input_dir + lammps_dump+'.%d' %step
        atoms, boundary, dxyz = loadData(data, R1, R2)
        num_meshes_by_axis = buildMesh(boundary, mesh_size)
        num_meshes = np.prod(num_meshes_by_axis)
        print('data:', data)
        print('first atom:', atoms[1])
        print('boundary:', boundary); print('Lx, Ly, Lz:', dxyz)
        print('num_meshes_by_axis:', num_meshes_by_axis); print('num_meshes:', num_meshes)

        # Generate atomic configuration and save configuration in npz and lammps format
        normalizeCoord(satom_coords, boundary, dxyz, atoms)
        atoms_lammps_file = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
        AtomGenerateConfiguration(atoms, boundary, dxyz, atoms_lammps_file)
        extend_atoms, extend_boundary, extend_dxyz = Enlargebox(atoms, boundary, dxyz, dL)
        extend_atoms_lammps_file = extend_atom_dir + 'coords_extend_atoms_%s_%d.dump.gz' % (keyword, step)
        AtomGenerateConfiguration(extend_atoms, extend_boundary, extend_dxyz, extend_atoms_lammps_file)
        print('atoms.shape:', atoms.shape);
        print('extend_atoms.shape:', extend_atoms.shape)
        print('\nGenerate atomic configuration:', atoms_lammps_file)
        print('first atom:', atoms[1]);  print('boundary:', boundary);  print('Lx, Ly, Lz:', dxyz)
        print('\nGenerate extend atomic configuration:', extend_atoms_lammps_file)
        print('first atom:', extend_atoms[1]); print('extend_boundary:', extend_boundary); print('Lx, Ly, Lz:', extend_dxyz)

        tp_shift_atom = time.time()
        print('time shift atoms:', tp_shift_atom - tp_load_data)

        # find empty meshes
        in_any_atom = [False] * num_meshes
        atom_groups = []
        for atom in extend_atoms[:len(atoms)]:
            atom_ids = []
            lo_i = max(0, int((atom['x'] - atom['r'] * schmitt_radius*alpha - boundary[0][0]) / mesh_size) - 1)
            hi_i = min(num_meshes_by_axis[0],
                       int(math.ceil((atom['x'] + atom['r'] * schmitt_radius*alpha - boundary[0][0]) / mesh_size)) + 1)
            lo_j = max(0, int((atom['y'] - atom['r'] * schmitt_radius*alpha - boundary[1][0]) / mesh_size) - 1)
            hi_j = min(num_meshes_by_axis[1],
                       int(math.ceil((atom['y'] + atom['r'] * schmitt_radius*alpha - boundary[1][0]) / mesh_size)) + 1)
            lo_k = max(0, int((atom['z'] - atom['r'] * schmitt_radius*alpha - boundary[2][0]) / mesh_size) - 1)
            hi_k = min(num_meshes_by_axis[2],
                       int(math.ceil((atom['z'] + atom['r'] * schmitt_radius*alpha - boundary[2][0]) / mesh_size)) + 1)
            for i in range(lo_i, hi_i):
                for j in range(lo_j, hi_j):
                    for k in range(lo_k, hi_k):
                        mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
                        mesh_coord = getMeshCoordinate(i, j, k, boundary, mesh_size)
                        state = inAtom(mesh_coord, atom, mesh_size, schmitt_radius*alpha)
                        if state:
                            atom_ids.append(mesh_id)
                        if in_any_atom[mesh_id]:
                            continue
                        in_any_atom[mesh_id] = inAtom(mesh_coord, atom, mesh_size, schmitt_radius)
            atom_groups.append(atom_ids)

        # Save the atom_groups
        atom_groups_file = atom_group_dir + 'atom_group_%s_%d.npz' % (keyword, step)
        print('Save atom_groups:',  atom_groups_file)
        np.savez_compressed(atom_groups_file, atom_groups)

        for atom in extend_atoms[len(atoms):]:
            lo_i = max(0, int((atom['x'] - atom['r'] * schmitt_radius - boundary[0][0]) / mesh_size) - 1)
            hi_i = min(num_meshes_by_axis[0],
                       int(math.ceil((atom['x'] + atom['r'] * schmitt_radius - boundary[0][0]) / mesh_size)) + 1)
            lo_j = max(0, int((atom['y'] - atom['r'] * schmitt_radius - boundary[1][0]) / mesh_size) - 1)
            hi_j = min(num_meshes_by_axis[1],
                       int(math.ceil((atom['y'] + atom['r'] * schmitt_radius - boundary[1][0]) / mesh_size)) + 1)
            lo_k = max(0, int((atom['z'] - atom['r'] * schmitt_radius - boundary[2][0]) / mesh_size) - 1)
            hi_k = min(num_meshes_by_axis[2],
                       int(math.ceil((atom['z'] + atom['r'] * schmitt_radius - boundary[2][0]) / mesh_size)) + 1)
            for i in range(lo_i, hi_i):
                for j in range(lo_j, hi_j):
                    for k in range(lo_k, hi_k):
                        mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
                        if in_any_atom[mesh_id]:
                            continue
                        mesh_coord = getMeshCoordinate(i, j, k, boundary, mesh_size)
                        in_any_atom[mesh_id] = inAtom(mesh_coord, atom, mesh_size, schmitt_radius)

        tp_dist_check = time.time()
        print('time elapsed checking distance:', tp_dist_check - tp_shift_atom)

        # Step 2. Calculate the free volume groups and output them
        print('\nStep 2. Calculate the free volume groups and output them')
        checked_ids = [False] * num_meshes
        volume_groups = []
        for i in range(num_meshes_by_axis[0]):
            for j in range(num_meshes_by_axis[1]):
                for k in range(num_meshes_by_axis[2]):
                    mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
                    if not checked_ids[mesh_id] and not in_any_atom[mesh_id]:
                        connected_mesh_ids = getConnectedMeshIds(i, j, k, in_any_atom, num_meshes_by_axis, checked_ids)
                        # volume = len(connected_mesh_ids)
                        volume_groups.append(connected_mesh_ids)

        # Save the volume_groups and its size
        volume_groups_file = output_dir + 'volume_group_%s_%d.npz' % (keyword, step)
        print('\nSave volume_groups:', volume_groups_file)
        np.savez_compressed( volume_groups_file, volume_groups)

        tp_bfs = time.time()
        print('time for bfs:', tp_bfs - tp_dist_check)

elif status == '2':
    ############### Define the environment for each atom ##########################
    # exports the maximum volume  of all the volume groups to the output directory.
    # 'coords_volume_%s_%d.dump.gz' % (keyword, i) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 2

    print('\n status 2: Define the environment for each atom.')

    for step in range(startstep, endstep, dstep):

        tp_load_data = time.time()

        data = output_dir + 'coords_atoms_%s_%d.dump.gz'%(keyword, step)
        init_data = output_dir + 'coords_atoms_%s_0.dump.gz'%keyword
        print('processing Mises strain calculation at step:', step)
        atoms, boundary, dxyz = loadData(data, R1, R2)
        mises_data = AtomicStrain(data, init_data)

        volume_groups_file = output_dir + 'volume_group_%s_%d.npz' %(keyword, step)
        volume_groups = np.load(volume_groups_file, allow_pickle=True)['arr_0']
        volume_list = np.array([len(volume_groups[v]) for v in range(len(volume_groups))])
        idxi = volume_list >= 10
        volume_groups, volume_list = volume_groups[idxi], volume_list[idxi]
        print('volume_group.size:', len(volume_groups))

        atom_groups_file = atom_group_dir + 'atom_group_%s_%d.npz' % (keyword, step)
        atom_groups = np.load(atom_groups_file, allow_pickle=True)['arr_0']
        atom_outvol = [0.0] * len(atom_groups)

        for i in range(len(atom_groups)):
            atom_group = set(atom_groups[i])
            for j in range(len(volume_groups)):
                volume_group = set(volume_groups[j])
                vol = len(volume_group)/1000.0
                if len(atom_group & volume_group)>=10:
                    atom_outvol[i] = max(atom_outvol[i], vol)

        atom_outvol_file = atom_outvol_dir + 'atom_mises_outvol_%s_%d.npz' % (keyword, step)
        np.savez_compressed(atom_outvol_file, mises_data=mises_data, atom_outvol=atom_outvol)

        # atom_outvol_file = atom_outvol_dir + 'atom_mises_outvol_%s_%d.npz' % (keyword, step)
        # atom_outvol_data = np.load(atom_outvol_file)
        # mises_data = atom_outvol_data['mises_data']
        # atom_outvol = atom_outvol_data['atom_outvol']

        print('finished calculation.')
        Satoms_lammps_file = atom_outvol_dir + 'coords_Satoms_%s_%d.dump.gz' % (keyword, step)
        SatomGenerateConfiguration(atoms, boundary, dxyz, mises_data, atom_outvol, Satoms_lammps_file)
        print('\nGenerate S atom configuration:', Satoms_lammps_file)
        print('first atom:', atoms[1]); print('boundary:', boundary); print('Lx, Ly, Lz:', dxyz)

        tp_finish = time.time()
        print('time finishing calculation:', tp_finish- tp_load_data)

elif status == '3':
    ############### Generate configurations for all the volume groups ##########################
    # exports the configurations of all the volume groups to the output directory.
    # 'coords_volume_%s_%d.dump.gz' % (keyword, i) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 2

    print('\n status 3: Generate configurations for all the volume groups.')

    for step in range(startstep, endstep, dstep):
        volume_group_file = output_dir + 'volume_group_%s_%d.npz' % (keyword, step)
        print('volume_group_file:', volume_group_file)
        volume_group = np.load(volume_group_file, allow_pickle=True)['arr_0']
        volume_list = np.array([len(volume_group[v]) for v in range(len(volume_group))])
        volume_id = np.array([v for v in range(len(volume_group))])

        if len(volume_group):
            mesh_coords = GetVolumeAndCoordinate(volume_group, volume_id, boundary, num_meshes_by_axis, mesh_size)
            mesh_lammps_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, step)
            MeshGenerateConfiguration(mesh_coords, boundary, dxyz, mesh_lammps_file)
            print('\nFinish generation of volume groups:', mesh_lammps_file)
        else:
            print('No volume group configuration were generated at: %d\n' % step)

elif status == '4':
    ############### Calculate the voulume group size distribution ##########################
    # exports the configurations of all the volume groups to the output directory.
    # 'volume_distri_%s_%d.npz' % (keyword, step) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 3

    print('\n status 4: Calculate the voulume group size distribution.')

    for step in range(startstep, endstep, dstep):
        volume_group_file = output_dir + 'volume_group_%s_%d.npz' % (keyword, step)
        print('volume_group_file:', volume_group_file)
        volume_group = np.load(volume_group_file, allow_pickle=True)['arr_0']
        volume_list = np.array([len(volume_group[v]) for v in range(len(volume_group))])
        volume_id = np.array([v for v in range(len(volume_group))])
        hist, bins = np.histogram(volume_list, bins=100, density=False)
        print('hist, bins:', hist, bins)
        np.savez_compressed(output_dir + 'volume_distri_%s_%d.npz' % (keyword, step), hist=hist, bins=bins)
        print('Finish calculation of volume group size distribution at step %d:'%step)

elif status == '5':
    ############### Filter evolved volume groups ##########################
    # exports the evolved volume groups to the intermediate directory.
    # 'coords_volume_group_%s_%d.dump.gz'%(keyword, i) would be obtained.
    # This data later is being used in the calculation of ith volume group.
    # run by: ovitos Workflow_volume_analysis.py -n 4

    print('\n status 5: filter the evolved volume groups.')

    boundary, dxyz = GetBoxSize(last_data)
    num_meshes_by_axis = buildMesh(boundary, mesh_size)
    print('boundary:', boundary)
    print('dxyz:', dxyz)
    print('num_meshes_by_axis:', num_meshes_by_axis)

    # Step 1: Load the initial configuration's volume information (volume_group and volume_size) and
    # Initialize the volume label for the configuration.
    print('\nStep 1: Load the initial volume information and initialize the volume label')

    volume_group = np.load(output_dir + 'volume_group_%s_0.npz' % keyword, allow_pickle=True)['arr_0']
    volume_size = np.array([len(volume_group[i]) for i in range(len(volume_group))])
    idx = volume_size >= vol_th_hi
    volume_group, volume_list = volume_group[idx], volume_size[idx]
    volume_group_id = np.zeros((volume_list.size, num_frame), dtype=np.int)
    volume_group_id[:, 0] = np.arange(1, volume_list.size + 1, dtype=np.int)
    volume_groups_evol = [volume_list.size]
    print('volume_list_0.size:', volume_list.size)
    np.savez_compressed(temp_dir + 'volume_group_new_%s_0.npz' % keyword, volume_group)
    np.savez_compressed(temp_dir + 'volume_group_id_%s_0.npz' % keyword, volume_group_id)

    # Step 2: Classify the volume information of ith configuration; 2. same, 0; different, 1
    print('\nStep 2: Classify the volume information of ith configuration; 2. same, 0; different, 1\n')

    num_frame = 16

    for i in range(1, num_frame):
        volume_group_i = np.load(output_dir + 'volume_group_%s_%d.npz' % (keyword, i), allow_pickle=True)['arr_0']
        volume_list_i = np.array([len(volume_group_i[i]) for i in range(len(volume_group_i))])
        idxi = volume_list_i >= vol_th_hi
        volume_group_i, volume_list_i = volume_group_i[idxi], volume_list_i[idxi]
        print('volume_list_%d.size:'%i, volume_list_i.size)
        for j in range(volume_list.size):
            group1 = set(volume_group[j])
            k = 0
            while k < volume_list_i.size:
                group2 = set(volume_group_i[k])
                if len(group1 & group2) * 1.0 / len(group1) >= trace_th_hi and \
                        len(group1 & group2) * 1.0 / len(group2) >= trace_th_hi:
                    volume_group_id[j, i] = volume_group_id[j, i - 1]
                    volume_group[j] = group2
                    volume_list[j] = volume_list_i[k]
                    volume_group_i = np.delete(volume_group_i, k, axis=0)
                    volume_list_i = np.delete(volume_list_i, k, axis=0)
                    break
                # elif len(group1&group2)*1.0/len(group1)>=trace_th_hi:
                #     volume_group_id[j, i] = volume_group_id[j, i-1]
                #     volume_group[j] = group2
                #     volume_list[j] = volume_list_i[k]
                #     volume_group_i = np.delete(volume_group_i, k, axis=0)
                #     volume_list_i = np.delete(volume_list_i, k, axis=0)
                #     break
                else:
                    k += 1

        volume_group_id_i = np.zeros((volume_list_i.size, num_frame), dtype=np.int)
        volume_group_id_i[:, i] = np.arange(volume_list.size + 1,
                                            volume_list.size + volume_list_i.size + 1, dtype=np.int)
        volume_group_id = np.vstack((volume_group_id, volume_group_id_i))
        volume_group = np.append(volume_group, volume_group_i)
        volume_list = np.append(volume_list, volume_list_i)
        volume_groups_evol.append(volume_list.size)
        print('%dth volume_list_i.size:' % i, volume_list_i.size)
        print('%dth volume_list.size:' % i, volume_list.size)
        print('%dth volume_group_id_i.shape:' % i, volume_group_id_i.shape)
        print('%dth volume_group_id.shape:' % i, volume_group_id.shape)

        print('Save files for volume_groups, volume_group_id...\n')
        np.savez_compressed(temp_dir + 'volume_group_new_%s_%d.npz' % (keyword, i), volume_group)

    print('\nSave files for volume_group_id_%s.npz...'%keyword)
    np.savez_compressed(temp_dir + 'volume_group_id_%s.npz' % keyword, volume_group_id)

    volume_group_is_ever = np.zeros(len(volume_group_id), dtype=np.int)
    for i in range(len(volume_group_id)):
        volume_group_id_i = volume_group_id[i].tolist()
        if volume_group_id_i.count(0) == 0:
            volume_group_is_ever[i] = 1

    print('volume_group_is_ever.shape:', volume_group_is_ever.shape)
    print('volume_group_is_ever:', volume_group_is_ever)
    print('\nSave files for volume_group_is_ever_%s.npz...' % keyword)
    np.savez_compressed(temp_dir + 'volume_group_is_ever_%s.npz' % keyword, volume_group_is_ever)

    # Step 3 Generate mesh configuration and save configuration in npz and lammps format
    print('\nStep3: Generate mesh configuration and save configuration in npz and lammps format\n')

    for i in range(num_frame):
        group_i = np.load(temp_dir + 'volume_group_new_%s_%d.npz' % (keyword, i), allow_pickle=True)['arr_0']
        a = len(group_i)
        group_id = np.load(temp_dir + 'volume_group_id_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a]
        volume_group_is_ever = np.load(temp_dir + 'volume_group_is_ever_%s.npz' % (keyword),
                                       allow_pickle=True)['arr_0'][:a]
        print('volume_group_i.shape:', group_i.shape)
        print('len(volume_group_i[0]):', len(group_i[0]))
        print('volume_group_id_i.shape:', group_id.shape)
        flag0 = volume_group_is_ever == 0
        group_i_change = group_i[flag0]
        group_id_change = group_id[flag0]
        print('volume_group_id_i_change.shape:', group_id_change.shape)

        print('\nGenerate %dth meshes configuration:' % i)
        hollow_labels_coords = GetVolumeAndCoordinate(group_i_change, group_id_change[:, i],
                                                      boundary, num_meshes_by_axis, mesh_size)
        meshes_lammps_file = volume_group_dir + 'coords_volume_group_%s_%d.dump.gz' % (keyword, i)
        MeshGenerateConfiguration(hollow_labels_coords, boundary, dxyz, meshes_lammps_file)

elif status == '6':
    ############### Trace ith volume group ##########################
    # exports the ith volume groups to the group_i directory.
    # 'coords_%s_volume_group_id_%d_%d.dump.gz' % (keyword, id, j) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 5

    print('\nstatus 6: Trace ith volume group')
    print('\nStep 4: Load the initial volume information and initialize the volume label\n')

    volume_group_0 = np.load(temp_dir + 'volume_group_new_%s_0.npz' % keyword, allow_pickle=True)['arr_0']
    a = len(volume_group_0)
    volume_group_id_0 = np.load(temp_dir + 'volume_group_id_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a][:, 0]
    volume_group_is_ever = np.load(temp_dir + 'volume_group_is_ever_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a]
    flag0 = volume_group_is_ever == 0
    volume_group = volume_group_0[flag0]
    volume_group_id = volume_group_id_0[flag0]
    print('volume_group_0.shape:', volume_group.shape)

    num_frame = 16

    for i in range(len(volume_group)):
        volume_group_i = [volume_group[i]]
        volume_group_id_i = [volume_group_id[i]]
        id = volume_group_id[i]
        ## Question: As I tried to trace the volume_group_j in frame 2 compared to frame 1, I shall record the id of group
        for j in range(num_frame):
            volume_group_j = np.load(output_dir + 'volume_group_%s_%d.npz' % (keyword, j), allow_pickle=True)['arr_0']
            volume_list_j = np.array([len(volume_group_j[a]) for a in range(len(volume_group_j))])
            idxi = volume_list_j >= vol_th_lo
            volume_group_j, volume_list_j = volume_group_j[idxi], volume_list_j[idxi]
            print('volume_list_%d.size:'%j, volume_list_j.size)

            temp_volume_group_i = []
            temp_volume_group_id_i = []
            for g in range(len(volume_group_i)):
                group1 = set(volume_group_i[g])
                k = 0
                while k < volume_list_j.size:
                    group2 = set(volume_group_j[k])
                    if len(group1 & group2) * 1.0 / len(group1) >= trace_th_hi and \
                            len(group1 & group2) * 1.0 / len(group2) >= trace_th_hi:
                        temp_volume_group_i.append(group2)
                        temp_volume_group_id_i.append(k + 1)
                        k += 1
                        break
                    elif len(group1 & group2) * 1.0 / len(group1) >= trace_th_lo or \
                            len(group1 & group2) * 1.0 / len(group2) >= trace_th_lo:
                        temp_volume_group_i.append(group2)
                        temp_volume_group_id_i.append(k + 1)
                        k += 1
                    else:
                        k += 1
            volume_group_i = temp_volume_group_i
            volume_group_id_i = temp_volume_group_id_i
            print('volume_group_id_%d:'%i, volume_group_id_i)

            print('\nGenerate %dth meshes configuration, volume_group id: %d' % (j, id))
            group_id_coords = GetVolumeAndCoordinate(volume_group_i, volume_group_id_i, boundary,
                                                     num_meshes_by_axis, mesh_size)
            group_id_lammps_file = group_i_dir + 'coords_%s_volume_group_id_%d_%d.dump.gz' % (keyword, id, j)
            MeshGenerateConfiguration(group_id_coords, boundary, dxyz, group_id_lammps_file)

else:
    print('non-recognized status #')
