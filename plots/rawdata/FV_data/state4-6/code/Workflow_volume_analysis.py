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
frames = setup_args.frames
target_frame = setup_args.target_frame

R1 = setup_args.R1
R2 = setup_args.R2
schmitt_radius = setup_args.schmitt_radius
alpha = setup_args.alpha
mesh_size = setup_args.mesh_size
dL = setup_args.dL
mises_crit = setup_args.mises_crit

vol_th_hi = setup_args.vol_th_hi
vol_th_lo = setup_args.vol_th_lo
trace_th_hi = setup_args.trace_th_hi
trace_th_lo = setup_args.trace_th_lo

# set up for the directories
input_direct = setup_args.input_direct
output_dir = setup_args.output_dir
temp_dir = setup_args.temp_dir
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

########################## Main Script ##############################
if status == '0':
    print('\n status 0: Affine transform the configurations.')
    ###############Affine transformation#############
    Sdata = np.zeros((len(frames), 4), dtype=float)
    input_dir = input_direct + 'frame_%d/' % target_frame
    target_data = input_dir + 'dump.neb.final.mg.0.gz'
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        for s in range(num_frame):
            step = s
            data = input_dir + 'dump.neb.final.mg.%d.gz' % step
            ref_cell = NodeCell(target_data)
            atoms, boundary, dxyz = AffineTransform(data, ref_cell)
            new_data = input_dir + 'dump.neb.final.mg.%d.gz' % step
            AtomGenerateConfiguration(atoms, boundary, dxyz, new_data)
            # print('atoms[0], boundary[0], dxyz:', boundary[0], dxyz)

    ###############Calculate the number of S atoms#############
    Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
    Eb_data = np.load(Eb_data_file)['arr_0']
    Eb = (Eb_data - Eb_data[0]) * 1000
    state = np.argmax(Eb)
    if state == 0 or state == 15:
        state = 7
    steps = [0, state, 15]
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        Sdata[f, 0] = frame / 2500
        for s in range(len(steps)):
            step = steps[s]
            data = input_dir + 'dump.neb.final.mg.%d.gz' % step
            ref_data = input_dir + 'dump.neb.final.mg.0.gz'
            atoms, boundary, dxyz = loadDataMore(data, ref_data, R1, R2)
            id_satoms = atoms['atomic strain'] >= mises_crit
            satoms = atoms[id_satoms]
            Sdata[f, s+1] = satoms.size

    Sdata_file = output_dir + 'Sdata_T20K_state4_6_0.62_0.46.dat'
    np.savetxt(Sdata_file, Sdata, fmt='%.4f %d %d %d')

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

    ref_frames = frames
    if num_frame == 16:
        frames = ref_frames[:2]
    elif num_frame == 2:
        frames = ref_frames[2:4]
    elif num_frame == 3:
        frames = ref_frames[4:]

    print('frames:', frames)
    output_dir = os.path.join(input_direct, 'code/output/')
    os.makedirs(output_dir, exist_ok=True)
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/'%frame

        ref_data = os.path.join(input_dir, 'dump.neb.final.mg.0.gz')
        last_data = os.path.join(input_dir, 'dump.neb.final.mg.15.gz')
        satom_coords = SelectMiseAtom(last_data, ref_data)

        tp_1 = time.time()

        # Load a simulation series of a Cu-Zr metallic glass.
        Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        if state == 0 or state == 15:
            state = 7
        steps = [0, state, 15]
        print('steps:', steps)
        for s in range(len(steps)):
            step = steps[s]
            keyword = 'T%dK_frame_%d_%d' % (temp, frame, s)
            tp_load_data = time.time()

            print('\nprocessing configuration at frame: %d, step: %d\n'%(frame, step))
            data = input_dir + 'dump.neb.final.mg.%d.gz' %step
            atoms, boundary, dxyz = loadDataMore(data, ref_data, R1, R2)
            num_meshes_by_axis = buildMesh(boundary, mesh_size)
            num_meshes = np.prod(num_meshes_by_axis)
            print('data:', data)
            print('first atom:', atoms[1])
            print('boundary:', boundary); print('Lx, Ly, Lz:', dxyz)
            print('num_meshes_by_axis:', num_meshes_by_axis); print('num_meshes:', num_meshes)

            # Generate atomic configuration and save configuration in npz and lammps format
            normalizeCoord(satom_coords, boundary, dxyz, atoms)
            atoms_lammps_file = output_dir + 'coords_atoms_%s.dump.gz' % (keyword)
            SatomGenerateConfiguration(atoms, boundary, dxyz, atoms['atomic strain'], atoms_lammps_file)

            # continue

            extend_atoms, extend_boundary, extend_dxyz = Enlargebox(atoms, boundary, dxyz, dL)
            extend_atoms_lammps_file = output_dir + 'coords_extend_atoms_%s.dump.gz' % (keyword)
            AtomGenerateConfiguration(extend_atoms, extend_boundary, extend_dxyz, extend_atoms_lammps_file)

            tp_shift_atom = time.time()
            print('time shift atoms:', tp_shift_atom - tp_load_data)

            # find empty meshes
            in_any_atom = [False] * num_meshes
            # atom_groups = []
            # for atom in extend_atoms[:len(atoms)]:
            #     atom_ids = []
            #     lo_i = max(0, int((atom['x'] - atom['r'] * schmitt_radius*alpha - boundary[0][0]) / mesh_size) - 1)
            #     hi_i = min(num_meshes_by_axis[0],
            #                int(math.ceil((atom['x'] + atom['r'] * schmitt_radius*alpha - boundary[0][0]) / mesh_size)) + 1)
            #     lo_j = max(0, int((atom['y'] - atom['r'] * schmitt_radius*alpha - boundary[1][0]) / mesh_size) - 1)
            #     hi_j = min(num_meshes_by_axis[1],
            #                int(math.ceil((atom['y'] + atom['r'] * schmitt_radius*alpha - boundary[1][0]) / mesh_size)) + 1)
            #     lo_k = max(0, int((atom['z'] - atom['r'] * schmitt_radius*alpha - boundary[2][0]) / mesh_size) - 1)
            #     hi_k = min(num_meshes_by_axis[2],
            #                int(math.ceil((atom['z'] + atom['r'] * schmitt_radius*alpha - boundary[2][0]) / mesh_size)) + 1)
            #     for i in range(lo_i, hi_i):
            #         for j in range(lo_j, hi_j):
            #             for k in range(lo_k, hi_k):
            #                 mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
            #                 mesh_coord = getMeshCoordinate(i, j, k, boundary, mesh_size)
            #                 state = inAtom(mesh_coord, atom, mesh_size, schmitt_radius*alpha)
            #                 if state:
            #                     atom_ids.append(mesh_id)
            #                 if in_any_atom[mesh_id]:
            #                     continue
            #                 in_any_atom[mesh_id] = inAtom(mesh_coord, atom, mesh_size, schmitt_radius)
            #     atom_groups.append(atom_ids)
            #
            # # Save the atom_groups
            # atom_groups_file = atom_group_dir + 'atom_group_%s_%d.npz' % (keyword, step)
            # print('Save atom_groups:',  atom_groups_file)
            # np.savez_compressed(atom_groups_file, atom_groups)

            # for atom in extend_atoms[len(atoms):]:
            for atom in extend_atoms:
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
                            volume_groups.append(connected_mesh_ids)

            # Save the volume_groups and its size
            volume_groups_file = output_dir + 'volume_group_%s.npz' % (keyword)
            print('\nSave volume_groups:', volume_groups_file)
            np.savez_compressed(volume_groups_file, volume_groups)

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

        file_data = input_dir + lammps_dump + '.%d' % step
        pe_data = PotentialEnergy(file_data)

        data = output_dir + 'coords_atoms_%s_%d.dump.gz'%(keyword, step)
        init_data = output_dir + 'coords_atoms_%s_0.dump.gz'%keyword
        print('processing Mises strain calculation at step:', step)
        atoms, boundary, dxyz = loadData(data, R1, R2)
        mises_data, disp_data = AtomicStrain(data, init_data)

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
        np.savez_compressed(atom_outvol_file, mises_data=mises_data, atom_outvol=atom_outvol,
                            pe_data=pe_data, disp_data=disp_data)

        # atom_outvol_file = atom_outvol_dir + 'atom_mises_outvol_%s_%d.npz' % (keyword, step)
        # atom_outvol_data = np.load(atom_outvol_file)
        # mises_data = atom_outvol_data['mises_data']
        # atom_outvol = atom_outvol_data['atom_outvol']

        print('finished calculation.')
        Satoms_lammps_file = atom_outvol_dir + 'coords_Satoms_%s_%d.dump.gz' % (keyword, step)
        SatomGenerateConfiguration(atoms, boundary, dxyz, mises_data, atom_outvol, pe_data, Satoms_lammps_file)
        print('\nGenerate S atom configuration:', Satoms_lammps_file)
        print('first atom:', atoms[1]); print('boundary:', boundary); print('Lx, Ly, Lz:', dxyz)

        tp_finish = time.time()
        print('time finishing calculation:', tp_finish- tp_load_data)

elif status == '100':
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        if state == 0 or state == 15:
            state = 7
        steps = [0, state, 15]
        print('steps:', steps)

        for s in range(len(steps)):
            step = steps[s]
            os.rename(output_dir + 'volume_group_%s_step_%d.npz' % (keyword, step),
                      output_dir + 'volume_group_%s_%d.npz' % (keyword, s))
            os.rename(output_dir + 'coords_volume_%s_step_%d.dump.gz' % (keyword, step),
                      output_dir + 'coords_volume_%s_%d.npz' % (keyword, s))
            os.rename(output_dir + 'volume_group_%s_step_%d.npz' % (keyword, step),
                      output_dir + 'volume_group_%s_%d.npz' % (keyword, s))

            # Step 1: Load the initial configuration's volume information (volume_group and volume_size) and
            # Initialize the volume label for the configuration.
            print('\nFinish renaming')

elif status == '3':
    ############### Transfer the volume groups to volume map and Generate configurations for all the volume groups ##########################
    # exports the mesh_id of all the volume groups to the volume map.
    # 'volume_distri_%s_%d.npz' % (keyword, step) would be obtained.
    # exports the configurations of all the volume groups to the output directory.
    # 'coords_volume_%s_%d.dump.gz' % (keyword, i) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 3

    print('\n status 3: Generate configurations for all the volume groups.')
    VFdata = np.zeros((len(frames), 7), dtype=float)
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        VFdata[f, 0] = frame / 2500

        Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        if state == 0 or state == 15:
            state = 7
        steps = [0, state, 15]
        print('\nsteps:', steps)

        for s in range(len(steps)):
            step = steps[s]
            keyword = 'T%dK_frame_%d_%d' % (temp, frame, s)

            data = input_dir + 'dump.neb.final.mg.%d.gz' % step
            ref_data = input_dir + 'dump.neb.final.mg.0.gz'
            atoms, boundary, dxyz = loadDataMore(data, ref_data, R1, R2)
            cell_vol = np.prod(dxyz)
            num_meshes_by_axis = buildMesh(boundary, mesh_size)

            vol_map = []
            volume_group_file = output_dir + 'volume_group_%s.npz' % (keyword)
            print('\nvolume_group_file:', volume_group_file)
            volume_groups = np.load(volume_group_file, allow_pickle=True)['arr_0']
            volume_list = np.array([len(volume_groups[v]) for v in range(len(volume_groups))])
            idx = volume_list >= 1
            volume_groups = volume_groups[idx]
            volume_id = np.array([v for v in range(len(volume_groups))])

            if len(volume_groups):
                mesh_coords = GetVolumeAndCoordinate(volume_groups, volume_id, boundary, num_meshes_by_axis, mesh_size)
                mesh_lammps_file = output_dir + 'coords_volume_%s.dump.gz' % (keyword)
                MeshGenerateConfiguration(mesh_coords, boundary, dxyz, mesh_lammps_file)
                print('\nFinish generation of mesh volume:', mesh_lammps_file)

                for group in volume_groups:
                    vol_map = np.hstack((vol_map, group))
                VFdata[f, s + 1] = vol_map.size * 0.008 / cell_vol
                vol_map_file = output_dir + 'vol_map_%s.npz' % (keyword)
                np.savez_compressed(vol_map_file, vol_map)
                print('vol:', len(vol_map) / 1000)
                print('Save vol_map_file:', vol_map_file)

            else:
                print('No volume group configuration were generated at: %d\n' % step)

            id_satoms = atoms['atomic strain'] >= mises_crit
            satoms = atoms[id_satoms]
            VFdata[f, s + 4] = satoms.size

    VFdata_file = output_dir + 'VFdata_T20K_state4_6_0.62_0.46.dat'
    np.savetxt(VFdata_file, VFdata, fmt='%.4f %.4f %.4f %.4f %.3f %.3f %.3f')

elif status == '30':
    ############### Transfer the volume groups to volume map and Generate configurations for all the volume groups ##########################
    # exports the mesh_id of all the volume groups to the volume map.
    # 'volume_distri_%s_%d.npz' % (keyword, step) would be obtained.
    # exports the configurations of all the volume groups to the output directory.
    # 'coords_volume_%s_%d.dump.gz' % (keyword, i) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 3

    print('\n status 30: Generate configurations for all the volume groups.')
    Sdata = np.zeros((len(frames), 2), dtype=float)

    for f in range(len(frames)):
        frame = frames[f]
        Sdata[f, 0] = frame / 2500
        input_dir = input_direct + 'frame_%d/' % frame
        if f == 0:
            ref_data = input_dir + 'dump.neb.final.mg.15.gz'
        else:
            data = input_dir + 'dump.neb.final.mg.15.gz'
            atoms, boundary, dxyz = loadDataMore(data, ref_data, R1, R2)
            id_satoms = atoms['atomic strain'] >= mises_crit
            satoms = atoms[id_satoms]
            Sdata[f, 1] = satoms.size

    Sdata_file = output_dir + 'Sdata_T20K_state4_6_0.62_0.46.dat'
    np.savetxt(Sdata_file, Sdata, fmt='%.4f %.3f')

elif status == '4':
    ############### Calculate the voronoi indices for each atom ##########################
    voro_index = [[0, 2, 8, 1], [0, 0, 12, 0], [0, 2, 8, 2], [0, 3, 6, 3], [0, 1, 10, 2],
                  [0, 3, 6, 4], [0, 1, 10, 4], [0, 0, 12, 4], [0, 1, 10, 5], [0, 2, 8, 6]]
    print('voro_index:', voro_index)

    tp_0 = time.time()
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        keyword = 'T%dK_frame_%d' % (temp, frame)

        # Load a simulation series of a Cu-Zr metallic glass.
        Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        if state == 0 or state == 15:
            state = 7
        steps = [0, state, 15]
        print('\nsteps:', steps)
        voro_evol = np.zeros((len(voro_index), 3), dtype=float)

        for s in range(len(steps)):
            step = steps[s]

            print('\nprocessing configuration at step: %d' % step)

            data = input_dir + 'dump.neb.final.mg.%d.gz' % step
            ref_data = input_dir + 'dump.neb.final.mg.0.gz'
            atoms, boundary, dxyz = loadData(data, R1, R2)

            idcu = atoms['type'] == 1
            Ncu = atoms[idcu].size
            idzr = atoms['type'] == 2
            Nzr = atoms[idzr].size
            # calculate the voronoi index and counts
            voro_indices = VoroIndiceCal(data, R1, R2)
            # print('voro_indices:', voro_indices)
            indices_all, counts_all, voro_index, num = VoroIndiceHist(voro_indices[:, 2:], voro_index)
            # print('indices_all, counts_all:', indices_all[:10], counts_all[:10])
            # print('voro_index, num', voro_index, num)

            voro_evol[:, s] = num/atoms.size

        print('voro_evol:', voro_evol)
        voro_evol_file = output_dir + 'voro_evol_%s.dat' % (keyword)
        np.savetxt(voro_evol_file, voro_evol,
               header='Number of atoms: %d, Ncu = %d, Nzr = %d' % (atoms.size, Ncu, Nzr),
               fmt='%.3f %.3f %.3f')

    tp_finished = time.time()
    print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '5':
    ############### Filter evolved volume groups ##########################
    # exports the evolved volume groups to the intermediate directory.
    # 'coords_volume_group_%s_%d.dump.gz'%(keyword, i) would be obtained.
    # This data later is being used in the calculation of ith volume group.
    # run by: ovitos Workflow_volume_analysis.py -n 5

    print('\n status 5: filter the evolved volume groups.')
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        keyword = 'T%dK_frame_%d' % (temp, frame)

        last_data = input_dir + 'dump.neb.final.mg.15.gz'
        atoms, boundary, dxyz = loadData(last_data, R1, R2)
        num_meshes_by_axis = buildMesh(boundary, mesh_size)

        print('boundary:', boundary)
        print('dxyz:', dxyz)
        print('num_meshes_by_axis:', num_meshes_by_axis)

        volume_group = np.load(output_dir + 'volume_group_%s_0.npz' % keyword, allow_pickle=True)['arr_0']
        volume_size = np.array([len(volume_group[i]) for i in range(len(volume_group))])
        print('original free volume group size:', volume_size.size)
        idx = volume_size >= vol_th_lo
        volume_group, volume_list = volume_group[idx], volume_size[idx]
        volume_group_id = np.zeros((volume_list.size, num_frame), dtype=np.int)
        volume_group_id[:, 0] = np.arange(1, volume_list.size + 1, dtype=np.int)
        volume_groups_evol = [volume_list.size]
        print('volume_list_0.size:', volume_list.size)
        print('volume_group_0.id:', volume_group_id[:, 0])
        np.savez_compressed(temp_dir + 'volume_group_new_%s_0.npz' % keyword, volume_group)
        np.savez_compressed(temp_dir + 'volume_group_id_%s_0.npz' % keyword, volume_group_id)

        # Step 2: Classify the volume information of ith configuration; 2. same, 0; different, 1
        print('\nStep 2: Classify the volume information of ith configuration; 2. same, 0; different, 1\n')

        for i in range(1, num_frame):
            volume_group_i = np.load(output_dir + 'volume_group_%s_%d.npz' % (keyword, i), allow_pickle=True)['arr_0']
            volume_list_i = np.array([len(volume_group_i[i]) for i in range(len(volume_group_i))])
            idxi = volume_list_i >= vol_th_lo
            volume_group_i, volume_list_i = volume_group_i[idxi], volume_list_i[idxi]
            print('volume_list_%d.size:'%i, volume_list_i.size)
            for j in range(volume_list.size):
                group1 = set(volume_group[j])
                k = 0
                while k < volume_list_i.size:
                    group2 = set(volume_group_i[k])
                    # if the kth free volume group in the ith configuration is similar to,
                    #    the jth free volume group in the (i-1)th configuration,
                    if len(group1 & group2) * 1.0 / len(group1) >= trace_th_hi and \
                            len(group1 & group2) * 1.0 / len(group2) >= trace_th_hi:
                        volume_group_id[j, i] = volume_group_id[j, i - 1]
                        volume_group[j] = volume_group_i[k]
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

            print('\nvolume_group_id.shape, volume_group_id_i.shape', volume_group_id.shape, volume_group_id_i.shape)
            volume_group_id = np.vstack((volume_group_id, volume_group_id_i))
            volume_group = np.append(volume_group, volume_group_i)
            volume_list = np.append(volume_list, volume_list_i)
            volume_groups_evol.append(volume_list.size)
            print('%dth volume_list_i.size:' % i, volume_list_i.size)
            print('%dth volume_list.size:' % i, volume_list.size)

            print('Save files for volume_groups, volume_group_id...')
            np.savez_compressed(temp_dir + 'volume_group_new_%s_%d.npz' % (keyword, i), volume_group)

        print('Save files for volume_group_id_%s.npz...'%keyword)
        np.savez_compressed(temp_dir + 'volume_group_id_%s.npz' % keyword, volume_group_id)

        volume_group_is_ever = np.zeros(len(volume_group_id), dtype=np.int)
        for i in range(len(volume_group_id)):
            volume_group_id_i = volume_group_id[i].tolist()
            if volume_group_id_i.count(0) == 0:
                volume_group_is_ever[i] = 1
            else:
                print('volume_group_id_i:', volume_group_id_i)

        print('\nvolume_group_is_ever.shape:', volume_group_is_ever.shape)
        print('volume_group_is_ever:', volume_group_is_ever)
        print('Save files for volume_group_is_ever_%s.npz...' % keyword)
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
            meshes_lammps_file = output_dir + 'coords_volume_group_evolve_%s_%d.dump.gz' % (keyword, i)
            MeshGenerateConfiguration(hollow_labels_coords, boundary, dxyz, meshes_lammps_file)

            flag = volume_group_is_ever != 0
            group_i_freeze = group_i[flag]
            group_id_freeze = group_id[flag]
            hollow_labels_coords = GetVolumeAndCoordinate(group_i_freeze, group_id_freeze[:, i],
                                                          boundary, num_meshes_by_axis, mesh_size)
            meshes_lammps_file = output_dir + 'coords_volume_group_freeze_%s_%d.dump.gz' % (keyword, i)
            MeshGenerateConfiguration(hollow_labels_coords, boundary, dxyz, meshes_lammps_file)

elif status == '6':
    ############### Trace ith volume group ##########################
    # exports the ith volume groups to the group_i directory.
    # 'coords_%s_volume_group_id_%d_%d.dump.gz' % (keyword, id, j) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 6

    print('\nstatus 6: Trace ith volume group')
    print('\nStep 4: Load the initial volume information and initialize the volume label\n')
    for f in range(len(frames)):
        frame = frames[f]
        input_dir = input_direct + 'frame_%d/' % frame
        keyword = 'T%dK_frame_%d' % (temp, frame)

        volume_group_0 = np.load(temp_dir + 'volume_group_new_%s_0.npz' % keyword, allow_pickle=True)['arr_0']
        a = len(volume_group_0)
        volume_group_id_0 = np.load(temp_dir + 'volume_group_id_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a][:, 0]
        volume_group_is_ever = np.load(temp_dir + 'volume_group_is_ever_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a]
        flag0 = volume_group_is_ever == 0
        volume_group = volume_group_0[flag0]
        volume_group_id = volume_group_id_0[flag0]
        print('volume_group_0.shape:', volume_group.shape)

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
                group_id_lammps_file = temp_dir + 'coords_%s_volume_group_id_%d_%d.dump.gz' % (keyword, id, j)
                MeshGenerateConfiguration(group_id_coords, boundary, dxyz, group_id_lammps_file)

elif status == '7':
    ############### Trace ith volume group ##########################
    # exports the ith volume groups to the group_i directory.
    # 'coords_%s_volume_group_id_%d_%d.dump.gz' % (keyword, id, j) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 7

    print('\nstatus 7: Trace ith volume group')

    boundary, dxyz = GetBoxSize(last_data)
    num_meshes_by_axis = buildMesh(boundary, mesh_size)
    print('boundary:', boundary)
    print('dxyz:', dxyz)
    print('num_meshes_by_axis:', num_meshes_by_axis)

    print('\nStep 7: Load the initial volume information and initialize the volume label\n')

    volume_group_ref = np.load(temp_dir + 'volume_group_new_%s_0.npz' % keyword, allow_pickle=True)['arr_0']
    a = len(volume_group_ref)
    volume_group_id_ref = np.load(temp_dir + 'volume_group_id_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a][:, 0]
    volume_group_is_ever = np.load(temp_dir + 'volume_group_is_ever_%s.npz' % keyword, allow_pickle=True)['arr_0'][:a]
    flag0 = volume_group_is_ever == 0
    volume_group = volume_group_ref[flag0]
    volume_group_id = volume_group_id_ref[flag0]
    print('volume_group_ref.shape:', volume_group.shape)

    num_frame = 16
    for j in range(num_frame):
        volume_group_j = np.load(output_dir + 'volume_group_%s_%d.npz' % (keyword, j), allow_pickle=True)['arr_0']
        volume_list_j = np.array([len(volume_group_j[a]) for a in range(len(volume_group_j))])
        idxi = volume_list_j >= vol_th_lo
        volume_group_j, volume_list_j = volume_group_j[idxi], volume_list_j[idxi]
        print('volume_list_%d.size:' % j, volume_list_j.size)

        temp_volume_group_i = []
        temp_volume_group_id_i = []

        for i in range(len(volume_group)):
            volume_group_i = [volume_group[i]]
            volume_group_id_i = [volume_group_id[i]]
            id = volume_group_id[i]
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

        print('\nGenerate %dth meshes configuration'%j)
        group_id_coords = GetVolumeAndCoordinate(volume_group_i, volume_group_id_i, boundary,
                                                 num_meshes_by_axis, mesh_size)
        group_id_lammps_file = group_i_dir + 'coords_%s_volume_group_%d.dump.gz' % (keyword, j)
        MeshGenerateConfiguration(group_id_coords, boundary, dxyz, group_id_lammps_file)

elif status == '8':
    ############### Calculate the voulume group size distribution ##########################
    # exports the configurations of all the volume groups to the output directory.
    # 'volume_distri_bin_0_1_%s_%d.npz' % (keyword, step) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 8

    print('\n status 8: Calculate the voulume group size distribution')
    print('frames:\n', frames)

    # ave_sum_vol = np.zeros((len(frames), 5), dtype=float)
    # for f in range(len(frames)):
    #     frame = frames[f]
    #     keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
    #     sum_vol = np.zeros((num_frame, 3), dtype=float)
    #     for step in range(num_frame):
    keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
    Eb_data_file = os.path.join(input_direct, 'frame_%d/saved_Eb.npz' % frame)
    Eb_data = np.load(Eb_data_file)['arr_0']
    Eb = (Eb_data - Eb_data[0]) * 1000
    state = np.argmax(Eb)
    steps = [0, state, 15]
    sum_vol = np.zeros((len(steps), 6), dtype=float)
    for i in range(len(steps)):
        # load data
        step = steps[i]
        init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
        data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
        atoms, boundary, dxyz = loadData(data, R1, R2)
        mises_data, disp_data = AtomicStrain(data, init_data)
        id_satoms = mises_data >= mises_crit
        satoms = atoms[id_satoms]

        volume_data = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, step)
        print('\nvolume_data:', volume_data)
        volumeY = loadVolume(volume_data)
        dV = dxyz[0] * dxyz[2] * 1.0
        y0 = np.int(boundary[1][0])
        y1 = math.ceil(boundary[1][1])
        num_bins = np.int(y1 - y0)
        print('boundary:', boundary)
        print('dxyz:', dxyz)
        print('dV=dx*dz*1.0:', dV)
        print('y0, y1, bins:', y0, y1, num_bins)

        mises_hist, mises_bins = np.histogram(mises_data, bins=800, range=(0, 0.02), density=False)
        print('mises value range:', np.min(mises_data), np.max(mises_data))
        np.savetxt(hist_dir + 'distri_mises_%s_%d.dat' % (keyword, step),
                   np.column_stack((mises_bins[1:], mises_hist, mises_hist/atoms.size)),
                   fmt='%.4f %d %.4f', header='bins Ns Ns/Natom')

        # export the distribution of Natom, Ncu, free volume along Y direction.
        atoms_hist, bins = np.histogram(atoms['y'], bins=num_bins, range=(y0, y1), density=False)
        volume_hist, bins = np.histogram(volumeY, bins=num_bins, range=(y0, y1), density=False)
        id1 = atoms['type'] == 1
        Cu_atoms = atoms[id1]
        Cu_hist, Cu_bins = np.histogram(Cu_atoms['y'], bins=num_bins, range=(y0, y1), density=False)

        np.savetxt(hist_dir + 'distriY_volume_%s_%d.dat' % (keyword, step),
                   np.column_stack((bins[1:], atoms_hist, Cu_hist, atoms_hist/dV, Cu_hist/atoms_hist*100,
                                    volume_hist/1000, volume_hist/1000.0/atoms_hist)),
                   fmt='%.4f %d %d %.4f %.4f %.4f %.4f', header='bins Natom Ncu density f(Cu) Vf Vf/Natom')

        volume_group_file = output_dir + 'volume_group_%s_%d.npz' % (keyword, step)
        print('volume_group_file:', volume_group_file)
        volume_group = np.load(volume_group_file, allow_pickle=True)['arr_0']
        volume_list = np.array([len(volume_group[v])/1000 for v in range(len(volume_group))])
        cell_vol = np.prod(dxyz)
        free_vol = np.sum(volume_list)
        voro_vol = np.sum(atoms['vol'])
        sum_vol[i] = [step, cell_vol, free_vol, voro_vol, free_vol/atoms.size, voro_vol/atoms.size]

        print('volume_list size range:', np.max(volume_list), np.min(volume_list))
        hist_total, bins_total = np.histogram(volume_list, bins=190, range=(0.1, 10), density=False)
        hist_1, bins_1 = np.histogram(volume_list, bins=100, range=(0.1, 1.0), density=False)
        hist_2, bins_2 = np.histogram(volume_list, bins=90, range=(1.0, 10), density=False)
        np.savez_compressed(hist_dir + 'distri_volume_bin_total_%s_%d.npz' % (keyword, step), hist=hist_total, bins=bins_total)
        np.savez_compressed(hist_dir + 'distri_volume_bin_0_1_%s_%d.npz' % (keyword, step), hist=hist_1, bins=bins_1)
        np.savez_compressed(hist_dir + 'distri_volume_bin_1_9_%s_%d.npz' % (keyword, step), hist=hist_2, bins=bins_2)
        print('Finish calculation of volume group size distribution at step %d:' % step)

        # print('sum_vol:', sum_vol)
        # ave_sum_vol[f] = [frame/2500.0, frame, np.mean(sum_vol[:, 1]), np.mean(sum_vol[:, 2]),
        #                   np.mean(sum_vol[:, 2])/np.mean(sum_vol[:, 1])]
    sum_vol_file = hist_dir + 'sum_vol_%s.dat'%keyword
    np.savetxt(sum_vol_file, sum_vol, fmt='%d %.4f %.4f %.4f %.4f %.4f',
                header='step, cell_vol,  free_vol,  voro_vol, ave_free_vol, ave_voro_vol')
    print('sum_vol_file:', sum_vol_file)
    # ave_sum_vol_file = hist_dir + 'ave_sum_vol_T%dK_strain_%.2f.dat'%(temp, strain)
    # np.savetxt(ave_sum_vol_file, ave_sum_vol, fmt='%.4f, %d, %.4f, %.4f, %.4f',
    #            header='strain, frame, ave_cell_vol, ave_free_vol, f_free_vol')
    # print('\nave_sum_vol_file:', ave_sum_vol_file)

elif status == '10':
    ############### Calculate the average vol for the whole matrix region  ##########################
    # exports the activated volume around the activated stz to the output directory.
    # 'feature_events_T%dK_%.2f.dat'%(temp, strain) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 10
    print('\n status 10: Calculate the average vol for the whole matrix region.')
    print('frames:', frames)
    tp_0 = time.time()

    dt = np.dtype([('state', '<i4'), ('Eb', '<f4'), ('region', '<i4'), ('max_dr', '<f4'), ('stz_size', '<i4'),
                   ('cluster_size', '<i4'), ('activated_voro', '<f4'), ('activated_vol', '<f4'),
                   ('ave_activated_voro', '<f4'), ('ave_activated_vol', '<f4')])
    # print('feature dtype:', feature.dtype)
    num_refs = 9
    max_dr_list = [4.0, 4.5, 5.0, 5.5, 6.0]
    # max_dr_list = [0]
    for max_dr in max_dr_list:
        print('max_dr:', max_dr)
        for f in range(len(frames)):
            # pick the activated state
            frame = frames[f]
            input_dir = input_direct + 'frame_%d/' % frame
            keyword = 'T%dK_frame_%d' % (temp, frame)
            Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
            Eb_data = np.load(Eb_data_file)['arr_0']
            Eb = (Eb_data - Eb_data[0]) * 1000
            state = np.argmax(Eb)
            if state == 0 or state == 15:
                state = 7
            steps = [0, state, 15]
            Nf = num_refs + 3
            feature_vol = np.zeros(len(steps) * Nf, dtype=dt)

            for m in range(len(steps)):
                step = steps[m]
                feature_ref = np.zeros(num_refs, dtype=dt)
                print('\nprocessing the calculation at the activated state:', keyword, step)
                vols_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, m)
                vols = loadVolumePosition(vols_file)
                print('vols_file:', vols_file)
                print('vols.size:', vols.size)
                print('vols.shape[0]:', vols.shape[0])
                data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, m)
                init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
                final_data = output_dir + 'coords_atoms_%s_2.dump.gz' % keyword

                # define the stz region using the critial value, mises_crit
                atoms, boundary, dxyz = loadData(data, R1, R2)
                mises_data, disp_data = AtomicStrain(data, init_data)
                id_satoms = mises_data >= mises_crit
                satoms = atoms[id_satoms]


                last_atoms, boundary, dxyz = loadData(final_data, R1, R2)
                mises_data, disp_data = AtomicStrain(final_data, init_data)
                satom = last_atoms[np.argmax(mises_data)]
                id_satoms = mises_data >= mises_crit
                last_satoms = last_atoms[id_satoms]

                 ## select the satoms in the data

                # calculate the volume of bulk
                total_voro = np.sum(atoms['vol'] - 4 / 3 * 3.14 * atoms['r'] * atoms['r'] * atoms['r'])
                total_vol = vols.size * 0.008
                print('atoms.size:', atoms.size)
                print('total_vol/atoms.size:', total_vol/atoms.size)

                feature_vol[m * Nf] = (step, Eb[step], 0, 0, satoms.size,
                                       atoms.size, total_voro, total_vol,
                                       total_voro / atoms.size, total_vol / atoms.size)

                # calculate the volume of matrix
                index = [False] * atoms.size
                index_model = index
                coord_model = [satom['x'], satom['y'], satom['z']]
                print('coord_model:', coord_model)
                for atom in atoms:
                    dr = AtomicSpacing(coord_model, atom)
                    if dr <= 20.0:
                        index_model = np.logical_or(index_model, atoms == atom)
                model_atoms = atoms[index_model]
                model_voro = np.sum(model_atoms['vol'] - 4 / 3 * 3.14 * model_atoms['r'] * model_atoms['r'] * model_atoms['r'])
                model_vol = 0
                model_size = model_atoms.size

                # Fix the spherical space,
                for vol in vols:
                    if TwoAtoms(satom, vol, 20.0):
                        model_vol += 1
                print('model_atoms.size:', model_atoms.size)
                # print('model_voro/model_atoms.size:', model_voro / model_atoms.size)
                print('model_vol/model_atoms.size:', model_vol*0.008/model_atoms.size)

                feature_vol[m * Nf + 1] = (step, Eb[step], 1, 0, satoms.size,
                                         model_size, model_voro, model_vol*0.008,
                                         model_voro / model_size, model_vol*0.008 / model_size)

                # calculate the volume of the stz and referenced regions.
                # select num_refs regions
                index_refs = index * num_refs
                coord_ref0 = [np.mean(last_satoms['x']), np.mean(last_satoms['y']), np.mean(last_satoms['z'])]
                coord_ref1 = [8.5, 8.5, 8.5]
                coord_ref2 = [33.5, 33.5, 8.5]
                coord_ref3 = [33.5, 8.5, 33.5]
                coord_ref4 = [8.5, 33.5, 33.5]
                coord_ref5 = [8.5, 8.5, 33.5]
                coord_ref6 = [8.5, 33.5, 8.5]
                coord_ref7 = [33.5, 8.5, 8.5]
                coord_ref8 = [33.5, 33.5, 33.5]
                ref_coords = [coord_ref0, coord_ref1, coord_ref2, coord_ref3, coord_ref4,
                              coord_ref5, coord_ref6, coord_ref7, coord_ref8]

                dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32), ('r', np.float32)])
                Catoms = np.empty(num_refs, dtype=dtype)
                for i in range(num_refs):
                    cm_coord = ref_coords[i]
                    Catom = nearAtom(cm_coord, atoms)
                    Catoms[i] = (Catom['x'], Catom['y'], Catom['z'], Catom['r'])
                print('coord_ref0:', coord_ref0)
                print('Catoms.size, Catoms:', Catoms.size, Catoms)
                ######################define the max_dr##########################
                # max_dr = getCutoff(Catoms[0], last_satoms)
                # print('max_dr:', max_dr)
                ######################define the max_dr##########################

                # calculate the index in selected region
                index_ref_atoms = index
                for atom in atoms:
                    for i in range(num_refs):
                        if TwoAtoms(Catoms[i], atom, max_dr):
                            index_refs[i] = np.logical_or(index_refs[i], atoms == atom)
                            index_ref_atoms = np.logical_or(index_refs[i], index_ref_atoms)

                # output the selected regions into one dump file
                mises_data_i, disp_data_i = AtomicStrain(data, init_data)
                ref_atoms = atoms[index_ref_atoms]
                ref_mises_data = mises_data_i[index_ref_atoms]
                ref_atoms_lammps_file = temp_dir + 'coords_ref_atoms_%s_%d.dump.gz' % (keyword, m)
                SatomGenerateConfiguration(ref_atoms, boundary, dxyz, ref_mises_data, ref_atoms_lammps_file)

                for p in range(num_refs):
                    ref_atoms = atoms[index_refs[p]]
                    Catom = Catoms[p]
                    activated_voro = np.sum(
                        ref_atoms['vol'] - 4 / 3 * 3.14 * ref_atoms['r'] * ref_atoms['r'] * ref_atoms['r'])
                    activated_vol = 0
                    cluster_size = ref_atoms.size

                    # Fix the spherical space,
                    # cm_coord, r=max_dr,
                    vols_index = [True]*vols.size
                    xyz = ['x', 'y', 'z']
                    for i in range(len(xyz)):
                        vols_index = np.logical_and(vols_index, \
                                     np.logical_and(vols[xyz[i]] >= Catom[xyz[i]] - max_dr*schmitt_radius,
                                                    vols[xyz[i]] <= Catom[xyz[i]] + max_dr*schmitt_radius))
                    ref_vols = vols[vols_index]
                    for vol in ref_vols:
                        if TwoAtoms(Catom, vol, max_dr):
                            activated_vol +=1

                    feature_ref[p] = (step, Eb[step], p, max_dr, satoms.size,
                                      cluster_size, activated_voro, activated_vol * 0.008,
                                      activated_voro / cluster_size, activated_vol * 0.008 / cluster_size)
                print('feature_ref:', feature_ref)
                matrix_size = model_size - feature_ref[0]['cluster_size']
                matrix_voro = model_voro - feature_ref[0]['activated_voro']
                matrix_vol = model_vol*0.008 - feature_ref[0]['activated_vol']
                feature_vol[m * Nf + 2] = (step, Eb[step], 2, max_dr, satoms.size,
                                             matrix_size, matrix_voro, matrix_vol,
                                             matrix_voro / matrix_size, matrix_vol / matrix_size)
                feature_vol[(m * Nf + 3): (m+1) * Nf] = feature_ref

            # feature_vol_file = output_dir + 'feature_vol_%s.dat' % (keyword)
            # print('\nSaved feature_vol_file:', feature_vol_file)
            # np.savetxt(feature_vol_file, feature_vol,
            #            fmt='%d, %.4f, %d, %.2f, %d, %d, %.4f, %.4f, %.4f, %.4f',
            #            header='state, Eb, region, stz_size,'
            #                   'cluster_size, activated_voro, activated_vol'
            #                   'ave_activated_voro, ave_activated_vol')
            # np.savez_compressed(output_dir + 'feature_vol_%s.npz' % (keyword),
            #                     feature_vol, fmt='%d, %.4f, %d, %.2f, %d, %d, %.4f, %.4f, %.4f, %.4f')


            feature_vol_file = output_dir + 'feature_vol_%s_dr_%.1f.dat' % (keyword, max_dr)
            print('\nSaved feature_vol_file:', feature_vol_file)
            np.savetxt(feature_vol_file, feature_vol,
                       fmt='%d, %.4f, %d, %.2f, %d, %d, %.4f, %.4f, %.4f, %.4f',
                       header='state, Eb, region, stz_size,'
                              'cluster_size, activated_voro, activated_vol'
                              'ave_activated_voro, ave_activated_vol')
            np.savez_compressed(output_dir + 'feature_vol_%s_dr_%.1f.npz' % (keyword, max_dr),
                                feature_vol, fmt='%d, %.4f, %d, %.2f, %d, %d, %.4f, %.4f, %.4f, %.4f')
            tp_finished = time.time()
            print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '11':
    ############### Calculate the free volume for the specific region ##########################
    # exports the activated volume around the specific region to the output directory.
    # '_T%dK_%.2f.dat'%(temp, strain) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 11
    print('\n status 11: Calculate the free volume for the specific region.')
    print('frames:', frames)
    tp_0 = time.time()

    dt = np.dtype([('state', '<i4'), ('Eb', '<f4'), ('region', '<i4'),
                   ('stz_size', '<i4'), ('activated_voro', '<f4'), ('activated_vol', '<f4'),
                   ('ave_activated_voro', '<f4'), ('ave_activated_vol', '<f4')])

    num_refs = 4
    max_dr = 6.0

    for f in range(len(frames)):
        # pick the activated state
        frame = frames[f]
        keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
        Eb_data_file = os.path.join(input_direct, 'frame_%d/saved_Eb.npz' % frame)
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        steps = [0, state, 15]

        for m in range(len(steps)):
            step = steps[m]
            feature_ref = np.zeros(num_refs, dtype=dt)
            print('processing the calculation at the activated state:', step)
            vols_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, step)
            print('\nvols_file:', vols_file)
            vols = loadVolumePosition(vols_file)
            data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
            atoms, boundary, dxyz = loadData(data, R1, R2)
            extend_vols, extend_boundary, extend_dxyz = Enlargebox(vols, boundary, dxyz, dL)
            extend_atoms, extend_boundary, extend_dxyz = Enlargebox(atoms, boundary, dxyz, dL)
            data1 = ref_atom_dir + 'coords_region1_%s_15.dump.gz' % (keyword)
            data2 = ref_atom_dir + 'coords_region2_%s_15.dump.gz' % (keyword)
            if not os.path.exists(data1):
                break
            fatoms1, boundary, dxyz = loadData(data1, R1, R2)
            fatoms2, boundary, dxyz = loadData(data2, R1, R2)

            index_atoms1 = [False] * atoms.size
            index_atoms2 = [False] * atoms.size

            for i in range(atoms.size):
                atom = atoms[i]
                if atom['id'] in fatoms1['id']:
                    index_atoms1 = np.logical_or(index_atoms1, atoms['id'] == atom['id'])
                elif atom['id'] in fatoms2['id']:
                    index_atoms2 = np.logical_or(index_atoms2, atoms['id'] == atom['id'])
            atoms1 = atoms[index_atoms1]
            atoms2 = atoms[index_atoms2]
            print('atoms1.size:', atoms1.size, atoms1[:10])
            print('atoms2.size:', atoms2.size, atoms2[:10])
            print('atoms.size:', atoms)
            # select num_refs regions to trace the voronoi indices evolution
            coord_ref1 = [np.mean(atoms1['x']), np.mean(atoms1['y']), np.mean(atoms1['z'])]
            coord_ref2 = [np.mean(atoms2['x']), np.mean(atoms2['y']), np.mean(atoms2['z'])]
            ref_coords = [coord_ref1, coord_ref2, coord_ref1, coord_ref2]
            print('ref_coords:', ref_coords)
            max_drs = [5.5, 5.5, 6.0, 6.0]

            # calculate the volume of the stz and referenced regions.
            # select num_refs regions

            # calculate the index in selected region
            index = [False] * atoms.size
            index_regions = index * num_refs
            for atom in atoms:
                for i in range(num_refs):
                    max_dr = max_drs[i]
                    dr = AtomicSpacing(ref_coords[i], atom)
                    if dr <= max_dr:
                        index_regions[i] = np.logical_or(index_regions[i], atoms == atom)

            # calculate the volume of the stz and referenced regions.
            for p in range(num_refs):
                max_dr = max_drs[p]
                ref_atoms = atoms[index_regions[p]]
                cm_coord = ref_coords[p]
                activated_voro = np.sum(
                    ref_atoms['vol'] - 4 / 3 * 3.14 * ref_atoms['r'] * ref_atoms['r'] * ref_atoms['r'])
                activated_vol = 0
                stz_size = ref_atoms.size

                # Fix the spherical space,
                # cm_coord, r=max_dr,
                vols_index = [True]*vols.size
                xyz = ['x', 'y', 'z']
                for i in range(len(cm_coord)):
                    vols_index = np.logical_and(vols_index,
                                 np.logical_and(vols[xyz[i]] >= cm_coord[i] - max_dr*schmitt_radius,
                                                vols[xyz[i]] <= cm_coord[i] + max_dr*schmitt_radius))
                ref_vols = vols[vols_index]
                for vol in ref_vols:
                    if inMesh(cm_coord, vol, mesh_size, max_dr):
                        activated_vol +=1

                print('stz.size:', stz_size)
                # print('activated_voro/stz_size:', activated_voro / stz_size)
                print('activated_vol/stz_size:', activated_vol / 1000.0/stz_size)

                feature_ref[p] = (step, Eb[step], p,
                                  stz_size, activated_voro, activated_vol / 1000.0,
                                  activated_voro / stz_size, activated_vol / 1000.0 / stz_size)

                # output the selected regions into one dump file
                init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
                final_data = output_dir + 'coords_atoms_%s_15.dump.gz' % keyword
                mises_data_i, disp_data_i = AtomicStrain(data, init_data)
                ref_mises_data = mises_data_i[index_regions[p]]
                ref_atoms_lammps_file = ref_atom_dir + 'coords_region_%d_%s_%d_%.1f.dump.gz' % (p, keyword, step, max_dr)
                SatomGenerateConfiguration(ref_atoms, boundary, dxyz, ref_mises_data, ref_atoms_lammps_file)

            feature_ref_file = hist_dir + 'feature_events_regions_%s_%d.dat' % (keyword, step)
            print('\nSaved feature_ref_file:', feature_ref_file)
            np.savez_compressed(hist_dir + 'feature_events_regions_%s_%d.npz' % (keyword, step), feature_ref)
            np.savetxt(feature_ref_file, feature_ref,
                       fmt='%d, %.4f, %d, %d, %.4f, %.4f, %.4f, %.4f',
                       header='state, Eb, region,'
                              'stz_size, activated_voro, activated_vol'
                              'ave_activated_voro, ave_activated_vol')
        tp_finished = time.time()
        print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '12':
    ############### Define the environment for each atom ##########################
    # exports the activated volume around the activated stz to the output directory.
    # 'feature_events_T%dK_%.2f.dat'%(temp, strain) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 12
    print('\n status 12: Define the environment for each atom.')
    print('frames:', frames)
    tp_0 = time.time()

    max_dr_list = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
    for max_dr in max_dr_list:
        print('max_dr:', max_dr)
        for f in range(len(frames)):
            # pick the activated state
            frame = frames[f]
            keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
            Eb_data_file = os.path.join(input_direct, 'frame_%d/saved_Eb.npz' % frame)
            Eb_data = np.load(Eb_data_file)['arr_0']
            Eb = (Eb_data - Eb_data[0]) * 1000
            state = np.argmax(Eb)
            steps = [0]
            print('steps:', steps)

            for m in range(len(steps)):
                step = steps[m]
                print('\nprocessing the calculation at the activated state:', step)
                data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
                init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
                final_data = output_dir + 'coords_atoms_%s_15.dump.gz' % keyword
                mises_array, disp_array = AtomicStrain(final_data, init_data)
                print('range of mises_array:', min(mises_array), max(mises_array))
                vols_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, step)
                vols = loadVolumePosition(vols_file)
                atoms, boundary, dxyz = loadDataMore(data, mises_array, R1, R2)
                print('vols_file:', vols_file)
                mises_data, disp_data = AtomicStrain(data, init_data)
                print('range of mises_data:', min(mises_data), max(mises_data))
                print('range of atomic strain:', min(atoms['atomic strain']), max(atoms['atomic strain']))
                extend_vols, extend_boundary, extend_dxyz = Enlargebox(vols, boundary, dxyz, 8.0)
                extend_atoms, extend_boundary, extend_dxyz = Enlargebox(atoms, boundary, dxyz, 8.0)

                tp_1 = time.time()
                print('time loading configurations:', tp_1 - tp_0)

                # Define the number density, free volume density,

                xyz = ['x', 'y', 'z']
                for dn in range(5):
                    dl = dn*1.0 - 2
                    count = np.zeros((atoms.size, 3))
                    for k in range(atoms.size):
                        atom = atoms[k]
                        atom_coord = [atom['x'], atom['y'] + dl, atom['z']]
                        neigh_atom_index = [True] * extend_atoms.size
                        vols_index = [True] * extend_vols.size
                        for i in range(len(atom_coord)):
                            neigh_atom_index = np.logical_and(neigh_atom_index,
                                               np.logical_and(extend_atoms[xyz[i]] >= atom_coord[i] - max_dr* schmitt_radius,
                                                              extend_atoms[xyz[i]] <= atom_coord[i] + max_dr* schmitt_radius))
                            vols_index = np.logical_and(vols_index,
                                         np.logical_and(extend_vols[xyz[i]] >= atom_coord[i] - max_dr * schmitt_radius,
                                                        extend_vols[xyz[i]] <= atom_coord[i] + max_dr * schmitt_radius))
                        neigh_atom = extend_atoms[neigh_atom_index]
                        neigh_vol = extend_vols[vols_index]
                        for atomj in neigh_atom:
                            dr = AtomicSpacing(atom_coord, atomj)
                            if dr <= max_dr:
                                count[k, 0] +=1
                                if atomj['atomic strain'] >= mises_crit*0.8:
                                    count[k, 1] +=1

                        for vol in neigh_vol:
                            if inMesh(atom_coord, vol, mesh_size, max_dr):
                                count[k, 2] +=1

                    tp_2 = time.time()
                    print('\ntime calculating:', tp_2 - tp_1)

                    print('count:', count[:20])
                    print('range of Natom:', min(count[:, 0]), max(count[:, 0]))
                    print('range of stz_size:', min(count[:, 1]), max(count[:, 1]))
                    print('range of volume:', min(count[:, 2]), max(count[:, 2]))
                    dt = np.dtype([('Natom', int), ('stz_size', int), ('FreeVol', float), ('Ave_FreeVol', float),
                                   ('vol', float), ('delta_voro', float), ('atomic strain', float)])
                    atom_feature = np.empty(atoms.size, dtype=dt)
                    atom_feature['Natom'] = count[:, 0]
                    atom_feature['stz_size'] = count[:, 1]
                    atom_feature['FreeVol'] = count[:, 2] / 1000.0
                    atom_feature['Ave_FreeVol'] = count[:, 2] / 1000.0 / count[:, 0]
                    atom_feature['vol'] = atoms['vol']
                    atom_feature['delta_voro'] = atoms['vol'] - SphericalVol(atoms['r'])
                    atom_feature['atomic strain'] = atoms['atomic strain']

                    density_atoms_lammps_file = hist_dir + 'open_vol_max_dr_%s_%.1f_dl_%.1f.dump.gz' % (keyword, max_dr, dl)
                    DensityAtomGenerateConfiguration(atoms, boundary, dxyz, atom_feature, density_atoms_lammps_file)

                    np.savez_compressed(hist_dir + 'open_vol_max_dr_%s_%.1f_dl_%.1f.npz' % (keyword, max_dr, dl),
                                        atom_feature, fmt='%d, %d, %.4f, %.4f, %.4f, %.4f, %.4f')

        tp_finished = time.time()
        print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '13':
    ############### Define the environment for each atom ##########################
    # exports the activated volume around the activated stz to the output directory.
    # 'feature_events_T%dK_%.2f.dat'%(temp, strain) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 13
    print('\n status 13: Define the environment for each atom.')
    print('frames:', frames)
    tp_0 = time.time()

    dt = np.dtype([('state', '<i4'), ('Eb', '<f4'), ('region', '<i4'),
                   ('stz_size', '<i4'), ('activated_voro', '<f4'), ('activated_vol', '<f4'),
                   ('ave_activated_voro', '<f4'), ('ave_activated_vol', '<f4')])
    # print('feature dtype:', feature.dtype)
    num_refs = 9
    max_dr_list = [4.0, 4.5, 5.0, 5.5, 6.0]
    for max_dr in max_dr_list:
        print('max_dr:', max_dr)

        for f in range(len(frames)):
            # pick the activated state
            frame = frames[f]
            input_dir = input_direct + 'frame_%d/' % frame
            keyword = 'T%dK_frame_%d' % (temp, frame)
            Eb_data_file = os.path.join(input_dir, 'saved_Eb.npz')
            Eb_data = np.load(Eb_data_file)['arr_0']
            Eb = (Eb_data - Eb_data[0]) * 1000
            state = np.argmax(Eb)
            if state == 0 or state == 15:
                state = 7
            steps = [0, state, 15]
            feature_matrix = np.zeros(len(steps) * 4, dtype=dt)
            feature_stz = np.zeros(len(steps) * num_refs, dtype=dt)

            for m in range(len(steps)):
                step = steps[m]
                feature_ref = np.zeros(num_refs, dtype=dt)
                print('\nprocessing the calculation at the activated state:', keyword, step)
                vols_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, m)
                print('vols_file:', vols_file)
                vols = loadVolumePosition(vols_file)
                print('vols.size:', vols.size)
                print('vols.shape[0]:', vols.shape[0])
                data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, m)
                init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
                final_data = output_dir + 'coords_atoms_%s_2.dump.gz' % keyword

                # define the stz region using the critial value, mises_crit
                atoms, boundary, dxyz = loadData(data, R1, R2)
                last_atoms, boundary, dxyz = loadData(final_data, R1, R2)
                mises_data, disp_data = AtomicStrain(final_data, init_data)
                satom = last_atoms[np.argmax(mises_data)]  ## select the satom in the last data
                id_satoms = mises_data >= mises_crit
                satoms = atoms[id_satoms]  ## select the satoms in the data

                # calculate the volume of bulk
                total_voro = np.sum(atoms['vol'] - 4 / 3 * 3.14 * atoms['r'] * atoms['r'] * atoms['r'])
                total_vol = vols.size * 0.008
                print('atoms.size:', atoms.size)
                print('total_voro/atoms.size:', total_voro / atoms.size)
                print('vols.shape:', vols.shape)

                feature_matrix[m * 4] = (step, Eb[step], 10,
                                         atoms.size, total_voro, total_vol,
                                         total_voro / atoms.size, total_vol / atoms.size)

                # calculate the volume of matrix
                index = [False] * atoms.size
                index_model = index
                coord_model = [satom['x'], satom['y'], satom['z']]
                print('coord_model:', coord_model)
                for atom in atoms:
                    dr = AtomicSpacing(coord_model, atom)
                    if dr <= 20.0:
                        index_model = np.logical_or(index_model, atoms == atom)
                model_atoms = atoms[index_model]
                model_voro = np.sum(model_atoms['vol'] - 4 / 3 * 3.14 * model_atoms['r'] * model_atoms['r'] * model_atoms['r'])
                model_vol = 0
                model_size = model_atoms.size

                # Fix the spherical space,
                for vol in vols:
                    if TwoAtoms(satom, vol, 20.0):
                        model_vol += 1
                print('model_atoms.size:', model_atoms.size)
                # print('model_voro/model_atoms.size:', model_voro / model_atoms.size)
                print('model_vol/model_atoms.size:', model_vol*0.008/model_atoms.size)

                feature_matrix[m * 4 + 1] = (step, Eb[step], 20,
                                             model_size, model_voro, model_vol * 0.008,
                                             model_voro / model_size, model_vol * 0.008 / model_size)

                # # calculate the volume of the stz and referenced regions.
                # # select num_refs regions
                # index_refs = index * num_refs
                # N_refs = [0] * num_refs
                # coord_ref0 = [np.mean(satoms['x']), np.mean(satoms['y']), np.mean(satoms['z'])]
                # print('coord_ref0:', coord_ref0)
                # ref_coords = [coord_ref0]

                index_refs = index * num_refs
                N_refs = [0] * num_refs
                # coord_ref0 = [satom['x'], satom['y'], satom['z']]
                # coord_ref0 = [np.mean(boundary[0]), np.mean(boundary[1]), np.mean(boundary[2])]
                coord_ref0 = [np.mean(satoms['x']), np.mean(satoms['y']), np.mean(satoms['z'])]
                print('coord_ref0:', coord_ref0)
                coord_ref1 = [8.5, 8.5, 8.5]
                coord_ref2 = [33.5, 33.5, 8.5]
                coord_ref3 = [33.5, 8.5, 33.5]
                coord_ref4 = [8.5, 33.5, 33.5]
                coord_ref5 = [8.5, 8.5, 33.5]
                coord_ref6 = [8.5, 33.5, 8.5]
                coord_ref7 = [33.5, 8.5, 8.5]
                coord_ref8 = [33.5, 33.5, 33.5]
                ref_coords = [coord_ref0, coord_ref1, coord_ref2, coord_ref3, coord_ref4,
                              coord_ref5, coord_ref6, coord_ref7, coord_ref8]

                # calculate the index in selected region
                index_ref_atoms = index
                for atom in atoms:
                    for i in range(num_refs):
                        dr = AtomicSpacing(ref_coords[i], atom)
                        if dr <= max_dr:
                            index_refs[i] = np.logical_or(index_refs[i], atoms == atom)
                            index_ref_atoms = np.logical_or(index_refs[i], index_ref_atoms)

                # output the selected regions into one dump file
                mises_data_i, disp_data_i = AtomicStrain(data, init_data)
                ref_atoms = atoms[index_ref_atoms]
                ref_mises_data = mises_data_i[index_ref_atoms]
                ref_atoms_lammps_file = temp_dir + 'coords_ref_atoms_%s_%d_max_dr_%.1f.dump.gz' % (keyword, m, max_dr)
                SatomGenerateConfiguration(ref_atoms, boundary, dxyz, ref_mises_data, ref_atoms_lammps_file)

                for p in range(num_refs):
                    ref_atoms = atoms[index_refs[p]]
                    cm_coord = ref_coords[p]
                    activated_voro = np.sum(
                        ref_atoms['vol'] - 4 / 3 * 3.14 * ref_atoms['r'] * ref_atoms['r'] * ref_atoms['r'])
                    activated_vol = 0
                    stz_size = ref_atoms.size

                    # Fix the spherical space,
                    # cm_coord, r=max_dr,
                    vols_index = [True]*vols.size
                    xyz = ['x', 'y', 'z']
                    for i in range(len(cm_coord)):
                        vols_index = np.logical_and(vols_index,
                                     np.logical_and(vols[xyz[i]] >= cm_coord[i] - max_dr*schmitt_radius,
                                                    vols[xyz[i]] <= cm_coord[i] + max_dr*schmitt_radius))
                    ref_vols = vols[vols_index]
                    for vol in ref_vols:
                        if inMesh(cm_coord, vol, mesh_size, max_dr):
                            activated_vol +=1

                    feature_ref[p] = (step, Eb[step], p,
                                      stz_size, activated_voro, activated_vol * 0.008,
                                      activated_voro / stz_size, activated_vol * 0.008 / stz_size)

                matrix_size = model_size - feature_ref[0]['stz_size']
                matrix_voro = model_voro - feature_ref[0]['activated_voro']
                matrix_vol = model_vol*0.008 - feature_ref[0]['activated_vol']
                feature_matrix[m * 4 + 2] = (step, Eb[step], 30,
                                             matrix_size, matrix_voro, matrix_vol,
                                             matrix_voro / matrix_size, matrix_vol / matrix_size)
                feature_matrix[m * 4 + 3] = feature_ref[0]
                feature_stz[m*num_refs:(m+1)*num_refs] = feature_ref
                print('stz.size:', feature_ref[0]['stz_size'])
                print('activated_vol/stz_size:', feature_ref[0]['ave_activated_vol'])

            feature_matrix_file = output_dir + 'feature_matrix_events_%s_max_dr_%.1f.dat' % (keyword, max_dr)
            print('\nSaved feature_matrix_file:', feature_matrix_file)
            np.savetxt(feature_matrix_file, feature_matrix,
                       fmt='%d, %.4f, %d, %d, %.4f, %.4f, %.4f, %.4f',
                       header='state, Eb, region,'
                              'stz_size, activated_voro, activated_vol'
                              'ave_activated_voro, ave_activated_vol')
            np.savez_compressed(output_dir + 'feature_matrix_events_%s_max_dr_%.1f.npz' % (keyword, max_dr),
                                feature_matrix, fmt='%d, %d, %.4f, %.4f, %.4f, %.4f, %.4f')

            feature_stz_file = output_dir + 'feature_stz_events_%s_max_dr_%.1f.dat' % (keyword, max_dr)
            print('\nSaved feature_stz_file:', feature_stz_file)
            np.savetxt(feature_stz_file, feature_stz,
                       fmt='%d, %.4f, %d, %d, %.4f, %.4f, %.4f, %.4f',
                       header='state, Eb, region,'
                              'stz_size, activated_voro, activated_vol'
                              'ave_activated_voro, ave_activated_vol')
            np.savez_compressed(output_dir + 'feature_stz_events_%s_max_dr_%.1f.npz' % (keyword, max_dr),
                                feature_stz, fmt='%d, %d, %.4f, %.4f, %.4f, %.4f, %.4f')

            tp_finished = time.time()
            print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '14':
    ############### Define the environment for each atom ##########################
    # exports the activated volume around the activated stz to the output directory.
    # 'feature_events_T%dK_%.2f.dat'%(temp, strain) would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 14
    print('\n status 14: Define volume distribution of one specific atom.')
    print('frames:', frames)
    tp_0 = time.time()

    R = 20.0
    bins = 200
    dr = R/bins
    for f in range(len(frames)):
        # pick the activated state
        frame = frames[f]
        keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
        Eb_data_file = os.path.join(input_direct, 'frame_%d/saved_Eb.npz' % frame)
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        steps = [0]
        print('steps:', steps)

        for m in range(len(steps)):
            step = steps[m]
            print('\nprocessing the calculation at the activated state:', step)
            data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
            init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
            final_data = output_dir + 'coords_atoms_%s_15.dump.gz' % keyword
            vols_file = output_dir + 'coords_volume_%s_%d.dump.gz' % (keyword, step)
            vols = loadVolumePosition(vols_file)

            atoms, boundary, dxyz = loadData(data, R1, R2)
            last_atoms, boundary, dxyz = loadData(final_data, R1, R2)
            mises_data, disp_data = AtomicStrain(final_data, init_data)
            satom = atoms[np.argmax(mises_data)]

            print('atoms[satom]:', atoms[10])
            print('last_atoms[satom]:', last_atoms[10])
            print('range of mises_data:', min(mises_data), max(mises_data))
            extend_vols, extend_boundary, extend_dxyz = Enlargebox(vols, boundary, dxyz, 25.0)
            extend_atoms, extend_boundary, extend_dxyz = Enlargebox(atoms, boundary, dxyz, 25.0)

            tp_1 = time.time()
            print('time loading configurations:', tp_1 - tp_0)

            vol_distri = np.zeros((bins, 6), dtype=float)
            vol_distri[:, 0] = np.arange(0, R, dr)

            for i in range(9):
                dl = i*0.5 -2
                coord_atom = [satom['x'], satom['y']+dl, satom['z']]
                print('dl, coord_atom:', dl, coord_atom)
                count = 0
                count2 = 0
                for atomj in atoms:
                    rij = AtomicSpacing(coord_atom, atomj)
                    if rij>=R:
                        continue
                    else:
                        count +=1
                        n = int(rij//dr)
                        vol_distri[n, 1] +=1
                print('count_atoms:', count)
                for volj in vols:
                    Rij = AtomicSpacing(coord_atom, volj)
                    if Rij>=R:
                        continue
                    else:
                        count2 +=1
                        n = int(Rij // dr)
                        vol_distri[n, 2] += 1
                print('count_vol:', count2)

                vol_distri[:, 3] = vol_distri[:, 1]
                vol_distri[:, 4] = vol_distri[:, 2]
                print(vol_distri[:, 1])
                for i in range(1, bins):
                    vol_distri[i, 3] += vol_distri[i-1, 3]
                    vol_distri[i, 4] += vol_distri[i-1, 4]
                for i in range(bins):
                    vol_distri[i, 5] = vol_distri[i, 4]/vol_distri[i, 3]/1000

                tp_2 = time.time()
                print('\ntime calculating:', tp_2 - tp_1)

                np.savez_compressed(hist_dir + 'open_vol_distri_%s_dl_%.1f.npz' % (keyword, dl),
                                    vol_distri, fmt='%.1f, %d, %.4f, %d, %.4f, %.4f')
                vol_distri_file = hist_dir + 'open_vol_distri_%s_dl_%.1f.txt' % (keyword, dl)
                np.savetxt(vol_distri_file, vol_distri, fmt='%.1f, %d, %.4f, %d, %.4f, %.4f')

        tp_finished = time.time()
        print('tp_finished - tp_0:', tp_finished - tp_0)

elif status == '20':
    ############### Calculate the voronoi indices for each atom ##########################
    # exports the fraction of voronoi indices in the whole sample, the activated stz,
    # and randomly selected regions for comparison, separately.
    # 'voro_indices_fraction_all_%s.dat' %keyword would be obtained.
    # run by: ovitos Workflow_volume_analysis.py -n 20
    print('\n status 20: Calculate the voronoi indices and obtain a distribution.')
    print('frames:', frames)
    voro_index = [[0, 2, 8, 1], [0, 0, 12, 0], [0, 2, 8, 2], [0, 3, 6, 3], [0, 1, 10, 2],
                  [0, 3, 6, 4], [0, 1, 10, 4], [0, 0, 12, 4], [0, 1, 10, 5], [0, 2, 8, 6]]

    tp_0 = time.time()
    num_refs = 9
    for f in range(len(frames)):
        # pick the activated state
        frame = frames[f]
        keyword = 'T%dK_%.2f_frame_%d' % (temp, strain, frame)
        Eb_data_file = os.path.join(input_direct, 'frame_%d/saved_Eb.npz' % frame)
        Eb_data = np.load(Eb_data_file)['arr_0']
        Eb = (Eb_data - Eb_data[0]) * 1000
        state = np.argmax(Eb)
        steps = [0, state, 15]
        voro_index_fraction_all = np.zeros((len(voro_index), len(steps)), dtype=float)

        for m in range(len(steps)):
            step = steps[m]
            print('processing the calculation at the activated state:', step)
            data = output_dir + 'coords_atoms_%s_%d.dump.gz' % (keyword, step)
            init_data = output_dir + 'coords_atoms_%s_0.dump.gz' % keyword
            final_data = output_dir + 'coords_atoms_%s_15.dump.gz' % keyword
            atoms, boundary, dxyz = loadData(data, R1, R2)
            mises_data, disp_data = AtomicStrain(final_data, init_data)
            satom = atoms[np.argmax(mises_data)]
            id_satoms = mises_data >= mises_crit
            satoms = atoms[id_satoms]

            # According to the activated stz, define the mass center and the activated space
            voro_indices = VoroIndiceCal(data, R1, R2)
            indices_all, counts_all, fraction_all = VoroIndiceHist(voro_indices[:, 2:], voro_index)
            voro_index_fraction_all[:, m] = fraction_all
            print('satom.voro_indice:', voro_indices[np.argmax(mises_data)])

            # select num_refs regions to trace the voronoi indices evolution
            index = [False] * atoms.size
            index_refs = index*num_refs
            N_refs = [0]*num_refs
            # coord_ref0 = [satom['x'], satom['y'], satom['z']]
            # coord_ref0 = [np.mean(boundary[0]), np.mean(boundary[1]), np.mean(boundary[2])]
            coord_ref0 = [np.mean(satoms['x']), np.mean(satoms['y']), np.mean(satoms['z'])]
            print('coord_ref0:', coord_ref0)
            print('satom:', satom)
            coord_ref0 = [np.mean(satoms['x']), np.mean(satoms['y']), np.mean(satoms['z'])]
            coord_ref1 = [8.5, 8.5, 8.5]
            coord_ref2 = [33.5, 33.5, 8.5]
            coord_ref3 = [33.5, 8.5, 33.5]
            coord_ref4 = [8.5, 33.5, 33.5]
            coord_ref5 = [8.5, 8.5, 33.5]
            coord_ref6 = [8.5, 33.5, 8.5]
            coord_ref7 = [33.5, 8.5, 8.5]
            coord_ref8 = [33.5, 33.5, 33.5]
            for i in range(num_refs):
                ref_coords = [coord_ref0, coord_ref1, coord_ref2, coord_ref3, coord_ref4,
                                     coord_ref5, coord_ref6, coord_ref7, coord_ref8]
            max_dr = 6

            # calculate the voro_indices in selected region
            for atom in atoms:
                for i in range(num_refs):
                    dr = AtomicSpacing(ref_coords[i], atom)
                    if dr <= max_dr:
                        index_refs[i] = np.logical_or(index_refs[i], atoms == atom)
                        index = np.logical_or(index_refs[i], index)

            # output the selected regions into one dump file
            mises_data_i, disp_data_i = AtomicStrain(data, init_data)
            ref_atoms = atoms[index]
            ref_mises_data = mises_data_i[index]
            ref_atoms_lammps_file = ref_atom_dir + 'coords_ref_atoms_%s_%d.dump.gz' % (keyword, step)
            SatomGenerateConfiguration(ref_atoms, boundary, dxyz, ref_mises_data, ref_atoms_lammps_file)

            # calculate the voronoi cluster for specific ith atom in the stz region.
            stz_atoms = atoms[index_refs[0]]
            stz_indices = voro_indices[index_refs[0]]
            for i in range(stz_atoms.size):
                voro_index_i = list(stz_indices[i][2:])
                atomi = stz_atoms[i]
                N = np.sum(voro_index_i)
                cluster_i = ClusterCal(data, atomi, N)
                voro_index_i_list = [str(x) for x in voro_index_i]
                voro_index_i_str = '-'.join(voro_index_i_list)
                cluster_i_file = cluster_i_dir + '%s_cluster_id_%d_%d_%s.xyz' \
                                 %(keyword, atomi['id'], step, voro_index_i_str)
                np.savetxt(cluster_i_file, cluster_i, fmt='%d %.4f %.4f %.4f',
                           header='%d\n #id:%d, type:%d, index:%s'
                                  '#type x y z' % (N+1, atomi['id'], atomi['type'], voro_index_i))

            # Calculate the voronoi index for the num_refs selected regions
            voro_index_fraction_refs = np.zeros((len(voro_index), num_refs), dtype=float)
            for i in range(num_refs):
                voro_indices_ref = voro_indices[index_refs[i]]
                indices_ref, counts_ref, fraction_ref = VoroIndiceHist(voro_indices_ref[:, 2:], voro_index)
                voro_index_fraction_refs[:, i] = fraction_ref

            voro_index_fraction_refs_file = voro_vol_dir + 'voro_indices_fraction_refs_%s_%d.dat' % (keyword, step)
            np.savetxt(voro_index_fraction_refs_file, voro_index_fraction_refs,
                       fmt='%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f')
        voro_index_fraction_all_file = voro_vol_dir + 'voro_indices_fraction_all_%s.dat' %keyword
        np.savetxt(voro_index_fraction_all_file, voro_index_fraction_all, fmt='%.2f %.2f %.2f')

        tp_finished = time.time()
        print('tp_finished - tp_0:', tp_finished - tp_0)

else:
    print('non-recognized status #')
