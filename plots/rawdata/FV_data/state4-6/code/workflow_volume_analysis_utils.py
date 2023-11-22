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

def read_setup(filename, return_comments=False):
    with open(filename, 'r') as fdir:
        input_lines = fdir.read().splitlines()
        output_lines= []
        comment_lines=[]
        for rawline in input_lines:
            line = rawline.strip()
            if line.startswith('#'):
                comment_line = line.strip('# ')
                comment_id = len(output_lines)
                comment_lines.append((comment_id, comment_line))
            if (line != '') and (not line.startswith('#')):
                output_lines.append(line)
        if return_comments:
            return output_lines, comment_lines
        else:
            return output_lines

def arguments():
    parser = argparse.ArgumentParser()
    # global arguments
    parser.add_argument('-n', '--nstatus', default=0, type=str,
                        help='status number for running the script')
    parser.add_argument('-i', '--input', default='setup_volume_analysis.txt', type=str,
                        help='setup file')
    parser.add_argument('--temp', default=2, type=int,
                        help='finite temperature')
    parser.add_argument('--strain', default=0.24, type=float,
                        help='strain where transition event actually happens')
    parser.add_argument('--frame', default=0, type=int,
                        help='frame where NEB calculations is calculated')
    parser.add_argument('--frames', nargs='+', type=int,
                        default=[0, 184, 226, 452, 678, 904],
                        help='frame where NEB calculations is calculated')
    parser.add_argument('--target_frame', default=0, type=int,
                        help='frame where NEB calculations is calculated')
    parser.add_argument('--num_frame', default=16, type=int,
                        help='total frames in the input data')
    parser.add_argument('--drange', nargs=3, default=[0, 16, 1], type=int,
                        help='dump range for calculation')
    parser.add_argument('--ref_dump', default='dump.neb.final.mg.0', type=str,
                        help='ref lammps dump file name')
    parser.add_argument('--last_dump', default='dump.neb.final.mg.15', type=str,
                        help='last lammps dump file name')
    parser.add_argument('--lammps_dump', default='dump.neb.final.mg.', type=str,
                        help='lammps dump file name')

    # directories
    parser.add_argument('--input_direct', default='', type=str,
                        help='directories for input dump files')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='directories for output data')
    parser.add_argument('--temp_dir', default='output', type=str,
                        help='directories for output data')
    parser.add_argument('--extend_atom_direct', default='extend_atom', type=str,
                        help='directories for extend atoms in output_direct')
    parser.add_argument('--ref_atom_direct', default='ref_atom', type=str,
                        help='directories for ref atoms in output_direct')
    parser.add_argument('--voro_vol_direct', default='voro_vol', type=str,
                        help='directories for voro vol in output_direct')
    parser.add_argument('--cluster_i_direct', default='cluster_i', type=str,
                        help='directories for cluster_i in output_direct')
    parser.add_argument('--atom_group_direct', default='atom_group', type=str,
                        help='directories for atom groups in output_direct')
    parser.add_argument('--atom_outvol_direct', default='atom_outvol', type=str,
                        help='directories for atom outvol in output_direct')
    parser.add_argument('--volume_group_direct', default='volume_group', type=str,
                        help='directories for volume_groups in output_direct')
    parser.add_argument('--group_i_direct', default='group_i', type=str,
                        help='directories for ith volume group in output_direct')
    parser.add_argument('--hist_direct', default='vol_hist', type=str,
                        help='directories for ith volume group in output_direct')
    parser.add_argument('--temp_direct', default='temp', type=str,
                        help='directories for temporary files in output_direct')

   # status in [2,3]

    parser.add_argument('--R1', default=1.28, type=float,
                        help='Radii for type=1 atom')
    parser.add_argument('--R2', default=1.62, type=float,
                        help='Radii for type=2 atom')
    parser.add_argument('--schmitt_radius', default=1.12, type=float,
                        help='a factor for adjusting the atom radius')
    parser.add_argument('--alpha', default=1.1, type=float,
                        help='a factor for adjusting the atom radius to find the outer volume for each atom')
    parser.add_argument('--mesh_size', default=0.1, type=float,
                        help='side length of the meshed finite element box')
    parser.add_argument('--dL', default=3.0, type=float,
                        help='extend the original box to L+dL*2')
    parser.add_argument('--mises_crit', default=0.0100, type=float,
                        help='set up the criteria for the active atoms')
    parser.add_argument('--vol_th_hi', default=500, type=int,
                        help='threshold for the changing volume group')
    parser.add_argument('--vol_th_lo', default=300, type=int,
                        help='threshold for components in ith volume group')
    parser.add_argument('--trace_th_hi', default=0.9, type=float,
                        help='above which two volume groups shall be the same')
    parser.add_argument('--trace_th_lo', default=0.4, type=float,
                        help='above which one volume group shall belong to the previous one')
    return parser

# define function for loading lammps data and return the boundary and data files.
def loadData(file_data: str, R1, R2):
    node = import_file(file_data)
    n_particles = node.source.number_of_particles
    # dt = np.dtype([('id', '<i4'), ('type', '<i1'), ('x', float), ('y', float), ('z', float),
    #                ('r', float), ('c_pe_peratom', float), ('vol', float)])
    dt = np.dtype([('id', '<i4'), ('type', '<i1'), ('x', float), ('y', float), ('z', float),
                   ('r', float), ('vol', float)])
    data = np.empty(n_particles, dtype=dt)

    ##Obtain the boundary and dxyz
    atom_type = node.source.particle_properties.particle_type.type_list
    atom_type[0].radius = R1  # Cu atom
    atom_type[1].radius = R2  # Zr atom
    # set up the voronoi tessellation modifier
    vol_mod = VoronoiAnalysisModifier(compute_indices=True, edge_count=10,
                                      relative_face_threshold=0.0, use_radii=True)
    node.modifiers.append(vol_mod)
    node.compute()

    nodecell = node.source.cell
    cell = nodecell.matrix
    dxyz = [cell[0][0], cell[1][1], cell[2][2]]
    boundary = [[cell[0, -1], cell[0, -1] + dxyz[0]],
                [cell[1, -1], cell[1, -1] + dxyz[1]],
                [cell[2, -1], cell[2, -1] + dxyz[2]],
                ]

    data['id'] = node.output.particle_properties['Particle Identifier'].array
    data['type'] = node.output.particle_properties['Particle Type'].array
    positions = node.output.particle_properties['Position'].array
    data['x'], data['y'], data['z'] = positions[:, 0], positions[:, 1], positions[:, 2]
    data['vol'] = node.output.particle_properties['Atomic Volume'].array
    id1 = data['type'] == 1
    id2 = data['type'] == 2
    data['r'][id1] = R1
    data['r'][id2] = R2

    data = np.hstack((data[id1], data[id2]))
    return data, boundary, dxyz

def loadDataMore(file_data: str, ref_data, R1, R2):
    node = import_file(file_data)
    n_particles = node.source.number_of_particles

    # define the features for the data
    dt = np.dtype([('id', '<i4'), ('type', '<i1'), ('x', float), ('y', float), ('z', float),
                   ('r', float), ('vol', float), ('atomic strain', float)])
    data = np.empty(n_particles, dtype=dt)

    # ## set up the mises modifiers and voronoi volume modifiers
    mise_mod = AtomicStrainModifier(cutoff=3.8, eliminate_cell_deformation=True)
    mise_mod.reference = FileSource()
    mise_mod.reference.load(ref_data)
    node.modifiers.append(mise_mod)

    atom_type = node.source.particle_properties.particle_type.type_list
    atom_type[0].radius = R1  # Cu atom
    atom_type[1].radius = R2  # Zr atom
    # set up the voronoi tessellation modifier
    vol_mod = VoronoiAnalysisModifier(compute_indices=True, edge_count=10,
                                      relative_face_threshold=0.0, use_radii=True)
    node.modifiers.append(vol_mod)
    node.compute()

    nodecell = node.source.cell
    cell = nodecell.matrix
    dxyz = [cell[0][0], cell[1][1], cell[2][2]]
    boundary = [[cell[0, -1], cell[0, -1] + dxyz[0]],
                [cell[1, -1], cell[1, -1] + dxyz[1]],
                [cell[2, -1], cell[2, -1] + dxyz[2]],
                ]

    data['id'] = node.output.particle_properties['Particle Identifier'].array
    data['type'] = node.output.particle_properties['Particle Type'].array
    positions = node.output.particle_properties['Position'].array
    data['x'], data['y'], data['z'] = positions[:, 0], positions[:, 1], positions[:, 2]
    data['vol'] = node.output.particle_properties['Atomic Volume'].array
    data['atomic strain'] = node.output.particle_properties['Shear Strain'].array
    id1 = data['type'] == 1
    id2 = data['type'] == 2
    data['r'][id1] = R1
    data['r'][id2] = R2

    data = np.hstack((data[id1], data[id2]))
    return data, boundary, dxyz

def loadVolume(file_data: str):
    node = import_file(file_data)
    ##Obtain the boundary and dxyz
    node.compute()
    positions = node.output.particle_properties['Position'].array
    volumeY = positions[:, 1]
    return volumeY

def loadVolumePosition(file_data: str):
    node = import_file(file_data)
    n_particles = node.source.number_of_particles
    ##Obtain the boundary and dxyz
    node.compute()
    dt = np.dtype([('x', float), ('y', float), ('z', float)])
    data = np.empty(n_particles, dtype=dt)
    positions = node.output.particle_properties['Position'].array
    data['x'], data['y'], data['z'] = positions[:, 0], positions[:, 1], positions[:, 2]
    return data

def NodeCell(target_data):
    node = import_file(target_data)
    node.compute()
    nodecell = node.source.cell
    ref_cell = nodecell.matrix
    # print('ref_cell.shape:', ref_cell)
    return ref_cell

def AffineTransform(file_data, ref_cell):
    node = import_file(file_data)
    nodecell = node.source.cell
    cell = nodecell.matrix
    n_particles = node.source.number_of_particles

    # define the features for the data
    dt = np.dtype([('id', '<i4'), ('type', '<i1'), ('x', float), ('y', float), ('z', float)])
    data = np.empty(n_particles, dtype=dt)

    mod = AffineTransformationModifier(
        transform_particles=True,
        transform_box=True,
        relative_mode=False,
        target_cell=ref_cell,
    )

    node.modifiers.append(mod)
    node.compute()

    nodecell = node.output.cell
    cell = nodecell.matrix
    print('cell[0]:', cell[0])

    dxyz = [cell[0][0], cell[1][1], cell[2][2]]
    boundary = [[cell[0, -1], cell[0, -1] + dxyz[0]],
                [cell[1, -1], cell[1, -1] + dxyz[1]],
                [cell[2, -1], cell[2, -1] + dxyz[2]],
                ]
    data['id'] = node.output.particle_properties['Particle Identifier'].array
    data['type'] = node.output.particle_properties['Particle Type'].array
    positions = node.output.particle_properties['Position'].array
    data['x'], data['y'], data['z'] = positions[:, 0], positions[:, 1], positions[:, 2]
    return data, boundary, dxyz

def AtomicStrain(file_data, ref_data):
    node = import_file(file_data)
    # Load the displacement modifier (respect to t = 0)
    disp_mod = CalculateDisplacementsModifier()
    disp_mod.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToReference
    disp_mod.reference.load(ref_data)
    node.modifiers.append(disp_mod)

    mise_mod = AtomicStrainModifier(cutoff=3.8, eliminate_cell_deformation=True)
    mise_mod.reference = FileSource()
    mise_mod.reference.load(ref_data)
    node.modifiers.append(mise_mod)

    node.compute()
    mises_data = node.output.particle_properties['Shear Strain'].array
    disp_data = node.output.particle_properties['Displacement Magnitude'].array
    return mises_data, disp_data

def SphericalVol(r):
    return 4/3*3.14159*r*r*r

def VoroVol(file_data):
    node = import_file(file_data)
    atom_type = node.source.particle_properties.particle_type.type_list
    atom_type[0].radius = R1  # Cu atom
    atom_type[1].radius = R2  # Zr atom
    # set up the voronoi tessellation modifier
    vol_mod = VoronoiAnalysisModifier(compute_indices=True, edge_count=10,
                                      relative_face_threshold=0.0, use_radii=True)
    node.modifiers.append(vol_mod)
    node.compute()
    voro_vol = node.output.particle_properties['Atomic Volume'].array
    return voro_vol

def VoroIndiceCal(file_data, R1, R2):
    node = import_file(file_data)
    n_particles = node.source.number_of_particles
    data = np.zeros((n_particles, 6), dtype=int)
    # Set atomic radii (required for polydisperse Voronoi tessellation).
    atom_type = node.source.particle_properties.particle_type.type_list
    atom_type[0].radius = R1  # Cu atom
    atom_type[1].radius = R2  # Zr atom
    # set up the voronoi tessellation modifier
    vol_mod = VoronoiAnalysisModifier(compute_indices=True, edge_count=10,
                                      edge_threshold=0.1, use_radii=True)
    node.modifiers.append(vol_mod)
    node.compute()

    # Access computed Voronoi indices.
    # This is an (N) x (M) array, where M is the maximum face order.
    data[:, 0] = node.output.particle_properties['Particle Identifier'].array
    data[:, 1] = node.output.particle_properties['Particle Type'].array
    voro_list = node.output.particle_properties['Voronoi Index'].array
    data[:, 2:] = voro_list[:, 2:6]
    return data

def VoroIndiceHist(voro_indices, voro_index):
    # This helper function takes a two-dimensional array and computes a frequency
    # histogram of the data rows using some NumPy magic.
    # It returns two arrays (of equal length):
    # 1. The list of unique data rows from the input array
    # 2. The number of occurences of each unique row
    # Both arrays are sorted in descending order such that the most frequent rows are listed first.
    def row_histogram(a):
        ca = np.ascontiguousarray(a).view([('', a.dtype)] * a.shape[1])
        unique, indices, inverse = np.unique(ca, return_index=True, return_inverse=True)
        counts = np.bincount(inverse)
        sort_indices = np.argsort(counts)[::-1]
        return (a[indices[sort_indices]], counts[sort_indices])

    # Compute frequency histogram.
    unique_indices, counts = row_histogram(voro_indices)
    num = np.zeros((len(voro_index)))
    for i in range(num.size):
        for j in range(counts.size):
            if list(voro_index[i]) == list(unique_indices[j]):
                num[i] = counts[j]
                continue
    return unique_indices, counts, voro_index, num

def ClusterCal(file_data, atomi, N):
    from ovito.data import NearestNeighborFinder
    # print('calculating neighboring id of particle:', atomi['id'])
    node = import_file(file_data)
    node.compute()

    id_list = node.output.particle_properties['Particle Identifier'].array
    cluster = np.empty((N+1, 4))
    cluster[0] = [atomi['type'], atomi['x'], atomi['y'], atomi['z']]
    i = 1
    finder = NearestNeighborFinder(N, node.output)
    for neigh in finder.find(np.where(id_list == atomi['id'])[0][0]):
        cluster[i, 0] = node.output.particle_properties.particle_type.array[neigh.index]
        cluster[i, 1:] = node.output.particle_properties['Position'].array[neigh.index]
        i+=1
    return cluster

def PotentialEnergy(file_data):
    node = import_file(file_data)
    node.compute()
    pe_data = node.output.particle_properties['c_pe_peratom'].array
    return pe_data

def GetBoxSize(filepath:str):
    def parseBoundary(line: str) -> Tuple[int]:
        elems = line.split(' ')
        return (float(elems[0]), float(elems[1]))
    with open(filepath, 'r') as f:
        text = f.read()
    lines = text.split('\n')
    boundary = list(map(parseBoundary, lines[5:8]))
    dxyz = list(map(lambda x: x[1] - x[0], boundary))
    return boundary, dxyz

# define function for calculating atomic mises strain, return the atom coordinates with largest mises strain value.
def SelectMiseAtom(last_data, ref_data) -> List[float]:
    node = import_file(last_data)
    mise_mod = AtomicStrainModifier(cutoff=3.8, eliminate_cell_deformation=True)
    mise_mod.reference = FileSource()
    mise_mod.reference.load(ref_data)
    node.modifiers.append(mise_mod)
    node.compute()

    positions = node.output.particle_properties['Position'].array
    mises_list = node.output.particle_properties['Shear Strain'].array
    idx = mises_list == max(mises_list)
    satom_coords = positions[idx].flatten()
    satom_coords.tolist()
    return satom_coords

def Enlargebox(atoms, boundary, dxyz, dL):
    extend_atoms = atoms
    axis = ['x', 'y', 'z']
    for i in range(len(axis)):
        axs_lo = np.logical_and(extend_atoms[axis[i]] >= boundary[i][0],
                                extend_atoms[axis[i]] <= boundary[i][0] + dL)
        atoms_lo = extend_atoms[axs_lo]
        atoms_lo[axis[i]] += dxyz[i]
        extend_atoms= np.hstack((extend_atoms, atoms_lo))
        axs_hi = np.logical_and(extend_atoms[axis[i]] >= boundary[i][1] - dL,
                                extend_atoms[axis[i]] <= boundary[i][1])
        atoms_hi = extend_atoms[axs_hi]
        atoms_hi[axis[i]] -= dxyz[i]
        extend_atoms = np.hstack((extend_atoms, atoms_hi))

    extend_boundary = [[boundary[0][0] - dL, boundary[0][1] + dL],
                       [boundary[1][0] - dL, boundary[1][1] + dL],
                       [boundary[2][0] - dL, boundary[2][1] + dL]]
    extend_dxyz = [dxyz[0] + dL*2, dxyz[1] + dL*2, dxyz[2] + dL*2]

    return extend_atoms, extend_boundary, extend_dxyz

# define function for shifting the S atom to the (L/2, L/2, L/2) and applied the pbc condition.
def normalizeCoord(satom: List[float], boundary: List[Tuple[int]], dxyz, atoms):
    for atom in atoms:
        atom['x'] = atom['x'] - satom[0] + boundary[0][0] + dxyz[0] * 0.5 - dxyz[0] * round((atom['x'] - satom[0]) / dxyz[0])
        atom['y'] = atom['y'] - satom[1] + boundary[1][0] + dxyz[1] * 0.5 - dxyz[1] * round((atom['y'] - satom[1]) / dxyz[1])
        atom['z'] = atom['z'] - satom[2] + boundary[2][0] + dxyz[2] * 0.5 - dxyz[2] * round((atom['z'] - satom[2]) / dxyz[2])
    return True

def MeshGenerateConfiguration(meshes, boundary, dxyz, lammps_file):
    # The number of particles we are going to create.
    num_particles = len(meshes[:, 0])
    print('Number of meshes:', num_particles)

    # Create particle properties
    id_prop = ParticleProperty.create(ParticleProperty.Type.Identifier, num_particles)
    id_prop.marray[:] = meshes[:, 0]
    type_prop = ParticleProperty.create(ParticleProperty.Type.ParticleType, num_particles)
    type_prop.marray[:] = meshes[:, 1]
    position_prop = ParticleProperty.create(ParticleProperty.Type.Position, num_particles)
    position_prop.marray[:] = meshes[:, 2:5]
    volume_prop = ParticleProperty.create_user('Volume', 'int', num_particles)
    volume_prop.marray[:] = meshes[:, 5]
    # st_prop = ParticleProperty.create_user('Structure Type', 'int', num_particles)
    # st_prop.marray[:] = meshes[:, 6]

    # Create the simulation box.
    cell = SimulationCell()
    cell.matrix = [[dxyz[0], 0, 0, boundary[0][0]],
                   [0, dxyz[1], 0, boundary[1][0]],
                   [0, 0, dxyz[2], boundary[2][0]],
                   ]
    cell.pbc = (True, True, True)

    # Create a data collection to hold the particle properties and simulation cell. Create a node
    data = DataCollection()
    data.add(id_prop); data.add(type_prop); data.add(position_prop); data.add(volume_prop)
    #data.add(st_prop)
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'Volume'])

def AtomGenerateConfiguration(atoms, boundary, dxyz, lammps_file):
    # The number of particles we are going to create.
    num_particles = len(atoms)
    # print('Number of atoms:', num_particles)

    # Create particle properties
    id_prop = ParticleProperty.create(ParticleProperty.Type.Identifier, num_particles)
    id_prop.marray[:] = atoms['id']
    type_prop = ParticleProperty.create(ParticleProperty.Type.ParticleType, num_particles)
    type_prop.marray[:] = atoms['type']
    position_prop = ParticleProperty.create(ParticleProperty.Type.Position, num_particles)
    position_prop.marray[:, 0] = atoms['x']
    position_prop.marray[:, 1] = atoms['y']
    position_prop.marray[:, 2] = atoms['z']

    # Create the simulation box.
    cell = SimulationCell()
    cell.matrix = [[dxyz[0], 0, 0, boundary[0][0]],
                   [0, dxyz[1], 0, boundary[1][0]],
                   [0, 0, dxyz[2], boundary[2][0]],
                   ]
    cell.pbc = (True, True, True)

    # Create a data collection to hold the particle properties and simulation cell. Create a node
    data = DataCollection()
    data.add(id_prop); data.add(type_prop); data.add(position_prop)
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z'])

def allAtomGenerateConfiguration(atoms, boundary, dxyz, lammps_file):
    # The number of particles we are going to create.
    num_particles = len(atoms)
    print('Number of atoms:', num_particles)

    # Create particle properties
    id_prop = ParticleProperty.create(ParticleProperty.Type.Identifier, num_particles)
    id_prop.marray[:] = atoms['id']
    type_prop = ParticleProperty.create(ParticleProperty.Type.ParticleType, num_particles)
    type_prop.marray[:] = atoms['type']
    position_prop = ParticleProperty.create(ParticleProperty.Type.Position, num_particles)
    position_prop.marray[:, 0] = atoms['x']
    position_prop.marray[:, 1] = atoms['y']
    position_prop.marray[:, 2] = atoms['z']
    pe_prop = ParticleProperty.create_user('c_pe_peratom', 'float', num_particles)
    pe_prop.marray[:] = atoms['c_pe_peratom']
    voro_prop = ParticleProperty.create_user('vol', 'float', num_particles)
    voro_prop.marray[:] = atoms['vol']

    # Create the simulation box.
    cell = SimulationCell()
    cell.matrix = [[dxyz[0], 0, 0, boundary[0][0]],
                   [0, dxyz[1], 0, boundary[1][0]],
                   [0, 0, dxyz[2], boundary[2][0]],
                   ]
    cell.pbc = (True, True, True)

    # Create a data collection to hold the particle properties and simulation cell. Create a node
    data = DataCollection()
    data.add(id_prop); data.add(type_prop); data.add(position_prop)
    data.add(pe_prop); data.add(voro_prop)
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'c_pe_peratom', 'vol'])

def SatomGenerateConfiguration(atoms, boundary, dxyz, ref_mises_data, lammps_file):
    # The number of particles we are going to create.
    num_particles = len(atoms)
    print('Number of atoms:', num_particles)

    # Create particle properties
    id_prop = ParticleProperty.create(ParticleProperty.Type.Identifier, num_particles)
    id_prop.marray[:] = atoms['id']
    type_prop = ParticleProperty.create(ParticleProperty.Type.ParticleType, num_particles)
    type_prop.marray[:] = atoms['type']
    position_prop = ParticleProperty.create(ParticleProperty.Type.Position, num_particles)
    position_prop.marray[:, 0] = atoms['x']
    position_prop.marray[:, 1] = atoms['y']
    position_prop.marray[:, 2] = atoms['z']
    mises_prop = ParticleProperty.create_user('atomic strain', 'float', num_particles)
    # mises_prop.marray[:] = atoms['atomic strain']
    mises_prop.marray[:] = ref_mises_data

    # Create the simulation box.
    cell = SimulationCell()
    cell.matrix = [[dxyz[0], 0, 0, boundary[0][0]],
                   [0, dxyz[1], 0, boundary[1][0]],
                   [0, 0, dxyz[2], boundary[2][0]],
                   ]
    cell.pbc = (True, True, True)

    # Create a data collection to hold the particle properties and simulation cell. Create a node
    data = DataCollection()
    data.add(id_prop); data.add(type_prop); data.add(position_prop);
    data.add(mises_prop)
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'atomic strain'])

def DensityAtomGenerateConfiguration(atoms, boundary, dxyz, atom_feature, lammps_file):
    # The number of particles we are going to create.
    num_particles = len(atoms)
    print('Number of atoms:', num_particles)

    # Create particle properties
    id_prop = ParticleProperty.create(ParticleProperty.Type.Identifier, num_particles)
    id_prop.marray[:] = atoms['id']
    type_prop = ParticleProperty.create(ParticleProperty.Type.ParticleType, num_particles)
    type_prop.marray[:] = atoms['type']
    position_prop = ParticleProperty.create(ParticleProperty.Type.Position, num_particles)
    position_prop.marray[:, 0] = atoms['x']
    position_prop.marray[:, 1] = atoms['y']
    position_prop.marray[:, 2] = atoms['z']
    Natom_prop = ParticleProperty.create_user('Natom', 'int', num_particles)
    Natom_prop.marray[:] = atom_feature['Natom']
    stz_size_prop = ParticleProperty.create_user('stz_size', 'int', num_particles)
    stz_size_prop.marray[:] = atom_feature['stz_size']
    FreeVol_prop = ParticleProperty.create_user('FreeVol', 'float', num_particles)
    FreeVol_prop.marray[:] = atom_feature['FreeVol']
    Ave_FreeVol_prop = ParticleProperty.create_user('Ave_FreeVol', 'float', num_particles)
    Ave_FreeVol_prop.marray[:] = atom_feature['Ave_FreeVol']
    vol_prop = ParticleProperty.create_user('vol', 'float', num_particles)
    vol_prop.marray[:] = atom_feature['vol']
    delta_voro_prop = ParticleProperty.create_user('delta_voro', 'float', num_particles)
    delta_voro_prop.marray[:] = atom_feature['delta_voro']
    mises_prop = ParticleProperty.create_user('atomic strain', 'float', num_particles)
    mises_prop.marray[:] = atom_feature['atomic strain']

    # Create the simulation box.
    cell = SimulationCell()
    cell.matrix = [[dxyz[0], 0, 0, boundary[0][0]],
                   [0, dxyz[1], 0, boundary[1][0]],
                   [0, 0, dxyz[2], boundary[2][0]],
                   ]
    cell.pbc = (True, True, True)

    # Create a data collection to hold the particle properties and simulation cell. Create a node
    data = DataCollection()
    data.add(id_prop); data.add(type_prop); data.add(position_prop);data.add(stz_size_prop);
    data.add(Natom_prop); data.add(FreeVol_prop);data.add(Ave_FreeVol_prop);
    data.add(vol_prop);data.add(delta_voro_prop);data.add(mises_prop);
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z',
     'Natom', 'stz_size', 'FreeVol', 'Ave_FreeVol', 'vol', 'delta_voro', 'atomic strain'])

def buildMesh(boundary: List[Tuple[int]], mesh_size_in_nm: float) -> List[int]:
    return list(map(lambda x: math.ceil((x[1] - x[0]) / mesh_size_in_nm), boundary))

def getMeshId(i: int, j: int, k: int, num_meshes_by_axis: List[int]) -> int:
    return i * (num_meshes_by_axis[1] * num_meshes_by_axis[2]) + j * num_meshes_by_axis[2] + k

def getMeshIdReverse(mesh_id: int, num_meshes_by_axis: List[int]) -> List[int]:
    return [
        mesh_id // (num_meshes_by_axis[1] * num_meshes_by_axis[2]),
        (mesh_id%(num_meshes_by_axis[1] * num_meshes_by_axis[2])) // (num_meshes_by_axis[2]),
        mesh_id%num_meshes_by_axis[2],
    ]

def getMeshCoordinate(i: int, j: int, k: int, boundary: List[Tuple[int]], mesh_size_in_nm: float) -> List[float]:
    return [
        boundary[0][0] + mesh_size_in_nm * (i + 0.5),
        boundary[1][0] + mesh_size_in_nm * (j + 0.5),
        boundary[2][0] + mesh_size_in_nm * (k + 0.5),
    ]

def getCutoff(satom, last_satoms):
    cutoff = 0.0
    for atom in last_satoms:
        min_dx = satom['x'] - atom['x']
        min_dy = satom['y'] - atom['y']
        min_dz = satom['z'] - atom['z']
        min_distance = math.sqrt(min_dx * min_dx + min_dy * min_dy + min_dz * min_dz)
        if min_distance > 8.5:
            cutoff = 8.5
            break
        elif min_distance >= cutoff:
            cutoff = min_distance
    if cutoff <= 3.8:
        cutoff = 3.8
    return cutoff

def nearAtom(cm_coord, atoms):
    dr = 20
    for atom in atoms:
        dx = cm_coord[0] - atom['x']
        dy = cm_coord[1] - atom['y']
        dz = cm_coord[2] - atom['z']
        min_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if min_distance < dr:
            dr = min_distance
            Catom = atom
    return Catom

def inAtom(mesh_coord: List[float], atom, mesh_size_in_nm, schmitt_radius) -> bool:
    # center_dx = mesh_coord[0] - atom['x']
    # center_dy = mesh_coord[1] - atom['y']
    # center_dz = mesh_coord[2] - atom['z']
    # min_dx = min(abs(center_dx - mesh_size_in_nm * 0.5), abs(center_dx + mesh_size_in_nm * 0.5))
    # min_dy = min(abs(center_dy - mesh_size_in_nm * 0.5), abs(center_dy + mesh_size_in_nm * 0.5))
    # min_dz = min(abs(center_dz - mesh_size_in_nm * 0.5), abs(center_dz + mesh_size_in_nm * 0.5))
    min_dx = mesh_coord[0] - atom['x']
    min_dy = mesh_coord[1] - atom['y']
    min_dz = mesh_coord[2] - atom['z']
    min_distance = math.sqrt(min_dx * min_dx + min_dy * min_dy + min_dz * min_dz)
    return min_distance <= atom['r'] * schmitt_radius

def inMesh(cm_coord, mesh_coords, mesh_size_in_nm, max_dr):
    dx = mesh_coords[0] - cm_coord[0]
    dy = mesh_coords[1] - cm_coord[1]
    dz = mesh_coords[2] - cm_coord[2]
    dr = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dr <= max_dr

def TwoAtoms(atomi, atomj, max_dr):
    dx = atomi['x'] - atomj['x']
    dy = atomi['y'] - atomj['y']
    dz = atomi['z'] - atomj['z']
    dij = math.sqrt(dx * dx + dy * dy + dz * dz)
    return dij <= max_dr

def AtomicSpacing(cm_coord, atom):
    dx = cm_coord[0] - atom['x']
    dy = cm_coord[1] - atom['y']
    dz = cm_coord[2] - atom['z']
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def getConnectedMeshIds(i: int, j: int, k: int, in_any_atom: List[bool], num_meshes_by_axis: List[int], checked_ids: List[bool]) -> List[int]:
    def visitNeighbor(i: int, j: int, k: int):
        neighbor_mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
        if not checked_ids[neighbor_mesh_id] and not in_any_atom[neighbor_mesh_id]:
            checked_ids[neighbor_mesh_id] = True
            frontier.put((i, j, k))

    connected_mesh_ids = []
    frontier = queue.Queue()
    frontier.put((i, j, k))
    mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
    checked_ids[mesh_id] = True
    while not frontier.empty():
        i, j, k = frontier.get()
        mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
        connected_mesh_ids.append(mesh_id)
        if 0 < i:
            visitNeighbor(i - 1, j, k)
        if i < num_meshes_by_axis[0] - 1:
            visitNeighbor(i + 1, j, k)
        if 0 < j:
            visitNeighbor(i, j - 1, k)
        if j < num_meshes_by_axis[1] - 1:
            visitNeighbor(i, j + 1, k)
        if 0 < k:
            visitNeighbor(i, j, k - 1)
        if k < num_meshes_by_axis[2] - 1:
            visitNeighbor(i, j, k + 1)
    return connected_mesh_ids

def getHistogram(volume_of_hollow_spaces: List[int], numBuckets: int, percentile: float = 99) -> Tuple[List[int], List[int], int]:
    vol_perc = np.percentile(volume_of_hollow_spaces, percentile)
    valid_vols = list(filter(lambda vol: vol <= vol_perc, volume_of_hollow_spaces))
    hist, bin_edges = np.histogram(valid_vols, numBuckets)
    return hist, bin_edges, vol_perc

def GetVolumeAndCoordinate(volume_group, volume_group_id, boundary, num_meshes_by_axis, mesh_size_in_nm: float):
    id_arr = []
    group_id_arr = []
    x_arr = []
    y_arr = []
    z_arr = []
    vol_arr = []
    for group in range(len(volume_group)):
        group_id = volume_group_id[group]
        for mesh_id in volume_group[group]:
            i, j, k = getMeshIdReverse(mesh_id, num_meshes_by_axis)
            id_arr.append(mesh_id)
            group_id_arr.append(group_id)
            x_arr.append(boundary[0][0] + mesh_size_in_nm * (i + 0.5))
            y_arr.append(boundary[1][0] + mesh_size_in_nm * (j + 0.5))
            z_arr.append(boundary[2][0] + mesh_size_in_nm * (k + 0.5))
            vol_arr.append(len(volume_group[group]))
    return np.concatenate((np.array([id_arr], dtype=np.int).T, np.array([group_id_arr], dtype=np.int).T,
                           np.array([x_arr]).T, np.array([y_arr]).T, np.array([z_arr]).T,
                           np.array([vol_arr]).T), axis=1)
