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
    parser.add_argument('--output_direct', default='output', type=str,
                    help='directories for output data')
    parser.add_argument('--extend_atom_direct', default='extend_atom', type=str,
                    help='directories for extend atoms in output_direct')
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
    dt = np.dtype([('id', '<i4'), ('type', '<i1'), ('x', float), ('y', float), ('z', float), ('r', float)])
    data = np.empty(n_particles, dtype=dt)

    ##Obtain the boundary and dxyz
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
    id1 = data['type'] == 1
    id2 = data['type'] == 2
    data['r'][id1] = R1
    data['r'][id2] = R2
    return data, boundary, dxyz

def AtomicStrain(file_data, ref_data):
    node = import_file(file_data)
    # Load the displacement modifier (respect to t = 0)
    mise_mod = AtomicStrainModifier(cutoff=3.8, eliminate_cell_deformation=True)
    mise_mod.reference = FileSource()
    mise_mod.reference.load(ref_data)
    node.modifiers.append(mise_mod)
    node.compute()
    mises_data = node.output.particle_properties['Shear Strain'].array
    return mises_data

def Maxdisplacement(file_data, ref_data):
    node = import_file(file_data)
    # Load the displacement modifier (respect to t = 0)
    disp_mod = CalculateDisplacementsModifier()
    disp_mod.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToReference
    disp_mod.reference.load(file_data)
    node.modifiers.append(disp_mod)

    mise_mod = AtomicStrainModifier(cutoff=3.8, eliminate_cell_deformation=True)
    mise_mod.reference = FileSource()
    mise_mod.reference.load(ref_data)
    node.modifiers.append(mise_mod)
    node.compute()
    disp_data = node.output.particle_properties['Displacement Magnitude'].array
    return disp_data

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
        axs_lo = extend_atoms[axis[i]] <= boundary[i][0] + dL
        atoms_lo = extend_atoms[axs_lo]
        atoms_lo[axis[i]] += dxyz[i]
        atoms_lo['id'] += atoms.shape
        extend_atoms= np.hstack((extend_atoms, atoms_lo))
        axs_hi = extend_atoms[axis[i]] >= boundary[i][1] - dL
        atoms_hi = extend_atoms[axs_hi]
        atoms_hi[axis[i]] -= dxyz[i]
        atoms_lo['id'] += atoms.shape
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

def SatomGenerateConfiguration(atoms, boundary, dxyz, mises_data, atom_outvol, lammps_file):
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
    mises_prop.marray[:] = mises_data
    outvol_prop = ParticleProperty.create_user('outvol', 'float', num_particles)
    outvol_prop.marray[:] = atom_outvol

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
    data.add(mises_prop); data.add(outvol_prop);
    data.add(cell)
    node = ovito.ObjectNode()
    node.source = data

    # Save LAMMPS data file
    export_file(node, lammps_file, 'lammps_dump', columns=
    ['Particle Identifier', 'Particle Type', 'Position.X', 'Position.Y', 'Position.Z', 'atomic strain', 'outvol'])

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

def inAtom(mesh_coord: List[float], atom, mesh_size_in_nm, schmitt_radius) -> bool:
    center_dx = mesh_coord[0] - atom['x']
    center_dy = mesh_coord[1] - atom['y']
    center_dz = mesh_coord[2] - atom['z']
    min_dx = min(abs(center_dx - mesh_size_in_nm * 0.5), abs(center_dx + mesh_size_in_nm * 0.5))
    min_dy = min(abs(center_dy - mesh_size_in_nm * 0.5), abs(center_dy + mesh_size_in_nm * 0.5))
    min_dz = min(abs(center_dz - mesh_size_in_nm * 0.5), abs(center_dz + mesh_size_in_nm * 0.5))
    min_distance = math.sqrt(min_dx * min_dx + min_dy * min_dy + min_dz * min_dz)
    return min_distance <= atom['r'] * schmitt_radius

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
        if group_id>0:
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

def CenterBox(lx, rx, ly, ry, lz, rz, boundary):
    center_ids = []
    lo_i = int((lx - boundary[0][0]) / mesh_size) - 1
    hi_i = int(math.ceil((rx - boundary[0][0]) / mesh_size)) + 1
    lo_j = int((ly - boundary[0][0]) / mesh_size) - 1
    hi_j = int(math.ceil((ry - boundary[0][0]) / mesh_size)) + 1
    lo_k = int((lz - boundary[0][0]) / mesh_size) - 1
    hi_k = int(math.ceil((rz - boundary[0][0]) / mesh_size)) + 1
    for i in range(lo_i, hi_i):
        for j in range(lo_j, hi_j):
            for k in range(lo_k, hi_k):
                mesh_id = getMeshId(i, j, k, num_meshes_by_axis)
                center_ids.append(mesh_id)
    return center_ids