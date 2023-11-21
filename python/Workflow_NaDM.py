import os, sys, shutil, glob, argparse, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--reference_frame',    default=0,     type=int)
ap.add_argument('--frame_offset',    type=int)
ap.add_argument('--relaxed_status',     default='0K',  type=str)
ap.add_argument('--temp',               default=2,     type=int)
ap.add_argument('--isample',            default=0,     type=int)
ap.add_argument('--nskip',    nargs=2,  default=[1,1], type=int)
ap.add_argument('--nstart',             default=0,     type=int)
ap.add_argument('--erate',              default=1e7,   type=float)
ap.add_argument('--ref_erate',                         type=float)
ap.add_argument('--qrate',              default=2.8e7, type=float)
ap.add_argument('--unloading',          type=float)
ap.add_argument('--ref_loading',        action='store_true')
ap.add_argument('--unloading_to',       default=0,     type=int)
ap.add_argument('--unloading2',         type=float)
ap.add_argument('--unloading_to2',      default=0,     type=int)
ap.add_argument('--max_tensile_strain', default=4.0,   type=float)
ap.add_argument('--plot_strain',nargs=2,default=[0,4], type=float)
ap.add_argument('--summarize',          action='store_true')
parsed_info_args = ap.parse_args()

print(parsed_info_args)

erate = parsed_info_args.erate
qr = parsed_info_args.qrate
isample = parsed_info_args.isample
temp = parsed_info_args.temp
max_tensile_strain = parsed_info_args.max_tensile_strain
plot_strain = parsed_info_args.plot_strain

from workflow_utils import MetallicGlassWorkflow
from log import log as lammps_log

print('** Information for CuZr metallic glass')

mgw_tensile_test = MetallicGlassWorkflow(MLmat_dir   = os.environ['MLMAT_DIR'],
                                         natom       = 5000,                      # Atom number
                                         x1          = 0.645,                     # Composition: Cu64.5Zr35.5
                                         qrate       = qr,                        # Quenching rate
                                         temperature = temp,                      # Tensile test temperature
                                         erate       = erate,                     # Tensile test strain rate
                                         max_tensile_strain = max_tensile_strain,
                                         plot_strain_range = plot_strain,
                                         poisson_ratio=0.4,
                                         sample_id   = isample,
                                         potfilename = 'CuZr.eam.fs',
                                         use_scratch = True)

mgw_tensile_test.print_directories()

if parsed_info_args.ref_erate is not None:
    mgw_tensile_ref = MetallicGlassWorkflow(MLmat_dir   = os.environ['MLMAT_DIR'],
                                            natom       = 5000,                      # Atom number
                                            x1          = 0.645,                     # Composition: Cu64.5Zr35.5
                                            qrate       = qr,                        # Quenching rate
                                            temperature = temp,                      # Tensile test temperature
                                            erate       = parsed_info_args.ref_erate,# Tensile test strain rate
                                            max_tensile_strain = max_tensile_strain,
                                            plot_strain_range = plot_strain,
                                            poisson_ratio=0.4,
                                            sample_id   = isample,
                                            potfilename = 'CuZr.eam.fs',
                                            use_scratch = True)
    mgw_tensile_ref.print_directories()
    _, reference = mgw_tensile_ref.tensile_test_directories(relaxed=parsed_info_args.relaxed_status)
else:
    reference = None

from ovito.io import import_file
from ovito.pipeline import FileSource
from ovito.modifiers import CalculateDisplacementsModifier

######## Extract states based on max displacement #######
time_init = time.time()
_, dumpfile = mgw_tensile_test.tensile_test_directories(relaxed=parsed_info_args.relaxed_status, unloading=parsed_info_args.unloading, unloading_to=parsed_info_args.unloading_to, unloading2=parsed_info_args.unloading2, unloading_to2=parsed_info_args.unloading_to2)
if parsed_info_args.ref_loading:
    _, reference = mgw_tensile_test.tensile_test_directories(relaxed=parsed_info_args.relaxed_status)

# Load dump file
if not os.path.exists(dumpfile): dumpfile = dumpfile + '.gz'
pipeline = import_file(dumpfile)
print('Loaded dump file %s: %.2fs'%(dumpfile, time.time()-time_init))
mod = CalculateDisplacementsModifier()
if reference is not None:
    tic = time.time()
    if not os.path.exists(reference): reference = reference + '.gz';
    mod.reference = FileSource()
    mod.reference.load(reference)
    print('Loaded reference file %s: %.2fs'%(reference, time.time()-tic))
pipeline.modifiers.append(mod)

def calculate_displacement(dumpfile, frame_offset=-1, reference_frame=0, use_frame_offset=False,
                           affine_mapping=CalculateDisplacementsModifier.AffineMapping.ToReference,
                           reference=None, minimum_image_convention=True, saved_dir='maxdisp',
                           overwrite=False, verbose=False, return_timestep=False):
    use_frame_offset_str = 'offset_%d'%frame_offset if use_frame_offset else 'reference_%d'%reference_frame
    saved_dir  = os.path.join(os.path.dirname(dumpfile), saved_dir)
    os.makedirs(saved_dir, exist_ok=True)
    saved_file = os.path.join(saved_dir, 'disp_%s.npz')%use_frame_offset_str
    if not overwrite and os.path.exists(saved_file):
        rawdata = np.load(saved_file)
        disp_data, disp_max_id, timestep = rawdata['disp_data'], rawdata['disp_max_id'], rawdata['timestep']
    else:
        # Setup CalculateDisplacementsModifier
        mod.affine_mapping   = affine_mapping
        mod.frame_offset     = frame_offset
        mod.reference_frame  = reference_frame
        mod.use_frame_offset = use_frame_offset

        # Define the range of frames to explore
        if use_frame_offset:
            lower_limit = max(0, -frame_offset)
            upper_limit = min(pipeline.source.num_frames, pipeline.source.num_frames-frame_offset)
        else:
            lower_limit, upper_limit = 0, pipeline.source.num_frames
        frame_range = range(lower_limit, upper_limit, parsed_info_args.nskip[0])

        # Calculate Displacements for every frame
        disp_data   = np.zeros(pipeline.source.num_frames)
        disp_max_id = np.zeros(pipeline.source.num_frames, dtype=int)
        timestep    = np.zeros(pipeline.source.num_frames, dtype=int)
        for frame in frame_range:
            if verbose: print('processing frame%8d...'%frame, end='\r')
            data = pipeline.compute(frame)
            disp = data.particles['Displacement'][...]
            disp_corrected = disp - np.mean(disp, axis=0)
            disp_mag = np.linalg.norm(disp_corrected, axis=1)
            disp_argmax = disp_mag.argmax()
            disp_data[frame]   = disp_mag[disp_argmax]
            disp_max_id[frame] = data.particles['Particle Identifier'][...][disp_argmax]
            timestep[frame] = data.attributes['Timestep']

        # Return data within calculated frame range
        disp_data, disp_max_id, timestep = (disp_data[frame_range], disp_max_id[frame_range], timestep[frame_range])
        np.savez_compressed(saved_file, disp_data=disp_data, disp_max_id=disp_max_id, timestep=timestep)
        if verbose: print('saved file %s'%saved_file)

    if return_timestep:
        return disp_data, disp_max_id, timestep
    else:
        return disp_data, disp_max_id

def calculate_NaDM(dumpfile, saved_file='NaDM.npz', saved_dir='maxdisp', overwrite=False, verbose=False, return_timestep=False):
    saved_file = os.path.join(os.path.dirname(dumpfile), saved_file)
    if not overwrite and os.path.exists(saved_file):
        if verbose: print('loaded saved file: %s'%saved_file)
        rawdata = np.load(saved_file)
        NaDM, NaDM_id = rawdata['NaDM'], rawdata['NaDM_id']
    else:
        nframetotal = pipeline.source.num_frames if ((mod.reference is None) or (parsed_info_args.unloading is not None)) else mod.reference.num_frames
        if verbose:
            print('nframetotal = %d'%nframetotal)
            print('pipeline.source.num_frames = %d'%pipeline.source.num_frames)
            # print('mod.reference.num_frames = %d'%mod.reference.num_frames)
        NaDM_ref_range = list(range(parsed_info_args.nstart, nframetotal, parsed_info_args.nskip[1]))
        NaDM_frame_range = list(range(0, pipeline.source.num_frames, parsed_info_args.nskip[0]))
        NaDM_size = (len(NaDM_ref_range), len(NaDM_frame_range))
        print(NaDM_size)
        if len(NaDM_ref_range) * len(NaDM_frame_range) > 4e8:
            NaDM = np.memmap(saved_file, dtype='float32', mode='w+', shape=NaDM_size)
            NaDM_id = np.memmap(saved_file+'_id.npz', mode='w+', shape=NaDM_size)
        else:
            NaDM = np.zeros(NaDM_size)
            NaDM_id = np.zeros(NaDM_size, dtype=int)
        for i in range(len(NaDM_ref_range)):
            reference_frame = NaDM_ref_range[i]
            if verbose: print('reference frame %8d...'%reference_frame)
            time_init = time.time()
            disp_data, disp_max_id = calculate_displacement(dumpfile, reference_frame=reference_frame, saved_dir=saved_dir,
                                                            overwrite=overwrite, verbose=verbose)
            print('Time Spend on calculate_displacement: %.2fs'%(time.time() - time_init))
            NaDM[i, :] = disp_data
            NaDM_id[i, :] = disp_max_id
        if parsed_info_args.nstart == 0:
            if len(NaDM_ref_range) * len(NaDM_frame_range) > 4e8:
                del NaDM
                del NaDM_id
                return None
            else:
                np.savez_compressed(saved_file, NaDM=NaDM, NaDM_id=NaDM_id)
    total_steps = int((NaDM.shape[0] - 1)*plot_strain[1]/max_tensile_strain) + 1
    return NaDM[:total_steps, :total_steps], NaDM_id[:total_steps, :total_steps]

if parsed_info_args.ref_erate is not None:
    saved_file = 'NaDM_%.0e.npz'%parsed_info_args.ref_erate
    saved_dir  = 'maxdisp_%.0e'%parsed_info_args.ref_erate
elif parsed_info_args.ref_loading:
    saved_file = 'NaDM_refload.npz'
    saved_dir  = 'maxdisp_refload'
elif parsed_info_args.frame_offset is not None:
    saved_file = 'NaDM_frame_offset_%d'%parsed_info_args.frame_offset
    saved_dir  = 'maxdisp_%d'%parsed_info_args.frame_offset
else:
    saved_file = 'NaDM.npz'
    saved_dir  = 'maxdisp'

if parsed_info_args.summarize:
    NaDM, NaDM_id = calculate_NaDM(dumpfile, saved_file=saved_file, saved_dir=saved_dir, verbose=True)
elif parsed_info_args.frame_offset is not None:
    disp_data, disp_max_id = calculate_displacement(dumpfile, use_frame_offset=True, saved_dir=saved_dir, frame_offset=parsed_info_args.frame_offset, verbose=True)
else:
    disp_data, disp_max_id = calculate_displacement(dumpfile, reference_frame=parsed_info_args.reference_frame, saved_dir=saved_dir, verbose=True)
print('Total Time Spend: %.2fs'%(time.time() - time_init))
