# Workflow for MetallicGlass simulation

import os, subprocess, sys, argparse, shutil, gzip, time
import log
import numpy as np
from workflow_utils import MetallicGlassWorkflow, test_env_vars

ap = argparse.ArgumentParser()
ap.add_argument('--temp_list',   nargs='*', default=[50, ], type=int, help='A list of temperature for calculation')
ap.add_argument('--overwrite',   action='store_true')
ap.add_argument('--use_refdump', action='store_true')
ap.add_argument('--no_submit',   action='store_true')
ap.add_argument('--use_scratch', action='store_true')
ap.add_argument('--use_nvt',     action='store_true')
ap.add_argument('--unloading',   type=float, help='start the unloading simulation from given strain, must smaller than max_tensile_strain. If not given, no unloading is performed')
ap.add_argument('--run_0K_simulation', action='store_true')
ap.add_argument('--run_0K_unloading',    type=int)
ap.add_argument('--unloading2',  type=float, help='start the 0K simulation of a given strain from the unloaded simulation')
ap.add_argument('--run_0K_unloading2',   type=int)
ap.add_argument('--potfilename',         default='CuZr.eam.fs', type=str)
ap.add_argument('--elements', nargs=2,   default=['Cu', 'Zr'],  type=str)
ap.add_argument('--partition_mc2',       default='cpu',         type=str)
ap.add_argument('--natom',               default=5000,  type=int)
ap.add_argument('--x1',                  default=0.645, type=float)
ap.add_argument('--qrate',               default=1e10,  type=float)
ap.add_argument('--erate',               default=1e7,   type=float)
ap.add_argument('--sample_id',           default=0,     type=int)
ap.add_argument('--poisson_ratio',       default=0.4,   type=float)
ap.add_argument('--pressure_anneal',     default=0.0,   type=float) # in units of bar
ap.add_argument('--timestep_anneal',     default=2.5e-3,type=float)
ap.add_argument('--timestep_tensile',    default=1.0e-3,type=float)
ap.add_argument('--dumpfreq_tensile',    default=400   ,type=int)
ap.add_argument('--dumpfreq_unloading',                 type=int)
ap.add_argument('--equilibrate_time',    default=1000,  type=int)
ap.add_argument('--max_tensile_strain',  default=4,     type=float, help='maximum strain value for tensile test, default is 4')
ap.add_argument('--relax_Npart',         default=1,     type=int, help='separate energy minimization job into N parts, default is 1')
ap.add_argument('--max_analysis_strain', type=float, help='maximum strain value for analysis, default is the same as max_tensile_strain')
ap.add_argument('--analysis_nslice',     default=10,    type=int, help='number of slices we use to save disp and phop data, default is 1')
parsed_info_args = ap.parse_args()

if parsed_info_args.max_analysis_strain is None: parsed_info_args.max_analysis_strain = parsed_info_args.max_tensile_strain
if parsed_info_args.dumpfreq_unloading  is None: parsed_info_args.dumpfreq_unloading  = parsed_info_args.dumpfreq_tensile
cluster_info = {}; do_submit = use_scratch = False;   # By default, for testing on local machines
npound = 60                             # number of pound signs for printing header
convert_s2ps = 1.0e12                   # unit conversion
use_gpu = False

hostname = os.getenv('HOSTNAME')
if hostname:
    if hostname[:3] == 'mc2':           # for mc2 cluster
        cluster_info = {'partition': parsed_info_args.partition_mc2, 'nodes': 1, 'tasks-per-node': 24} # SLURM system
        if parsed_info_args.partition_mc2[:3] == 'gpu':
            cluster_info['gres'] = 'gpu:1'
            cluster_info['exclude'] = 'gpu-200-[3-4]' #'gpu-200-3' # node 3 not working
            # cluster_info['exclude'] = 'gpu-200-3' # node 3 not working
            cluster_info.pop('tasks-per-node', None)
            use_gpu = True
        do_submit = use_scratch = True
    elif hostname[:3] == 'mc3':           # for mc3 cluster
        cluster_info = {'partition': parsed_info_args.partition_mc2, 'nodes': 1, 'tasks-per-node': 32} # SLURM system
        if parsed_info_args.partition_mc2[:3] == 'gpu':
            # cluster_info['gres'] = 'gpu:1'
            cluster_info.pop('tasks-per-node', 8)
            use_gpu = True
        do_submit = use_scratch = True
    elif hostname[:2] == 'sh':         # for sherlock cluster
        cluster_info = {'partition': 'mc', 'nodes': 1, 'tasks-per-node': 20, 'time': '120:00:00'} # SLURM system
        do_submit = use_scratch = True

temp_list =           parsed_info_args.temp_list
overwrite =           parsed_info_args.overwrite
use_refdump_as_init = parsed_info_args.use_refdump         # use the init_config_2.8e7.lammps as initial configuration
if parsed_info_args.max_analysis_strain > parsed_info_args.max_tensile_strain:
    raise ValueError('max_analysis_strain larger than max_tensile_strain')

if parsed_info_args.no_submit:     do_submit = False
if parsed_info_args.sample_id < 0: parsed_info_args.sample_id = None
if parsed_info_args.use_scratch:   use_scratch = True

# ## 1. Set up paths and prepare folders

# Check the necessary environment variables
if not test_env_vars(['LAMMPS_DIR', 'LAMMPS_SYS', 'LAMMPS_BIN', 'MLMAT_DIR']):
    print('MAKE SURE following environment variables are set before running the script:')
    print('  export LAMMPS_DIR=$HOME/Codes/lammps  # Where you want to install lammps')
    print('  export LAMMPS_SYS=icc_mpich           # If on sherlock and mc2, use icc_mpich')
    print(' #export LAMMPS_SYS=serial              # If on local machine without SLURM, use serial')
    print('  export LAMMPS_BIN=$LAMMPS_DIR/src/lmp_$LAMMPS_SYS  # path to the lammps binary')
    print('  export MLMAT_DIR=${path to your MLmat repository}')
    print('#'*npound)

# Install LAMMPS
lammpsdir = os.environ['LAMMPS_DIR']
lammps_sys= os.environ['LAMMPS_SYS']
lammps_bin= os.environ['LAMMPS_BIN']
if not os.path.exists(lammpsdir):
    print('#'*npound); print('Download LAMMPS from github repository'); print('#'*npound);
    directory, root = os.path.split(lammpsdir)
    os.makedirs(directory, exist_ok=True)
    subprocess.run('git clone -b stable https://github.com/lammps/lammps.git'.split().append(lammpsdir))
if not os.path.exists(lammps_bin):
    print('#'*npound); print('Build LAMMPS using environment variable settings'); print('#'*npound);
    os.chdir(os.path.join(lammpsdir, 'src'))
    subprocess.run(['make', 'yes-manybody'])     # install MANYBODY package to use EAM potential
    subprocess.run(['make', lammps_sys])

for temp in temp_list:

    print('#'*npound); print("1. Set up paths and prepare folders"); print('#'*npound)

    MLmat_dir = os.getenv('MLMAT_DIR'); os.chdir(MLmat_dir);
    print('MLmat_dir =', MLmat_dir)
    mgw = MetallicGlassWorkflow(MLmat_dir          = MLmat_dir,
                                natom              = parsed_info_args.natom,
                                x1                 = parsed_info_args.x1,
                                qrate              = parsed_info_args.qrate,              # K/s
                                temperature        = temp,         # K
                                erate              = parsed_info_args.erate,              #  /s
                                sample_id          = parsed_info_args.sample_id,
                                poisson_ratio      = parsed_info_args.poisson_ratio,
                                max_tensile_strain = parsed_info_args.max_tensile_strain, # percent
                                potfilename        = parsed_info_args.potfilename,
                                unloading          = parsed_info_args.unloading,
                                use_scratch        = use_scratch,                         # use $SCRATCH to save simulation data
                                use_gpu            = use_gpu)
    mgw.print_sample_properties()
    mgw.print_directories()
    sample_qrate = 1.0e10

    if mgw.qrate == 1.0e10:
        sample_prep_id = 4.5
    elif mgw.qrate == 5.0e8:
        sample_prep_id = 5.5
    elif mgw.qrate == 6.3e7:
        sample_prep_id = 6.5
    elif mgw.qrate == 2.8e7:
        sample_prep_id = 7.5
        sample_prep_id = 7  # For initial configurations with lower energy
    elif mgw.qrate == 1.0e7:
        sample_prep_id = 8.5
    elif mgw.qrate == 8.0e6:
        sample_prep_id = 9.5
    elif mgw.qrate == 5.0e6:
        sample_prep_id = 10.5
        sample_prep_id = 10 # For initial configurations with lower energy
    else:
        sample_qrate = mgw.qrate
        sample_prep_id = 4.5
        sample_prep_id = 4  # For initial configurations with lower energy

    mgw_sample_prep = MetallicGlassWorkflow(MLmat_dir          = MLmat_dir,
                                            natom              = parsed_info_args.natom,
                                            x1                 = parsed_info_args.x1,
                                            qrate              = sample_qrate,        # K/s
                                            sample_id          = parsed_info_args.sample_id,
                                            use_scratch        = use_scratch)   # use $SCRATCH to save simulation data
    if use_refdump_as_init:
        init_config_file = os.path.join(mgw.lammps_templates, 'init_config_Cu%.1fZr%.1f_%.1e.lammps'%(mgw_sample_prep.x1*100, 100-mgw_sample_prep.x1*100, mgw.qrate))
    else:
        init_config_file = os.path.join(mgw_sample_prep.datadir, 'dump.%s'%sample_prep_id, "restart.%s.2"%sample_prep_id)
    if not os.path.exists(init_config_file):
        print('** Checking', init_config_file)
        print('Sample preparation workflow is not done!')
        exit(0)

    # 2. Heat up the sample to target temperature at 1e10K/s, anneal at the target temperature for 1ns

    print('#'*npound); 
    print("2. Heat up the sample to target temperature at 1e10K/s, anneal at the target temperature for 1ns"); 
    print('#'*npound);

    stepid = 2

    timestep = parsed_info_args.timestep_anneal # ps
    if parsed_info_args.use_nvt:
        T0 = T1 = T2 = mgw.temperature
    else:
        T0, T1, T2 = 2, mgw.temperature, mgw.temperature
        T0, T1, T2 = mgw.temperature, mgw.temperature, mgw.temperature
        if T1 < T0: T0 = T1
    period_1 = (T1 - T0) / mgw_sample_prep.qrate *convert_s2ps  # ps
    period_2 = 1000 # ps
    infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
    dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
    os.makedirs(dumpdir, exist_ok=True)

    lammps_vars = {'T0':                                T0,
                   'T1':                                T1,
                   'T2':                                T2,
                   'timestep':                    timestep, # default: 2.5e-3
                   'Tdamp':                   100*timestep, # default: 0.25
                   'Pdamp':                  1000*timestep, # default: 2.5
                   'total_steps_1': int(period_1/timestep),
                   'total_steps_2': int(period_2/timestep),
                   'thermo_step':                      100,
                   # parameters for energy minimization before heating up
                   'do_minimize':                        1,
                   'etol_minimize':                   1e-3,
                   'ftol_minimize':                   1e-4,
                   'maxiter_minimize':                1000,
                   'maxeval_minimize':                1000,
                   # parameters for setting velocity 
                   'set_velocity':                       1,
                   'rand_seed':     np.random.randint(100000000),
                   # path for saving dump and restart files
                   'dump_freq':     int(period_2/timestep/5),
                   'dumpdir':                      dumpdir,
                   'restart_freq':                       0,
                   'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                   'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

    if parsed_info_args.use_nvt:
        lammps_vars['Pdamp'] = -1
    if use_refdump_as_init:
        init_file_key = 'init_config'
    else:
        init_file_key = 'restart_file'
    if mgw.qrate == 5e6:
        init_file_key = 'init_config'
        init_config_file = os.path.join(mgw_sample_prep.datadir, 'dump.%s'%sample_prep_id, "lmp.%s.2"%sample_prep_id)
    potential =   {init_file_key: init_config_file,
                   'pair_style': 'eam/fs',
                   'potfile': mgw.potfile,
                   'elements': parsed_info_args.elements}
    cluster_info["job-name"] = "step-%s"%stepid

    replace_str = {'iso 0.0 0.0': 'aniso %f %f'%(parsed_info_args.pressure_anneal, parsed_info_args.pressure_anneal)}
    mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, 
                              replace_str = replace_str, overwrite=overwrite,
                              paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                              submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                checkpoint_success=lammps_vars['restart_file_2'], 
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): continue
    restart_for_nextstep = os.path.join(dumpdir, "restart.%s.2"%stepid)
    log_file_name = os.path.join(mgw.lammps_scripts, 'log.%s'%stepid)

    # Calculate the average simulation box size for the NPT annealing

    if os.path.exists(log_file_name):
        log_npt = log.log(log_file_name)
        step, Temp, press, lx, ly, lz = log_npt.get('Step', 'Temp', 'Press', 'Lx', 'Ly', 'Lz')
        equil_steps = int(0.5*lammps_vars['total_steps_2']/lammps_vars['thermo_step'])
        equil_lx = np.mean(lx[-equil_steps:])
        equil_ly = np.mean(ly[-equil_steps:])
        equil_lz = np.mean(lz[-equil_steps:])
        print('**   equil_lx = %s **'%equil_lx)
        print('**   equil_ly = %s **'%equil_ly)
        print('**   equil_lz = %s **'%equil_lz)
    else:
        print('** log file: %s does not exist! **'%log_file_name)
        continue
        
    # 3. NVT annealing using average simulation box size for 1ns and tensile test with Poisson's ratio = 0.4

    print('#'*npound)
    print("3. NVT annealing using average simulation box size for 1ns and tensile test with Poisson's ratio = 0.4")
    print('#'*npound)

    stepid = 3

    timestep = parsed_info_args.timestep_tensile; 
    dump_freq = parsed_info_args.dumpfreq_tensile;  # ps, 0.4ps
    # timestep = 2.5e-3; dump_freq = 100;  # ps, 0.25ps
    Ttarget  = mgw.temperature     # K
    period_1 = parsed_info_args.equilibrate_time # ps, by default 1000ps
    period_2 = mgw.max_tensile_strain/100/mgw.erate *convert_s2ps # ps
    infile_template = os.path.join(mgw.lammps_templates, 'in.tensile.mg')
    dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
    os.makedirs(dumpdir, exist_ok=True)

    lammps_vars = {'Ttarget':                      Ttarget,
                   'timestep':                    timestep, # default: 2.5e-3
                   'Tdamp':                   100*timestep, # default: 0.25
                   # Change box size to equilibrated box size at beginning
                   'change_box_init':                    1,
                   'equil_lx':                    equil_lx,
                   'equil_ly':                    equil_ly,
                   'equil_lz':                    equil_lz,
                   # Set up tensile test parameters
                   'N_deform':                           1, # change box size every step
                   'ex_trate':                mgw.ex_trate,
                   'ey_trate':                mgw.ey_trate,
                   'ez_trate':                mgw.ez_trate,
                   'total_steps_1': int(period_1/timestep), # equilibrium steps
                   'total_steps_2': int(period_2/timestep), # tensile test steps
                   'thermo_step':                      100,
                   # path for saving dump and restart files
                   'dump_freq':                  dump_freq,
                   'dumpdir':                      dumpdir,
                   'restart_freq':                       0,
                   'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                   'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

    potential =   {'restart_file': restart_for_nextstep,
                   'pair_style': 'eam/fs',
                   'potfile': mgw.potfile,
                   'elements': parsed_info_args.elements}
    cluster_info["job-name"] = "step-%s"%stepid
    # Restrict job submission to only node gpu-200-3 and gpu-200-4 (Yifan 2020.11.24)
    gpu_id = 3 if mgw.qrate == 2.8e7 else 4
    cluster_info['nodelist'] = ''# 'gpu-200-%d'%gpu_id

    if parsed_info_args.equilibrate_time == 0:
        replace_str = {'write_restart   ${restart_file_1}': 
                       'velocity        all create %d %d dist gaussian'%(Ttarget, np.random.randint(10000))}
    else:
        replace_str = {}
    mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite,
                              replace_str=replace_str,
                              paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                              submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)
    # Remove node restriction (Yifan 2020.11.24)
    cluster_info.pop('nodelist', None)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                checkpoint_success=lammps_vars['restart_file_2'], 
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): continue
    restart_for_nextstep = os.path.join(dumpdir, "restart.%s.1"%stepid)
    restart_for_unloading= os.path.join(dumpdir, "restart.%s.2"%stepid)
    nfile_total = lammps_vars['total_steps_2']//lammps_vars['dump_freq']

    # 3.0. 0K tensile test (molecular statics) using the box size history as Step 3

    if parsed_info_args.run_0K_simulation:
        
        print('#'*npound)
        print("3.0. 0K tensile test (molecular statics) using the box size history from Step 3")
        print('#'*npound)
        
        stepid = 3.0

        if parsed_info_args.partition_mc2[:3] == 'gpu':
            infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
        else:
            infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg')
        dumpdir   = os.path.join(mgw.datadir, 'dump.3')
        logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
        relaxdir0K= os.path.join(mgw.datadir, 'dump.%s'%stepid)
        os.makedirs(relaxdir0K, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        relaxed_prev_restart = os.path.join(relaxdir0K, 'relaxed.prev.restart')
        if not os.path.exists(restart_for_nextstep):
            print('** initial configuration %s in step 3 does not exist!')
            continue
        else:
            shutil.copy(restart_for_nextstep, relaxed_prev_restart)

        potential =   {'restart_file':       '${restart_file}',   # beginning of the tensile test
                       'pair_style': 'eam/fs',
                       'potfile': mgw.potfile,
                       'elements': parsed_info_args.elements}

        lammps_vars = {'istart':                             0,
                       'iend':                     nfile_total,
                       'unloading':                          0,
                       # energy minimization parameters
                       'etol_minimize':                    0.0,
                       'ftol_minimize':                1.0e-11,
                       'maxiter_minimize':           100000000,
                       'maxeval_minimize':         10000000000,
                       'write_restart':                      1,
                       'restart_file':    relaxed_prev_restart,
                       'dump_freq':                  dump_freq,
                       'logdir':                        logdir,
                       'inputdir':                     dumpdir,
                       'relaxdir':                  relaxdir0K}

        cluster_info["job-name"] = "step-%s"%(stepid)

        mgw.generate_lammps_input(infile_template, infile="in.%s"%(stepid), log_file_name=None, overwrite=overwrite,
                                  replace_str = {'in.relax.mg': "in.%s"%(stepid), 
                                                 '${istart}':   '%s'%(0), 
                                                 '${iend}':     '%s'%(nfile_total),
                                                 'replace yes': 'replace no'}, # use the box size from finite temperature tensile test
                                  paramsfile="in.params.%s"%(stepid), variables=lammps_vars, potential=potential,
                                  submitfile="submit.sh.%s"%(stepid), queue_system='slurm', cluster_info=cluster_info)

        if do_submit:
            if parsed_info_args.partition_mc2[:3] == 'gpu':
                checkpoint_success = os.path.join(relaxdir0K, 'relaxed.dump')
            else:
                checkpoint_success = os.path.join(relaxdir0K, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
            checkpoint_success = os.path.join(logdir, 'log.finished')
            status0K = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                      checkpoint_success=checkpoint_success,
                                      checkpoint_continue=relaxed_prev_restart,
                                      queue_system='slurm', cluster_info=cluster_info)
        else:
            status0K = {'status': 'no_submit'}

        if not mgw.print_instructions(status0K, submitfile="submit.sh.%s"%stepid): continue

    # 4. Energy minimization dump files generated from Step 3 tensile test

    print('#'*npound)
    print("4. Energy minimization dump files generated from Step 3 tensile test")
    print('#'*npound)

    stepid = 4

    if parsed_info_args.partition_mc2[:3] == 'gpu':
        infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
    else:
        infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg')
    dumpdir   = os.path.join(mgw.datadir, 'dump.3')
    logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
    relaxdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
    os.makedirs(relaxdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    potential =   {'restart_file': restart_for_nextstep,
                   'pair_style': 'eam/fs',
                   'potfile': mgw.potfile,
                   'elements': parsed_info_args.elements}
    
    # Split relax jobs into Npart parts
    Npart = parsed_info_args.relax_Npart
    # nfile_total = lammps_vars['total_steps_2']//lammps_vars['dump_freq']
    nfile = nfile_total // Npart
    all_parts_finished = True
    
    for ipart in range(Npart):
        lammps_vars = {'istart':                   ipart*nfile,
                       'iend':                 (ipart+1)*nfile,
                       'unloading':                          0,
                       # energy minimization parameters
                       'etol_minimize':                    0.0,
                       'ftol_minimize':                1.0e-11,
                       'maxiter_minimize':           100000000,
                       'maxeval_minimize':         10000000000,
                       'write_restart':                      0,
                       'dump_freq':                  dump_freq,
                       'logdir':                        logdir,
                       'inputdir':                     dumpdir,
                       'relaxdir':                    relaxdir}

        cluster_info["job-name"] = "step-%s.%s"%(stepid, ipart)

        mgw.generate_lammps_input(infile_template, infile="in.%s.%s"%(stepid, ipart), log_file_name=None, overwrite=overwrite,
                                  replace_str = {'in.relax.mg': "in.%s.%s"%(stepid, ipart), '${istart}': '%s'%(ipart*nfile), '${iend}': '%s'%((ipart+1)*nfile)},
                                  paramsfile="in.params.%s.%s"%(stepid, ipart), variables=lammps_vars, potential=potential,
                                  submitfile="submit.sh.%s.%s"%(stepid, ipart), queue_system='slurm', cluster_info=cluster_info)

        if do_submit:
            if parsed_info_args.partition_mc2[:3] == 'gpu':
                checkpoint_success = os.path.join(relaxdir, 'relaxed.dump')
            else:
                checkpoint_success = os.path.join(relaxdir, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
            checkpoint_success = os.path.join(logdir, 'log.finished')
            status = mgw.submit_job(submitfile="submit.sh.%s.%s"%(stepid, ipart), 
                                    checkpoint_success=checkpoint_success, 
                                    queue_system='slurm', cluster_info=cluster_info)
        else:
            status = {'status': 'no_submit'}

        if not mgw.print_instructions(status, submitfile="submit.sh.%s.%s"%(stepid, ipart)):
            all_parts_finished = False
            continue

    if not all_parts_finished: continue

    if parsed_info_args.unloading is not None:

        if parsed_info_args.unloading > parsed_info_args.max_tensile_strain:
            print('Unloading strain is larger than max_tensile_strain, skip...')
            continue

        if nfile_total == 0:
            unloading_strain_resolution = 1
        else:
            unloading_strain_resolution = np.ceil(np.log10(nfile_total/parsed_info_args.max_tensile_strain)).astype(int)
        # print(unloading_strain_resolution)
        resolution_str = '.%df'%unloading_strain_resolution
        load_dump_i    = np.round(nfile_total*parsed_info_args.unloading/parsed_info_args.max_tensile_strain).astype(int)
        load_dump_idx  = load_dump_i*parsed_info_args.dumpfreq_tensile
        load_dump_init = os.path.join(mgw.datadir, 'dump.3', 'dump.%d.gz')%load_dump_idx
        unloading_str = ('u_%'+resolution_str+'-0')%parsed_info_args.unloading

        # 4.0u. 0K unloading test (molecular statics) after finite temperature loading

        if parsed_info_args.run_0K_unloading is not None:
            print('#'*npound)
            print("4.0u. 0K unloading test (molecular statics) using the box size history from Step 3")
            print('#'*npound)

            unloading_str = ('u_%'+resolution_str+'-%d')%(parsed_info_args.unloading, parsed_info_args.run_0K_unloading)
            stepid = '4.0' + unloading_str

            finite_temp_loading = os.path.join(mgw.datadir, 'dump.3')
            nfile_unload = round(mgw.unloading/100/mgw.erate *convert_s2ps/parsed_info_args.timestep_tensile)//parsed_info_args.dumpfreq_tensile
            loadingdir = os.path.join(mgw.datadir, 'dump.4')
            loadinglogdir = os.path.join(mgw.datadir, 'log.4')
            if parsed_info_args.partition_mc2[:3] == 'gpu':
                infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
                loading_logfile = os.path.join(loadinglogdir, 'log.relax')
                unloading_init_file = os.path.join(loadingdir, 'relaxed.dump')
                loadinglog = log.log(loading_logfile, verbose=False)
                unloading_init_timestep = loadinglog.get('Step')[load_dump_i*2+1]
            else:
                infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg')
                unloading_init_file  = os.path.join(loadingdir, 'relaxed.%d.dump.gz')%load_dump_idx
                with gzip.open(unloading_init_file, 'rt') as f:
                    line = f.readline()
                    if not line.split() == ['ITEM:', 'TIMESTEP']: raise TypeError('dump file not understandable')
                    unloading_init_file = int(f.readline().split()[0])
            dumpdir   = os.path.join(mgw.datadir, 'dump.3')

            logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
            relaxdir0K= os.path.join(mgw.datadir, 'dump.%s'%stepid)
            os.makedirs(relaxdir0K, exist_ok=True)
            os.makedirs(logdir, exist_ok=True)
            relaxed_prev_restart = os.path.join(relaxdir0K, 'relaxed.prev.restart')
            if not os.path.exists(restart_for_unloading):
                print('** final configuration %s in step 3 does not exist!')
                continue
            else:
                shutil.copy(restart_for_unloading, relaxed_prev_restart)

            potential =   {'restart_file':       '${restart_file}',   # beginning of the tensile test
                           'pair_style': 'eam/fs',
                           'potfile': mgw.potfile,
                           'elements': parsed_info_args.elements}

            lammps_vars = {'istart':                             0,
                           'iend':                    nfile_unload,
                           'unloading':                          1,
                           'loadingdir':       finite_temp_loading,
                           # energy minimization parameters
                           'etol_minimize':                    0.0,
                           'ftol_minimize':                1.0e-11,
                           'maxiter_minimize':           100000000,
                           'maxeval_minimize':         10000000000,
                           'write_restart':                      1,
                           'restart_file':    relaxed_prev_restart,
                           'dump_freq':                  dump_freq,
                           'logdir':                        logdir,
                           'inputdir':                     dumpdir,
                           'relaxdir':                  relaxdir0K}

            cluster_info["job-name"] = "step-%s"%(stepid)

            unload_end  = round(nfile_total//parsed_info_args.max_tensile_strain*parsed_info_args.run_0K_unloading)
            replace_str = {'in.relax.mg': "in.%s"%(stepid), 
                           '${istart} ${iend}':   '%s %s'%(0, nfile_unload - unload_end), 
                           'read_dump       ${inputdir}/dump.${filenum}.gz ${filenum} x y z box yes replace yes': '# read_dump     ${inputdir}/dump.${filenum}.gz ${filenum} x y z box yes replace yes', # not using filenum
                           'reset_timestep 1': 'read_dump       %s %d x y z box yes replace yes'%(unloading_init_file, unloading_init_timestep), # load corresponding dump file before unloading starts
                           "print 'final step'": 'reset_timestep 1', 
                           '${i} == ${iend}': '${i} == %s'%(nfile_unload - unload_end)}
            if parsed_info_args.run_0K_unloading > parsed_info_args.unloading: # 0K loading to run_0K_unloading
                replace_str['${istart} ${iend}'] = '%s %s'%(0, unload_end - nfile_unload)
                replace_str['${iend}-${i}'] = '${iend}+${i}'
                replace_str['${i} == ${iend}'] = '${i} == %s'%(unload_end - nfile_unload)
                replace_str.pop("print 'final step'", None)
            mgw.generate_lammps_input(infile_template, infile="in.%s"%(stepid), log_file_name=None, overwrite=overwrite,
                                      replace_str = replace_str,
                                      paramsfile="in.params.%s"%(stepid), variables=lammps_vars, potential=potential,
                                      submitfile="submit.sh.%s"%(stepid), queue_system='slurm', cluster_info=cluster_info)

            if do_submit:
                if parsed_info_args.partition_mc2[:3] == 'gpu':
                    checkpoint_success = os.path.join(relaxdir0K, 'relaxed.dump')
                else:
                    checkpoint_success = os.path.join(relaxdir0K, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
                checkpoint_success = os.path.join(logdir, 'log.finished')
                status0K = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                          checkpoint_success=checkpoint_success,
                                          checkpoint_continue=relaxed_prev_restart,
                                          queue_system='slurm', cluster_info=cluster_info)
            else:
                status0K = {'status': 'no_submit'}

            if not mgw.print_instructions(status0K, submitfile="submit.sh.%s"%stepid): continue

        else:
            # 3u. Unloading test with Poisson's ratio = 0.4

            print('#'*npound)
            print("3u. Unloading test with Poisson's ratio = 0.4")
            print('#'*npound)

            stepid = '3' + unloading_str

            timestep = parsed_info_args.timestep_tensile;
            dump_freq = parsed_info_args.dumpfreq_unloading;  # ps, 0.4ps
            # timestep = 2.5e-3; dump_freq = 100;  # ps, 0.25ps
            Ttarget  = mgw.temperature     # K
            period_1 = 0                   # ps
            period_2 = mgw.unloading/100/mgw.erate *convert_s2ps # ps
            infile_template = os.path.join(mgw.lammps_templates, 'in.tensile.mg')
            dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
            os.makedirs(dumpdir, exist_ok=True)

            lammps_vars = {'Ttarget':                      Ttarget,
                           'timestep':                    timestep, # default: 2.5e-3
                           'Tdamp':                   100*timestep, # default: 0.25
                           'change_box_init':                    2, # load initial box from dump file
                           'load_dump_init':        load_dump_init,
                           'load_dump_idx':          load_dump_idx,
                           'N_deform':                           1, # change box size every step
                           'ex_trate':        1/(1+mgw.ex_trate)-1,
                           'ey_trate':        1/(1+mgw.ey_trate)-1,
                           'ez_trate':        1/(1+mgw.ez_trate)-1,
                           'total_steps_1': int(period_1/timestep),
                           'total_steps_2': int(period_2/timestep),
                           'thermo_step':                      100,
                           # path for saving dump and restart files
                           'dump_freq':                  dump_freq,
                           'dumpdir':                      dumpdir,
                           'restart_freq':                       0,
                           'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                           'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

            potential =   {'restart_file': restart_for_unloading,
                           'pair_style': 'eam/fs',
                           'potfile': mgw.potfile,
                           'elements': parsed_info_args.elements}
            cluster_info["job-name"] = "step-%s"%stepid
            # Restrict job submission to only node gpu-200-3 (Yifan 2020.11.24)
            gpu_id = 3 if mgw.qrate == 2.8e7 else 4
            cluster_info['nodelist'] = '' # 'gpu-200-%d'%gpu_id

            mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite,
                                      paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                                      submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)
            # Remove node restriction (Yifan 2020.11.24)
            cluster_info.pop('nodelist', None)

            if do_submit:
                status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                        checkpoint_success=lammps_vars['restart_file_2'], 
                                        queue_system='slurm', cluster_info=cluster_info)
            else:
                status = {'status': 'no_submit'}

            if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): continue
            nfile_unload = lammps_vars['total_steps_2']//lammps_vars['dump_freq']
            restart_for_unloading_relax = lammps_vars['restart_file_2']

            # 3.0u. 0K unloading test with Poisson's ratio = 0.4

            if parsed_info_args.run_0K_simulation:
                print('#'*npound)
                print("3.0u. 0K unloading test (molecular statics) using the box size history from Step 3")
                print('#'*npound)

                stepid = '3.0' + unloading_str

                finite_temp_loading = os.path.join(mgw.datadir, 'dump.3')
                loadinglogdir= os.path.join(mgw.datadir, 'log.3.0')
                loadingdir= os.path.join(mgw.datadir, 'dump.3.0')
                if parsed_info_args.partition_mc2[:3] == 'gpu':
                    infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
                    loading_logfile = os.path.join(loadinglogdir, 'log.relax')
                    unloading_init_file = os.path.join(loadingdir, 'relaxed.dump')
                    loadinglog = log.log(loading_logfile, verbose=False)
                    unloading_init_timestep = loadinglog.get('Step')[load_dump_i*2+1]
                else:
                    infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg')
                    unloading_init_file  = os.path.join(loadingdir, 'relaxed.%d.dump.gz')%load_dump_idx
                    with gzip.open(unloading_init_file, 'rt') as f:
                        line = f.readline()
                        if not line.split() == ['ITEM:', 'TIMESTEP']: raise TypeError('dump file not understandable')
                        unloading_init_file = int(f.readline().split()[0])
                dumpdir   = os.path.join(mgw.datadir, 'dump.3')

                logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
                relaxdir0K= os.path.join(mgw.datadir, 'dump.%s'%stepid)
                os.makedirs(relaxdir0K, exist_ok=True)
                os.makedirs(logdir, exist_ok=True)
                relaxed_prev_restart = os.path.join(relaxdir0K, 'relaxed.prev.restart')
                if not os.path.exists(restart_for_unloading):
                    print('** final configuration %s in step 3 does not exist!')
                    continue
                else:
                    shutil.copy(restart_for_unloading, relaxed_prev_restart)

                potential =   {'restart_file':       '${restart_file}',   # beginning of the tensile test
                               'pair_style': 'eam/fs',
                               'potfile': mgw.potfile,
                               'elements': parsed_info_args.elements}

                lammps_vars = {'istart':                             0,
                               'iend':                    nfile_unload,
                               'unloading':                          1,
                               'loadingdir':       finite_temp_loading,
                               # energy minimization parameters
                               'etol_minimize':                    0.0,
                               'ftol_minimize':                1.0e-11,
                               'maxiter_minimize':           100000000,
                               'maxeval_minimize':         10000000000,
                               'write_restart':                      1,
                               'restart_file':    relaxed_prev_restart,
                               'dump_freq':                  dump_freq,
                               'logdir':                        logdir,
                               'inputdir':                     dumpdir,
                               'relaxdir':                  relaxdir0K}

                cluster_info["job-name"] = "step-%s"%(stepid)

                mgw.generate_lammps_input(infile_template, infile="in.%s"%(stepid), log_file_name=None, overwrite=overwrite,
                                          replace_str = {'in.relax.mg': "in.%s"%(stepid), 
                                                         '${istart}':   '%s'%(0), 
                                                         '${iend}':     '%s'%(nfile_unload),
                                                         'replace yes': 'replace no', # use the box size from finite temperature tensile test
                                                         'reset_timestep 1': 'read_dump       %s %d x y z box yes replace yes'%(unloading_init_file, unloading_init_timestep), # load corresponding dump file before unloading starts
                                                         "print 'final step'": 'reset_timestep 1'},
                                          paramsfile="in.params.%s"%(stepid), variables=lammps_vars, potential=potential,
                                          submitfile="submit.sh.%s"%(stepid), queue_system='slurm', cluster_info=cluster_info)


                if do_submit:
                    if parsed_info_args.partition_mc2[:3] == 'gpu':
                        checkpoint_success = os.path.join(relaxdir0K, 'relaxed.dump')
                    else:
                        checkpoint_success = os.path.join(relaxdir0K, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
                    checkpoint_success = os.path.join(logdir, 'log.finished')
                    status0K = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                              checkpoint_success=checkpoint_success,
                                              checkpoint_continue=relaxed_prev_restart,
                                              queue_system='slurm', cluster_info=cluster_info)
                else:
                    status0K = {'status': 'no_submit'}

                if not mgw.print_instructions(status0K, submitfile="submit.sh.%s"%stepid): continue

            # 4u. Energy minimization dump files generated from step 3u tensile test

            print('#'*npound)
            print("4u. Energy minimization dump files generated from step 3u tensile test")
            print('#'*npound)

            stepid = '4' + unloading_str

            if parsed_info_args.partition_mc2[:3] == 'gpu':
                infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
            else:
                infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg')
            finite_temp_loading = os.path.join(mgw.datadir, 'dump.3')
            dumpdir   = os.path.join(mgw.datadir, 'dump.3' + unloading_str)
            logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
            relaxdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
            os.makedirs(relaxdir, exist_ok=True)
            os.makedirs(logdir, exist_ok=True)
            
            potential =   {'restart_file': restart_for_unloading_relax,
                           'pair_style': 'eam/fs',
                           'potfile': mgw.potfile,
                           'elements': parsed_info_args.elements}
            
            # Split relax jobs into Npart parts
            Npart = parsed_info_args.relax_Npart
            # nfile_total = lammps_vars['total_steps_2']//lammps_vars['dump_freq']
            nfile = nfile_unload // Npart
            all_parts_finished = True
            
            for ipart in range(Npart):
                lammps_vars = {'istart':                   ipart*nfile,
                               'iend':                 (ipart+1)*nfile,
                               'unloading':                          0,       # Not using box size from loading trajectory
                               'loadingdir':       finite_temp_loading,
                               # energy minimization parameters
                               'etol_minimize':                    0.0,
                               'ftol_minimize':                1.0e-11,
                               'maxiter_minimize':           100000000,
                               'maxeval_minimize':         10000000000,
                               'write_restart':                      0,
                               'dump_freq':                  dump_freq,
                               'logdir':                        logdir,
                               'inputdir':                     dumpdir,
                               'relaxdir':                    relaxdir}

                cluster_info["job-name"] = "step-%s.%s"%(stepid, ipart)

                mgw.generate_lammps_input(infile_template, infile="in.%s.%s"%(stepid, ipart), log_file_name=None, overwrite=overwrite,
                                          replace_str = {'in.relax.mg': "in.%s.%s"%(stepid, ipart), 
                                                         '${istart}':   '%s'%(ipart*nfile), 
                                                         '${iend}':     '%s'%((ipart+1)*nfile),
                                                         'reset_timestep 1': "print 'first step'",
                                                         "print 'final step'": 'reset_timestep 1',
                                                         # remove dump command (20-11-08, Yifan)
                                                        #  'dump            minimization': '#dump minimization',
                                                        #  'dump_modify     minimization': '#dump_modify minimization'
                                                        },
                                          paramsfile="in.params.%s.%s"%(stepid, ipart), variables=lammps_vars, potential=potential,
                                          submitfile="submit.sh.%s.%s"%(stepid, ipart), queue_system='slurm', cluster_info=cluster_info)

                if do_submit:
                    if parsed_info_args.partition_mc2[:3] == 'gpu':
                        checkpoint_success = os.path.join(relaxdir, 'relaxed.dump')
                    else:
                        checkpoint_success = os.path.join(relaxdir, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
                    checkpoint_success = os.path.join(logdir, 'log.finished')
                    status = mgw.submit_job(submitfile="submit.sh.%s.%s"%(stepid, ipart), 
                                            checkpoint_success=checkpoint_success, 
                                            queue_system='slurm', cluster_info=cluster_info)
                else:
                    status = {'status': 'no_submit'}

                if not mgw.print_instructions(status, submitfile="submit.sh.%s.%s"%(stepid, ipart)):
                    all_parts_finished = False
                    continue

            if not all_parts_finished: continue

            if parsed_info_args.unloading2 is not None:

                if parsed_info_args.unloading2 > parsed_info_args.unloading:
                    print('Unloading strain 2 is larger than unloading strain, skip...')
                    continue

                if nfile_total == 0:
                    unloading_strain_resolution = 1
                else:
                    unloading_strain_resolution = np.ceil(np.log10(nfile_total/parsed_info_args.max_tensile_strain)).astype(int)
                # print(unloading_strain_resolution)
                resolution_str = '.%df'%unloading_strain_resolution
                load_dump_i    = np.round(nfile_total*(parsed_info_args.unloading - parsed_info_args.unloading2)/parsed_info_args.max_tensile_strain).astype(int)
                load_dump_idx  = load_dump_i*parsed_info_args.dumpfreq_tensile
                load_dump_init = os.path.join(mgw.datadir, 'dump.3', 'dump.%d.gz')%load_dump_idx
                unloading_str = ('u_%'+resolution_str+'-0')%parsed_info_args.unloading

                # 4.0u. 0K unloading test (molecular statics) after finite temperature unloading

                if parsed_info_args.run_0K_unloading2 is not None:
                    print('#'*npound)
                    print("4.0u. 0K unloading test (molecular statics) using the box size history from Step 3u")
                    print('#'*npound)

                    unloading2_str = ('u_%'+resolution_str+'-%d')%(parsed_info_args.unloading2, parsed_info_args.run_0K_unloading2)
                    stepid = '3' + unloading_str + '.4.0' + unloading2_str

                    finite_temp_loading = os.path.join(mgw.datadir, 'dump.3')
                    nfile_unload = round(parsed_info_args.unloading2/100/mgw.erate *convert_s2ps/parsed_info_args.timestep_tensile)//parsed_info_args.dumpfreq_tensile
                    unloadingdir = os.path.join(mgw.datadir, 'dump.4' + unloading_str)
                    unloadinglogdir = os.path.join(mgw.datadir, 'log.4' + unloading_str)
                    if parsed_info_args.partition_mc2[:3] == 'gpu':
                        infile_template = os.path.join(mgw.lammps_templates, 'in.relax.mg.gpu')
                        loading_logfile = os.path.join(unloadinglogdir, 'log.relax')
                        unloading_init_file = os.path.join(unloadingdir, 'relaxed.dump')
                        loadinglog = log.log(loading_logfile, verbose=False)
                        unloading_init_timestep = loadinglog.get('Step')[load_dump_i*2+1]
                        # print(load_dump_i, loadinglog.get('Step')[load_dump_i*2+1])
                    else:
                        raise ValueError('non gpu not supported yet for this step!')
                    dumpdir   = os.path.join(mgw.datadir, 'dump.3')

                    logdir    = os.path.join(mgw.datadir,  'log.%s'%stepid)
                    relaxdir0K= os.path.join(mgw.datadir, 'dump.%s'%stepid)
                    os.makedirs(relaxdir0K, exist_ok=True)
                    os.makedirs(logdir, exist_ok=True)
                    relaxed_prev_restart = os.path.join(relaxdir0K, 'relaxed.prev.restart')
                    if not os.path.exists(restart_for_unloading):
                        print('** final configuration %s in step 3 does not exist!')
                        continue
                    else:
                        shutil.copy(restart_for_unloading, relaxed_prev_restart)

                    potential =   {'restart_file':       '${restart_file}',   # beginning of the tensile test
                                'pair_style': 'eam/fs',
                                'potfile': mgw.potfile,
                                'elements': parsed_info_args.elements}

                    lammps_vars = {'istart':                          0,
                                'iend':                    nfile_unload,
                                'unloading':                          1,
                                'loadingdir':       finite_temp_loading,
                                # energy minimization parameters
                                'etol_minimize':                    0.0,
                                'ftol_minimize':                1.0e-11,
                                'maxiter_minimize':           100000000,
                                'maxeval_minimize':         10000000000,
                                'write_restart':                      1,
                                'restart_file':    relaxed_prev_restart,
                                'dump_freq':                  dump_freq,
                                'logdir':                        logdir,
                                'inputdir':                     dumpdir,
                                'relaxdir':                  relaxdir0K}

                    cluster_info["job-name"] = "step-%s"%(stepid)

                    unload_end  = round(nfile_total//parsed_info_args.max_tensile_strain*parsed_info_args.run_0K_unloading2)
                    replace_str = {'in.relax.mg': "in.%s"%(stepid), 
                                '${istart} ${iend}':   '%s %s'%(0, nfile_unload - unload_end), 
                                'read_dump       ${inputdir}/dump.${filenum}.gz ${filenum} x y z box yes replace yes': '# read_dump     ${inputdir}/dump.${filenum}.gz ${filenum} x y z box yes replace yes', # not using filenum
                                'reset_timestep 1': 'read_dump       %s %d x y z box yes replace yes'%(unloading_init_file, unloading_init_timestep), # load corresponding dump file before unloading starts
                                "print 'final step'": 'reset_timestep 1', 
                                '${i} == ${iend}': '${i} == %s'%(nfile_unload - unload_end)}
                    if parsed_info_args.run_0K_unloading2 > parsed_info_args.unloading2: # 0K loading to run_0K_unloading
                        replace_str['${istart} ${iend}'] = '%s %s'%(0, unload_end - nfile_unload)
                        replace_str['${iend}-${i}'] = '${iend}+${i}'
                        replace_str['${i} == ${iend}'] = '${i} == %s'%(unload_end - nfile_unload)
                        replace_str.pop("print 'final step'", None)
                    mgw.generate_lammps_input(infile_template, infile="in.%s"%(stepid), log_file_name=None, overwrite=overwrite,
                                            replace_str = replace_str,
                                            paramsfile="in.params.%s"%(stepid), variables=lammps_vars, potential=potential,
                                            submitfile="submit.sh.%s"%(stepid), queue_system='slurm', cluster_info=cluster_info)

                    if do_submit:
                        if parsed_info_args.partition_mc2[:3] == 'gpu':
                            checkpoint_success = os.path.join(relaxdir0K, 'relaxed.dump')
                        else:
                            checkpoint_success = os.path.join(relaxdir0K, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq))
                        checkpoint_success = os.path.join(logdir, 'log.finished')
                        print(checkpoint_success)
                        status0K = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                                checkpoint_success=checkpoint_success,
                                                checkpoint_continue=relaxed_prev_restart,
                                                queue_system='slurm', cluster_info=cluster_info)
                    else:
                        status0K = {'status': 'no_submit'}

                    if not mgw.print_instructions(status0K, submitfile="submit.sh.%s"%stepid): continue

    # gpu jobs cannot do soft/hard analysis yet
    if parsed_info_args.partition_mc2[:3] == 'gpu':
        print('#'*npound)
        print('T = %dK tensile test finished'%temp)
        print('#'*npound)
        continue

    # 5.0. Analyze 0K simulation and obtain the phop values, and soft/hard labels

    if parsed_info_args.run_0K_simulation and status0K['status'] in ['finished', 'no_submit']:

        print('#'*npound)
        print("5.0. Analyze 0K simulation and obtain the phop values, and soft/hard labels")
        print('#'*npound)

        stepid = 5.0
        nslice = parsed_info_args.analysis_nslice

        # Split relax jobs into Npart parts
        arguments = {'--data_dir':    mgw.datadir,
                     '--lammps_dump': os.path.join(os.path.basename(relaxdir0K), 'relaxed.*.dump.gz'),
                     '--post_dir':    'post_0K',
                     '--disp_data':   'disp/disp.data',
                     '--phop_data':   'phop/phop.data',
                     '--phop_window': int(2/timestep/lammps_vars['dump_freq']),
                     '--meta_data':   'meta/meta.data',
                     '--phop_th':     0.05,
                     '--feature_dir': 'feature'}

        cluster_info["job-name"] = "step-%s"%(stepid); # cluster_info['partition'] = 'gpu-tesla'
        analysis_file_fraction = parsed_info_args.max_analysis_strain/mgw.max_tensile_strain
        mgw.generate_analysis_code(n_frame_total=int(nfile_total*analysis_file_fraction), n_frame_slice=nslice, n_particle_total=mgw.natom, n_particle_slice=nslice, paramsfile='setup_0K.txt',
                                   variables=arguments, overwrite=overwrite, submitfile='submit.sh.%s'%stepid, queue_system='slurm', cluster_info=cluster_info)

        if do_submit:
            status0K = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                      checkpoint_success=os.path.join(mgw.datadir, arguments['--post_dir'], arguments['--meta_data']+'.full.npz'),
                                      checkpoint_continue=os.path.join(relaxdir0K, 'relaxed.%s.dump.gz'%(lammps_vars['iend']*dump_freq)),
                                      queue_system='slurm', cluster_info=cluster_info)
        else:
            status0K = {'status': 'no_submit'}

        if not mgw.print_instructions(status0K): continue

    # 5. Analyze the relaxed configuration and obtain the phop values, and soft/hard labels

    print('#'*npound)
    print("5. Analyze the relaxed configuration and obtain the phop values, and soft/hard labels")
    print('#'*npound)

    stepid = 5
    nslice = parsed_info_args.analysis_nslice

    # Split relax jobs into Npart parts
    arguments = {'--data_dir':    mgw.datadir,
                 '--lammps_dump': os.path.join(os.path.basename(relaxdir), 'relaxed.*.dump.gz'),
                 '--post_dir':    'post',
                 '--disp_data':   'disp/disp.data',
                 '--phop_data':   'phop/phop.data',
                 '--phop_window': int(2/timestep/lammps_vars['dump_freq']),
                 '--meta_data':   'meta/meta.data',
                 '--phop_th':     0.05,
                 '--feature_dir': 'feature'}

    cluster_info["job-name"] = "step-%s"%(stepid); # cluster_info['partition'] = 'gpu-tesla'
    analysis_file_fraction = parsed_info_args.max_analysis_strain/mgw.max_tensile_strain
    mgw.generate_analysis_code(n_frame_total=int(nfile_total*analysis_file_fraction), n_frame_slice=nslice, n_particle_total=mgw.natom, n_particle_slice=nslice,
                               variables=arguments, overwrite=overwrite, submitfile='submit.sh.%s'%stepid, queue_system='slurm', cluster_info=cluster_info)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                checkpoint_success=os.path.join(mgw.datadir, arguments['--post_dir'], arguments['--meta_data']+'.full.npz'),
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status): continue

    print('#'*120)
    print('** Finish tensile test workflow of temperature %sK'%mgw.temperature)
    print('#'*120)
    print()

