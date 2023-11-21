# Workflow for MetallicGlass simulation

import os, subprocess, argparse
import numpy as np
from workflow_utils import MetallicGlassWorkflow, test_env_vars

ap = argparse.ArgumentParser()
ap.add_argument('--overwrite',   action='store_true')
ap.add_argument('--no_submit',   action='store_true')
ap.add_argument('--use_scratch', action='store_true')
ap.add_argument('--potfilename',         default='CuZr.eam.fs', type=str)
ap.add_argument('--elements', nargs=2,   default=['Cu', 'Zr'],  type=str)
ap.add_argument('--partition_mc2',       default='cpu',         type=str)
ap.add_argument('--natom',               default=5000,  type=int)
ap.add_argument('--x1',                  default=0.645, type=float)
ap.add_argument('--qrate',               default=1e10,  type=float)
ap.add_argument('--sample_id',           default=0,     type=int)
ap.add_argument('--timestep_anneal',     default=2.5e-3,type=float)
ap.add_argument('--temp_anneal',         default=700,   type=int, help='temperature for annealing, 700K for CuZr metallic glass')
ap.add_argument('--initsize',            default=[58, 58, 58], nargs=3,   type=int, help='size of the random initial configuration')

parsed_info_args = ap.parse_args()

cluster_info = {}; do_submit = use_scratch = False;   # By default, for testing on local machines
npound = 60                             # number of pound signs for printing header
convert_us2ps= 1.0e6
convert_s2ps = 1.0e12                   # unit conversion
use_gpu = False

hostname = os.getenv('HOSTNAME')
if hostname:
    if hostname[:3] == 'mc2':           # for mc2 cluster
        cluster_info = {'partition': parsed_info_args.partition_mc2, 'nodes': 1, 'tasks-per-node': 24} # SLURM system
        if parsed_info_args.partition_mc2[:3] == 'gpu':
            cluster_info['gres'] = 'gpu:1'
            if parsed_info_args.partition_mc2 == 'gpu-geforce':
                # Restrict job submission to only node gpu-200-4 (Yifan 2020.11.24)
                cluster_info['nodelist'] = 'gpu-200-4'
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
        cluster_info = {'partition': 'mc', 'nodes': 1, 'tasks-per-node': 20, 'time': '7-00:00:00'} # SLURM system
        do_submit = use_scratch = True

overwrite =           parsed_info_args.overwrite
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
    print('#'*npound)
    print('Build LAMMPS using environment variable settings')
    print('#'*npound)
    raise FileNotFoundError('Binary %s is not found'%lammps_bin)
    # os.chdir(os.path.join(lammpsdir, 'src'))
    # subprocess.run(['make', 'yes-manybody'])     # install MANYBODY package to use EAM potential
    # subprocess.run(['make', lammps_sys])

print('#'*npound); print("1. Set up paths and prepare folders"); print('#'*npound)

MLmat_dir = os.getenv('MLMAT_DIR'); os.chdir(MLmat_dir);
print('MLmat_dir =', MLmat_dir)
mgw = MetallicGlassWorkflow(MLmat_dir          = MLmat_dir,
                            natom              = parsed_info_args.natom,
                            x1                 = parsed_info_args.x1,
                            qrate              = parsed_info_args.qrate,         # K/s
                            sample_id          = parsed_info_args.sample_id,
                            potfilename        = parsed_info_args.potfilename,
                            use_scratch        = use_scratch,  # use $SCRATCH to save simulation data
                            use_gpu            = use_gpu)
mgw.print_sample_properties()
mgw.print_directories()

# 2. Generate initial configuration and save to file
print('#'*npound)
print("2. Generate initial configuration and save to file")
print('#'*npound)
# default initLx = initLy = initLz = 58
initsize = tuple(parsed_info_args.initsize) #(initLx, initLy, initLz)
mgw.generate_initial_config(initsize=initsize, filename='init_config_%dx%dx%d'%initsize)
init_config_file = mgw.init_config_file

# 3. Heat up the sample from 300 K to 2000 K and anneal at 2000 K for 3 ns by LAMMPS

print('#'*npound); 
print("3. Heat up the sample from 300 K to 2000 K and anneal at 2000 K for 3 ns by LAMMPS"); 
print('#'*npound);

stepid = 3

timestep = parsed_info_args.timestep_anneal # ps
period_1 = 100  # ps
period_2 = 3000 # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = 300, 2000, 2000
dumpdir  = os.path.join(mgw.datadir, 'dump.%d'%stepid)
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
               'dump_freq':     int(period_1/timestep/5),
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep/5),
               'restart_file_1':  os.path.join(dumpdir, "restart.%d.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%d.2"%stepid) }

potential =   {'init_config': init_config_file,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%d"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%d"%stepid, log_file_name='log.%d'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%d"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%d"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%d"%stepid, checkpoint_success=lammps_vars['restart_file_2'], queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): exit(0)

restart_for_nextstep = os.path.join(dumpdir, "restart.%d.2"%stepid)

# 4. Cool down the sample from 2000 - parsed_info_args.temp_anneal - 2 K by cooling rate of qrate K/s by LAMMPS

stepid = 4

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = 2000, parsed_info_args.temp_anneal, 2
period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #130000  # ps
period_2 = (T1 - T2)/mgw.qrate*convert_s2ps #69800 # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound); 
print("%s. Cool down the sample from %s - %s - %s K by cooling rate of %.1f K/s by LAMMPS"%(stepid, T0, T1, T2, mgw.qrate))
print('#'*npound);

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%s"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=lammps_vars['restart_file_1'], 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)

# Use the 1e10K/s restart file for the following calculation
restart_for_nextstep = os.path.join(dumpdir, "restart.%s.1"%stepid)
restart_final = os.path.join(dumpdir, "restart.%s.2"%stepid)

# 4.5 Cool down the sample from parsed_info_args.temp_anneal - 300K by cooling rate of qrate K/s, and anneal at 300K for 1ns

stepid = 4.5

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = parsed_info_args.temp_anneal, 300, 300
if T0 == T1:
    period_1 = 500
    period_2 = 500
else:
    period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #40000  # ps
    period_2 = 1000  # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound); 
print("%s. Cool down the sample from %s - %s K by cooling rate of %.1f K/s, anneal at %sK for 1ns"%(stepid, T0, T1, mgw.qrate, T2))
print('#'*npound);

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%s"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=restart_for_nextstep, 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)

if mgw.qrate != 1e10:
    print('Finished. No need to continue')
    exit(0)

# 5. Anneal the sample at parsed_info_args.temp_anneal K for maximum of 0.1 us and cool down to 2K (effective qrate=5.0e8 K/s)

stepid  = 5

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, 2
period_1 = 100000 # ps
period_2 = (T1 - T2)/mgw.qrate*convert_s2ps #69800 # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%d'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound)
print("%s. Continue annealing the sample at %sK for %.1f us and cool down to 2K"%(stepid, T1, period_1/convert_us2ps))
print('#'*npound)

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%d.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%d.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%d"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%d"%stepid, log_file_name='log.%d'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%d"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%d"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%d"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=lammps_vars['restart_file_1'], 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): exit(0)

restart_for_nextstep = os.path.join(dumpdir, "restart.%d.1"%stepid)

# 5.5 Cool down the sample from parsed_info_args.temp_anneal - 300K by cooling rate of qrate K/s, and anneal at 300K for 1ns

stepid = 5.5

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = parsed_info_args.temp_anneal, 300, 300
period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #40000  # ps
period_2 = 1000  # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound); 
print("%s. Cool down the sample from %s - %s K by cooling rate of %.1f K/s, anneal at %sK for 1ns"%(stepid, T0, T1, mgw.qrate, T2))
print('#'*npound);

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%s"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=restart_for_nextstep, 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)

# 6. Continue annealing the sample at parsed_info_args.temp_anneal K for 0.4 us and cool down to 2K (effective qrate=6.3e7 K/s)

stepid  = 6

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, 2
period_1 = 400000 # ps
period_2 = (T1 - T2)/mgw.qrate*convert_s2ps #69800 # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%d'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound)
print("%s. Continue annealing the sample at %sK for %.1f us and cool down to 2K"%(stepid, T1, period_1/convert_us2ps))
print('#'*npound)

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%d.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%d.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%d"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%d"%stepid, log_file_name='log.%d'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%d"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%d"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%d"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=lammps_vars['restart_file_1'], 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): exit(0)
restart_for_nextstep = os.path.join(dumpdir, "restart.%d.1"%stepid)

# 6.5 Cool down the sample from parsed_info_args.temp_anneal - 300K by cooling rate of qrate K/s, and anneal at 300K for 1ns

stepid = 6.5

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = parsed_info_args.temp_anneal, 300, 300
period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #40000  # ps
period_2 = 1000  # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound) 
print("%s. Cool down the sample from %s - %s K by cooling rate of %.1f K/s, anneal at %sK for 1ns"%(stepid, T0, T1, mgw.qrate, T2))
print('#'*npound)

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%s"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=restart_for_nextstep, 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)

# 7.1-2. Continue annealing the sample at parsed_info_args.temp_anneal K for 0.5 us (x2 times)

for isubstep in [1, 2]:
    stepid  = '7.%s'%isubstep

    timestep = parsed_info_args.timestep_anneal # ps
    infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
    T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, parsed_info_args.temp_anneal
    period_1 = 250000 # ps
    period_2 = 250000 # ps
    dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
    os.makedirs(dumpdir, exist_ok=True)

    print('#'*npound)
    print("%s. Continue annealing the sample at %sK for %.1f us"%(stepid, T1, (period_1+period_2)/convert_us2ps))
    print('#'*npound)
    
    lammps_vars = {'T0':                                T0,
                   'T1':                                T1,
                   'T2':                                T2,
                   'timestep':                    timestep, # default: 2.5e-3
                   'Tdamp':                   100*timestep, # default: 0.25
                   'Pdamp':                  1000*timestep, # default: 2.5
                   'total_steps_1': int(period_1/timestep),
                   'total_steps_2': int(period_2/timestep),
                   'thermo_step':                      500,
                   # parameters for energy minimization before heating up
                   'do_minimize':                        0,
                   # parameters for setting velocity 
                   'set_velocity':                       0,
                   # path for saving dump and restart files
                   # 'dump_freq':     int(period_1/timestep),
                   'dump_freq':                      10000,
                   'dumpdir':                      dumpdir,
                   'restart_freq':  int(period_1/timestep),
                   'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                   'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

    potential =   {'restart_file': restart_for_nextstep,
                   'pair_style': 'eam/fs',
                   'potfile': mgw.potfile,
                   'elements': parsed_info_args.elements}
    cluster_info["job-name"] = "step-%s"%stepid

    mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                              paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                              submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                checkpoint_success=lammps_vars['restart_file_2'], 
                                checkpoint_continue=lammps_vars['restart_file_2'], 
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)
    restart_for_nextstep = os.path.join(dumpdir, "restart.%s.2"%stepid)

# 7. Continue annealing the sample at parsed_info_args.temp_anneal K for 0.5 us and cool down to 2K (effective qrate=2.8e7K/s)

stepid  = 7

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, 2
period_1 = 500000 # ps
period_2 = (T1 - T2)/mgw.qrate*convert_s2ps #69800 # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%d'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound)
print("%s. Continue annealing the sample at %sK for %.1f us and cool down to 2K"%(stepid, T1, period_1/convert_us2ps))
print('#'*npound)

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%d.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%d.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%d"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%d"%stepid, log_file_name='log.%d'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%d"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%d"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%d"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=lammps_vars['restart_file_1'], 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): exit(0)
restart_for_nextstep = os.path.join(dumpdir, "restart.%s.1"%stepid)

# 7.5 Cool down the sample from parsed_info_args.temp_anneal - 300K by cooling rate of qrate K/s, and anneal at 300K for 1ns

stepid = 7.5

timestep = parsed_info_args.timestep_anneal # ps
infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
T0, T1, T2 = parsed_info_args.temp_anneal, 300, 300
period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #40000  # ps
period_2 = 1000  # ps
dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
os.makedirs(dumpdir, exist_ok=True)

print('#'*npound) 
print("%s. Cool down the sample from %s - %s K by cooling rate of %.1f K/s, anneal at %sK for 1ns"%(stepid, T0, T1, mgw.qrate, T2))
print('#'*npound)

lammps_vars = {'T0':                                T0,
               'T1':                                T1,
               'T2':                                T2,
               'timestep':                    timestep, # default: 2.5e-3
               'Tdamp':                   100*timestep, # default: 0.25
               'Pdamp':                  1000*timestep, # default: 2.5
               'total_steps_1': int(period_1/timestep),
               'total_steps_2': int(period_2/timestep),
               'thermo_step':                      500,
               # parameters for energy minimization before heating up
               'do_minimize':                        0,
               # parameters for setting velocity 
               'set_velocity':                       0,
               # path for saving dump and restart files
               # 'dump_freq':     int(period_1/timestep),
               'dump_freq':                      10000,
               'dumpdir':                      dumpdir,
               'restart_freq':  int(period_1/timestep),
               'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
               'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

potential =   {'restart_file': restart_for_nextstep,
               'pair_style': 'eam/fs',
               'potfile': mgw.potfile,
               'elements': parsed_info_args.elements}
cluster_info["job-name"] = "step-%s"%stepid

mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                          paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                          submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

if do_submit:
    status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                            checkpoint_success=lammps_vars['restart_file_2'], 
                            checkpoint_continue=restart_for_nextstep, 
                            queue_system='slurm', cluster_info=cluster_info)
else:
    status = {'status': 'no_submit'}

if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)

for istep in [8, 9, 10]:
    # istep.1-2. Continue annealing the sample at parsed_info_args.temp_anneal K for 1.0 us (x2 times)

    for isubstep in [1, 2]:
        stepid  = '%s.%s'%(istep, isubstep)

        timestep = parsed_info_args.timestep_anneal # ps
        infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
        T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, parsed_info_args.temp_anneal
        period_1 = 500000 # ps
        period_2 = 500000 # ps
        dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
        os.makedirs(dumpdir, exist_ok=True)

        print('#'*npound)
        print("%s. Continue annealing the sample at %sK for %.1f us"%(stepid, T1, (period_1+period_2)/convert_us2ps))
        print('#'*npound)
        
        lammps_vars = {'T0':                                T0,
                    'T1':                                T1,
                    'T2':                                T2,
                    'timestep':                    timestep, # default: 2.5e-3
                    'Tdamp':                   100*timestep, # default: 0.25
                    'Pdamp':                  1000*timestep, # default: 2.5
                    'total_steps_1': int(period_1/timestep),
                    'total_steps_2': int(period_2/timestep),
                    'thermo_step':                      500,
                    # parameters for energy minimization before heating up
                    'do_minimize':                        0,
                    # parameters for setting velocity 
                    'set_velocity':                       0,
                    # path for saving dump and restart files
                    # 'dump_freq':     int(period_1/timestep),
                    'dump_freq':                      10000,
                    'dumpdir':                      dumpdir,
                    'restart_freq':  int(period_1/timestep),
                    'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                    'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

        potential =   {'restart_file': restart_for_nextstep,
                    'pair_style': 'eam/fs',
                    'potfile': mgw.potfile,
                    'elements': parsed_info_args.elements}
        cluster_info["job-name"] = "step-%s"%stepid

        mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                                paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                                submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

        if do_submit:
            status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                    checkpoint_success=lammps_vars['restart_file_2'], 
                                    checkpoint_continue=lammps_vars['restart_file_2'], 
                                    queue_system='slurm', cluster_info=cluster_info)
        else:
            status = {'status': 'no_submit'}

        if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)
        restart_for_nextstep = os.path.join(dumpdir, "restart.%s.2"%stepid)

    # istep. Continue annealing the sample at parsed_info_args.temp_anneal K for 1.0 us and cool down to 2K

    stepid  = istep

    timestep = parsed_info_args.timestep_anneal # ps
    infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
    T0, T1, T2   = parsed_info_args.temp_anneal, parsed_info_args.temp_anneal, 2
    period_1 = 1000000 # ps
    period_2 = (T1 - T2)/mgw.qrate*convert_s2ps #69800 # ps
    dumpdir  = os.path.join(mgw.datadir, 'dump.%d'%stepid)
    os.makedirs(dumpdir, exist_ok=True)

    print('#'*npound)
    print("%s. Continue annealing the sample at %sK for %.1f us and cool down to 2K"%(stepid, T1, period_1/convert_us2ps))
    print('#'*npound)

    lammps_vars = {'T0':                                T0,
                'T1':                                T1,
                'T2':                                T2,
                'timestep':                    timestep, # default: 2.5e-3
                'Tdamp':                   100*timestep, # default: 0.25
                'Pdamp':                  1000*timestep, # default: 2.5
                'total_steps_1': int(period_1/timestep),
                'total_steps_2': int(period_2/timestep),
                'thermo_step':                      500,
                # parameters for energy minimization before heating up
                'do_minimize':                        0,
                # parameters for setting velocity 
                'set_velocity':                       0,
                # path for saving dump and restart files
                # 'dump_freq':     int(period_1/timestep),
                'dump_freq':                      10000,
                'dumpdir':                      dumpdir,
                'restart_freq':  int(period_1/timestep),
                'restart_file_1':  os.path.join(dumpdir, "restart.%d.1"%stepid),
                'restart_file_2':  os.path.join(dumpdir, "restart.%d.2"%stepid) }

    potential =   {'restart_file': restart_for_nextstep,
                'pair_style': 'eam/fs',
                'potfile': mgw.potfile,
                'elements': parsed_info_args.elements}
    cluster_info["job-name"] = "step-%d"%stepid

    mgw.generate_lammps_input(infile_template, infile="in.%d"%stepid, log_file_name='log.%d'%stepid, overwrite=overwrite, 
                            paramsfile="in.params.%d"%stepid, variables=lammps_vars, potential=potential, 
                            submitfile="submit.sh.%d"%stepid, queue_system='slurm', cluster_info=cluster_info)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%d"%stepid, 
                                checkpoint_success=lammps_vars['restart_file_2'], 
                                checkpoint_continue=lammps_vars['restart_file_1'], 
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status, log_file_name='log.%d'%stepid, submitfile="submit.sh.%d"%stepid): exit(0)
    restart_for_nextstep = os.path.join(dumpdir, "restart.%s.1"%stepid)

    # istep.5 Cool down the sample from parsed_info_args.temp_anneal - 300K by cooling rate of qrate K/s, and anneal at 300K for 1ns

    stepid = '%s.5'%istep

    timestep = parsed_info_args.timestep_anneal # ps
    infile_template = os.path.join(mgw.lammps_templates, 'in.anneal.npt.mg')
    T0, T1, T2 = parsed_info_args.temp_anneal, 300, 300
    period_1 = (T0 - T1)/mgw.qrate*convert_s2ps #40000  # ps
    period_2 = 1000  # ps
    dumpdir  = os.path.join(mgw.datadir, 'dump.%s'%stepid)
    os.makedirs(dumpdir, exist_ok=True)

    print('#'*npound) 
    print("%s. Cool down the sample from %s - %s K by cooling rate of %.1f K/s, anneal at %sK for 1ns"%(stepid, T0, T1, mgw.qrate, T2))
    print('#'*npound)

    lammps_vars = {'T0':                                T0,
                'T1':                                T1,
                'T2':                                T2,
                'timestep':                    timestep, # default: 2.5e-3
                'Tdamp':                   100*timestep, # default: 0.25
                'Pdamp':                  1000*timestep, # default: 2.5
                'total_steps_1': int(period_1/timestep),
                'total_steps_2': int(period_2/timestep),
                'thermo_step':                      500,
                # parameters for energy minimization before heating up
                'do_minimize':                        0,
                # parameters for setting velocity 
                'set_velocity':                       0,
                # path for saving dump and restart files
                # 'dump_freq':     int(period_1/timestep),
                'dump_freq':                      10000,
                'dumpdir':                      dumpdir,
                'restart_freq':  int(period_1/timestep),
                'restart_file_1':  os.path.join(dumpdir, "restart.%s.1"%stepid),
                'restart_file_2':  os.path.join(dumpdir, "restart.%s.2"%stepid) }

    potential =   {'restart_file': restart_for_nextstep,
                'pair_style': 'eam/fs',
                'potfile': mgw.potfile,
                'elements': parsed_info_args.elements}
    cluster_info["job-name"] = "step-%s"%stepid

    mgw.generate_lammps_input(infile_template, infile="in.%s"%stepid, log_file_name='log.%s'%stepid, overwrite=overwrite, 
                            paramsfile="in.params.%s"%stepid, variables=lammps_vars, potential=potential, 
                            submitfile="submit.sh.%s"%stepid, queue_system='slurm', cluster_info=cluster_info)

    if do_submit:
        status = mgw.submit_job(submitfile="submit.sh.%s"%stepid, 
                                checkpoint_success=lammps_vars['restart_file_2'], 
                                checkpoint_continue=restart_for_nextstep, 
                                queue_system='slurm', cluster_info=cluster_info)
    else:
        status = {'status': 'no_submit'}

    if not mgw.print_instructions(status, log_file_name='log.%s'%stepid, submitfile="submit.sh.%s"%stepid): exit(0)
