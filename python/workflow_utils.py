''' Utility class and functions used in Workflow.py
'''

import os, shutil, subprocess, glob
import numpy as np
import matplotlib.pyplot as plt
from log import log as lammps_log
from poisson_disk.poisson_disk import Grid3D, sample_poisson_3d

# Default directory for the MLmat package
MLmat_dir_default = os.getenv('MLMAT_DIR')
ovitodir_default  = os.getenv("OVITOS_BIN")
lammpsbin_default = os.getenv("LAMMPS_BIN")
convert_s2ps = 1.0e12                   # unit conversion
default_timestep = 1e-3                 # ps

# Default cluster information for generating submit file (SLURM system, mc2 cluster)
cluster_info_default = {'job-name': 'mg', 'partition': 'cpu', 'nodes': 1, 'tasks-per-node': 24}

def test_env_vars(envvars):
    testflag = True
    for envvar in envvars:
        if envvar not in os.environ.keys():
            print('Environment variable "'+envvar+'" not set')
            testflag = False
        else:
            print('Environment variable "'+envvar+'" set to '+os.environ[envvar])
    return testflag

def generate_header_str(header_dict, bashstr='/bin/sh', option='SBATCH'):
    '''Generate string for submit file given the settings in dictionary format
        By default, we use /bin/sh as bash and SBATCH for set up
    '''
    option_list = ['#%s --%s=%s'%(option, key, str(header_dict[key])) for key in header_dict]
    return '\n'.join(['#!/bin/sh', ] + option_list)

class MetallicGlassWorkflow:
    '''
    The workflow utility class for metallic glass
    
    Use the class by setting up an instance:
    ```
    mgw = MetallicGlassWorkflow(MLmat_dir, natom, Lx, Ly, Lz, qrate, erate, max_tensile_strain)
    ```

    Attributes
    ----------
    natom : int
        Number of atoms in the box
    qrate : float
        Effective quenching rate of the sample (K/s)
    erate : float
        Strain rate of the tensile test (s^(-1))
    max_tensile_strain : int
        Maximum tensile strain during the tensile test
    use_scratch : bool
        If True, use $SCRATCH directory to save data

    Methods
    -------
    get_dirname()
        Returns the directory of the case
    '''
    def __init__(self, MLmat_dir=MLmat_dir_default, ovitodir=ovitodir_default, lammpsbin=lammpsbin_default, natom=5000, qrate=1e10, sample_id=None, erate=None, poisson_ratio=None, max_tensile_strain=None, plot_strain_range=None, temperature=None, x1 =0.645, potfilename='CuZr.eam.fs', use_scratch=True, unloading=False, use_gpu=False, plot_strain_skip=1, run_id=None):
        ''' Initialization for the workflow class
        '''
        # Settings for the simulation
        self.natom = natom
        self.x1  = x1 
        self.qrate = qrate
        self.erate = erate
        self.poisson_ratio = poisson_ratio
        self.max_tensile_strain = max_tensile_strain
        if max_tensile_strain is not None:
            if plot_strain_range is None:
                self.plot_strain_range = [0, max_tensile_strain]
            else:
                self.plot_strain_range = plot_strain_range
        else:
            self.plot_strain_range = None
        self.plot_strain_skip = plot_strain_skip
        self.temperature = temperature
        self.unloading   = unloading
        self.use_gpu     = use_gpu
        self.timestep    = default_timestep
        
        # Set up directories and potential files
        self.MLmat_dir = MLmat_dir
        self.ovitodir = ovitodir
        self.lammpsbin = lammpsbin
        self.sample_id = sample_id
        self.run_id    = run_id
        self.homedir = os.path.join(self.MLmat_dir, 'MetallicGlass')
        self.codedir = os.path.join(self.homedir, 'python')
        self.lammps_templates = os.path.join(self.homedir, 'lammps_scripts')
        self.potfile = os.path.join(self.lammps_templates, potfilename)
        self.datarootdir = os.path.join(self.homedir, 'data')
        # Use $SCRATCH to save data
        if use_scratch and 'SCRATCH' in os.environ:
            self.scratch_datadir = os.path.join(os.environ['SCRATCH'], 'MLmat_data', 'MetallicGlass')
            os.makedirs(self.scratch_datadir, exist_ok=True)
            if not os.path.exists(self.datarootdir):
                os.symlink(self.scratch_datadir, self.datarootdir)
        self.datadir = os.path.join(self.datarootdir, 'natom-%d'%self.natom, 'qrate-%.1e'%self.qrate)
        if self.sample_id is not None:
            self.datadir = os.path.join(self.datadir, 'sample_%d'%self.sample_id)
        if self.erate is not None and self.max_tensile_strain is not None and self.poisson_ratio is not None:
            self.datadir = os.path.join(self.datadir, 'erate-%.1e'%self.erate, 'strain_0-%.1f'%self.max_tensile_strain)
            if self.run_id is not None:
                self.datadir = self.datadir + '_%d'%self.run_id
            self.ey_trate= self.erate / convert_s2ps
            self.ex_trate= self.ez_trate = -self.poisson_ratio * self.ey_trate
        if self.temperature is not None:
            self.datadir = os.path.join(self.datadir, 'T%dK'%self.temperature)
        os.makedirs(self.datadir, exist_ok=True)

        self.lammps_scripts = os.path.join(self.datadir, 'lammps_scripts')
        self.analysis_scripts = os.path.join(self.datadir, 'analysis_scripts')
        os.makedirs(self.lammps_scripts, exist_ok=True)
        os.makedirs(self.analysis_scripts, exist_ok=True)

    def print_sample_properties(self):
        print('natom =', self.natom)
        print('x1  =', self.x1 )
        print('qrate = %.1e'%self.qrate)
        if self.erate is not None and self.max_tensile_strain is not None:
            print('erate = %.1e'%self.erate)
            print(r'max_tensile_strain = %.1f%%'%self.max_tensile_strain)
        if self.temperature is not None:
            print('temperature = %s'%self.temperature)
        
    def print_directories(self):
        print('datadir =', self.datadir)

    def preparation_directory(self, qrate=None, sample_id=None, stage='quench'):
        ''' load temperature energy data in preparation phase '''
        if qrate is None: qrate = self.qrate
        if   qrate >= 1.0e9:
            step_prep_id = 4
        elif qrate == 5.0e8:
            step_prep_id = 5
        elif qrate == 6.3e7:
            step_prep_id = 6
        elif qrate == 2.8e7:
            step_prep_id = 7
        elif qrate == 1e7:
            step_prep_id = 8
        elif qrate == 8e6:
            step_prep_id = 9
        elif qrate == 5e6:
            step_prep_id = 10

        if   stage == 'quench':
            stepid = step_prep_id + 0
        elif stage == 'anneal':
            stepid = step_prep_id + 0.5
        elif stage == 'heatup':
            stepid = 3

        if qrate < 1e9: 
            datadir = os.path.join(self.datarootdir, 'natom-%d'%self.natom, 'qrate-%.1e'%1e10)
        else:
            datadir = os.path.join(self.datarootdir, 'natom-%d'%self.natom, 'qrate-%.1e'%qrate)
        if sample_id is not None:
            datadir = os.path.join(datadir, 'sample_%d'%sample_id)
        log_file_name = os.path.join(datadir, 'lammps_scripts', 'log.%s')%stepid
        return log_file_name

    def preparation_data(self, log_file_name, verbose=False, loadrange=(0, None, 1)):
        log = lammps_log(log_file_name, verbose=verbose)
        nstart, nend, nskip = loadrange
        if nend is None: nend = log.nlen
        templist = np.reshape(log.get('Temp')[nstart:nend], (-1, nskip)).mean(axis=-1)
        steplist = np.reshape(log.get('Step')[nstart:nend], (-1, nskip)).mean(axis=-1)
        englist  = np.reshape(log.get('PotEng')[nstart:nend], (-1, nskip)).mean(axis=-1)
        return steplist, templist, englist

    def tensile_test_directories(self, relaxed='', unloading=None, unloading_to=0, unloading2=None, unloading_to2=0, digit=4):
        ''' For running on GPU only '''
        if unloading is not None:
            unloading_digit = r'%%.%df'%digit
            unloading_str = ('u_'+unloading_digit+'-%d')%(unloading, unloading_to)
        else:
            unloading_str = ''
        if relaxed == '0K':
            stepid = '3.0'
        elif relaxed == 'relaxed':
            stepid = '4'
        elif relaxed == '0K-unloading':
            stepid = '4'
            if unloading is not None:
                stepid = stepid + '.0'
        else:
            stepid = '3'

        dumpdir= os.path.join(self.datadir,'dump.%s'%(stepid+unloading_str))

        if stepid == '3':
            logdir = os.path.join(self.lammps_scripts, 'log.%s')%(stepid+unloading_str)
            dumpdir= os.path.join(dumpdir, 'dump.*.gz')
        else:
            logdir = os.path.join(self.datadir, 'log.%s', 'log.relax')%(stepid+unloading_str)
            dumpdir= os.path.join(dumpdir, 'relaxed.dump')

        if unloading2 is not None:
            unloading_digit = r'%%.%df'%digit
            unloading_str = ('u_'+unloading_digit+'-%d')%(unloading, unloading_to)
            unloading2_str = ('u_'+unloading_digit+'-%d')%(unloading2, unloading_to2)
            stepid = '3' + unloading_str + '.4.0' + unloading2_str
            logdir = os.path.join(self.datadir, 'log.%s'%(stepid), 'log.relax')
            dumpdir = os.path.join(self.datadir, 'dump.%s'%(stepid), 'relaxed.dump')

        return (logdir, dumpdir)

    def stress_strain_energy(self, relaxed='', unloading=None, unloading_to=0, unloading2=None, unloading_to2=0, skip_header=1, skip_step=2, strain_range=None, overwrite=False, digit=4, lognum=None, return_Ly=False, verbose=False, backup=False):
        ''' For running on GPU only '''
        log_file_name, dumpdir = self.tensile_test_directories(relaxed, unloading=unloading, unloading_to=unloading_to, unloading2=unloading2, unloading_to2=unloading_to2, digit=digit)
        if verbose: print(log_file_name)
        if strain_range is None: strain_range = self.plot_strain_range
        if os.path.exists(log_file_name) and backup: # backup logfile
            shutil.move(log_file_name, log_file_name + '_%d'%len(glob.glob(log_file_name + '_[0-9]')))
        if lognum is None:
            logfiles = glob.glob(log_file_name + '_[0-9]') # support at most 10 runs
            if os.path.exists(log_file_name): logfiles.append(log_file_name)
        else:
            logfiles = [log_file_name + '_%d'%lognum, ]
        if len(logfiles) == 0:
            if return_Ly:
                return None, None, None, None
            else:
                return None, None, None
        
        strain_avg = []
        stress_avg = []
        energy_avg = []
        Ly_avg     = []
        for log_file_name in logfiles:
            if verbose: print(log_file_name)
            saved_file = log_file_name + '.npz'
            if not overwrite and os.path.exists(saved_file):
                if verbose: print('loaded', os.path.basename(saved_file))
                strain, stress, energy = np.vsplit(np.load(saved_file)['arr_0'], 3)
                strain, stress, energy = strain.flatten(), stress.flatten(), energy.flatten()
                Ly = np.load(saved_file)['arr_1']
            else:
                if os.path.exists(saved_file): os.remove(saved_file)
                log = lammps_log(log_file_name, verbose=False, remove_duplicate=False)
                if relaxed == '':
                    if unloading is None:
                        skip_header = 10002 # skip the annealing
                    else:
                        skip_header = 1
                nstart, nend, nskip = skip_header, log.nlen, skip_step
                if verbose:
                    print(nstart, nend, nskip)
                totalstep = self.max_tensile_strain/100/self.erate*convert_s2ps/self.timestep
                if relaxed == '':
                    strain = np.array(log.get('Step'), dtype=int)[nstart:nend]
                    if unloading is not None: strain = (strain[-1] - strain)
                    strain = strain[::nskip]
                else:
                    strain = np.array(log.get('Step'), dtype=int)[nstart-1:nend:nskip]
                strain = (strain/totalstep)*self.max_tensile_strain
                stress =-np.array(log.get('Pyy'))[nstart:nend:nskip]/10000  # return GPa
                Ly = np.array(log.get('Ly'))[nstart:nend:nskip]
                if 'PotEng' in log.names:
                    engstr = 'PotEng'
                elif 'c_pe_all' in log.names:
                    engstr = 'c_pe_all'
                else:
                    engstr = 'c_pe'
                energy = np.array(log.get(engstr))[nstart:nend:nskip]
                if len(strain_avg) > 0 and strain.size < strain_avg[0].size: continue
                np.savez_compressed(saved_file, np.vstack([strain, stress, energy]), Ly)
                if verbose: print('saved', os.path.basename(saved_file))
            strain_range_id = np.logical_and(strain >= strain_range[0], strain <= strain_range[1])
            strain, stress, energy, Ly = (strain[strain_range_id], stress[strain_range_id], energy[strain_range_id], Ly[strain_range_id])
            s = self.plot_strain_skip
            strain_avg.append(strain[::s])
            stress_avg.append(stress[::s])
            energy_avg.append(energy[::s])
            Ly_avg.append(Ly[::s])

        if return_Ly:
            return np.array(strain_avg), np.array(stress_avg), np.array(energy_avg), np.array(Ly_avg)
        else:
            return np.array(strain_avg), np.array(stress_avg), np.array(energy_avg)

    def postproc_directories(self, run0K=False):
        postdir = os.path.join(self.datadir, 'post')
        if run0K:
            postdir = postdir+'_0K'
        dispdir = os.path.join(postdir, 'disp')
        phopdir = os.path.join(postdir, 'phop')
        metadir = os.path.join(postdir, 'meta')
        dispfiles = sorted(glob.glob(os.path.join(dispdir, 'disp.data.particle.*')), key=os.path.getmtime)
        phopfiles = sorted(glob.glob(os.path.join(phopdir, 'phop.data.particle.*')), key=os.path.getmtime)
        epotfiles = sorted(glob.glob(os.path.join(phopdir, 'epot.data.particle.*')), key=os.path.getmtime)
        metafile  = os.path.join(metadir, 'meta.data.full.npz')
        return (dispfiles, phopfiles, epotfiles, metafile)

    def save_config_to_lammps_old(self, node, cell, npzfile, lammpsfile, 
                              codedir=None, datadir=None, ovitodir=None):
        ''' Save generated Poisson-disk configuration to npz and lammps file (ovito 2.7)
        '''
        if codedir  is None:  codedir = self.codedir
        if datadir  is None:  datadir = self.datadir
        if ovitodir is None: ovitodir = self.ovitodir

        print('save configuration to file...')
        print('datadir = ', datadir)
        os.makedirs(datadir, exist_ok = True)
        pyscript   = os.path.join(codedir, 'npz2lammps_data.ovito.py')
        
        # save data in npz format
        np.savez_compressed(npzfile, node=node, cell=cell)
        # save data in lammps format using ovito
        cmdlist = [os.path.join(ovitodir, 'ovitos'), pyscript, npzfile, lammpsfile]
        print(' '.join(cmdlist))
        subprocess.run(cmdlist)

    def save_config_to_lammps(self, node, cell, npzfile, lammpsfile, 
                              codedir=None, datadir=None, ovitodir=None):
        ''' Save generated Poisson-disk configuration to npz and lammps file (ovito 3)
        '''
        from ovito.data import Particles, ParticleType, SimulationCell
        from ovito.data import DataCollection
        from ovito.io import export_file

        if codedir  is None:  codedir = self.codedir
        if datadir  is None:  datadir = self.datadir
        if ovitodir is None: ovitodir = self.ovitodir

        print('save configuration to file...')
        print('datadir = ', datadir)
        os.makedirs(datadir, exist_ok = True)

        # Create the data collection containing a Particles object:
        particles = Particles()
        data = DataCollection()
        data.objects.append(particles)

        # XYZ coordinates of the three atoms to create:
        pos = node['Position']

        # Create the particle position property:
        pos_prop = particles.create_property('Position', data=pos)

        # Create the particle type property and insert two atom types:
        type_prop = particles.create_property('Particle Type')
        type_prop.types.append(ParticleType(id = 1, name = 'Cu'))
        type_prop.types.append(ParticleType(id = 2, name = 'Zr'))
        type_prop[...] = node['Particle Type']

        # Create the simulation box:
        simcell = SimulationCell(pbc = (True, True, True))
        simcell[...] = cell
        data.objects.append(simcell)

        # Save the data to a lammps data file
        export_file(data, lammpsfile, 'lammps/data', atom_style='atomic', precision=17)

    def visualize_sample_prep(self, step_id_list, plot_groups=[range(2,3), range(4,11), range(11,14)], nframes=(None, None, 1000), fig=None, suptitle=None, figsize=(8, 6), log_verbose=True, bbox_to_anchor=None):
        if fig is None:
            fig = plt.figure(figsize=figsize); ax0 = fig.add_subplot(211);
            nfig= len(plot_groups)
            axs = [None, ]*nfig
            for i in range(nfig):
                axs[i] = fig.add_subplot(2, nfig, i+(nfig+1))
        else:
            ax = fig.axes; ax0 = ax[0]; axs = ax[1:]; nfig = len(axs)
        for step_id in step_id_list:
            log_file_name = os.path.join(self.lammps_scripts, 'log.%s'%step_id)
            if not os.path.exists(log_file_name): continue
            log = lammps_log(log_file_name, verbose=log_verbose)
            # print(log.names)
            nstart, nend, nskip = nframes; nstart = 0;
            if nend is None: nend  = (len(log.get('Step'))//nskip)*nskip

            ts = np.reshape(log.get('Temp')[nstart:nend], (-1, nskip)).mean(axis=-1)
            steps = np.array(log.get('Step')); #steps -= steps[0]; 
            ax0.plot(np.reshape(steps[nstart:nend], (-1, nskip)).mean(axis=-1), ts, 'C0')
            ax0.legend(['Temp(K)', ]); ax0.set_xlabel('Step');

            for ifig in range(nfig):
                # print('axis %d:'%ifig, [log.names[i] for i in plot_groups[ifig]])
                for i in plot_groups[ifig]:
                    vs = np.reshape(log.get(log.names[i])[nstart:nend], (-1, nskip)).mean(axis=-1)
                    if log.names[i] == 'PotEng':
                        log.names[i] = 'pe/atom'
                        vs = vs/self.natom
                    if log.names[i] in ['Pxx', 'Pyy', 'Pzz', 'Pxy', 'Pyz', 'Pxz', 'Press']:
                        log.names[i] = log.names[i] + '(GPa)'
                        vs = vs/10000 # from bar to GPa
                    # axs[ifig].plot(ts, vs, 'C%d'%(i - 2), label=log.names[i]+'-%s'%step_id)
                    axs[ifig].plot(ts, vs, label=log.names[i]+'-%s'%step_id)
                axs[ifig].legend(bbox_to_anchor=bbox_to_anchor); axs[ifig].set_xlabel('Temp(K)');
        if suptitle is not None:
            fig.suptitle(suptitle)
        return fig

    def generate_initial_config(self, initsize=(58, 58, 58), natom=None, x1 =None, filename='init_config', options={'nDim': 3, 'rmin': 2.3, 'rmax': 2.5, 'ntrial': 30}):
        ''' Generate random 3D glass configuration using Poisson-disk sampling
        '''
        if natom is None: natom = self.natom
        if x1    is None: x1    = self.x1 
        fullpath = os.path.join(self.datadir, filename)
        npzfile = fullpath + '.npz'
        self.init_config_file = fullpath + '.lmp'

        # If configuration already generated, return
        if os.path.exists(self.init_config_file):
            print('Initial configuration %s already generated!'%self.init_config_file)
            if os.path.exists(npzfile):
                node = np.load(npzfile)['node']
                cell = np.load(npzfile)['cell']
                return node, cell
            else:
                return
        
        # **Set up atom number, box size, and components of Metallic Glass**
        print('Generate random 3D glass configuration using Poisson-disk sampling...')
        x2 = 1 - x1 
        n1, n2 = int(natom*x1 ), int(natom*x2)
        print('    n1 = ', n1, 'n2 = ', n2)

        # **Generate initial configuration using [fast poisson disk sampling](http://devmag.org.za/2009/05/03/poisson-disk-sampling)**
        Lx, Ly, Lz = initsize
        rgrid = Grid3D(initsize, options['rmax'])
        coords = sample_poisson_3d(Lx, Ly, Lz, rgrid, options['ntrial']) # sample points until the entire space is filled up
        np.random.shuffle(coords)
        print('Poisson-disk sampling generates {} points'.format(len(coords)))

        ptype = np.ones(natom, dtype=int); ptype[n1:] = 2;
        np.random.shuffle(ptype)
        node = np.empty(natom, dtype=[('Particle Identifier', int), ('Particle Type', int), ('Position', float, 3)])
        node['Particle Identifier'] = np.arange(natom, dtype=int) + 1
        node['Particle Type'] = ptype
        node['Position'][:, :options['nDim']] = np.array(coords[:natom])
        cell = [[Lx,0,0,0], [0,Ly,0,0], [0,0,Lz,0]]

        # Save configuration in npz and lammps format
        print('Saving configuration file %s...'%self.init_config_file)
        self.save_config_to_lammps(node, cell, npzfile=npzfile, lammpsfile=self.init_config_file)

        return node, cell

    def generate_lammps_input(self, templatefile, infile = None, log_file_name=None, replace_str={}, overwrite=False,
                              paramsfile = None, variables = None, potential = None,                              
                              submitfile='submit.sh', queue_system='slurm', cluster_info=None):
        ''' Generate a copy of lammps input file into the data directory
            1. Copy `templatefile` to `lammps_scripts` folder
            2. Create `paramsfile` using `variable` and `potential` information
            3. Create `submitfile` using `cluster_info` information
        '''

        # Copy `templatefile` to `lammps_scripts` folder
        if infile is None: infile = os.path.basename(templatefile)
        scripts   = self.lammps_scripts
        inputcopy = os.path.join(scripts, infile)
        if paramsfile is not None:    replace_str['in.params'] = os.path.join(self.lammps_scripts, paramsfile)
        if log_file_name is not None: replace_str['log.lammps']= os.path.join(self.lammps_scripts, log_file_name)
        if (not overwrite) and os.path.exists(inputcopy):
            print('%s has already been created.'%inputcopy)
        else:
            with open(inputcopy, 'w') as foutput:
                with open(templatefile, 'r') as finput:
                    inputstr = finput.read()
                for key in replace_str:
                    inputstr = inputstr.replace(key, replace_str[key])
                foutput.write(inputstr)
            print('Creating %s done'%inputcopy)

        # Create `paramsfile` using `variable` and `potential` information
        if paramsfile is not None:
            paramsfile = os.path.join(self.lammps_scripts, paramsfile)
            if (not overwrite) and os.path.exists(paramsfile):
                print('%s has already been created.'%paramsfile)
            else:
                with open(paramsfile, 'w') as f:
                    if variables is not None:
                        for key, value in variables.items():
                            if type(value) is str:
                                print('variable %s string'%key, value, file=f)
                            else:
                                print('variable %s equal'%key,  value, file=f)
                    if potential is not None:
                        if 'init_config' in potential:
                            print('read_data', potential['init_config'], file=f)
                        elif 'restart_file' in potential:
                            print('read_restart', potential['restart_file'], file=f)
                        else:
                            raise KeyError('No input parameter!')
                        print('pair_style %s'%potential['pair_style'], file=f)
                        print('pair_coeff * * ', potential['potfile'], ' '.join(potential['elements']), file=f)
                print('Creating %s done'%paramsfile)

        # Create `submitfile` using `cluster_info` information
        if cluster_info is not None:
            if queue_system == "slurm":
                if os.path.dirname(submitfile) == '': submitfile = os.path.join(scripts, submitfile)
                header = generate_header_str(cluster_info)
                if (not overwrite) and os.path.exists(submitfile):
                    print('%s has already been created.'%submitfile)
                else:
                    if self.use_gpu:
                        gpu_insert = '-sf gpu -pk gpu 1'
                    else:
                        gpu_insert = ''
                    if 'nodes' in cluster_info and 'tasks-per-node' in cluster_info:
                        ncpu = cluster_info['nodes']*cluster_info['tasks-per-node']
                        with open(submitfile, 'w') as f:
                            print(header, file=f)
                            print('mpirun -np %d'%ncpu, self.lammpsbin, gpu_insert, '-in', inputcopy, file=f)
                            print('wait', file=f)
                    elif 'nodes' in cluster_info:
                        with open(submitfile, 'w') as f:
                            print(header, file=f)
                            print('mpirun -np 1', self.lammpsbin, gpu_insert, '-in', inputcopy, file=f)
                            print('wait', file=f)
                    else:
                        with open(submitfile, 'w') as f:
                            print(self.lammpsbin, gpu_insert, '-in', inputcopy, file=f)
                    print('Creating %s done'%submitfile)
            else:
                raise NotImplementedError("Unknown queue_system, submit file not created!")

    def generate_analysis_code(self, n_frame_total, n_frame_slice, n_particle_total, n_particle_slice, paramsfile = 'setup.txt', 
                               variables={}, overwrite=False, submitfile='submit.sh', queue_system='slurm', cluster_info=None):
        ''' Generate a copy of analysis code into the data directory
            1. Create `submitfile` using `cluster_info` and frame slice information
        '''
        # if max_tensile_strain is None: max_tensile_strain = self.max_tensile_strain
        frame_m = np.ceil(n_frame_total/n_frame_slice).astype(np.int)
        particle_m = np.ceil(n_particle_total/n_particle_slice).astype(np.int)

        # Copy `templatefile` to `analysis_scripts` folder
        for infile in ['postproc_utils.py', 'postproc.ovito.py']:
            templatefile    = os.path.join(self.codedir, infile)
            inputcopy       = os.path.join(self.analysis_scripts, infile)
            if os.path.exists(inputcopy):
                print('%s exists'%inputcopy)
            else:
                shutil.copyfile(templatefile, inputcopy)
                print('%s has been copied'%inputcopy)

        # create setup file
        paramsfile = os.path.join(self.analysis_scripts, paramsfile)
        if (not overwrite) and os.path.exists(paramsfile):
            print('%s has already been created.'%paramsfile)
        else:
            with open(paramsfile, 'w') as f:
                for key, value in variables.items():
                    print('%s %s'%(key, value), file=f)
            print('Creating %s done'%paramsfile)

        # Create `submitfile` using `cluster_info` information
        if cluster_info is not None:
            if queue_system == "slurm":
                if os.path.dirname(submitfile) == '': submitfile = os.path.join(self.analysis_scripts, submitfile)
                header = generate_header_str(cluster_info)
                if (not overwrite) and os.path.exists(submitfile):
                    print('%s has already been created.'%submitfile)
                else:
                    with open(submitfile, 'w') as f:
                        print(header, file=f)
                        # print('cd', self.analysis_scripts, file=f)
                        print('# Step 1. Calculate displacement trajectory:',   file=f)
                        for i in range(n_frame_slice):
                            print(os.path.join(self.ovitodir, 'ovitos'), inputcopy, '-i %s'%paramsfile, '-n 1 -s %d %d %d &'%(i, n_frame_slice, frame_m), file=f)
                        print('wait', file=f)
                        print('# Step 2. Calculate phop:',                      file=f)
                        for i in range(n_frame_slice):
                            print(os.path.join(self.ovitodir, 'ovitos'), inputcopy, '-i %s'%paramsfile, '-n 2 -s %d %d %d &'%(i, n_frame_slice, frame_m), file=f)
                        print('wait', file=f)
                        print('# Step 1.5. change frame slices (%d, %d) to particle slices (%d, %d) for displacement'%(n_particle_total, frame_m, particle_m, n_frame_total), file=f)
                        print(os.path.join(self.ovitodir, 'ovitos'), inputcopy, '-i %s'%paramsfile, '-n 1.5 -s %d %d 0 &'%(n_frame_slice, n_particle_slice), file=f)
                        print('# Step 2.5. change frame slices (%d, %d) to particle slices (%d, %d) for phop'%(n_particle_total, frame_m, particle_m, n_frame_total), file=f)
                        print(os.path.join(self.ovitodir, 'ovitos'), inputcopy, '-i %s'%paramsfile, '-n 2.5 -s %d %d 0 &'%(n_frame_slice, n_particle_slice), file=f)
                        print('wait', file=f)
                        print('# Step 3. pick all soft and hard particles from phop data', file=f)
                        print(os.path.join(self.ovitodir, 'ovitos'), inputcopy, '-i %s'%paramsfile, '-n 3 -s -1 %d %d'%(n_particle_slice, particle_m), file=f)
                    print('Creating %s done'%submitfile)
            else:
                raise NotImplementedError("Unknown queue_system, submit file not created!")

    def submit_job(self, submitfile, checkpoint_success, checkpoint_continue=None, queue_system='slurm', cluster_info=cluster_info_default):
        ''' This function does the following operations:
            1. Submit `submitfile` to the `queue_system`;
            2. If already submitted, check its running status;
            3. Return the current running status of the job `submitfile`
        '''
        # check whether `submitfile` exists
        if   os.path.exists(os.path.join(self.lammps_scripts, submitfile)):
            submitfile = os.path.join(self.lammps_scripts, submitfile)
        elif os.path.exists(os.path.join(self.analysis_scripts, submitfile)):
            submitfile = os.path.join(self.analysis_scripts, submitfile)
        else:
            raise FileNotFoundError(submitfile, 'does not exist!')

        if queue_system == 'slurm':
            # Submit `submitfile` to the `queue_system`
            status_file = submitfile + '.status'
            status = {}
            if not os.path.exists(status_file): # not submitted, submit the job
                os.chdir(os.path.dirname(submitfile)) # make sure slurm stdout is in the data folder
                cmdlist = ['sbatch', submitfile]
                output = subprocess.run(cmdlist, stdout=subprocess.PIPE).stdout.decode()
                if output.split()[:-1] == ['Submitted', 'batch', 'job']:
                    status['jobid'] = output.split()[-1]
                else:
                    raise ValueError('Fail submitting job %s!'%submitfile)
            else:                               # job submitted, return status
                with open(status_file, 'r') as fstatus:
                    for line in fstatus:
                        status[line.split()[0]] = line.split()[1]
            
            # Check if the job is still running
            cmd = 'squeue -u %s'%os.environ['USER']
            output = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode()
            status['status'] = 'submitted'

            for line in output.splitlines():
                if len(line.split()) > 0 and line.split()[0] == status['jobid']:
                    status['status'] = 'running'
                    break

            # Check the stdout file and checkpoint_success file to determine 'finish' status
            if not status['status'] == 'running':
                if os.path.exists(checkpoint_success):
                    status['status'] = 'finished'
                else:
                    status['status'] = 'failed'
            else:
                if (checkpoint_continue is not None) and os.path.exists(checkpoint_continue):
                    status['status'] = 'continue'

            # Write the status
            with open(status_file, 'w') as fstatus:
                for key in status:
                    print(key, status[key], file=fstatus)
        else:
            raise NotImplementedError("Unknown queue_system, submit file not created!")

        return status

    def print_instructions(self, status, log_file_name=None, submitfile=None, checkpoint_continue=''):
        ''' Print instructions of different job status
            return True to continue the next step
            return False to pause the python script
        '''
        if status['status'] == 'no_submit':
            print('** do_submit set to False, not submitting jobs. **')
            return True
        elif status['status'] == 'finished':
            print('** Job %s successfully finished. Continue... **'%status['jobid'])
            return True
        elif status['status'] == 'submitted':
            print('** Job %s is submitted, but fail to run (cannot find slurm-%s.out). **'%(status['jobid'], status['jobid']))
        elif status['status'] == 'running':
            print('** Job %s is running. Please wait for the job to finish. **'%status['jobid'])
        elif status['status'] == 'continue':
            print('** Job %s is running. Checkpoint %s exists. Continue to next step... **'%(status['jobid'], checkpoint_continue))
            return True
        elif status['status'] == 'failed':
            print('** Job %s is failed, please check the output files:'%status['jobid'])
            print('**     cat %s'%os.path.join(self.lammps_scripts, 'slurm-%s.out'%status['jobid']))
            print('** If you want to rerun the job, delete following files and run the python script again:')
            print('**     rm %s'%os.path.join(self.lammps_scripts, 'slurm-%s.out'%status['jobid']))
            if log_file_name:
                print('**      rm %s'%os.path.join(self.lammps_scripts, log_file_name))
            if submitfile:
                print('**      rm %s.status'%os.path.join(self.lammps_scripts, submitfile))
                print('** Job %s is failed **'%submitfile)
        else:
            print('** Unknown status %s **'%status['status'])
        return False
