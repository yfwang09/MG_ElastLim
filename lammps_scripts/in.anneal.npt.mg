# LAMMPS input file template of CuZr metallic glass tensile test 

log             log.lammps
units           metal
boundary        p p p

atom_style      atomic

include         in.params

timestep        ${timestep}

group Cu type 1
group Zr type 2
compute msd all msd com yes
compute msd_Cu Cu msd com yes
compute msd_Zr Zr msd com yes

thermo          ${thermo_step}
thermo_style    custom step temp pe etotal press pxx pyy pzz pxy pxz pyz lx ly lz vol c_msd[4] c_msd_Cu[4] c_msd_Zr[4]
# thermo_modify   format float %23.17e

if "${do_minimize} == 1" then &
    "minimize	    ${etol_minimize} ${ftol_minimize} ${maxiter_minimize} ${maxeval_minimize}"

# change temperature from T0 to T1
if "${set_velocity} == 1" then &
	"velocity        all create ${T0} ${rand_seed} dist gaussian"

if "${Pdamp} > 0.0" then & 
   "fix         1 all npt temp ${T0} ${T1} ${Tdamp} iso 0.0 0.0 ${Pdamp}" &
else &
   "fix         1 all nvt temp ${T0} ${T1} ${Tdamp}"
compute         PE all pe/atom

dump            1 all custom ${dump_freq} ${dumpdir}/anneal.*.dump.gz id type x y z vx vy vz c_PE
if "${restart_freq} > 0" then&
    "restart    ${restart_freq} ${dumpdir}/anneal.*.restart"

run             ${total_steps_1}
write_restart   ${restart_file_1}
unfix           1

# change temperature from T1 to T2
if "${Pdamp} > 0.0" then & 
   "fix         2 all npt temp ${T1} ${T2} ${Tdamp} iso 0.0 0.0 ${Pdamp}" &
else &
   "fix         2 all nvt temp ${T1} ${T2} ${Tdamp}"

run             ${total_steps_2}
write_restart   ${restart_file_2}
unfix           2
