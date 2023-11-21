# MD simulation of CuZr metallic glass tensile test

log             log.lammps
units           metal
boundary        p p p

atom_style      atomic

include         in.params

timestep        ${timestep}
thermo          ${thermo_step}
thermo_style    custom step temp pe etotal press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify   format float %23.17e

## NVT equilibration with averaged box size ##
run             0
if  "${change_box_init}==1" then &
   "change_box  all x final 0.0 ${equil_lx} y final 0.0 ${equil_ly} z final 0.0 ${equil_lz} boundary p p p remap units box" &
elif ${change_box_init}==2 & 
   "read_dump   ${load_dump_init} ${load_dump_idx} x y z vx vy vz box yes replace yes"

fix             equil all nvt temp ${Ttarget} ${Ttarget} ${Tdamp}
run             ${total_steps_1}
unfix           equil

write_restart   ${restart_file_1}

############## tensile test ##################
fix             1 all deform ${N_deform} y trate ${ey_trate} x trate ${ex_trate} z trate ${ez_trate} remap v units box
fix             2 all nvt/sllod temp ${Ttarget} ${Ttarget} ${Tdamp}

# compute temperature and stress
compute         PE all pe/atom

thermo          ${thermo_step}
thermo_style    custom step c_2_temp pe etotal press pxx pyy pzz pxy pxz pyz lx ly lz vol

# dump atom configurations
dump            1 all custom ${dump_freq} ${dumpdir}/dump.*.gz id type x y z vx vy vz c_PE
dump_modify     1 first yes format float %23.17e
if "${restart_freq} > 0" then &
    "restart    ${restart_freq} ${dumpdir}/tensile.*.restart"

# tensile test
reset_timestep  0
run             ${total_steps_2}

write_restart   ${restart_file_2}
minimize        0 1e-11 100000000 10000000000
