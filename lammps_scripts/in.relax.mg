# Relaxation of ZrCu metallic glass configurations

label loopi
variable i loop ${istart} ${iend}
    clear

    units           metal
    boundary        p p p
    atom_style      atomic

    include         in.params
    log             ${logdir}/log.relax.${i}
    variable        filenum     equal   ${i}*${dump_freq}
    read_dump       ${inputdir}/dump.${filenum}.gz ${filenum} x y z box yes replace yes
    neigh_modify    every 1 delay 0 check yes
    compute         pe all pe/atom
    thermo          1000
    thermo_style    custom step temp pe etotal press pxx pyy pzz pxy pxz pyz lx ly lz vol
    thermo_modify   format float %23.17e

    dump            minimization all custom ${maxiter_minimize} ${relaxdir}/relaxed.${filenum}.dump.gz id type x y z vx vy vz c_pe
    dump_modify     minimization first no

    if "${i} == 0"  then "reset_timestep 1"  # To avoid saving step 0
    minimize	    ${etol_minimize} ${ftol_minimize} ${maxiter_minimize} ${maxeval_minimize}
    if "${write_restart} == 1" then &
       "write_restart ${restart_file}"

next i
jump in.relax.mg loopi
