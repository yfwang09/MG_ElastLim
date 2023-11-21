# NEB simulation of CuZr metallic glass

units           metal

atom_style      atomic
atom_modify     map array
boundary        p p p
atom_modify     sort 0 0.0

read_data       initial.mg.gz

# choose potential

pair_style      eam/fs
pair_coeff      * *  CuZr.eam.fs Cu Zr

include         in.neb.fixatoms

variable        u equal part
# dump            events all custom 100000 dump.neb.*.mg.$u id type x y z

reset_timestep  0

compute         pe_peratom all pe/atom
compute         pe_all all reduce sum c_pe_peratom
compute         sigma all stress/atom NULL
compute         sigma_all all reduce sum c_sigma[*]
fix             1 all neb 1.0 

thermo          100
thermo_style    custom step temp pe c_pe_all press pxx pyy pzz pxy pxz pyz lx ly lz vol c_sigma_all[1] c_sigma_all[2] c_sigma_all[3] c_sigma_all[4] c_sigma_all[5] c_sigma_all[6]
thermo_modify   format float %23.17e

# run NEB for 20000 steps or to force tolerance

timestep        0.001
min_style       fire   

neb             0.0 0.01 100 100 10 final final.mg.gz
# if "${savedump} > 0" then "write_dump      all custom dump.neb.final.mg.$u.gz id type x y z c_pe_peratom modify format float %23.17e"
if "${savedump} > 0" then "write_dump      all custom dump.neb.final.mg.$u.gz id type x y z c_pe_peratom c_sigma[1] c_sigma[2] c_sigma[3] c_sigma[4] c_sigma[5] c_sigma[6] modify format float %23.17e"
