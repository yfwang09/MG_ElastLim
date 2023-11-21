This folder contains the LAMMPS scripts, initial configurations, and the pair potentials for the MD simulations.

## LAMMPS pair potentials

eam/fs potentials for Cu-Zr and Ni-Nb metallic glass system are used. See the header of each file for the reference.

* CuZr.eam.fs
* NiNb.eam.fs
* Cu-Zr.eam.fs
* Cu-Zr_2.eam.fs
* Cu-Zr_3.eam.fs
* Cu-Zr_4.eam.fs

## LAMMPS input script templates

See the `data`` folder for the final generated lammps scripts using these templates.

* in.*

## Initial configurations

Configurations used for the tensile test simulations. File name includes the composition and the (effective) cooling rate (K/s).

The ${\rm Cu_{64.5}Zr_{35.5}}$ and ${\rm Ni_{60}Nb_{40}}$ samples are generated using the slow cooling schedule described in the method and supplementary note 2 of the paper.
