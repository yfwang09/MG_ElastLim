# MG_ElastLim

[![DOI](https://zenodo.org/badge/721847943.svg)](https://zenodo.org/doi/10.5281/zenodo.10184744)

This repository contains the simulation and analysis scripts, and the data for plotting in the paper:

Yifan Wang, Jing Liu, Jian-Zhong Jiang, and Wei Cai, Anomalous temperature dependence of elastic limit in metallic glasses, *accepted*

**Abstract:**
Understanding the atomistic mechanisms of inelastic deformation in metallic glasses (MGs) remains challenging due to their amorphous structure, in which local carriers of plasticity cannot be easily defined. Using molecular dynamics (MD) simulations, we analyzed the onset of inelastic deformation in CuZr MGs, specifically the temperature dependence of the elastic limit, in terms of localized shear transformation (ST) events. We find that although the ST events initiate at lower strain with increasing temperature, the elastic limit increases with temperature in certain temperature ranges. We explain this anomalous behavior through the framework of an energy-strain landscape (ESL) constructed from high-throughput strain-dependent energy barrier calculations for the competing ST events identified in the MD simulations. The ESL reveals that the anomalous temperature dependence of the elastic limit is caused by the transition of ST events from irreversible to reversible with increasing temperature. The critical temperature for this transition can be predicted by a strain-independent parameter of each ST event called the eigen-barrier, defined at the specific strain where the forward and backward barriers are equal. This work demonstrates the importance of accounting for the strain effects on the energy landscape and illustrates the advantage of ESL in understanding the fundamental deformation mechanisms of MGs.

## Structure of the code

* `python`: The workflow for the MD simulations and the post-processing of the data.
* `lammps_scirpts`: The LAMMPS input script templates, pair potentials, and initial configurations for the MD simulations.
* `data`: Due to the size limit, the raw simulation data (dump files, log files) are not saved. Only the simulation scripts for reproducing the simulation are provided.
* `plots`: The scripts and processed data for generating the plots in the paper.

Separate README files are provided in each folder for more details.

## Dependencies (tested versions)

* Python 3.10
* LAMMPS 3Mar2020
* numpy 1.21.2
* scipy 1.4.1
* matplotlib 3.4.3

## Contribution guidelines

Please contact the authors if you have questions or want to contribute.

## Authors

    Yifan Wang (yfwang09@stanford.edu)
    Jing Liu (liujinger@zju.edu.cn)
    Jian-Zhong Jiang (jiangjz@zju.edu.cn)
    Wei Cai (caiwei@stanford.edu)
