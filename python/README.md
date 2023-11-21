This folder contains the python scripts of the workflow for the MD simulations and the post-processing of the data.

## Structure of the code

* `Workflow_sample_preparation.py`: The workflow for preparing the initial configurations for the MD simulations.
* `Workflow_tensile_test.py`: The workflow for the tensile test simulations.
* `Workflow_NaDM.py`: The workflow for the non-affine displacement matrix analysis.
* `Workflow_volume_analysis.py`: The workflow for the free volume analysis.

See `example.shell` for examples of running the workflow.

## Dependencies

* `log.py`: adapted from the [lammps/pizza](https://github.com/lammps/lammps/tree/develop/tools/python/pizza) code, which is used to read the log file of LAMMPS.