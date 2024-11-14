# TODO

## build_RVIC_Param.py

1. buildFlowDirection_wbw in linux

2. build

3. RVIC modified

## calibrate.py

1. Spotpy

2. DEAP

## build_MeteForcing_nco.py

1. totally on linux

## build_GlobalParam.py

1. check the params

## python pkgs workflow

1. test github workflow

!note: not compile RVIC with VIC, otherwise you will not able to use the parallel (mpiexec)

two types:
compile VIC with RVIC:
you can run VIC with RVIC, and set different timestep

compile VIC witout RVIC:
you can run VIC under parallel, but you need to run RVIC additionally (rvic.convolution.convolution)
you need prepare rvic.convolution.cfg file
note that the VIC output dt (daily, hourly) should be same with the UHBOX dt (86400, 3600)
