# easy_vic_build

This is a open-source Python package for VIC model deployment  
![Architecture][docs/evb_architecture.tif]

## Installation

To install the package, run the following command:  
pip install easy_vic_build  
pip install easy_vic_build[nco]  
pip install easy_vic_build[rvic]  
pip install easy_vic_build[nco_rvic]

or  
pip install .whl  
pip install .whl[nco]  
pip install .whl[rvic]  
pip install .whl[nco_rvic]

based on the environment required by the Users

For development purposes, you can install it from the repository:  
git clone https://github.com/XudongZhengSteven/easy_vic_build  
cd easy_vic_build  
pip install -e .

## Usage

0. install
1. build_dpc
2. build_Domain
3. build_Param
4. build_hydroanalysis
5. plot_Basin_map (do hydroanalysis_for_basin first)
6. build_MeteForcing (or build_MeteForcing_nco)
7. build_RVIC_Param
8. build_GlobalParam
9. calibrate
10. plot_VIC_result

## Features

## Documentation

For detailed documentation, please refer to the documentation website or view the API documentation.

## Contributing

## Lincese

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- author: XudongZheng
- Email: zhengxd@sehemodel.club
- Github: @XudongZhengSteven

## note

1. test github workflow

!note: not compile RVIC with VIC, otherwise you will not able to use the parallel (mpiexec)

two types:
compile VIC with RVIC:
you can run VIC with RVIC, and set different timestep

compile VIC witout RVIC:
you can run VIC under parallel, but you need to run RVIC additionally (rvic.convolution.convolution)
you need prepare rvic.convolution.cfg file
note that the VIC output dt (daily, hourly) should be same with the UHBOX dt (86400, 3600)

## Citation
