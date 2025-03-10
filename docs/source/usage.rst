Usage
=====

After installation, you can use the package in the following sequence:

1. **Build DPC** (`build_dpc`)
2. **Build Domain** (`build_Domain`)
3. **Build Parameters** (`build_Param`)
4. **Perform Hydroanalysis** (`build_hydroanalysis`)
5. **Plot Basin Map** (`plot_Basin_map`)  
   (Note: You must first run `hydroanalysis_for_basin`)
6. **Build Meteorological Forcing** (`build_MeteForcing`) or (`build_MeteForcing_nco`)
7. **Build RVIC Parameters** (`build_RVIC_Param`)
8. **Build Global Parameters** (`build_GlobalParam`)
9. **Calibrate the Model** (`calibrate`)
10. **Plot VIC Results** (`plot_VIC_result`)