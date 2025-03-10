Notes
=====

1. **Testing GitHub Workflow**  
   This section is to test the GitHub Actions workflow.

2. **RVIC Compilation Note**  
   Please note that **RVIC** should not be compiled with **VIC** if you wish to use parallel processing (e.g., `mpiexec`).

   There are two types of compilation:

   - **Compile VIC with RVIC**:  
     You can run VIC with RVIC and set different timesteps.
   - **Compile VIC without RVIC**:  
     You can run VIC in parallel, but you will need to run RVIC separately (`rvic.convolution.convolution`).  
     Make sure to prepare the `rvic.convolution.cfg` configuration file.  
     Additionally, ensure that the VIC output timestep (daily or hourly) matches the UHBOX timestep (86400 seconds for daily, 3600 seconds for hourly).