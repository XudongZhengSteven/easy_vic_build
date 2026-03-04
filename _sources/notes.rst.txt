Notes
=====

RVIC and parallel runs
----------------------

When VIC is compiled with RVIC coupling, parallel execution behavior may differ
from standalone VIC runs. In some workflows, users run VIC in parallel first
and then run RVIC convolution separately.

If using separate RVIC convolution, ensure:

- VIC output timestep matches UHBOX timestep.
- ``rvic.convolution.cfg`` is configured with consistent temporal settings.

Project status
--------------

The project is under active development. APIs and workflow defaults may evolve
across versions.
