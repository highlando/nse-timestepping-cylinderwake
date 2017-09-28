NSE time integration schemes tested on the cylinder wake
---

Code to test the schemes 'IMEX-Euler', 'Projection2', and 'SIMPLE' for the time integration of incompressible Navier-Stokes equation. These methods are all of first order and semi-explicit in the sense that the nonlinearity is treated explicitly. The test can be conducted for direct solves and iterative solves for varying tolerances.

To compute reference solution, the fully implicit trapezoidal rule which is of order 2 is used.

To start the tests, run 
```
python3 check_cyl_errs.py
```
with the parameters set accordingly.

### Dependencies

 * numpy (v1.11.0)
 * scipy (v0.19.1)
 * fenics/dolfin (v2017.2)
