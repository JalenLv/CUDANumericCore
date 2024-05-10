# CUDANumericCore

CUDANumericCore (CNC) is a primitive library that implements basic numeric operations on CUDA.

It is designed to be used as a building block for more complex libraries that require basic numeric operations on CUDA.
CNC is written in C++ and CUDA C++.

Honestly, this library is a homebrew implementation of the standard CUDA libraries. It is not intended to be used in
production code. It is just a learning project.

By now, CNC implements the following libraries:

- `cncBLAS`: A CUDA implementation of the Basic Linear Algebra Subprograms (BLAS) library.

## cncBLAS

Due to my insufficient knowledge of complex numbers, the `cncBLAS` library supports real numbers functions and some
complex numbers functions that is within my knowledge.

All the standard level one BLAS functions except for some complex numbers functions are implemented.

Below is a list of the functions that are not yet implemented:

- `rot`
- `rotg`
- `rotm`
- `rotmg`

I am planning to implement these functions in the future, and I am currently working on the level two BLAS functions.
