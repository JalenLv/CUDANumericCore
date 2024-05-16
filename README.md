# CUDANumericCore

CUDANumericCore (CNC) is a primitive library that implements basic numeric operations on CUDA. It is designed to be used
as a building block for more complex libraries that require basic numeric operations on CUDA. CNC is written in C++ and
CUDA C++.

Honestly, this library is a homebrew implementation of the standard CUDA libraries. It is not intended to be used in
production code. It is just a learning project.

By now, CNC implements the following sublibraries:

- `cncBLAS`: A CUDA implementation of the Basic Linear Algebra Subprograms (BLAS) library.

## Installation

### Prerequisites

To build the project, you need to have the following tools installed:

- CUDA Toolkit
- C++ Compiler
- Git
- CMake
- Make or
- Ninja (Optional)

### Ubuntu

This repo is tested on Ubuntu 22.04.04. To install the library, you need to clone the repository and build the project
using CMake.

```bash
git clone --recursive https://github.com/JalenLv/CUDANumericCore.git
```

The `--recursive` flag is used to clone the submodules of the repository. If you forget to add this flag, you can
run `git submodule update --init --recursive` to clone the submodules.

Then `cd` into the project directory and create a build directory.

```bash
cd CUDANumericCore
mkdir build
cd build
```

Then run CMake to configure the project. If you have Ninja installed, you can use it as the generator to speed up the
compilation.

```bash
cmake .. [-G Ninja]
make      (if you are using makefile, or)
ninja     (if you are using ninja)
```

To run unit tests, you can run the following command: `./tests/blas/cncblasTest`, or simply `ctest` if you don't want to
bother with the test details.

The library is built as a shared library, and the shared library is located in the `build/core/<sublib>` directories,
named `lib<sublib>.so`.

You can use the library by adding it as a submodule and linking it to your own project.

```cmake
add_subdirectory(path/to/CUDANumericCore)
target_link_libraries(your_target <name_of_sublib>)
```

Or you can copy the shared library and its corresponding headers to your project directory and link it to your project.
Headers are located in the `core/<sublib>` and the `core/<sublib>/include` directory. You should copy both the public
and internal headers.

## Sublibraries

### cncBLAS

Due to my insufficient knowledge of complex numbers, the `cncBLAS` library supports real numbers functions and some
complex numbers functions that is within my knowledge.

All the standard level one BLAS functions except for some complex numbers functions are implemented.

Below is a list of the functions that are not yet implemented:

- `rot`
- `rotg`
- `rotm`
- `rotmg`

I am planning to implement these functions in the future, and I am currently working on the level two BLAS functions.
