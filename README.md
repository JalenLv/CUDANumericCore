# CUDANumericCore

CUDANumericCore (CNC) is a primitive library that implements basic numeric operations on CUDA. It is designed to be used
as a building block for more complex libraries that require basic numeric operations on CUDA. CNC is written in C++ and
CUDA C++.

Honestly, this library is a homebrew implementation of the standard CUDA libraries. It is not intended to be used in
production code. It is just a learning project.

By now, CNC implements the following sub-libraries:

- `cncBLAS`: A CUDA implementation of the Basic Linear Algebra Subprograms (BLAS) library.

## Installation

---

___Note: If you encounter any problems during the installation, please feel free to open an issue, or contact me
at [this email](mailto:sejalenlv@mail.scut.edu.cn). I will try my best to help you.___

---

### Prerequisites

To build the project, you need to have the following tools installed:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Built upon CUDA 12.4)
- C++ Compiler
- Git
- [CMake](https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3-linux-x86_64.sh) (Built upon CMake
  3.29.2. Minimum version required is 3.28)
- GNU make or
- Ninja (Optional)

### Ubuntu (or WSL2)

This repo is tested on native Ubuntu 22.04.04 with RTX4060. If you are running Windows, it is recommended to test this
repo under WSL2.

To install the library, you need to clone the repository and build the project using CMake.

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

The implementation of this repo relies on some of the recent features of CUDA, so if it complains about missing matched CUDA APIs, like `atomicAdd`, you may need to update your CUDA toolkit or try out this repo on a more recent GPU.

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

## Sub-libraries

### cncBLAS

Due to my insufficient knowledge of complex numbers, the `cncBLAS` library supports real numbers functions and some
complex numbers functions that is within my knowledge.

All the standard level one BLAS functions except for some complex numbers functions are implemented. Below is a list of
the functions that are not yet implemented:

- `rot`
- `rotg`
- `rotm`
- `rotmg`

I am planning to implement these functions in the future, and I am currently working on the level two BLAS functions.
Below are the level two BLAS functions that are implemented:

- `gbmv`
- `gemv`
- `ger`
