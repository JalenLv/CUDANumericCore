# CUDANumericCore

CUDANumericCore (CNC) is a primitive library that implements basic numeric operations backed on CUDA C/C++. It is designed to be used
as a building block for more complex libraries that require basic numeric operations on CUDA.

Honestly, this library is a homebrew implementation of the official CUDA libraries. It is not intended to be used in
production code. It is just a learning project.

So far, CNC implements the following sub-libraries:

- `cncBLAS`: A CUDA implementation of the Basic Linear Algebra Subprograms (BLAS) library.

## Installation (From Source)

---

***[Contact me](mailto:jalenlv543@gmail.com) if you have any questions or suggestions about the project. I am always happy to hear from you.***

---

### Dependencies

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (*sm_60* or higher is required)
- C/C++
- Git
- [CMake] (Minimum version required is 3.28. Don't know the exact minimum version.)
- GNU make or Ninja

### Ubuntu (or WSL2)

This project is tested on native Ubuntu 22.04.04 with RTX4060. WSL2 is recommended if you are using Windows.

To build the project, you need to clone the repository and build the project using CMake.

```bash
git clone --recursive https://github.com/JalenLv/CUDANumericCore.git
```

The project incorporates googletest as a submodule for unit testing purposes. If you forget to add this flag, you can
run `git submodule update --init --recursive` to clone the submodules.

Then `cd` into the project directory and build the project.

```bash
cd CUDANumericCore
cmake -S . -B build [-G Ninja]
cmake --build build -j$(nproc)
```

You can run the tests using `build/tests/blas/cncblasTest` or `ctest --test-dir build`.

The library is built as a shared library and the shared library is in `build/core/<sublib>`, named after `lib<sublib>.so`.

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

All the standard level-one BLAS functions except for some complex numbers functions are implemented. Below is a list of
the functions that are not yet implemented:

- `rot`
- `rotg`
- `rotm`
- `rotmg`

I am planning to implement these functions in the future, and I am currently working on level-two BLAS functions.
Below are the implemented level-two BLAS functions:

- `gbmv`
- `gemv`
- `ger`
