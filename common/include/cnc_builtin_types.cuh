/*
 * This header file defines basic types of the CUDANumaricCore library.
 */

#ifndef CNC_BUILTIN_TYPES_H
#define CNC_BUILTIN_TYPES_H

/*
 * Basic aliases for c++ standard types.
 */
typedef int cncInt;
typedef unsigned int cncUInt;
typedef long cncLong;
typedef unsigned long cncULong;
typedef float cncFloat;
typedef double cncDouble;

/*
 * Declaration of the complex number type.
 * Definition for cncFloat and cncDouble is provided.
 * Default to cncFloat.
 */
struct cncComplex {
  // Data members
  cncFloat real;
  cncFloat imag;

  // Constructors
  __host__ __device__ cncComplex(cncFloat r = 0, cncFloat i = 0);

  // Operators
  __host__ __device__ cncComplex operator+(const cncComplex &rhs) const;
  __host__ __device__ cncComplex operator-(const cncComplex &rhs) const;
  __host__ __device__ cncComplex operator*(const cncComplex &rhs) const;
  __host__ __device__ cncComplex operator/(const cncComplex &rhs) const;
};

struct cncComplexDouble {
  // Data members
  cncDouble real;
  cncDouble imag;

  // Constructors
  __host__ __device__ cncComplexDouble(cncDouble r = 0, cncDouble i = 0);

  // Operators
  __host__ __device__ cncComplexDouble operator+(const cncComplexDouble &rhs) const;
  __host__ __device__ cncComplexDouble operator-(const cncComplexDouble &rhs) const;
  __host__ __device__ cncComplexDouble operator*(const cncComplexDouble &rhs) const;
  __host__ __device__ cncComplexDouble operator/(const cncComplexDouble &rhs) const;
};

/*
 * Declarations for vector and matrix types and their operations.
 * The type of the elements is assumed to be cncFloat.
 * The declarations provide a simple vector and matrix interface to
 * ordinary C arrays.
 */
struct cncVector {
  // Data members
  cncFloat *data;
  size_t size;
  size_t stride;

  // Constructors
  __host__ __device__ cncVector(size_t s = 0);
  __host__ __device__ cncVector(cncFloat *d, size_t s);

  // Destructor
  __host__ __device__ ~cncVector();

  // Methods
  __host__ __device__ cncFloat &operator[](size_t i);
  __host__ __device__ const cncFloat &operator[](size_t i) const;
};

#endif // CNC_BUILTIN_TYPES_H
