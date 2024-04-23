#include "cnc_builtin_types.cuh"

__host__ __device__ cncComplex::cncComplex(cncFloat r, cncFloat i) : real(r), imag(i) {}

__host__ __device__ cncComplex cncComplex::operator+(const cncComplex &rhs) const {
  return cncComplex(real + rhs.real, imag + rhs.imag);
}

__host__ __device__ cncComplex cncComplex::operator-(const cncComplex &rhs) const {
  return cncComplex(real - rhs.real, imag - rhs.imag);
}

__host__ __device__ cncComplex cncComplex::operator*(const cncComplex &rhs) const {
  return cncComplex(real * rhs.real - imag * rhs.imag, real * rhs.imag + imag * rhs.real);
}

__host__ __device__ cncComplex cncComplex::operator/(const cncComplex &rhs) const {
  cncFloat denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
  return cncComplex((real * rhs.real + imag * rhs.imag) / denominator,
                    (imag * rhs.real - real * rhs.imag) / denominator);
}

__host__ __device__ cncComplexDouble::cncComplexDouble(cncDouble r, cncDouble i) : real(r), imag(i) {}

__host__ __device__ cncComplexDouble cncComplexDouble::operator+(const cncComplexDouble &rhs) const {
  return cncComplexDouble(real + rhs.real, imag + rhs.imag);
}

__host__ __device__ cncComplexDouble cncComplexDouble::operator-(const cncComplexDouble &rhs) const {
  return cncComplexDouble(real - rhs.real, imag - rhs.imag);
}

__host__ __device__ cncComplexDouble cncComplexDouble::operator*(const cncComplexDouble &rhs) const {
  return cncComplexDouble(real * rhs.real - imag * rhs.imag, real * rhs.imag + imag * rhs.real);
}

__host__ __device__ cncComplexDouble cncComplexDouble::operator/(const cncComplexDouble &rhs) const {
  cncDouble denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
  return cncComplexDouble((real * rhs.real + imag * rhs.imag) / denominator,
                          (imag * rhs.real - real * rhs.imag) / denominator);
}
