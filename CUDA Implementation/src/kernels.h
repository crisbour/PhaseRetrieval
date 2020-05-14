#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>

#define CUDART_INF_F            __int_as_float(0x7f800000)

/** @brief GPU kernel for decomposition of a complex matrix in amplitude and phase
 * 
 * @param d_signal - input:  Complex matrix in array format
 * @param d_amp    - output: Amplitude matrix __
 * @param d_phase  - output: Phase matrix __
 * @param dim      - input:  Number of elements in the matrix
 * @return __global__ Decomp_kernel 
 */
__global__ void Decomp_kernel(cufftComplex *d_signal,float *d_amp,float *d_phase,unsigned int dim);

/** @brief GPU kernel for composition of a complex matrix in amplitude and phase
 * 
 * @param d_signal - output: Complex matrix in array format
 * @param d_amp    - input:  Amplitude matrix __
 * @param d_phase  - input:  Phase matrix __
 * @param dim      - input:  Number of elements in the matrix
 * @return __global__ Comp_kernel 
 */
__global__
void Comp_kernel(cufftComplex *d_signal,float *d_amp,float *d_phase,unsigned int dim);

/** @brief Scale matrix after inverse Fourier Transform
 * 
 * @param d_signal  - input/output: Signal to be scaled
 * @param dim       - input:        Dimension of the signal
 * @return __global__ scale_kernel 
 */
__global__
void scale_kernel(cufftComplex *d_signal, unsigned int dim);

/** @brief Find the maximum in an array
 * 
 * @param d_in      - input:    Input array
 * @param d_max     - output:   Maximum value 
 * @param mutex     - global:   Mutual exclusion variable
 * @param length    - input:    Dimension of the array
 * @return __global__ max_kernel 
 */
__global__
void max_kernel(float *d_in,float *d_max,int *mutex,unsigned int length);

#endif