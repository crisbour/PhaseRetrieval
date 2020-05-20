/*
 * kernels.h
 *
 *  Created on: 9 May 2020
 *      Author: Cristian Bourceanu
 */
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

/**
 * @brief Square the amplitude and find the intensity matrix
 * 
 * @param d_amp 
 * @param d_int 
 * @return __global__ scaleFourier_kernel 
 */
__global__
void amplitudeToIntensity_kernel(float *d_amp, float *d_int,unsigned int dim);

/** 
 * @brief Scale matrix after inverse Fourier Transform
 * 
 * @param d_signal  - input/output: Signal to be scaled
 * @param dim       - input:        Dimension of the signal
 * @return __global__ scale_kernel 
 */
__global__
void scaleFourier_kernel(cufftComplex *d_signal, unsigned int dim);

/**
 * @brief Scale a amplitude matrix by a coefficient.(i.e. normalize amplitude)
 * 
 * @param d_signal 
 * @param dim 
 * @param scale_factor 
 * @return __global__ scaleAmp_kernel 
 */
__global__
void scaleAmp_kernel(float *d_signal, unsigned int dim,float scale_factor);

__global__
void weight_kernel(float *d_w, float *d_ampOut_before, float *d_inOut,float *d_din, unsigned int *d_ROI,unsigned int n_ROI);

/**
 * @brief Add real number to every element of an array.
 * 
 * @param d_signal 
 * @param dim 
 * @param add_factor 
 * @return __global__ scaleAmp_kernel 
 */
__global__
void addFloatArray_kernel(float *d_signal, unsigned int dim,float add_factor);

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

/** @brief Find the minimum in an array
 * 
 * @param d_in      - input:    Input array
 * @param d_max     - output:   Minimum value 
 * @param mutex     - global:   Mutual exclusion variable
 * @param length    - input:    Dimension of the array
 * @return __global__ max_kernel 
 */
__global__
void min_kernel(float *d_in,float *d_max,int *mutex,unsigned int length);

__global__
void minmax_kernel(float *d_signal,float *d_min, float *d_max,int *mutex, unsigned int length);

__global__
void minmaxROI_kernel(float *d_signal,float *d_min, float *d_max,int *mutex,unsigned int *ROI, unsigned int nROI);

__global__
void sumROI_kernel(float *d_signal,float *d_sum,int *mutex,unsigned int *ROI, unsigned int nROI);

__global__
void addROI_kernel(float *d_in,float scale_in,float *d_out,float scale_out,unsigned int *ROI, unsigned int nROI);

__global__
void accuracy_kernel(float *d_iOut,float *d_di,float *d_min,float *d_max,int *mutex,unsigned int *ROI, unsigned int nROI);

__global__
void efficiency_kernel(float *d_signal,float *d_sumSR,float *d_sum,int *mutex,unsigned int *ROI, unsigned int nROI, unsigned int length);

#endif