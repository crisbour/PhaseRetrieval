/*
 * algorithm.h
 *
 *  Created on: 11 May 2020
 *      Author: cristi
 */

#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <curand.h>
#include "cublas_v2.h"
#include "kernels.h"

//Macros to assert if cuda functions has been processed correctly 
//If an error occurs it halts the program and it prints
//the line of code where the error happend 
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)


//List of Phase Retrieval Algorithms available
//Choose one of those when you are setting up the solver
enum PR_Type{
	Gerchberg_Saxton,
	MRAF,
	Wang,
	HIO,
	Mine
};

/**
 * @brief Operational Blocks to run with CUDA for Phase Retrieval Algorithm
 * Functionalities:
 * 	-	"SLM_To_Obj": Inverse Fourier Transform from image plane to the pupils exit plane of the objective(Phase lagged from SLM)
 * 	-	"Obj_to_SLM": Fourier Transform from exit pupil plane of the objective to the image plane
 * 	-	"Compose" the complex signl from its amplitude and phase
 * 	-	"Decompose" the complex signal in its amplitude and phase
 * 	-	"RandomArray": Used to create a random phase for the desired pattern
 * 	-	"Normalize" input signal by the following formual: u=(u-min(u))/(max(u)-min(u)) 
 * 	-	"NormalizedIntensity"
 * 
 */

class OppBlocks{
protected:
	cufftHandle planFFT;
	cufftResult error;
	cublasStatus_t stat_cublas;
	cublasHandle_t handle_cublas;
	curandGenerator_t curand_gen;
	curandStatus_t stat_curand;
	int nx,ny;
	float *d_min,*d_max;
	int *d_mutex;

public:
	OppBlocks(int nx,int ny);
	~OppBlocks();
	void SLM_To_Obj(cuComplex *d_SLM,cuComplex *d_Obj);
	void Obj_to_SLM(cuComplex *d_Obj,cuComplex *d_SLM);
	void Compose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void Decompose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void RandomArray(float* d_array,float min, float max);
	void Normalize(float *d_quantity);
	void NormalizedIntensity(float *d_amp, float *d_int);
};

struct DeviceMemory{
	float *damp,*illum,*amp,*phase,*intensity;
	cuComplex *complex;
};
struct HostMemory{
	float *damp,*illum,*intensity,*phase,*amp;
	cuComplex *complex;
};
/** @brief Phase Retrieval Algorithm that inherits the
 * optimised CUDA operation blocks. It initalizes memory for the
 * the graphics card that will be used by the operation blocks.
 *
 */

class PhaseRetrieve:protected OppBlocks{
protected:
	// cuComplex *d_complex;
	// float *h_amp;		float *h_phase;
	// float *d_amp;		float *d_phase;
	// float *h_illum;		float *d_illum;
	// float *h_damp;		float *d_damp;
	// float *h_int,*d_int;
	DeviceMemory *device=new DeviceMemory;
	HostMemory *host=new HostMemory;
	float *h_out_img,*h_out_phase;

public:
	PhaseRetrieve(float *gray_img,int nx, int ny, PR_Type type=Gerchberg_Saxton);
	~PhaseRetrieve();
	void InitGPU(int device_id);
	void SetImage(float *gray_img);
	void SetIllumination(float *illum_img);		//To do
	void SetIllumination();
	void SetAlgorithm(PR_Type type);
	float* GetImage();
	float* GetPhaseMask();
	void Compute(int n_iter=0);
	void Test();
	unsigned int index(unsigned int i, unsigned int j);
};

// /** @brief Factory Method of the algorithm for PhaseRetrieve
//  *
//  */
// class PhaseRetrievalAlgorithm:protected OppBlocks{
// public:
// 	virtual void OneIteration()=0;
// 	virtual void GetData()=0;
// };

// class GS:protected PhaseRetrievalAlgorithm{
// 	void OneIteration();
// };


// class MRAF:protected PhaseRetrievalAlgorithm{

// }

// class Wang{
// protected:
// 	float *d_previous_phase;
// };

// class AlgorithmCreator{
// public:
// 	static PhaseRetrievalAlgorithm FactoryMethod(PR_Type type){
// 		if(type==Gerchberg_Saxton)	return new GS();
// 	}
// };

#endif /* ALGORITHM_H_ */
