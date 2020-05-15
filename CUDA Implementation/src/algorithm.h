/*
 * algorithm.h
 *
 *  Created on: 9 May 2020
 *      Author: Cristian Bourceanu
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
//the line of code where the error happend.
//It's not mandatory to use these, but it's a good defensive mechanism
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CUFFT_CALL(x) do { if((x)!=CUFFT_SUCCESS) { \
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
 * @brief Structuring of host and device data
 * 
 */
struct DeviceMemory{
	float *damp,*illum,*amp,*ampOut,*phSLM,*phImg,*intensity;
	cuComplex *complex;
	unsigned int *ROI;
};
struct HostMemory{
	float *damp,*illum,*amp,*ampOut,*phSLM,*phImg,*intensity;
	cuComplex *complex;
	int nx,ny,n_ROI=0;
	unsigned int *ROI=NULL;
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

class OpBlocks{
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
	OpBlocks(int nx,int ny);
	~OpBlocks();
	void ZeroArray(float* d_array,size_t n_bytes);
	void SLM_To_Obj(cuComplex *d_SLM,cuComplex *d_Obj);
	void Obj_to_SLM(cuComplex *d_Obj,cuComplex *d_SLM);
	void Compose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void Decompose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void RandomArray(float* d_array,float min, float max);
	void Normalize(float *d_quantity);
	void Intensity(float *d_amp,float *d_intensity);
	void NormalizedIntensity(float *d_amp, float *d_int);
	float Uniformity(float *d_signal,unsigned int *d_ROI,unsigned int n_ROI);
};








/** @brief Factory Method of the algorithm for PhaseRetrieve
 *
 */
class PhaseRetrievalAlgorithm{
protected:
	DeviceMemory *device;
	HostMemory *host;
	OpBlocks *operation;
public:
	PhaseRetrievalAlgorithm(OpBlocks *operation,DeviceMemory *device,HostMemory *host):device(device),host(host),operation(operation){};
	virtual ~PhaseRetrievalAlgorithm(){};
	virtual void OneIteration() = 0;
	virtual void Initialize() = 0;
};

class GS_ALG:public PhaseRetrievalAlgorithm{
public:
	GS_ALG(OpBlocks *operation,DeviceMemory *device,HostMemory *host):PhaseRetrievalAlgorithm(operation,device,host){};
	~GS_ALG(){};
	void OneIteration();
	void Initialize();
};

class MRAF_ALG:public PhaseRetrievalAlgorithm{
public:
	MRAF_ALG(OpBlocks *operation,DeviceMemory *device,HostMemory *host):PhaseRetrievalAlgorithm(operation,device,host){};
	~MRAF_ALG(){};
	void OneIteration();
	void Initialize();
};

class Wang_ALG:public PhaseRetrievalAlgorithm{
public:
	Wang_ALG(OpBlocks *operation,DeviceMemory *device,HostMemory *host):PhaseRetrievalAlgorithm(operation,device,host){};
	~Wang_ALG(){};
	void OneIteration(){};
	void Initialize(){};
};

class AlgorithmCreator{
public:
	PhaseRetrievalAlgorithm *FactoryMethod(OpBlocks *operation,DeviceMemory *device,HostMemory *host,PR_Type type){
		if(type == Gerchberg_Saxton)	return new GS_ALG(operation,device, host);
		if(type == MRAF)				return new MRAF_ALG(operation,device, host);
		if(type == Wang)				return new Wang_ALG(operation,device, host);
		printf("The algorithm specified is not a valid one! Check 'PR_Type' enumeration in algorithm.h !\n");
		exit(-1);
	}
};







/** @brief Phase Retrieval Algorithm that inherits the
 * optimised CUDA operation blocks. It initalizes memory for the
 * the graphics card that will be used by the operation blocks.
 *
 */

class PhaseRetrieve{
protected:
	unsigned int nx,ny;
	DeviceMemory *device=new DeviceMemory;
	HostMemory *host=new HostMemory;
	OpBlocks *operation = NULL;
	PhaseRetrievalAlgorithm *algorithm = NULL;
	float *h_out_img,*h_out_phase;

public:
	PhaseRetrieve(float *gray_img,unsigned int nx,unsigned int ny, PR_Type type=Gerchberg_Saxton);
	~PhaseRetrieve();
	void InitGPU(int device_id);
	void SetImage(float *gray_img);
	void SetIllumination(float *illum_img);		//To do
	void SetIllumination();
	void SetAlgorithm(PR_Type type);
	void FindROI(float threshold);
	void Compute(int n_iter=0);
	void Test();
	float* GetImage();
	float* GetPhaseMask();

	unsigned int index(unsigned int i, unsigned int j);
};

#endif /* ALGORITHM_H_ */
