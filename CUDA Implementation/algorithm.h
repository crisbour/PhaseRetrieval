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
#include <vector>
#include <queue>
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

// Execute instruction passed if verbosity is set
// E.g. printf some information: VERB_EXEC(printf("a=%d",1),1); will print "a=1"
// Whereas VERB_EXEC(printf("a=%d",1),0); will not do anything
#define VERB_EXEC(instruction,verbosity)do{ if(verbosity) { \
	(instruction); }}while(0)


//List of Phase Retrieval Algorithms available
//Choose one of those when you are setting up the solver
enum PR_Type{
	Gerchberg_Saxton,
	MRAF,
	UCMRAF,
	Weighted_GS
};

/**
 * @brief Structuring of host and device data
 * 
 */
struct DeviceMemory{
	float *damp,*illum,*amp,*ampOut,*phSLM,*phImg,*intensity,*dint,*weight;
	cuComplex *complex;
	unsigned int *ROI, *SR;
};
struct HostMemory{
	float *damp,*illum,*amp,*ampOut,*phSLM,*phImg,*intensity,*dint;
	cuComplex *complex;
	int nx,ny,n_ROI=0,n_SR=0;
	unsigned int *ROI=NULL,*SR=NULL;
	std::vector<float> uniformity;
	std::vector<float> efficiency;
	std::vector<float> accuracy;
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
	float *d_sum, *d_sumROI;
	int *d_mutex;

public:
	OpBlocks(int nx,int ny);
	~OpBlocks();
	cublasHandle_t& GetCUBLAS();
	void SLM_To_Obj(cuComplex *d_SLM,cuComplex *d_Obj);
	void Obj_to_SLM(cuComplex *d_Obj,cuComplex *d_SLM);
	void Compose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void Decompose(cuComplex *d_signal,float *d_amp,float *d_phase);
	void Sum(float *d_adto,float *d_increment);
	void Scale(float *d_signal,float scaling);
	void RandomArray(float* d_array,float min, float max);
	void ZeroArray(float* d_array,size_t n_bytes);
	void MapUnity(float *d_quantity, char verbose=0);
	void Normalize(float *d_quantity, char verbose=0);
	void Intensity(float *d_amp,float *d_intensity);
	void NormalizedIntensity(float *d_amp, float *d_int);
	float Uniformity(float *d_signal,unsigned int *d_ROI,unsigned int n_ROI, char verbose=0);
	float Efficiency(float *d_signal,unsigned int *d_ROI,unsigned int n_ROI,unsigned int length);
	float Accuracy(float *d_Out,float *d_In,unsigned int *d_ROI,unsigned int n_ROI);
	void PerformanceMetrics(DeviceMemory &device, HostMemory &host);
};








/** @brief Factory Method of the algorithm for PhaseRetrieve
 *
 */
class PhaseRetrievalAlgorithm{
protected:
	char name[20];
	unsigned int index_iter=0;
	unsigned int index_int=0;
	DeviceMemory &device;
	HostMemory &host;
	OpBlocks *operation;
public:
	PhaseRetrievalAlgorithm(OpBlocks *operation,DeviceMemory &device,HostMemory &host):device(device),host(host),operation(operation){};
	virtual ~PhaseRetrievalAlgorithm(){};
	virtual void OneIteration() = 0;
	virtual void Intialize(float param){};
	virtual void Initialize(){};
	void SetIndex(unsigned int index){index_iter=index;};
	unsigned int GetIndex()const{return index_iter;};
	void IncrementIndex(){index_iter++;};
	unsigned int isIntUpdated()const{return index_iter==index_int;};
	const char* GetName(){ return name;}
	void SetName(const char* _name){strcpy(name,_name);};
	void updatedInt(){index_int=index_iter;};
};

class GS_ALG:public PhaseRetrievalAlgorithm{
public:
	GS_ALG(OpBlocks *operation,DeviceMemory &device,HostMemory &host):PhaseRetrievalAlgorithm(operation,device,host){SetName("GS");};
	~GS_ALG(){};
	void OneIteration();
	void Initialize();
};

class MRAF_ALG:public PhaseRetrievalAlgorithm{
public:
	MRAF_ALG(OpBlocks *operation,DeviceMemory &device,HostMemory &host):PhaseRetrievalAlgorithm(operation,device,host){SetName("MRAF");};
	~MRAF_ALG(){};
	void OneIteration();
	void Initialize(float param);
	void Initialize();
protected:
	float m=0.5;
};

class UCMRAF_ALG:public PhaseRetrievalAlgorithm{
public:
	UCMRAF_ALG(OpBlocks *operation,DeviceMemory &device,HostMemory &host):PhaseRetrievalAlgorithm(operation,device,host){SetName("UCMRAF");};
	~UCMRAF_ALG(){};
	void OneIteration();
	void Initialize();
};

class WGS_ALG:public PhaseRetrievalAlgorithm{
public:
	WGS_ALG(OpBlocks *operation,DeviceMemory &device,HostMemory &host):PhaseRetrievalAlgorithm(operation,device,host){
		SetName("WGS");
		CUDA_CALL(cudaMalloc((void**)&device.weight,host.nx*host.ny*sizeof(float)));
	};
	~WGS_ALG(){
		CUDA_CALL(cudaFree(device.weight));
	};
	void OneIteration();
	void Initialize();
};

class AlgorithmCreator{
public:
	PhaseRetrievalAlgorithm *FactoryMethod(OpBlocks *operation,DeviceMemory &device,HostMemory &host,PR_Type type){
		if(type == Gerchberg_Saxton)	return new GS_ALG(operation,device, host);
		if(type == MRAF)				return new MRAF_ALG(operation,device, host);
		if(type == UCMRAF)				return new UCMRAF_ALG(operation,device, host);
		if(type == Weighted_GS)				return new WGS_ALG(operation,device, host);
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
	std::vector<std::vector<float>> metrics;
	char buffer[20];
	char type[20]; 
	unsigned int nx,ny;
	DeviceMemory device;
	HostMemory host;
	OpBlocks *operation = NULL;
	PhaseRetrievalAlgorithm *algorithm = NULL;
	float *h_out_img,*h_out_phase;
	float* ROI_Mask=NULL;

public:
	PhaseRetrieve(float *gray_img,unsigned int nx,unsigned int ny, PR_Type type=Gerchberg_Saxton);
	~PhaseRetrieve();
	void InitGPU(int device_id);
	void SetImage(float *gray_img);
	void SetIllumination(float *illum_img);		//To do
	void SetIllumination();
	void SetAlgorithm(PR_Type type);
	void SetROI(float x, float y, float r);
	void Compute(int n_iter=30);
	float* GetImage();
	float* GetPhaseMask();
	float* GetROIMask();
	const char* GetName(){  if(buffer[0]=='\0'){strcat(buffer,algorithm->GetName()); strcat(buffer,type);} return buffer;}
	std::vector<float>& GetUniformity();
	std::vector<std::vector<float>>& GetMetrics();

private:
	unsigned int index(unsigned int i, unsigned int j);
	void FindROI(float threshold);
	void FindSR(float threshold);
	void InitCompute();
	void PrepareResults();
	char stage=0;
};

#endif /* ALGORITHM_H_ */
