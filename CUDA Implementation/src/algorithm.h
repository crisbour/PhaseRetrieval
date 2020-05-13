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

//Macros
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

//List of Phase Retrieval Algorithms available
enum PR_Type{
	Gerchberg_Saxton,
	MRAF,
	Wang,
	HIO,
	Mine
};

/** @brief Operational Blocks to run with CUDA for Phase Retrieval Algorithm
 */

class OppBlocks{
protected:
	cufftHandle planFFT;
	cufftResult error;
	int nx,ny;

public:
	OppBlocks(int nx,int ny);
	~OppBlocks();
	void SLM_To_Obj(cufftComplex *d_SLM,cufftComplex *d_Obj);
	void Obj_to_SLM(cufftComplex *d_Obj,cufftComplex *d_SLM);
	void Compose(cufftComplex *d_signal,float *d_amp,float *d_phase);
	void Decompose(cufftComplex *d_signal,float *d_amp,float *d_phase);
	void Normalize(float *d_amp,float *d_min);
};


/** @brief Phase Retrieval Algorithm that inherits the
 * optimised CUDA operation blocks. It initalizes memory for the
 * the graphics card that will be used by the operation blocks.
 *
 */

class PhaseRetrieve:protected OppBlocks{
protected:
	cufftComplex *h_img;
	cufftComplex *h_fimg;
	float *h_amp;
	float *h_min;
	float *h_phase;
	cufftComplex *d_img;
	cufftComplex *d_fimg;
	float *d_amp;
	float *d_min;
	float *d_phase;
public:
	PhaseRetrieve(int nx, int ny, PR_Type type);
	~PhaseRetrieve();
	void InitGPUI();
	void Test();
	void printComplex(cufftComplex *data);
	void printFloat(float *data);
	unsigned int index(unsigned int i, unsigned int j);
};

/** @brief Factory Method of the algorithm for PhaseRetrieve
 *
 */
/*
class PhaseRetrievalAlgorithm{
protected:
	PhaseRetrieve *handler_app;
public:
	virtual void Compute()=0;
};

class GS:protected PhaseRetrievalAlgorithm{
	void Compute();
};

class AlgorithmCreator{
public:
	static PhaseRetrievalAlgorithm FactoryMethod(PR_Type type){
		if(type==Gerchberg_Saxton)	return new GS();
	}
};
*/

#endif /* ALGORITHM_H_ */
