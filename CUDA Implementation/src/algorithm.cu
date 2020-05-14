/*
 * algorithm.cu
 *
 *  Created on: 11 May 2020
 *      Author: cristi
 */
#include "algorithm.h"


OppBlocks::OppBlocks(int nx,int ny):nx(nx),ny(ny){
	if((error = cufftPlan2d(&planFFT,nx,ny, CUFFT_C2C))!=CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
	}
	CUDA_CALL(cudaMalloc((void**)&d_mutex,sizeof(int)));
}
OppBlocks::~OppBlocks(){
	cudaFree(d_mutex);
	cufftDestroy(planFFT);
	cufftDestroy(planFFT);
	printf("OppBlocks destructed successfully!\n");
}
void OppBlocks::SLM_To_Obj(cufftComplex *d_SLM,cufftComplex *d_Obj){
	cufftExecC2C(planFFT,d_SLM,d_Obj,CUFFT_INVERSE);
	scale_kernel<<<(nx*ny+1023)/1024,1024>>>(d_Obj,nx*ny);
}
void OppBlocks::Obj_to_SLM(cufftComplex *d_Obj,cufftComplex *d_SLM){
	cufftExecC2C(planFFT,d_Obj,d_SLM,CUFFT_FORWARD);
}

void OppBlocks::Compose(cufftComplex *d_signal,float *d_amp,float *d_phase){
	Comp_kernel<<<(nx*ny+1023)/1024,1024>>>(d_signal,d_amp,d_phase,nx*ny);
}
void OppBlocks::Decompose(cufftComplex *d_signal,float *d_amp,float *d_phase){
	Decomp_kernel<<<(nx*ny+1023)/1024,1024>>>(d_signal,d_amp,d_phase,nx*ny);
}
void OppBlocks::Normalize(float *d_amp,float *d_min){
	max_kernel<<<(nx*ny+1023),1024>>>(d_amp,d_min,d_mutex,nx*ny);
	printf("Dimensions:(%d,%d)\n",ny,nx);
}

PhaseRetrieve::PhaseRetrieve(int nx, int ny, PR_Type type):OppBlocks(nx,ny){
	InitGPU(0);

	//Host memory allocation
	h_img=(cufftComplex*)malloc(nx*ny*sizeof(cufftComplex));
	h_fimg=(cufftComplex*)malloc(nx*ny*sizeof(cufftComplex));
	h_amp=(float*)malloc(nx*ny*sizeof(float));
	h_phase=(float*)malloc(nx*ny*sizeof(float));

	//Device memory allocation
	CUDA_CALL(cudaMalloc((void**)&d_img,nx*ny*sizeof(cufftComplex)));
	CUDA_CALL(cudaMalloc((void**)&d_fimg,nx*ny*sizeof(cufftComplex)));
	CUDA_CALL(cudaMalloc((void**)&d_amp,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_min,sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_phase,nx*ny*sizeof(float)));
}

PhaseRetrieve::~PhaseRetrieve(){
	free(h_img); free(h_fimg);
	free(h_amp); free(h_phase);
	cudaFree(d_img);	cudaFree(d_fimg);
	cudaFree(d_amp);	cudaFree(d_phase);
	cudaFree(d_min);	
	printf("PhaseRetrieve destructed successfully!\n");
}
void PhaseRetrieve::InitGPU(int device_id){
	int devCount;
    cudaGetDeviceCount(&devCount);	//number of GPUs available
	if(device_id<devCount)		//check if there are enogh GPUs
        cudaSetDevice(device_id);
    else exit(1);
}

void PhaseRetrieve::printComplex(cufftComplex *data){
	for(int i=0;i<ny;i++){
		for(int j=0;j<nx;j++)
			printf("%f ",data[index(i,j)].x);
		printf("\n");
	}
	printf("\n");
}
void PhaseRetrieve::printFloat(float *data){
	for(int i=0;i<ny;i++){
		for(int j=0;j<nx;j++)
			printf("%f ",data[index(i,j)]);
		printf("\n");
	}
	printf("\n");
}
unsigned int PhaseRetrieve::index(unsigned int i, unsigned int j){
	return nx*i+j;
}
void PhaseRetrieve::Test(){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++){
			h_img[index(i,j)].x=0.0;
			h_img[index(i,j)].y=0.0;
		}

	for(int i=ny/4;i<3*ny/4;i++)
		for(int j=nx/4;j<3*nx/4;j++){
			h_img[index(i,j)].x=-(abs(i-ny/2)+abs(j-nx/2)+1.0);
			h_img[index(i,j)].y=-(abs(i-ny/2)+abs(j-nx/2)+1.0);
		}
	//printComplex(h_img);
	CUDA_CALL(cudaMemcpy(d_img,h_img,nx*ny*sizeof(cufftComplex),cudaMemcpyHostToDevice));

	Decompose(d_img,d_amp,d_phase);
	Compose(d_img,d_amp,d_phase);
	Normalize(d_amp,d_min);
	cudaDeviceSynchronize();
	printf("Let's see!\n");
	CUDA_CALL(cudaMemcpy(&h_min,d_min,sizeof(float),cudaMemcpyDeviceToHost));
	printf("Max= %f\n",h_min);
	for(int i=0;i<50;i++){
		Obj_to_SLM(d_img,d_fimg);
		SLM_To_Obj(d_fimg,d_img);
	}

	//CUDA_CALL(cudaMemcpy(h_min,d_min,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
	//printf("Min= %f\n",h_min[0]);

	CUDA_CALL(cudaMemcpy(h_amp,d_amp,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_phase,d_phase,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaMemcpy(h_fimg,d_img,nx*ny*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

	//printComplex(h_fimg);

	float err=0;float max=-10000000;
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++){
			err+=pow((h_fimg[nx*i+j].x-h_img[nx*i+j].x),2)+pow((h_fimg[nx*i+j].y-h_img[nx*i+j].y),2);
			if(max<h_amp[nx*i+j])
				max=h_amp[nx*i+j];
		}
	printf("CPU Max = %f\n",max);
	printf("Error squared: %f\n",err);
}
