/*
 * algorithm.cu
 *
 *  Created on: 6 May 2020
 *      Author: Cristian Bourceanu
 */
#include "algorithm.h"


//*********** OpBlocks Definitions ***********//

OpBlocks::OpBlocks(int nx,int ny):nx(nx),ny(ny){
	if((error = cufftPlan2d(&planFFT,nx,ny, CUFFT_C2C))!=CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
	}
	if((stat_cublas = cublasCreate(&handle_cublas))!=CUBLAS_STATUS_SUCCESS){
		printf("cuBLAS error: Handle creation failed");
	}
	if((stat_curand =curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT))!=CURAND_STATUS_SUCCESS){
		printf("cuRAND error: Generator creation failed");
	}
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL));
	CUDA_CALL(cudaMalloc((void**)&d_min,sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_max,sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_mutex,sizeof(int)));
}
OpBlocks::~OpBlocks(){
	CUDA_CALL(cudaFree(d_min));
	CUDA_CALL(cudaFree(d_max));
	CUDA_CALL(cudaFree(d_mutex));
	CUFFT_CALL(cufftDestroy(planFFT));
	CUBLAS_CALL(cublasDestroy(handle_cublas)); 
	printf("OpBlocks destructed successfully!\n");
}
void OpBlocks::SLM_To_Obj(cuComplex *d_SLM,cuComplex *d_Obj){
	CUFFT_CALL(cufftExecC2C(planFFT,d_SLM,d_Obj,CUFFT_INVERSE));
	//scaleFourier_kernel<<<(nx*ny+1023)/1024,1024>>>(d_Obj,nx*ny);
	float scale = 1.0/(nx*ny);
	cudaDeviceSynchronize();
	CUBLAS_CALL(cublasCsscal(handle_cublas,nx*ny,&scale,d_Obj,1));

}
void OpBlocks::Obj_to_SLM(cuComplex *d_Obj,cuComplex *d_SLM){
	CUFFT_CALL(cufftExecC2C(planFFT,d_Obj,d_SLM,CUFFT_FORWARD));
}

void OpBlocks::Compose(cuComplex *d_signal,float *d_amp,float *d_phase){
	Comp_kernel<<<(nx*ny+1023)/1024,1024>>>(d_signal,d_amp,d_phase,nx*ny);
}
void OpBlocks::Decompose(cuComplex *d_signal,float *d_amp,float *d_phase){
	Decomp_kernel<<<(nx*ny+1023)/1024,1024>>>(d_signal,d_amp,d_phase,nx*ny);
}
void OpBlocks::RandomArray(float* d_array,float min, float max){
	curandGenerateNormal(curand_gen,d_array,nx*ny,min,max);
}
void OpBlocks::Normalize(float *d_quantity){
	float h_min,h_max;

	max_kernel<<<32,1024>>>(d_quantity,d_max,d_mutex,nx*ny);
	cudaMemcpy(&h_max,d_max,sizeof(float),cudaMemcpyDeviceToHost);
	min_kernel<<<32,1024>>>(d_quantity,d_min,d_mutex,nx*ny);
	cudaMemcpy(&h_min,d_min,sizeof(float),cudaMemcpyDeviceToHost);
	float scale=1/(h_max-h_min);
	CUBLAS_CALL(cublasSscal(handle_cublas,nx*ny,&scale,d_quantity,1));
	cudaDeviceSynchronize();
	//scaleAmp_kernel<<<(nx*ny+1023)/1024,1024>>>(d_amp,nx*ny,h_max-h_min);
	addFloatArray_kernel<<<(nx*ny+1023)/1024,1024>>>(d_quantity,nx*ny,-h_min/(h_max-h_min));		//Couldn't find cublas to add scalar to an array
	printf("(min,max)=(%f,%f)\n",h_min,h_max);
}
void OpBlocks::NormalizedIntensity(float *d_amp,float *d_intensity){
	amplitudeToIntensity_kernel<<<(nx*ny+1023)/1024,1024>>>(d_amp,d_intensity,nx*ny);
	Normalize(d_intensity);
}




//*********** PhaseRetrieve ***********//

PhaseRetrieve::PhaseRetrieve(float *gray_img,unsigned int nx,unsigned int ny, PR_Type type):
nx(nx),ny(ny){
	InitGPU(0);

	//Host memory allocation
	host->complex=(cuComplex*)malloc(nx*ny*sizeof(cuComplex));
	host->illum=(float*)malloc(nx*ny*sizeof(float));
	host->damp=(float*)malloc(nx*ny*sizeof(float));
	host->amp=(float*)malloc(nx*ny*sizeof(float));
	host->phase=(float*)malloc(nx*ny*sizeof(float));
	host->intensity=(float*)malloc(nx*ny*sizeof(float));
	h_out_img=(float*)malloc(nx*ny*sizeof(float));
	h_out_phase=(float*)malloc(nx*ny*sizeof(float));

	//Device memory allocation
	CUDA_CALL(cudaMalloc((void**)&device->complex,nx*ny*sizeof(cuComplex)));
	CUDA_CALL(cudaMalloc((void**)&device->illum,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device->damp,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device->amp,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device->phase,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device->intensity,nx*ny*sizeof(float)));

	SetImage(gray_img);
	SetIllumination();
	operation=new OpBlocks(nx,ny);
	SetAlgorithm(type);
}

PhaseRetrieve::~PhaseRetrieve(){
	free(host->complex);	free(host->damp);		free(host->amp);		free(host->phase);
	free(host->intensity);		free(h_out_img);	free(h_out_phase);	free(host->illum);
	cudaFree(device->complex);	cudaFree(device->damp);
	cudaFree(device->amp);		cudaFree(device->phase);	
	cudaFree(device->intensity);		cudaFree(device->illum);
	delete[] device;
	delete[] host;
	delete operation;
	delete algorithm;
	printf("PhaseRetrieve destructed successfully!\n");
}
void PhaseRetrieve::InitGPU(int device_id){
	int devCount;
    cudaGetDeviceCount(&devCount);	//number of GPUs available
	if(device_id<devCount)		//check if there are enogh GPUs
        cudaSetDevice(device_id);
    else exit(1);
}
void PhaseRetrieve::SetAlgorithm(PR_Type type){
	if(algorithm)
		delete algorithm;
	algorithm=AlgorithmCreator().FactoryMethod(operation,device,host,type);
}
void PhaseRetrieve::SetImage(float *gray_img){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			host->damp[index(i,j)]=sqrt(gray_img[index(i,j)]);
	CUDA_CALL(cudaMemcpy(device->damp,host->damp,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
void PhaseRetrieve::SetIllumination(float *illum_img){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			host->illum[index(i,j)]=sqrt(illum_img[index(i,j)]);
	CUDA_CALL(cudaMemcpy(device->illum,host->illum,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
void PhaseRetrieve::SetIllumination(){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			host->illum[index(i,j)]=sqrt(255);
	CUDA_CALL(cudaMemcpy(device->illum,host->illum,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
unsigned int PhaseRetrieve::index(unsigned int i, unsigned int j){
	return nx*i+j;
}
void PhaseRetrieve::Test(){

	operation->RandomArray(device->phase,-M_PI,M_PI);

	for(int i=0;i<1000;i++){
		algorithm->OneIteration();
		operation->Decompose(device->complex,device->amp,device->phase);
		operation->Compose(device->complex,device->illum,device->phase);
		operation->SLM_To_Obj(device->complex,device->complex);
		operation->Decompose(device->complex,device->amp,device->phase);
	}

	operation->NormalizedIntensity(device->amp,device->intensity);

	CUDA_CALL(cudaMemcpy(host->amp,device->amp,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(host->intensity,device->intensity,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	operation->Obj_to_SLM(device->complex,device->complex);
	operation->Decompose(device->complex,device->amp,device->phase);

	operation->Normalize(device->phase);
	CUDA_CALL(cudaMemcpy(h_out_phase,device->phase,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	float err=0;
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++){
			err+=pow((host->damp[index(i,j)]-host->amp[index(i,j)]),2);
			h_out_img[index(i,j)]=255*host->intensity[index(i,j)];
			h_out_phase[index(i,j)]=255*h_out_phase[index(i,j)];
		}

	printf("Error squared: %f\n",err);
}

float* PhaseRetrieve::GetImage(){
	return h_out_img;
}

float* PhaseRetrieve::GetPhaseMask(){
	return h_out_phase;
}



/********** Algorithms Implementation ***************/

void GS_ALG::OneIteration(){
	operation->Compose(device->complex,device->damp,device->phase);
	operation->Obj_to_SLM(device->complex,device->complex);
}