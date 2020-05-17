/*
 * algorithm.cu
 *
 *  Created on: 9 May 2020
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
	CUDA_CALL(cudaMemset(d_mutex,0,sizeof(int)));
}
OpBlocks::~OpBlocks(){
	CUDA_CALL(cudaFree(d_min));
	CUDA_CALL(cudaFree(d_max));
	CUDA_CALL(cudaFree(d_mutex));
	CUFFT_CALL(cufftDestroy(planFFT));
	CUBLAS_CALL(cublasDestroy(handle_cublas)); 
	printf("OpBlocks destructed successfully!\n");
}
cublasHandle_t& OpBlocks::GetCUBLAS(){
	return handle_cublas;
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
void OpBlocks::Sum(float *d_adto,float *d_increment){
	const float one=1.0f;
	CUBLAS_CALL(cublasSaxpy(handle_cublas,nx*ny,&one,d_increment,1,d_adto,1));
}
void OpBlocks::Scale(float *d_signal,float scaling){
	CUBLAS_CALL(cublasSscal(handle_cublas,nx*ny,&scaling,d_signal,1));
}
void OpBlocks::RandomArray(float* d_array,float min, float max){
	curandGenerateNormal(curand_gen,d_array,nx*ny,min,max);
}
void OpBlocks::ZeroArray(float* d_array,size_t n_bytes){
	CUDA_CALL(cudaMemset(d_array,0,n_bytes));
}
void OpBlocks::Normalize(float *d_quantity){
	float h_min,h_max;
	minmax_kernel<<<32,1024>>>(d_quantity,d_min,d_max,d_mutex,nx*ny);
	cudaMemcpy(&h_max,d_max,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_min,d_min,sizeof(float),cudaMemcpyDeviceToHost);
	float scale=1/(h_max-h_min);
	CUBLAS_CALL(cublasSscal(handle_cublas,nx*ny,&scale,d_quantity,1));
	cudaDeviceSynchronize();
	addFloatArray_kernel<<<(nx*ny+1023)/1024,1024>>>(d_quantity,nx*ny,-h_min/(h_max-h_min));		//Couldn't find cublas to add scalar to an array
	//printf("(min,max)=(%f,%f)\n",h_min,h_max);
}
void OpBlocks::Intensity(float *d_amp,float *d_intensity){
	amplitudeToIntensity_kernel<<<(nx*ny+1023)/1024,1024>>>(d_amp,d_intensity,nx*ny);
}
void OpBlocks::NormalizedIntensity(float *d_amp,float *d_intensity){
	amplitudeToIntensity_kernel<<<(nx*ny+1023)/1024,1024>>>(d_amp,d_intensity,nx*ny);
	Normalize(d_intensity);
}
/**
 * @brief Uniformity within the region of interest
 * 
 * @param d_signal Signal whose host.uniformity is assesed
 * @param d_ROI 	Array of indexes of the elements in the ROI
 * @param n_ROI 	Length of d_ROI
 */
float OpBlocks::Uniformity(float *d_signal,unsigned int *d_ROI,unsigned int n_ROI){
	minmaxROI_kernel<<<32,1024>>>(d_signal,d_min,d_max,d_mutex,d_ROI,n_ROI);
	float h_min,h_max;
	CUDA_CALL(cudaMemcpy(&h_max,d_max,sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&h_min,d_min,sizeof(float),cudaMemcpyDeviceToHost));
	return 1-(h_max-h_min)/(h_max+h_min);
}



//*********** PhaseRetrieve ***********//

PhaseRetrieve::PhaseRetrieve(float *gray_img,unsigned int nx,unsigned int ny, PR_Type type):
nx(nx),ny(ny){
	InitGPU(0);

	//Host memory allocation
	host.complex=(cuComplex*)malloc(nx*ny*sizeof(cuComplex));
	host.illum=(float*)malloc(nx*ny*sizeof(float));
	host.dint=(float*)malloc(nx*ny*sizeof(float));
	host.damp=(float*)malloc(nx*ny*sizeof(float));
	host.amp=(float*)malloc(nx*ny*sizeof(float));
	host.ampOut=(float*)malloc(nx*ny*sizeof(float));
	host.phSLM=(float*)malloc(nx*ny*sizeof(float));
	host.phImg=(float*)malloc(nx*ny*sizeof(float));
	host.intensity=(float*)malloc(nx*ny*sizeof(float));
	h_out_img=(float*)malloc(nx*ny*sizeof(float));
	h_out_phase=(float*)malloc(nx*ny*sizeof(float));

	//Device memory allocation
	CUDA_CALL(cudaMalloc((void**)&device.complex,nx*ny*sizeof(cuComplex)));
	CUDA_CALL(cudaMalloc((void**)&device.illum,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.dint,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.damp,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.amp,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.ampOut,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.phSLM,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.phImg,nx*ny*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&device.intensity,nx*ny*sizeof(float)));

	host.nx=nx;	host.ny=ny;

	SetImage(gray_img);
	SetIllumination();
	operation=new OpBlocks(nx,ny);
	SetAlgorithm(type);
}

PhaseRetrieve::~PhaseRetrieve(){
	free(host.complex);	free(host.damp);		free(host.dint);		free(host.amp);		free(host.phSLM);
	free(host.intensity);		free(h_out_img);	free(h_out_phase);	free(host.illum);
	free(host.ampOut);		free(host.phImg);
	cudaFree(device.complex);	cudaFree(device.damp);		cudaFree(device.dint);
	cudaFree(device.amp);		cudaFree(device.phSLM);	
	cudaFree(device.intensity);		cudaFree(device.illum);
	cudaFree(device.ampOut);	cudaFree(device.phImg);
	if(host.ROI){	free(host.ROI);	cudaFree(device.ROI);}	
	
	delete algorithm;
	delete operation;
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
		for(int j=0;j<nx;j++){
			host.damp[index(i,j)]=sqrt(gray_img[index(i,j)]);
			host.dint[index(i,j)]=gray_img[index(i,j)];
		}
	CUDA_CALL(cudaMemcpy(device.damp,host.damp,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(device.dint,host.dint,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
void PhaseRetrieve::SetIllumination(float *illum_img){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			host.illum[index(i,j)]=sqrt(illum_img[index(i,j)]);
	CUDA_CALL(cudaMemcpy(device.illum,host.illum,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
void PhaseRetrieve::SetIllumination(){
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++)
			host.illum[index(i,j)]=sqrt(255);
	CUDA_CALL(cudaMemcpy(device.illum,host.illum,nx*ny*sizeof(float),cudaMemcpyHostToDevice));
}
void PhaseRetrieve::FindROI(float threshold){
	if(host.n_ROI==0){
		host.ROI=(unsigned int*)malloc(host.nx*host.ny*sizeof(unsigned int));
		for(unsigned int i=0;i<host.nx*host.ny;i++)
			if(host.damp[i]>threshold){
				host.ROI[host.n_ROI++]=i;
			}
		CUDA_CALL(cudaMalloc((void**)&device.ROI,host.n_ROI*sizeof(unsigned int)));
		CUDA_CALL(cudaMemcpy(device.ROI,host.ROI,host.n_ROI*sizeof(unsigned int),cudaMemcpyHostToDevice));
	}
}
unsigned int PhaseRetrieve::index(unsigned int i, unsigned int j){
	return nx*i+j;
}
void PhaseRetrieve::Test(){
	
	operation->RandomArray(device.phImg,-M_PI,M_PI);
	operation->RandomArray(device.phSLM,-M_PI,M_PI);

	FindROI(sqrt(255)/2);
	algorithm->Initialize();
	
	for(int i=0;i<50;i++){
		//if(i==4) SetAlgorithm(MRAF);
		algorithm->OneIteration();
		// if(host.n_ROI)
		// 	host.uniformity.push_back(operation->Uniformity(device.intensity,device.ROI,host.n_ROI));
	}

	operation->NormalizedIntensity(device.ampOut,device.intensity);

	CUDA_CALL(cudaMemcpy(host.intensity,device.intensity,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	operation->Normalize(device.ampOut);
	CUDA_CALL(cudaMemcpy(host.ampOut,device.ampOut,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	operation->Normalize(device.phSLM);
	CUDA_CALL(cudaMemcpy(h_out_phase,device.phSLM,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	operation->Normalize(device.damp);
	CUDA_CALL(cudaMemcpy(host.damp,device.damp,nx*ny*sizeof(float),cudaMemcpyDeviceToHost));

	float err=0;
	for(int i=0;i<ny;i++)
		for(int j=0;j<nx;j++){
			err+=pow((host.damp[index(i,j)]-host.ampOut[index(i,j)]),2);
			h_out_img[index(i,j)]=255*host.intensity[index(i,j)];
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

std::vector<float>& PhaseRetrieve::GetUniformity(){
	return host.uniformity;
}


/********** Algorithms Implementation ***************/
void GS_ALG::Initialize(){

}
void GS_ALG::OneIteration(){
	operation->Compose(device.complex,device.damp,device.phImg);
	operation->Obj_to_SLM(device.complex,device.complex);
	operation->Decompose(device.complex,device.amp,device.phSLM);
	operation->Compose(device.complex,device.illum,device.phSLM);
	operation->SLM_To_Obj(device.complex,device.complex);
	operation->Decompose(device.complex,device.ampOut,device.phImg);
	operation->Intensity(device.ampOut,device.intensity);
}

void MRAF_ALG::Initialize(){
	//operation->ZeroArray(device.ampOut,host.nx * host.ny);
}
void MRAF_ALG::OneIteration(){
	//MRAF Scaling the desired amplitude for correction
	const float lambda = 1.0f;
	operation->Scale(device.ampOut,lambda*(host.uniformity.back()-1));
	float scale_des=1;
	CUBLAS_CALL(cublasSaxpy(operation->GetCUBLAS(),host.nx*host.ny,&scale_des,device.damp,1,device.ampOut,1));

	operation->Compose(device.complex,device.ampOut,device.phImg);
	operation->Obj_to_SLM(device.complex,device.complex);
	operation->Decompose(device.complex,device.amp,device.phSLM);
	operation->Compose(device.complex,device.illum,device.phSLM);
	operation->SLM_To_Obj(device.complex,device.complex);
	operation->Decompose(device.complex,device.ampOut,device.phImg);
}
void WGS_ALG::Initialize(){
	//float *ampOut_before,*wamp;
}
void WGS_ALG::OneIteration(){
	//operation->Compose(device.complex,wamp,device.phImg);
	//cudaMemcpy(ampOut_before,device.ampOut,host.nx*host.ny*sizeof(float),cudaMemcpyDeviceToDevice);
	operation->Obj_to_SLM(device.complex,device.complex);
	operation->Decompose(device.complex,device.amp,device.phSLM);
	operation->Compose(device.complex,device.illum,device.phSLM);
	operation->SLM_To_Obj(device.complex,device.complex);
	operation->Decompose(device.complex,device.ampOut,device.phImg);
}