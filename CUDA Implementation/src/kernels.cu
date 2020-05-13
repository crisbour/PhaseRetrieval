#include "kernels.h"


__global__
void Decomp_kernel(cufftComplex *d_signal,float *d_amp,float *d_phase,unsigned int dim){
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	if(index<dim){
		d_amp[index]=sqrt(d_signal[index].x*d_signal[index].x+d_signal[index].y*d_signal[index].y);
		d_phase[index]=atan2f(d_signal[index].y,d_signal[index].x);
	}
}

__global__
void Comp_kernel(cufftComplex *d_signal,float *d_amp,float *d_phase,unsigned int dim){
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	if(index<dim){
		d_signal[index].x=d_amp[index]*cos(d_phase[index]);
		d_signal[index].y=d_amp[index]*sin(d_phase[index]);
	}
}

__global__
void scale_kernel(cufftComplex *d_signal, unsigned int dim){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim){
		d_signal[index].x=d_signal[index].x/dim;
		d_signal[index].y=d_signal[index].y/dim;
	}
	__syncthreads();
}

__global__
void max_kernel(float *d_in,float *d_max,int *mutex,unsigned int length){
    atomicExch(mutex,0);
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float temp = -CUDART_INF_F;
    while(index+offset<length){
        temp=fmaxf(temp,d_in[index+offset]);
        offset+=stride;
    }
    
	sdata[tid] = temp;
    __syncthreads();
    
    if(index<length)
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s)
            sdata[tid]=fmaxf(sdata[tid],sdata[tid+s]);
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_max = fmaxf(*d_max,sdata[0]);
        atomicExch(mutex,0);
    }
        
}
