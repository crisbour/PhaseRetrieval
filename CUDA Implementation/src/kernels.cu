/*
 * kernels.cu
 *
 *  Created on: 9 May 2020
 *      Author: Cristian Bourceanu
 */
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
void amplitudeToIntensity_kernel(float *d_amp, float *d_int,unsigned int dim){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim)
        d_int[index]=d_amp[index]*d_amp[index];
        
	__syncthreads();
}

__global__
void scaleFourier_kernel(cufftComplex *d_signal, unsigned int dim){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim){
		d_signal[index].x=d_signal[index].x/dim;
		d_signal[index].y=d_signal[index].y/dim;
	}
	__syncthreads();
}
__global__
void scaleAmp_kernel(float *d_signal, unsigned int dim,float scale_factor){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim)
		d_signal[index]*=scale_factor;
}
__global__
void addFloatArray_kernel(float *d_signal, unsigned int dim,float add_factor){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim)
		d_signal[index]+=add_factor;
}

__global__
void max_kernel(float *d_in,float *d_max,int *mutex,unsigned int length){
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float temp = -CUDART_INF_F;
    if(index==0){                //Set d_max to infinity only once to avoid racing condition
        *d_max=-CUDART_INF_F;
        atomicExch(mutex,0);
    }
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
    __syncthreads();
    
}

__global__
void min_kernel(float *d_in,float *d_min,int *mutex,unsigned int length){
    __shared__ float sdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float temp = CUDART_INF_F;
    if(index==0)              //Set d_max to infinity only once to avoid racing condition
        *d_min=CUDART_INF_F;

    while(index+offset<length){
        temp=fminf(temp,d_in[index+offset]);
        offset+=stride;
    }

	sdata[tid] = temp;
    __syncthreads();
    
    
    if(index<length)
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s)
            sdata[tid]=fminf(sdata[tid],sdata[tid+s]);
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_min = fminf(*d_min,sdata[0]);
        atomicExch(mutex,0);
    }
    __syncthreads();
}

__global__
void minmax_kernel(float *d_signal,float *d_min, float *d_max,int *mutex, unsigned int length){
    __shared__ float mindata[1024];
    __shared__ float maxdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float mintemp = CUDART_INF_F;
    float maxtemp = -CUDART_INF_F;
    if(index==0){              //Set d_max to infinity only once to avoid racing condition
        *d_min=CUDART_INF_F;
        *d_max=-CUDART_INF_F;
    }
    while(index+offset<length){
        mintemp=fminf(mintemp,d_signal[index+offset]);
        maxtemp=fmaxf(maxtemp,d_signal[index+offset]);
        offset+=stride;
    }

    mindata[tid] = mintemp;
    maxdata[tid] = maxtemp;
    __syncthreads();
    

    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            mindata[tid]=fminf(mindata[tid],mindata[tid+s]);
            maxdata[tid]=fmaxf(maxdata[tid],maxdata[tid+s]);
        }    
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_min = fminf(*d_min,mindata[0]);
        *d_max = fmaxf(*d_max,maxdata[0]);
        atomicExch(mutex,0);
    }
    __syncthreads();
}

__global__
void minmaxROI_kernel(float *d_signal,float *d_min, float *d_max,int *mutex,unsigned int *ROI, unsigned int nROI){
    __shared__ float mindata[1024];
    __shared__ float maxdata[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float mintemp = CUDART_INF_F;
    float maxtemp = -CUDART_INF_F;
    if(index==0){              //Set d_max to infinity only once to avoid racing condition
        *d_min=CUDART_INF_F;
        *d_max=-CUDART_INF_F;
    }
    while(index+offset<nROI){
        mintemp=fminf(mintemp,d_signal[ROI[index+offset]]);
        maxtemp=fmaxf(maxtemp,d_signal[ROI[index+offset]]);
        offset+=stride;
    }

    mindata[tid] = mintemp;
    maxdata[tid] = maxtemp;
    __syncthreads();
    
    
    if(index<nROI)
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            mindata[tid]=fminf(mindata[tid],mindata[tid+s]);
            maxdata[tid]=fmaxf(maxdata[tid],maxdata[tid+s]);
        }    
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_min = fminf(*d_min,mindata[0]);
        *d_max = fmaxf(*d_max,maxdata[0]);
        atomicExch(mutex,0);
    }
    __syncthreads();
}

__global__
void sumROI_kernel(float *d_signal,float *d_sum,int *mutex,unsigned int *ROI, unsigned int nROI){
    __shared__ float data[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float temp = 0;
    if(index==0){              //Set d_max to infinity only once to avoid racing condition
        *d_sum=0;
    }
    while(index+offset<nROI){
        temp+=fabsf(d_signal[ROI[index+offset]]);
        offset+=stride;
    }

    data[tid] = temp;
    __syncthreads();
    
    
    if(index<nROI)
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s)
            data[tid]=data[tid]+data[tid+s];  
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_sum += data[0];
        atomicExch(mutex,0);
    }
    __syncthreads();
}