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
}

__global__
void scaleFourier_kernel(cufftComplex *d_signal, unsigned int dim){
	unsigned int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<dim){
		d_signal[index].x=d_signal[index].x/dim;
		d_signal[index].y=d_signal[index].y/dim;
	}
}

__global__
void scaleVect_kernel(float *d_signal, unsigned int dim,float scale_factor){
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
void weight_kernel(float *d_w, float *d_ampOut_before, float *d_inOut,float *d_din, unsigned int *d_ROI,unsigned int n_ROI){
    unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<n_ROI){
        unsigned int index_ROI=d_ROI[index];
        d_w[index_ROI]=sqrtf(d_din[index_ROI]/d_inOut[index_ROI])*d_ampOut_before[index_ROI];
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
}

__global__
void efficiency_kernel(float *d_signal,float *d_sumSR,float *d_sum,int *mutex,unsigned int *ROI, unsigned int nROI, unsigned int length){
    __shared__ float data[1024];
    __shared__ float dataSR[1024];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int offset=0;

    float temp = 0;
    float tempSR = 0;
    if(index==0){              //Set d_max to infinity only once to avoid racing condition
        *d_sum=0;
        *d_sumSR=0;
    }
    while(index+offset<length){
        
        temp+=d_signal[index+offset];
        if(index+offset<nROI)
            tempSR+=d_signal[ROI[index+offset]];
        offset+=stride;
    }

    data[tid] = temp;
    dataSR[tid] = tempSR;
    __syncthreads();
    
    
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            data[tid]=data[tid]+data[tid+s];
            dataSR[tid]=dataSR[tid]+dataSR[tid+s];   
        }
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        *d_sum += data[0];
        *d_sumSR+=dataSR[0];
        atomicExch(mutex,0);
    }
}

__global__
void accuracy_kernel(float *d_iOut,float *d_di,float *d_sumerr2,int *mutex,unsigned int *ROI, unsigned int nROI){
    __shared__ float dataerr[256];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x*gridDim.x;
    unsigned int indexROI;
    unsigned int offset=0;

    float temperr = 0;
    if(index==0){            //Set d_max only once to avoid racing condition
        *d_sumerr2=0;
    }
    
    while(index+offset<nROI){
        indexROI=ROI[index+offset];
        temperr+=fdividef(powf(d_iOut[indexROI]-d_di[indexROI],2),powf(d_di[indexROI],2));
        offset+=stride;
    }
    dataerr[tid] = temperr;
    __syncthreads();
    
    
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            dataerr[tid]+=dataerr[tid+s]; 
        }
        __syncthreads();
    }
    
    if(tid == 0){
        while(atomicCAS(mutex,0,1));
        __threadfence();
        *d_sumerr2 += dataerr[0];
        atomicExch(mutex,0);
    }
}

__global__
void addROI_kernel(float *d_in,float scale_in,float *d_out,float scale_out,unsigned int *ROI, unsigned int nROI){
    unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
    if(index<nROI){
        unsigned int index_ROI=ROI[index];
        d_out[index_ROI]=scale_in*d_in[index_ROI]+scale_out*d_out[index_ROI];
    }
    __syncthreads();
}

__global__
void sqrt_kernel(float *d_signal_in,float *d_signal_out,unsigned int length){
    unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;

    if(index<length){
        d_signal_out[index]=sqrtf(d_signal_in[index]);
    }
    __syncthreads();
}