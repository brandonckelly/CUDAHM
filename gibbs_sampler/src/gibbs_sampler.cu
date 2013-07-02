/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

static const int p = 8;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

// calculate the logdensity of theta for each chi on the device, needed for updating theta
__global__
void g_logdens_pop(double* chi, int ndata, double* logdens_pop)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < ndata)
	{
		double this_chi[p];
		for (int j=0; j<p; j++) {
			this_chi[j] = chi[j * ndata + i];
		}
		double chi_sum = 0.0;
		for (int j = 0; j < p; ++j) {
			chi_sum += this_chi[j];
		}

		logdens_pop[i] = chi_sum;
	}
}

struct zsqr : public thrust::unary_function<double,double> {
    __device__ __host__
    double operator()(double* chi) {
    	double chi_sum = 0.0;
    	for (int j = 0; j < p; ++j) {
			chi_sum += chi[j];
		}
        return chi_sum;
    }
};

int main(void) {

	unsigned int ndata = 100000;

	thrust::host_vector<double> h_chi(ndata * p);
	thrust::fill(h_chi.begin(), h_chi.end(), 3.4);
	thrust::device_vector<double> d_chi = h_chi;

    // Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    double* p_chi = thrust::raw_pointer_cast(&d_chi[0]);
    thrust::host_vector<double> h_logdens(ndata * p);
    thrust::device_vector<double> d_logdens = h_logdens;
    double* p_logdens = thrust::raw_pointer_cast(&d_logdens[0]);
    g_logdens_pop<<<nBlocks,nThreads>>>(p_chi, ndata, p_logdens);
    cudaDeviceSynchronize();
    double logdens_global = thrust::reduce(d_logdens.begin(), d_logdens.end());

    double logdens_zsqr = 0.0;
    zsqr zsqr0;
    logdens_zsqr = thrust::transform_reduce(d_chi.begin(), d_chi.end(), zsqr0, 0.0, thrust::plus<double>);

    std::cout << "logdens_global: " << logdens_global << ", logdens_zsqr: " << logdens_zsqr << std::endl;

}
