/*
 * test_function_pointer.cu
 *
 *  Created on: Jul 27, 2013
 *      Author: brandonkelly
 */

// Cuda Includes
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Standard includes
#include <cmath>
#include <vector>
#include <stdio.h>
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef double (*pLogDensMeas)(double*, double*, double*, int, int);

__device__ double test_function(double* x1, double* x2, double* x3, int p, int m) {
	return x1[0];
}

__global__ void test_function_pointer(int ndata, double* x, pLogDensMeas logdens_meas)
{
	//printf("I am at line 45.");
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < ndata) {
        //printf("ID: %i\n", id);
        double x1[2];
        x1[0] = id;
        double x2[2];
        double x3[2];
        int  p = 2, m = 3;
        double result = logdens_meas(x1, x2, x3, p, m);
        x[id] = result;
	}
}

__constant__ pLogDensMeas p_test_function = test_function;

class TestClass {
public:
	TestClass(int ndata, thrust::host_vector<double>& h_data, dim3& nB, dim3& nT) :
		ndata_(ndata), h_data_(h_data), nBlocks_(nB), nThreads_(nT) {
		d_data_ = h_data_;
	    std::cout << "transferring function pointer to host...";
	    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&fptr_, p_test_function, sizeof(p_test_function)));
	    std::cout << "success" << std::endl;
	}
	void SetFunctionPtr(pLogDensMeas pfunction) {

	}
	void LaunchKernel() {
		double* p_data = thrust::raw_pointer_cast(&d_data_[0]);
	    std::cout << "launching the kernel...";
		test_function_pointer<<<nBlocks_,nThreads_>>>(ndata_, p_data, fptr_);
		CUDA_CHECK_RETURN(cudaPeekAtLastError());
		// Wait until RNG stuff is done running on the GPU, make sure everything went OK
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		std::cout << "success." << std::endl;
	    std::cout << "results of test_function_pointer: ";
	    for (int i = 0; i < ndata_; ++i) {
	    	std::cout << d_data_[i] << ", ";
	 	}
	    std::cout << std::endl;
	}

private:
	dim3& nBlocks_;
	dim3& nThreads_;
	int ndata_;
	thrust::host_vector<double>& h_data_;
	thrust::device_vector<double> d_data_;
	pLogDensMeas fptr_;
};


int main(int argc, char **argv) {

	int ndata = 256;

	// Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    pLogDensMeas h_pfunction;
    std::cout << "transferring function pointer to host...";
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_pfunction, p_test_function, sizeof(p_test_function)));
    std::cout << "success" << std::endl;

    thrust::host_vector<double> h_data(ndata, 0.0);
    thrust::device_vector<double> d_data = h_data;
    double* p_data = thrust::raw_pointer_cast(&d_data[0]);

    std::cout << "launching the kernel..." << std::endl;
    std::cout << "launching the kernel...";
    test_function_pointer<<<nBlocks,nThreads>>>(ndata, p_data, h_pfunction);
	CUDA_CHECK_RETURN(cudaPeekAtLastError());
	// Wait until RNG stuff is done running on the GPU, make sure everything went OK
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	std::cout << "success." << std::endl;
    std::cout << "results of test_function_pointer: ";
    for (int i = 0; i < ndata; ++i) {
    	std::cout << d_data[i] << ", ";
 	}
    std::cout << std::endl;


    // test class
    thrust::host_vector<double> h_data2(ndata, 0.0);
    TestClass Test(ndata, h_data2, nBlocks, nThreads);
    Test.LaunchKernel();
}
