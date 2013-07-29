/*
 * unit_tests.cu
 *
 *  Created on: Jul 12, 2013
 *      Author: brandonkelly
 */

// standard includes
#include <iostream>
// local includes
#include "UnitTests.hpp"
#include "GibbsSampler.hpp"

/* list of unit tests:
 *
 *	- test rank-1 cholesky update
 *	- make sure Chi::Propose follows a multivariate normal distribution
 *	- make sure Chi::Accept always accepts when the proposal and the current values are the same
 *	- make sure we accept and save a Chi value with a much higher posterior
 *	- Test Chi::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- make sure PopulationPar::Propose follow a multivariate normal distribution
 *	- make sure that PopulationPar::Accept always accepts when the logdensities are the same
 *	- make sure PopulationPar::Update always accepts when the proposed and current theta values are the same
 *	- make sure we accept and save a PopulationPar value with the posterior is much higher
 *	- Test PopulationPar::Adapt acceptance rate and covariance by running a simple MCMC sampler
 *	- Test DataAugmentation::GetChi
 *	- make sure DataAugmentation::Update always accepts when the proposed and current chi values are the same
 *	- make sure we accept and save a Chi value when the posterior is much higher
 *	- make sure we get the correct acceptance rate and covariance of the characteristics by keeping the population
 *	parameter fixed
 *
 */

// Global random number generator and distributions for generating random numbers on the host. The random number generator used
// is the Mersenne Twister mt19937 from the BOOST library. These are instantiated in data_augmentation.cu.
extern boost::random::mt19937 rng;

int main(int argc, char** argv)
{
	int ndata = 10;
	int mfeat = 3;
	int pchi = 3;
	int dtheta = 3;

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << "free: " << free / 1024 << ", total: " << total / 1024 << std::endl;

	// Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    /*
    hvector h_data(ndata, 0.0);
    dvector d_data = h_data;
    double* p_data = thrust::raw_pointer_cast(&d_data[0]);

    // Set up parallel random number generators on the GPU
    curandState* devStates;  // Create state object for random number generators on the GPU
    // Allocate memory on GPU for RNG states
    CUDA_CHECK_RETURN(cudaMalloc((void **)&devStates, nThreads.x * nBlocks.x *
    		sizeof(curandState)));
     // Initialize the random number generator states on the GPU
     initialize_rng<<<nBlocks,nThreads>>>(devStates);
     CUDA_CHECK_RETURN(cudaPeekAtLastError());
     // Wait until RNG stuff is done running on the GPU, make sure everything went OK
     CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    pLogDensMeas h_pfunction;
    std::cout << "transferring function pointer to host...";
    CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_pfunction, p_test_function, sizeof(p_test_function)));
    std::cout << "success" << std::endl;

    std::cout << "launching the kernel..." << std::endl;
    test_function_pointer<<<nBlocks,nThreads>>>(p_data, ndata); //, h_pfunction);
    std::cout << "results of test_function_pointer: ";
    for (int i = 0; i < ndata; ++i) {
		std::cout << p_data[i] << ", ";
	}
    std::cout << std::endl;

	*/

    double** meas_temp;
    double** meas_unc_temp;
	// fill data arrays
    meas_temp = new double* [ndata];
    meas_unc_temp = new double* [ndata];
    for (int i = 0; i < ndata; ++i) {
		meas_temp[i] = new double [mfeat];
		meas_unc_temp[i] = new double [mfeat];
		for (int j = 0; j < mfeat; ++j) {
			meas_temp[i][j] = 0.0;
			meas_unc_temp[i][j] = 0.0;
		}
	}

    DataAugmentation Daug(meas_temp, meas_unc_temp, ndata, mfeat, pchi, nBlocks, nThreads);
    PopulationPar Theta(dtheta, &Daug, nBlocks, nThreads);
    // Characteristic Chi(pchi, mfeat, dtheta, 1);

    int niter(10000), nburnin(2500);
    GibbsSampler Sampler(Daug, Theta, niter, nburnin);

    rng.seed(123456);

	{

		UnitTests Tests(ndata, nBlocks, nThreads);

		// test the rank-1 cholesky update
		// Tests.R1CholUpdate();

		// tests for the characteristic class
		// Tests.ChiPropose();
		// Tests.ChiAcceptSame();
		// Tests.ChiAdapt();

		// tests for population parameter class
		// Tests.ThetaPropose();
		// Tests.ThetaAcceptSame();
		// Tests.ThetaAdapt();

		// tests for the data augmentation class
		//Tests.DaugPopPtr();
		//Tests.DaugGetChi();
		Tests.DaugAcceptSame();
		Tests.DaugAcceptBetter();

		// print results
		Tests.Finish();

	}
    /*
	for (int i = 0; i < ndata; ++i) {
		delete [] meas_temp[i];
		delete [] meas_unc_temp[i];
	}
	delete meas_temp;
	delete meas_unc_temp;
     */

	std::cout << "...... Finished ......." << std::endl;

	CUDA_CHECK_RETURN(cudaDeviceReset());
	cudaMemGetInfo(&free, &total);
	std::cout << "free: " << free / 1024 << ", total: " << total / 1024 << std::endl;

}
