/*
 * unit_tests.cu
 *
 *  Created on: Jul 12, 2013
 *      Author: brandonkelly
 */

// standard includes
#include <iostream>
// local includes
#include "UnitTests.cuh"

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

int main(int argc, char** argv)
{
	int ndata = 10000;

	bool check_memory = false;  // set to true if you want to check how much memory is available without running the tests
	if (check_memory) {
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		std::cout << "free: " << free / 1024 << ", total: " << total / 1024 << std::endl;
	}

	// Cuda grid launch
    dim3 nThreads(256);
    dim3 nBlocks((ndata + nThreads.x-1) / nThreads.x);
    printf("nBlocks: %d\n", nBlocks.x);  // no more than 64k blocks!
    if (nBlocks.x > 65535)
    {
        std::cerr << "ERROR: Block is too large" << std::endl;
        return 2;
    }

    rng.seed(123456);  // keep the host-side seed constant to make the unit tests reproducible
	{
    	/*
    	 * RUN THE UNIT TESTS
    	 */
		UnitTests Tests(ndata, nBlocks, nThreads);

		bool save_meas = false;
		if (save_meas) {
			Tests.SaveMeasurements();
		}

		// test the rank-1 cholesky update
		Tests.R1CholUpdate();

		// test that pointers are correctly set
		Tests.GibbsSamplerPtr();

		// tests for the characteristic class
		Tests.ChiPropose();
		Tests.ChiAcceptSame();
		Tests.ChiAdapt();

		// tests for population parameter class
		Tests.ThetaPropose();
		Tests.ThetaAcceptSame();
		Tests.ThetaAdapt();

		// tests for device-side functions used in updated the characteristics
		Tests.DevicePropose();
		Tests.DeviceAccept();
		Tests.DeviceAdapt();

		// tests for the data augmentation class
		Tests.DaugGetChi();
		Tests.DaugLogDensPtr();
		Tests.DaugAcceptSame();
		Tests.DaugAcceptBetter();
//
//		// tests for the MCMC sampler
		Tests.FixedChar();
		Tests.FixedPopPar();
		Tests.NormNorm();

		// print results
		Tests.Finish();
	}

	if (check_memory) {
		size_t free, total;
		CUDA_CHECK_RETURN(cudaDeviceReset());
		cudaMemGetInfo(&free, &total);
		std::cout << "free: " << free / 1024 << ", total: " << total / 1024 << std::endl;
	}
}
