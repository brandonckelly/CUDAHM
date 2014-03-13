/*
 * cudahm_blueprint.cu
 *
 *  Created on: Mar 13, 2014
 *      Author: Brandon C. Kelly
 *
 * This file provides a blueprint for using the CUDAHM API. In order to use CUDAHM the user must supply a function
 * to compute the logarithm of the conditional probability densities of the measurements (y) given the
 * characteristics (chi), and the characteristics given the population parameter theta. The user must also set
 * the pointers c_pLogDensMeas and c_pLogDensPop to these functions, in order to correctly place these functions
 * in GPU constant memory. The purpose of this file is to provide an easy way of setting the pointers and stream-line
 * the use of CUDAHM to build MCMC sampler. Using this blueprint, the user need only modify the LogDensityMeas and
 * LogDensityPop functions to ensure that the pointers are set correctly.
 *
 * The MCMC sampler should be constructed within the main function using the GibbSampler class. Further details
 * are provided below. A complete working example is provided in normnorm.cu.
 *
 */

// standard library includes
#include <iostream>

// local CUDAHM include
#include "GibbsSampler.hpp"

/*
 * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in
 * kernels.cu and kernels.cuh. Needed by LogDensityPop, which computes the conditional posterior of the
 * characteristics given the population parameters: log p(chi_i|theta). This assumes a maximum of 100 elements
 * in the theta parameter vector.
 *
 * If you need more than this, you will have to change this manually here and in
 * the kernels.cuh and kernels.cu files.
 *
 * YOU SHOULD NOT MODIFY THIS UNLESS YOU KNOW WHAT YOU ARE DOING.
 */
extern __constant__ double c_theta[100];


/*
 * This function returns the logarithm of the conditional density of the measurements given the
 * characteristics for a single data point, log p(y_i | chi_i). This function must be supplied by the user
 * and written in CUDA. The input parameters are:
 *
 * 	chi      - Pointer to the values of the characteristics for the i^th data point.
 * 	meas     - Pointer to the measurements for the i^th data point.
 * 	meas_unc - Pointer to the standard deviations in the measurement errors for y_ij.
 * 	mfeat    - The number of features measured for each data point, i.e., the length of the meas array.
 *  pchi     - The number of characteristics for each data point, i.e., the length of the chi array.
 *
 */
__device__ __host__
double LogDensityMeas(double* chi, double* meas, double* meas_unc, int mfeat, int pchi)
{
	return 0.0;
}


/*
 * This function returns the logarithm of the conditional density of the characteristics given the
 * population parameter theta for a single data point, log p(chi_i | theta). This function must be supplied by
 * the user and written in CUDA. The input parameters are:
 *
 * 	chi       - Pointer to the values of the characteristics for the i^th data point.
 * 	theta     - Pointer to the population parameter.
 *  pchi      - The number of characteristics for each data point, i.e., the length of the chi array.
 *  dim_theta - The dimension of the population parameter vector theta, i.e., the length of the theta array.
 *
 */
__device__ __host__
double LogDensityPop(double* chi, double* theta, int pchi, int dim_theta)
{
	return 0.0;
}


/*
 * Pointers to the GPU functions used to compute the conditional log-densities for a single data point.
 * These functions live on the GPU in constant memory.
 *
 * IF YOU ARE USING LogDensityMeas and LogDensityPop TO COMPUTE YOUR CONDITIONAL DENSITIES, DO NOT MODIFY THESE.
 * Otherwise you will need to set these points to whichever functions you are using to compute these quantities.
 *
 */
__constant__ pLogDensMeas c_LogDensMeas = LogDensityMeas;  // log p(y_i|chi_i)
__constant__ pLogDensPop c_LogDensPop = LogDensityPop;  // log p(chi_i|theta)


int main(int argc, char** argv)
{
	std::cout << "This file provides a blueprint for using the CUDAHM API. On its own it does nothing except print this message."
			<< std::endl;
	/*
	 * First you need to set the values of the parameter dimensions as const int types. These must be supplied
	 * at compile-time in order to efficiently make use of GPU memory. See normnorm.cu for further details.
	 */

				// set the parameter dimensions here //

	/*
	 * Read in the data for the measurements, meas, and their standard deviations, meas_unc.
	 */

				// read in the measured feature data here //

	/*
	 * Set the number of MCMC iterations and the amount of thinning for the chi and theta samples.
	 *
	 * NOTE THAT IF YOU HAVE A LARGE DATA SET, YOU WILL PROBABLY WANT TO THIN THE CHI VALUES SIGNIFICANTLY SO
	 * YOU DO NOT RUN OUR OF MEMORY.
	 */

				// choose the number of MCMC iterations and the amount of thinning //

	/*
	 * Instantiate the GibbsSampler<mfeat, pchi, dtheta> object here. Once you've instantiated it, use the
	 * GibbSampler::Run() method to run the MCMC sampler. Finally, use GibbSampler::GetCharSampler() to get
	 * the sampled characteristics as a std::vector<std::vector<std::vector<double> > > (three nested vectors,
	 * dimensions nchi_samples x ndata x pchi) object. Similarly, use the GibbsSampler::GetPopSamples() to
	 * retrieve the sampled theta values as a std::vector<std::vector<double> > (two nested vectors,
	 * dimensions nsamples x dtheta) object.
	 */

				// Instantiate the GibbSampler object, run the MCMC sampler, and retrieve the samples. //

	/*
	 * Finally, do calculations with the MCMC samples or dump them to a file.
	 */

				// use MCMC samples or dump them to a file //
}

