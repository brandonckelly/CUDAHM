/*
 * normnorm.cu
 *
 *  Created on: Mar 12, 2014
 *      Author: Brandon C. Kelly
 *
 * This file illustrates how to setup a simple Normal-Normal and sample from this using CUDAHM. The model is:
 *
 * 		y_ij | chi_ij ~ N(chi_ij, sigma^2_ij), i = 1, ..., N; j = 1, ..., M   (measurement model)
 * 		chi_i | theta ~ N(theta, Covar)										  (population model for unknown characteristics)
 * 		theta ~ p(theta), p(theta) \propto 1								  (prior, uniform from -infinity to infinity)
 *
 * Under this model the N values of the unknown characteristics, chi, are drawn from a p-dimensional multivariate normal density
 * with unknown mean, theta, and known covariance matrix Covar. We desire to infer the values of the characteristics (chi) and
 * their mean (theta), but we do not observe them. Instead, we observe a measured feature vector of size M which is related to chi.
 * In this simple example, the measured features y_ij are simple the chi-values contaminated with Gaussian noise of known standard
 * deviation sigma_ij, and M = p. So, the goal is to use the measured features to constrain the unknown characteristics and their
 * mean vector. We do this by using CUDAHM to construct a Metropolis-within-Gibbs MCMC sampler to sample from the posterior
 * probability distribution of chi_1, ..., chi_N, theta | {y_ij | i = 1, ..., N; j = 1, ..., M}.
 *
 * The measurements and their standard deviations are contained within the file normnorm_example.dat. These were simulated using
 * values of N = 10000 and M = 3. The values of chi used to simulate the data were generated using a value of
 * theta = [1.2, -0.4, 3.4] and Covar = [[5.29, 0.3105, -15.41], [0.3105, 0.2025, 3.2562], [-15.41, 3.2562, 179.56]]. The chi values
 * are provided in the file true_chi_values.dat.
 *
 */

// std includes
#include <iostream>
#include <fstream>
#include <string>

// local includes
#include "../../../mwg/src/GibbsSampler.hpp"
#include "../../../mwg/src/kernels.cu"
#include "../../../data_proc_util/src/input_output.cpp"


// known dimensions of features, characteristics and population parameter
const int mfeat = 3;
const int pchi = 3;
const int dtheta = 3;

/*
 * Pointer to the population parameter (theta), stored in constant memory on the GPU. Originally defined in
 * kernels.cu and kernels.cuh. Needed by LogDensityPop, which computes the conditional posterior of the
 * characteristics given the population parameters: log p(chi_i|theta).
 */
// extern __constant__ double c_theta[100];

// calculate transpose(x) * covar_inv * x
__device__ __host__
double ChiSqr(double* x, double* covar_inv, int nx)
{
	double chisqr = 0.0;
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < nx; ++j) {
			chisqr += x[i] * covar_inv[i * nx + j] * x[j];
		}
	}
	return chisqr;
}

// compute the conditional log-posterior density of the measurements given the characteristic
__device__ __host__
double LogDensityMeas(double* chi, double* meas, double* meas_unc)
{
	double logdens = 0.0;
	for (int i = 0; i < pchi; ++i) {
		double chi_std = (meas[i] - chi[i]) / meas_unc[i];
		logdens += -0.5 * chi_std * chi_std;
	}

	return logdens;
}

// compute the conditional log-posterior density of the characteristic given the population mean
__device__ __host__
double LogDensityPop(double* chi, double* theta)
{
	// known inverse covariance matrix of the characteristics
	double covar_inv[9] =
	{
			0.64880351, -2.66823952, 0.10406763,
			-2.66823952, 17.94430089, -0.55439855,
			0.10406763, -0.55439855, 0.02455399
	};
	// subtract off the population mean
	double chi_cent[pchi];
	for (int i = 0; i < pchi; ++i) {
		chi_cent[i] = chi[i] - theta[i];
	}

	double logdens = -0.5 * ChiSqr(chi_cent, covar_inv, pchi);
	return logdens;
}

/*
 * Pointers to the device-side functions used to compute the conditional log-densities. These functions must be defined by the
 * user, as above.
 */
__constant__ pLogDensMeas c_LogDensMeas = LogDensityMeas;  // log p(y_i|chi_i)
__constant__ pLogDensPop c_LogDensPop = LogDensityPop;  // log p(chi_i|theta)
extern __constant__ double c_theta[100];

int main(int argc, char** argv)
{
	BaseDataAdapter dataAdapter;
	// allocate memory for measurement arrays
	vecvec meas;
	vecvec meas_unc;
	std::string filename("../data/normnorm_example.txt");
	int ndata = dataAdapter.get_file_lines(filename);
    // read in measurement data from text file
    dataAdapter.read_data(filename, meas, meas_unc, ndata, mfeat, false);
    // build the MCMC sampler
    int niter = 50000;
    int nburnin = niter / 2;

    int nchi_samples = 500;  // only keep 500 samples for the chi values to control memory usage and avoid numerous reads from GPU
    int nthin_chi = niter / nchi_samples;

    // instantiate the Metropolis-within-Gibbs sampler object
    GibbsSampler<mfeat, pchi, dtheta> Sampler(meas, meas_unc, niter, nburnin, nthin_chi);

    // launch the MCMC sampler
    Sampler.Run();

    // grab the samples
	const double * theta_samples = Sampler.GetPopSamples();  // vecvec is a typedef for std::vector<std::vector<double> >
	const double * chi_samples = Sampler.GetCharSamples();

    std::cout << "Writing results to text files..." << std::endl;

    // write the sampled theta values to a file. Output will have nsamples rows and dtheta columns.
    std::string thetafile("normnorm_thetas.dat");
	dataAdapter.write_thetas(thetafile, theta_samples, niter, dtheta);

    // write the posterior means and standard deviations of the characteristics to a file. output will have ndata rows and
    // 2 * pchi columns, where the column format is posterior mean 1, posterior sigma 1, posterior mean 2, posterior sigma 2, etc.
    std::string chifile("normnorm_chi_summary.dat");
	dataAdapter.write_chis(chifile, chi_samples, nchi_samples, ndata, pchi);

}
