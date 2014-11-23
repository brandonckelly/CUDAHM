/*
 * DustPopPar.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: brandonkelly
 */

#ifndef DUSTPOPPAR_HPP_
#define DUSTPOPPAR_HPP_

#include <limits>

#include "../../../mwg/src/parameters.cuh"

/*
 * Override PopulationPar's LogPrior method, since we do not want a uniform prior on theta. Also override the InitialValue method
 * to compute mean, variances, and correlations from the initial values for chi = (log C, beta, log T).
 *
 * Note that because the classes are templated, we need to use the 'this' pointer in order to access the data members from
 * the base class.
 */

template <int mfeat, int pchi, int dtheta>
class DustPopPar: public PopulationPar<mfeat, pchi, dtheta>
{
public:
	// constructor
	DustPopPar() : PopulationPar<mfeat, pchi, dtheta>()
	{
		// set the hyperprior parameters
		prior_mu_mean.resize(pchi);
		prior_mu_mean[0] = 15.0;  // prior mean on the centroid for the log C distribution
		prior_mu_mean[1] = 2.0;  // prior mean on the centroid for the beta distribution
		prior_mu_mean[2] = log(15.0);  // prior mean on the log T distribution

		prior_mu_var.resize(pchi);
		prior_mu_var[0] = 100.0;  // very broad prior on centroid for log C distribution since we don't have a good handle on log N_H
		prior_mu_var[1] = 1.0;  // don't really expected centroid of beta distribution to be significantly outside of [1,3]
		prior_mu_var[2] = 1.0 * log(10.0) * log(10.0);  // temperature centroid probably within a factor of 10 of 15 K

		// place very broad log-normal prior on scale parameters
		prior_sigma_mean.assign(pchi, 0.0);
		prior_sigma_var.assign(pchi, 25.0);
	}

	// Set the initial values to the sample means and variances
	void InitialValue() {
		double * chi = this->Daug->GetChi();
		thrust::fill(this->h_theta.begin(), this->h_theta.end(), 0.0);
		// first compute sample mean
		int ndata = this->Daug->GetDataDim();
		for (int i = 0; i < ndata; ++i) {
			for (int j = 0; j < pchi; ++j) {
				this->h_theta[j] += chi[ndata * j + i] / double(ndata);
			}
		}
		// now compute sample variances
		std::vector<double> chi_var(pchi, 0.0);
		for (int i = 0; i < ndata; ++i) {
			for (int j = 0; j < pchi; ++j) {
				chi_var[j] += (chi[ndata * j + i] - this->h_theta[j]) * (chi[ndata * j + i] - this->h_theta[j]) / ndata;
			}
		}
		this->h_theta[3] = log(sqrt(chi_var[0]));
		this->h_theta[4] = log(sqrt(chi_var[1]));
		this->h_theta[5] = log(sqrt(chi_var[2]));
	}

	void InitialCholFactor() {
		// set initial covariance matrix of the theta proposals as a diagonal matrix
		thrust::fill(this->cholfact.begin(), this->cholfact.end(), 0.0);
		int diag_index = 0;
		for (int k=0; k<dtheta; k++) {
			this->cholfact[diag_index] = 0.01;
			diag_index += k + 2;
		}
	}

	// return the log-density of the prior for the mean parameter
	double MeanPrior(std::vector<double> mu) {
		double logprior = 0.0;
		for (int j = 0; j < pchi; ++j) {
			// prior for mu is independent and normal
			//std::cout << "mu[" << j << "]: " << mu[j] << ", ";
			logprior += -0.5 * log(prior_mu_var[j]) - 0.5 * (mu[j] - prior_mu_mean[j]) * (mu[j] - prior_mu_mean[j]) / prior_mu_var[j];
		}
		//std::cout << std::endl;
		return logprior;
	}

	// return the log-density of the prior for the scale parameter
	double SigmaPrior(std::vector<double> logsigma) {
		double logprior = 0.0;
		for (int j = 0; j < pchi; ++j) {
			// prior for scale parameters is independent and log-normal
			//std::cout << "logsigma[" << j << "]: " << logsigma[j] << ", ";
			logprior += -0.5 * log(prior_sigma_var[j]) -
					0.5 * (logsigma[j] - prior_sigma_mean[j]) * (logsigma[j] - prior_sigma_mean[j]) / prior_sigma_var[j];
		}
		//std::cout << std::endl;
		return logprior;
	}

	// return the log-density for the prior on the correlations
	double CorrPrior(std::vector<double> arctanh_corr) {
		std::vector<double> corr(pchi);
		for (int j = 0; j < pchi; ++j) {
			corr[j] = tanh(arctanh_corr[j]);
			//std::cout << "corr[" << j << "]: " << corr[j] << ", ";
		}
		//std::cout << std::endl;
		// make sure correlation matrix is positive definite
		double determ =
				(1.0 - corr[2] * corr[2]) - corr[0] * (corr[0] - corr[2] * corr[1]) + corr[1] * (corr[0] * corr[2] - corr[1]);

		double logprior;
		if (determ > 0) {
			// correlation matrix is positive definite, so calculate the log prior density
			logprior = (0.5 * pchi * (pchi - 1.0) - 1.0) * log(determ);
			logprior -= 0.5 * (pchi + 1.0) * log(1.0 - corr[2] * corr[2]);
			logprior -= 0.5 * (pchi + 1.0) * log(1.0 - corr[1] * corr[1]);
			logprior -= 0.5 * (pchi + 1.0) * log(1.0 - corr[0] * corr[0]);
		} else {
			// correlation matrix is not positive definite
			logprior = -std::numeric_limits<double>::infinity();
		}
		return logprior;
	}

	double LogPrior(hvector theta) {
		std::vector<double> mu(theta.begin(), theta.begin() + pchi);
		std::vector<double> logsigma(theta.begin() + pchi, theta.begin() + 2 * pchi);
		std::vector<double> arctanh_corr(theta.begin() + 2 * pchi, theta.end());
		double mean_prior = MeanPrior(mu);
		double sigma_prior = SigmaPrior(logsigma);
		double corr_prior = CorrPrior(arctanh_corr);
		//std::cout << "Priors: " << mean_prior << ", " << sigma_prior << ", " << corr_prior << std::endl;
		return MeanPrior(mu) + SigmaPrior(logsigma) + CorrPrior(arctanh_corr);
	}

private:
	// hyperprior parameters
	std::vector<double> prior_mu_mean;
	std::vector<double> prior_mu_var;
	std::vector<double> prior_sigma_mean;
	std::vector<double> prior_sigma_var;
};

#endif /* DUSTPOPPAR_HPP_ */
