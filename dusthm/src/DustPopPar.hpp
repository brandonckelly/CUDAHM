/*
 * DustPopPar.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: brandonkelly
 */

#ifndef DUSTPOPPAR_HPP_
#define DUSTPOPPAR_HPP_

#include <limits>

#include "parameters.cuh"

/*
 * Override PopulationPar's LogPrior method, since we do not want a uniform prior on theta.
 */

template <int mfeat>
class DustPopPar: public PopulationPar<mfeat, 3, 9>
{
	// constructor
	DustPopPar(dim3& nB, dim3& nT) : PopulationPar<mfeat, 3, 9>(nB, nT), pchi_(3), dtheta_(9)
	{
		prior_mu_mean.resize(pchi_);
		prior_mu_mean[0] = 15.0;  // prior mean on the centroid for the log C distribution
		prior_mu_mean[1] = 2.0;  // prior mean on the centroid for the beta distribution
		prior_mu_mean[2] = log(15.0);  // prior mean on the log T distribution

		prior_mu_var.resize(pchi_);
		prior_mu_var[0] = 100.0;  // very broad prior on centroid for log C distribution since we don't have a good handle on log N_H
		prior_mu_var[1] = 1.0;  // don't really expected centroid of beta distribution to be significantly outside of [1,3]
		prior_mu_var[2] = 1.0 * log(10.0) * log(10.0);  // temperature centroid probably within a factor of 10 of 15 K

		// place very broad log-normal prior on scale parameters
		prior_sigma_mean.assign(pchi_, 0.0);
		prior_sigma_var.assign(pchi_, 25.0);
	}

	// return the log-density of the prior for the mean parameter
	double MeanPrior(std::vector<double> mu) {
		double logprior = 0.0;
		for (int j = 0; j < pchi_; ++j) {
			// prior for mu is independent and normal
			logprior += -0.5 * log(prior_mu_var[j]) - 0.5 * (mu[j] - prior_mu_mean[j]) * (mu[j] - prior_mu_mean[j]) / prior_mu_var[j];
		}
		return logprior;
	}

	// return the log-density of the prior for the scale parameter
	double SigmaPrior(std::vector<double> logsigma) {
		double logprior = 0.0;
		for (int j = 0; j < pchi_; ++j) {
			// prior for scale parameters is independent and log-normal
			logprior += -0.5 * log(prior_sigma_var[j]) -
					0.5 * (logsigma[j] - prior_sigma_mean[j]) * (logsigma[j] - prior_sigma_mean[j]) / prior_sigma_var[j];
		}
		return logprior;
	}

	// return the log-density for the prior on the correlations
	double CorrPrior(std::vector<double> arctanh_corr) {
		std::vector<double> corr(pchi_);
		for (int j = 0; j < pchi_; ++j) {
			corr[j] = (exp(2.0 * arctanh_corr[j]) - 1.0) / (exp(2.0 * arctanh_corr[j]) + 1.0);
		}
		// make sure correlation matrix is positive definite
		double determ =
				(1.0 - corr[2] * corr[2]) - corr[0] * (corr[0] - corr[2] * corr[3]) + corr[1] * (corr[0] * corr[2] - corr[1]);

		double logprior;
		if (determ > 0) {
			// correlation matrix is positive definite, so calculate the log prior density
			logprior = (0.5 * pchi_ * (pchi_ - 1.0) - 1.0) * log(determ);
			logprior -= 0.5 * (pchi_ + 1.0) * log(1.0 - corr[2] * corr[2]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[1] * corr[2] - corr[0]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[0] * corr[2] - corr[1]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[1] * corr[2] - corr[0]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(1.0 - corr[1] * corr[1]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[0] * corr[1] - corr[2]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[0] * corr[2] - corr[1]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(corr[0] * corr[1] - corr[2]);
			logprior -= 0.5 * (pchi_ + 1.0) * log(1.0 - corr[0] * corr[0]);
		} else {
			// correlation matrix is not positive definite
			logprior = -std::numeric_limits<double>::infinity();
		}
		return logprior;
	}

	double LogPrior(hvector theta) {
		std::vector<double> mu(theta.begin(), theta.begin() + pchi_ - 1);
		std::vector<double> logsigma(theta.begin() + pchi_, theta.begin() + 2 * pchi_ - 1);
		std::vector<double> arctanh_corr(theta.begin() + 2 * pchi_, theta.end());
		return MeanPrior(mu) + SigmaPrior(logsigma) + CorrPrior(arctanh_corr);
	}

private:
	std::vector<double> prior_mu_mean;
	std::vector<double> prior_mu_var;
	std::vector<double> prior_sigma_mean;
	std::vector<double> prior_sigma_var;
	int pchi_;
	int dtheta_;
};

#endif /* DUSTPOPPAR_HPP_ */
