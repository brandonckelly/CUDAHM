/*
 * GibbsSampler.hpp
 *
 *  Created on: Jul 25, 2013
 *      Author: Brandon C. Kelly
 */

#ifndef GIBBSSAMPLER_HPP_
#define GIBBSSAMPLER_HPP_

// boost includes
//#include <boost/timer/timer.hpp>
//#include <boost/progress.hpp>

// local includes
#include "parameters.hpp"

class GibbsSampler
{
public:
	// constructor
	GibbsSampler(DataAugmentation& Daug, PopulationPar& PopPar, int niter, int nburnin,
			int nthin_chi=10, int nthin_theta=1) :
		Daug_(Daug), PopPar_(PopPar), niter_(niter), nburnin_(nburnin), nthin_theta_(nthin_theta),
		nthin_chi_(nthin_chi)
	{
		int nchi_samples = niter / nthin_chi;
		int ntheta_samples = niter / nthin_theta;
		ChiSamples_.resize(nchi_samples);
		ThetaSamples_.resize(ntheta_samples);

		ntheta_samples_ = 0;
		nchi_samples_ = 0;
		current_iter_ = 1;
		fix_poppar = false; // default is to sample both the population parameters and characteristics
		fix_char = false;
	}

	// fix the population parameters throughout the sampler?
	void FixPopPar(bool fix=true) { fix_poppar = fix; }
	void FixChar(bool fix=true) { fix_char = fix; }

	// perform a single iterations of the Gibbs Sampler
	virtual void Iterate() {
		if (!fix_char) Daug_.Update();
		if (!fix_poppar) PopPar_.Update();
		current_iter_++;
	}

	// run the MCMC sampler
	void Run() {
		// start the timer, will report on timing automatically Run() is finished
		//boost::timer::auto_cpu_timer auto_timer;

		// first run for a burn-in period
		std::cout << "Doing " << nburnin_ << " iterations of burnin..." << std::endl;
		//boost::progress_display progress_bar(nburnin_); // show a progress bar
		for (int i = 0; i < nburnin_; ++i) {
			Iterate();
			//progress_bar++;
		}
		// reset the current iteration
		current_iter_ = 1;

		// run the main MCMC sampler
		std::cout << "Burnin finished." << std::endl;
		std::cout << "Now doing " << niter_ << " iterations of the Gibbs Sampler..." << std::endl;
		//progress_bar.restart(niter_);

		for (int i = 0; i < niter_; ++i) {
			Iterate();
			if (current_iter_ % nthin_theta_ == 0) {
				// save the value of the population parameter since we've done nthin_theta_ iterations since the last save
				ThetaSamples_[ntheta_samples_] = PopPar_.GetTheta();
				ntheta_samples_++;
			}
			if (current_iter_ % nthin_chi_ == 0) {
				// save the value of the characteristics
				ChiSamples_[nchi_samples_] = Daug_.GetChi();
				nchi_samples_++;
			}
			current_iter_++;
			//progress_bar++;
		}
		// report on the results
		Report();
	}

	// print out useful information on the MCMC sampler results
	virtual void Report() {
		std::cout << "MCMC Report: " << std::endl;
	}

	// grab the MCMC samples
	const vecvec& GetPopSamples() const { return ThetaSamples_; }
	const std::vector<vecvec>& GetCharSamples() const { return ChiSamples_; }

protected:
	int niter_, nburnin_, nthin_chi_, nthin_theta_; // total # of iterations, # of burnin iterations, and thinning amount
	int current_iter_, ntheta_samples_, nchi_samples_;
	bool fix_poppar, fix_char; // is set to true, then keep the values fixed throughout the MCMC sampler
	DataAugmentation& Daug_;
	PopulationPar& PopPar_;
	std::vector<vecvec> ChiSamples_;
	vecvec ThetaSamples_;
};

#endif /* GIBBSSAMPLER_HPP_ */

