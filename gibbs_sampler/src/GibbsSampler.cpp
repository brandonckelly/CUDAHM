/*
 * GibbsSampler.cpp
 *
 *  Created on: Jul 28, 2013
 *      Author: brandonkelly
 */

// local includes
#include "parameters.hpp"
#include "GibbsSampler.hpp"

GibbsSampler::GibbsSampler(DataAugmentation& Daug, PopulationPar& PopPar, int niter, int nburnin, int nthin_chi,
	int nthin_theta) : Daug_(Daug), PopPar_(PopPar), niter_(niter), nburnin_(nburnin), nthin_theta_(nthin_theta),
	nthin_chi_(nthin_chi)
{
	int nchi_samples = niter / nthin_chi;
	int ntheta_samples = niter / nthin_theta;
	ChiSamples_.resize(nchi_samples);
	ThetaSamples_.resize(ntheta_samples);
	LogDensMeas_Samples_.resize(nchi_samples);
	LogDensPop_Samples_.resize(ntheta_samples);

	ntheta_samples_ = 0;
	nchi_samples_ = 0;
	current_iter_ = 1;
	fix_poppar = false; // default is to sample both the population parameters and characteristics
	fix_char = false;
}

void GibbsSampler::Iterate()
{
	if (!fix_char) Daug_.Update();
	if (!fix_poppar) PopPar_.Update();
	current_iter_++;
}

void GibbsSampler::Run()
{
	// start the timer, will report on timing automatically Run() is finished
	//boost::timer::auto_cpu_timer auto_timer;

	// first run for a burn-in period
	std::cout << "Doing " << nburnin_ << " iterations of burnin..." << std::endl;
	//boost::progress_display progress_bar(nburnin_); // show a progress bar
	for (int i = 0; i < nburnin_; ++i) {
		if (i % 1000 == 0) {
			std::cout << i << "..." << std::endl;
		}
		Iterate();
		//progress_bar++;
	}
	// TODO: print out acceptance rates during burnin

	// reset the current iteration and acceptance rates
	current_iter_ = 1;
	Daug_.ResetAcceptance();
	PopPar_.ResetAcceptance();

	// run the main MCMC sampler
	std::cout << "Burnin finished." << std::endl;
	std::cout << "Now doing " << niter_ << " iterations of the Gibbs Sampler..." << std::endl;
	//progress_bar.restart(niter_);

	for (int i = 0; i < niter_; ++i) {
		if (i % 1000 == 0) {
			std::cout << i << "..." << std::endl;
		}
		Iterate();
		// TODO: How long does saving the values take? Maybe replace these with iterators?
		if (!fix_poppar && (current_iter_ % nthin_theta_ == 0)) {
			// save the value of the population parameter since we've done nthin_theta_ iterations since the last save
			ThetaSamples_[ntheta_samples_] = PopPar_.GetTheta();
			LogDensPop_Samples_[ntheta_samples_] = PopPar_.GetLogDens();
			ntheta_samples_++;
		}
		if (!fix_char && (current_iter_ % nthin_chi_ == 0)) {
			// save the value of the characteristics
			ChiSamples_[nchi_samples_] = Daug_.GetChi();
			LogDensMeas_Samples_[nchi_samples_] = Daug_.GetLogDens();
			nchi_samples_++;
		}
		//progress_bar++;
	}
	// report on the results
	Report();
}

void GibbsSampler::Report()
{
	std::cout << "MCMC Report: " << std::endl;
}
