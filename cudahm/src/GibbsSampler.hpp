/*
 * GibbsSampler.hpp
 *
 *  Created on: Jul 25, 2013
 *      Author: Brandon C. Kelly
 */

#ifndef GIBBSSAMPLER_HPP_
#define GIBBSSAMPLER_HPP_

// boost includes
#include <boost/shared_ptr.hpp>
//#include <boost/timer/timer.hpp>
//#include <boost/progress.hpp>

// local includes
#include "parameters.cuh"

template<int mfeat, int pchi, int dtheta>
class GibbsSampler
{
public:
	// constructor
	GibbsSampler(double** meas, double** meas_unc, int ndata, dim3& nB, dim3& nT, int niter, int nburnin,
			int nthin_chi=100, int nthin_theta=1) :
				niter_(niter), nburnin_(nburnin), nthin_theta_(nthin_theta), nthin_chi_(nthin_chi)
		{
			// construct DataAugmentation and PopulationPar objects
			Daug_.reset(new DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc, ndata, nB, nT));
			PopPar_.reset(new PopulationPar<mfeat, pchi, dtheta>(nB, nT));
			Daug_->SetPopulationPtr(PopPar_);
			PopPar_->SetDataAugPtr(Daug_);

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

			Initialize();
		}

	// fix the population parameters throughout the sampler?
	void FixPopPar(bool fix=true) { fix_poppar = fix; }
	void FixChar(bool fix=true) { fix_char = fix; }

	// perform a single iterations of the Gibbs Sampler
	virtual void Iterate() {
		if (!fix_char) Daug_->Update();
		if (!fix_poppar) PopPar_->Update();
		current_iter_++;
	}

	void Initialize() {
		// Initialize the parameter values and the log-densities
		if (!fix_char) Daug_->Initialize();
		if (!fix_poppar) PopPar_->Initialize();
	}

	// run the MCMC sampler
	void Run() {
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
		Daug_->ResetAcceptance();
		PopPar_->ResetAcceptance();

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
				ThetaSamples_[ntheta_samples_] = PopPar_->GetTheta();
				LogDensPop_Samples_[ntheta_samples_] = PopPar_->GetLogDens();
				ntheta_samples_++;
			}
			if (!fix_char && (current_iter_ % nthin_chi_ == 0)) {
				// save the value of the characteristics
				ChiSamples_[nchi_samples_] = Daug_->GetChi();
				LogDensMeas_Samples_[nchi_samples_] = Daug_->GetLogDens();
				nchi_samples_++;
			}
			//progress_bar++;
		}
		// report on the results
		Report();
	}

	// print out useful information on the MCMC sampler results
	virtual void Report() {
		std::cout << "MCMC Report: " << std::endl;
	}

	// save the sampled characteristic values? not saving them can speed up the sampler since we do not need to
	// read the values from the GPU
	void NoCharSave(bool nosave = true) {
		if (nosave) {
			Daug_->SetSaveTrace(false);
		} else {
			Daug_->SetSaveTrace(true);
		}
	}

	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > GetDaugPtr() { return Daug_; }
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > GetThetaPtr() { return PopPar_; }

	// grab the MCMC samples
	const vecvec& GetPopSamples() const { return ThetaSamples_; }
	const std::vector<vecvec>& GetCharSamples() const { return ChiSamples_; }
	const std::vector<double>& GetLogDensPop() const { return LogDensPop_Samples_; }
	const std::vector<double>& GetLogDensMeas() const { return LogDensMeas_Samples_; }

protected:
	int niter_, nburnin_, nthin_chi_, nthin_theta_; // total # of iterations, # of burnin iterations, and thinning amount
	int current_iter_, ntheta_samples_, nchi_samples_;
	bool fix_poppar, fix_char; // is set to true, then keep the values fixed throughout the MCMC sampler
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > Daug_;
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > PopPar_;
	std::vector<vecvec> ChiSamples_;
	vecvec ThetaSamples_;
	std::vector<double> LogDensMeas_Samples_;
	std::vector<double> LogDensPop_Samples_;
};

#endif /* GIBBSSAMPLER_HPP_ */

