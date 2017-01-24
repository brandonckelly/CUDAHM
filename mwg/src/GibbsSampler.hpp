/*
 * GibbsSampler.hpp
 *
 *  Created on: Jul 25, 2013
 *      Author: Brandon C. Kelly
 */

#ifndef GIBBSSAMPLER_HPP_
#define GIBBSSAMPLER_HPP_

// std includes
#include <exception>
#include <sstream>
#include <fstream>
#include <string>

// boost includes
#include <boost/shared_ptr.hpp>

// local includes
#include "parameters.cuh"

template<int mfeat, int pchi, int dtheta>
class GibbsSampler
{
public:
	GibbsSampler()
	{
	}

	// constructor for default DataAugmentation and PopulationPar classes
	GibbsSampler(vecvec& meas, vecvec& meas_unc, int niter, int nburnin, int nthin_chi=100, int nthin_theta=1,
			int nThreads=256) :
				niter_(niter), nburnin_(nburnin), nthin_theta_(nthin_theta), nthin_chi_(nthin_chi)
		{
			_InitializeMembers(meas.size(), nThreads);
			// construct DataAugmentation and PopulationPar objects
			Daug_.reset(new DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc));
			PopPar_.reset(new PopulationPar<mfeat, pchi, dtheta>());
			// set the CUDA grid launch parameters and initialize the random number generator on the GPU
			Daug_->SetCudaGrid(nB_, nT_);
			Daug_->InitializeDeviceRNG();
			PopPar_->SetCudaGrid(nB_, nT_);
			// make sure the parameter objects can talk to eachother
			Daug_->SetPopulationPtr(PopPar_);
			PopPar_->SetDataAugPtr(Daug_);

			// Set the initial values of the characteristics and thetas, as well as their proposal covariances
			Initialize();
		}

	// Constructor for subclassed DataAugmentation and PopulationPar classes. In this case the user must supply the
	// pointers to the instantiated subclasses of DataAugmentation and PopulatinPar.
	GibbsSampler(boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > Daug,
			boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > Theta,
			int niter, int nburnin, int nthin_chi=100, int nthin_theta=1, int nThreads=256) :
				niter_(niter), nburnin_(nburnin), nthin_theta_(nthin_theta), nthin_chi_(nthin_chi)
		{
			int ndata = Daug->GetDataDim();
			_InitializeMembers(ndata, nThreads);
			// construct DataAugmentation and PopulationPar objects
			Daug_ = Daug;
			PopPar_ = Theta;
			// set the CUDA grid launch parameters and initialize the random number generator on the GPU
			Daug_->SetCudaGrid(nB_, nT_);
			Daug_->InitializeDeviceRNG();
			PopPar_->SetCudaGrid(nB_, nT_);
			// make sure the parameter objects can talk to eachother
			Daug_->SetPopulationPtr(PopPar_);
			PopPar_->SetDataAugPtr(Daug_);

			// Set the initial values of the characteristics and thetas, as well as their proposal covariances
			Initialize();
		}

	// fix the population parameters throughout the sampler?
	void FixPopPar(bool fix=true) { fix_poppar = fix; }

	// fix the characteristics throughout the sampler?
	void FixChar(bool fix=true) { fix_char = fix; }

	// perform a single iterations of the Gibbs Sampler
	virtual void Iterate() {
		if (!fix_char) Daug_->Update();
		if (!fix_poppar) PopPar_->Update();
		current_iter_++;
	}

	// Initialize the parameter values, their proposal covariances, and the log-densities
	void Initialize() {
		if (!fix_char) Daug_->Initialize();
		if (!fix_poppar) PopPar_->Initialize();
	}

	void RunBurnInPeriod() {
		// first run for a burn-in period
		std::cout << "Doing " << nburnin_ << " iterations of burnin..." << std::endl;

		for (int i = 0; i < nburnin_; ++i) {
			if (i % 1000 == 0) {
				std::cout << i << "..." << std::endl;
			}
			Iterate();
		}
		// TODO: add timer
		// TODO: add a progress bar

		// Burn-in stage is finished, so reset the current iteration and acceptance rates
		std::cout << "Burnin finished." << std::endl;
		Report();
		std::cout << std::endl;
	}

	// run the MCMC sampler
	void Run() {
		RunBurnInPeriod();

		current_iter_ = 1;
		Daug_->ResetAcceptance();
		PopPar_->ResetAcceptance();

		// run the main MCMC sampler
		std::cout << "Now doing " << niter_ << " iterations of the Gibbs Sampler..." << std::endl;

		int ndata = Daug_->GetDataDim();
		int ntheta_samples_num = niter_ / nthin_theta_;
		int nchi_samples_num = niter_ / nthin_chi_;
		for (int i = 0; i < niter_; ++i) {
			if (i % 1000 == 0) {
				std::cout << i << "..." << std::endl;
			}
			Iterate();
			// TODO: How long does saving the values take? Maybe replace these with iterators?
			if (!fix_poppar && (current_iter_ % nthin_theta_ == 0)) {
				double * current_theta = PopPar_->GetTheta();
				// save the value of the population parameter since we've done nthin_theta_ iterations since the last save
				for (int j = 0; j < dtheta; j++)
				{
					int current_idx = j*ntheta_samples_num + ntheta_samples_;
					ThetaSamples_[current_idx] = current_theta[j];
				}
				LogDensPop_Samples_[ntheta_samples_] = PopPar_->GetLogDens();
				ntheta_samples_++;
			}
			if (!fix_char && (current_iter_ % nthin_chi_ == 0) && Daug_->SaveTrace()) {
				double * chi = Daug_->GetChi();
				// save the value of the characteristics
				for (int i = 0; i < ndata; ++i)
				{
					for (int j = 0; j < pchi; ++j)
					{
						int current_idx = (j*ndata + i) * nchi_samples_num + nchi_samples_;
						ChiSamples_[current_idx] = chi[ndata * j + i];
					}
				}
				LogDensMeas_Samples_[nchi_samples_] = Daug_->GetLogDens();
				nchi_samples_++;
			}
		}
		// report on the results
		Report();
	}

	// print out useful information on the MCMC sampler results
	virtual void Report() {
		int naccept_theta = PopPar_->GetNaccept();
		double arate_theta = double(naccept_theta) / (current_iter_ - 1);
		std::cout << "Acceptance rate for Population Parameter is " << arate_theta << std::endl;
		thrust::host_vector<int> naccept_chi = Daug_->GetNaccept();
		double arate_mean = 0.0;
		double arate_max = 0.0;
		double arate_min = 1.0;
		for (int i = 0; i < naccept_chi.size(); ++i) {
			double this_arate = double(naccept_chi[i]) / (current_iter_ - 1.0);
			arate_mean += this_arate / naccept_chi.size();
			arate_min = std::min(this_arate, arate_min);
			arate_max = std::max(this_arate, arate_max);
		}
		std::cout << "Mean acceptance rate for characteristics is " << arate_mean << std::endl;
		std::cout << "Minimum acceptance rate for characteristics is " << arate_min << std::endl;
		std::cout << "Maximum acceptance rate for characteristics is " << arate_max << std::endl;
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

	// grab the pointers to the DataAugmentation and PopulationPar objects
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > GetDaugPtr() { return Daug_; }
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > GetThetaPtr() { return PopPar_; }

	// grab the MCMC samples
	const double * GetPopSamples() const { return ThetaSamples_; }
	const double * GetCharSamples() const { return ChiSamples_; }
	const std::vector<double>& GetLogDensPop() const { return LogDensPop_Samples_; }
	const std::vector<double>& GetLogDensMeas() const { return LogDensMeas_Samples_; }

protected:
	dim3 nB_, nT_;  // CUDA grid launch parameters
	int niter_, nburnin_, nthin_chi_, nthin_theta_; // total # of iterations, # of burnin iterations, and thinning amount
	int current_iter_, ntheta_samples_, nchi_samples_;
	bool fix_poppar, fix_char; // is set to true, then keep the values fixed throughout the MCMC sampler
	boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > Daug_;
	boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > PopPar_;
	double * ChiSamples_;
	double * ThetaSamples_;
	std::vector<double> LogDensMeas_Samples_;
	std::vector<double> LogDensPop_Samples_;

	// initialize the values and sizes of the data members, called by the constructor
	void _InitializeMembers(int ndata, int nThreads) {
		// first do CUDA grid launch
		dim3 nT(nThreads);
		nT_ = nT;
		dim3 nB((ndata + nT.x-1) / nT.x);
		nB_ = nB;
		if (nB.x > 65535)
		{
			std::stringstream errmsg;
			errmsg << "ERROR: Block is too large:\n";
			errmsg << nB.x << " blocks. Max is 65535.\n";
			throw std::runtime_error(errmsg.str());
		}
		// set container sizes
		int nchi_samples = niter_ / nthin_chi_;
		int ntheta_samples = niter_ / nthin_theta_;
		ChiSamples_ = new double[nchi_samples*ndata*pchi];
		for (int idx = 0; idx < nchi_samples*ndata*pchi; idx++)
		{
			ChiSamples_[idx] = 0.0;
		}
		ThetaSamples_ = new double[ntheta_samples*dtheta];
		for (int idx = 0; idx < ntheta_samples*dtheta; idx++)
		{
			ThetaSamples_[idx] = 0.0;
		}
		LogDensMeas_Samples_.resize(nchi_samples);
		LogDensPop_Samples_.resize(ntheta_samples);

		ntheta_samples_ = 0;
		nchi_samples_ = 0;
		current_iter_ = 1;
		fix_poppar = false; // default is to sample both the population parameters and characteristics
		fix_char = false;
	}
};

template<int mfeat, int pchi, int dtheta>
class GibbsSamplerWithCompactMemoryUsage : public GibbsSampler<mfeat, pchi, dtheta>
{
public:
	GibbsSamplerWithCompactMemoryUsage(vecvec& meas, vecvec& meas_unc, int niter, int nburnin, std::string& thetafile, std::string& chifile,
		int nthin_chi = 100, int nthin_theta = 1, int nThreads = 256) : thetafile(thetafile), chifile(chifile)
	{
		niter_ = niter;
		nburnin_ = nburnin;
		nthin_theta_ = nthin_theta; 
		nthin_chi_ = nthin_chi;
		_InitializeMembers(meas.size(), nThreads);
		// construct DataAugmentation and PopulationPar objects
		Daug_.reset(new DataAugmentation<mfeat, pchi, dtheta>(meas, meas_unc));
		PopPar_.reset(new PopulationPar<mfeat, pchi, dtheta>());
		// set the CUDA grid launch parameters and initialize the random number generator on the GPU
		Daug_->SetCudaGrid(nB_, nT_);
		Daug_->InitializeDeviceRNG();
		PopPar_->SetCudaGrid(nB_, nT_);
		// make sure the parameter objects can talk to eachother
		Daug_->SetPopulationPtr(PopPar_);
		PopPar_->SetDataAugPtr(Daug_);

		// Set the initial values of the characteristics and thetas, as well as their proposal covariances
		Initialize();
	}

	GibbsSamplerWithCompactMemoryUsage(boost::shared_ptr<DataAugmentation<mfeat, pchi, dtheta> > Daug,
		boost::shared_ptr<PopulationPar<mfeat, pchi, dtheta> > Theta,
		int niter, int nburnin, std::string& thetafile, std::string& chifile, int nthin_chi = 100, int nthin_theta = 1, int nThreads = 256) :
		thetafile(thetafile), chifile(chifile)
	{
		niter_ = niter;
		nburnin_ = nburnin;
		nthin_theta_ = nthin_theta;
		nthin_chi_ = nthin_chi;
		int ndata = Daug->GetDataDim();
		_InitializeMembers(ndata, nThreads);
		// construct DataAugmentation and PopulationPar objects
		Daug_ = Daug;
		PopPar_ = Theta;
		// set the CUDA grid launch parameters and initialize the random number generator on the GPU
		Daug_->SetCudaGrid(nB_, nT_);
		Daug_->InitializeDeviceRNG();
		PopPar_->SetCudaGrid(nB_, nT_);
		// make sure the parameter objects can talk to eachother
		Daug_->SetPopulationPtr(PopPar_);
		PopPar_->SetDataAugPtr(Daug_);

		// Set the initial values of the characteristics and thetas, as well as their proposal covariances
		Initialize();
	}

	// run the MCMC sampler
	void Run() {
		RunBurnInPeriod();

		current_iter_ = 1;
		Daug_->ResetAcceptance();
		PopPar_->ResetAcceptance();

		// run the main MCMC sampler
		std::cout << "Now doing " << niter_ << " iterations of the Gibbs Sampler..." << std::endl;

		int ndata = Daug_->GetDataDim();
		int ntheta_samples_num = niter_ / nthin_theta_;
		int nchi_samples_num = niter_ / nthin_chi_;
		for (int i = 0; i < niter_; ++i) {
			if (i % 1000 == 0) {
				std::cout << i << "..." << std::endl;
			}
			Iterate();
			// TODO: How long does saving the values take? Maybe replace these with iterators?
			if (!fix_poppar && (current_iter_ % nthin_theta_ == 0)) {
				double * current_theta = PopPar_->GetTheta();
				std::string str = "";
				for (int j = 0; j < dtheta; j++)
				{
					str += std::to_string(current_theta[j]);
					str += " ";
				}
				str += "\n";
				(*outputThetaFile) << str;
				ntheta_samples_++;
			}
			if (!fix_char && (current_iter_ % nthin_chi_ == 0) && Daug_->SaveTrace()) {
				double * chi = Daug_->GetChi();
				std::string str = "";
				for (int i = 0; i < ndata; ++i)
				{
					for (int j = 0; j < pchi; ++j)
					{
						str += std::to_string(chi[ndata * j + i]);
						str += " ";
					}
				}
				str += "\n";
				(*outputChiFile) << str;
				nchi_samples_++;
			}
		}
		// report on the results
		Report();
		std::cout << "Writing results to text files..." << std::endl;
		(*outputThetaFile) << std::flush;
		(*outputChiFile) << std::flush;
	}

protected:
	std::string& thetafile;
	std::string& chifile;
	std::ofstream * outputThetaFile;
	std::ofstream * outputChiFile;
	double * post_mean_i;
	double * post_msqr_i;

	void _InitializeMembers(int ndata, int nThreads) {
		// first do CUDA grid launch
		dim3 nT(nThreads);
		nT_ = nT;
		dim3 nB((ndata + nT.x - 1) / nT.x);
		nB_ = nB;
		if (nB.x > 65535)
		{
			std::stringstream errmsg;
			errmsg << "ERROR: Block is too large:\n";
			errmsg << nB.x << " blocks. Max is 65535.\n";
			throw std::runtime_error(errmsg.str());
		}

		post_mean_i = new double[ndata*pchi];
		for (int idx = 0; idx < ndata*pchi; idx++)
		{
			post_mean_i[idx] = 0.0;
		}

		post_msqr_i = new double[ndata*pchi];
		for (int idx = 0; idx < ndata*pchi; idx++)
		{
			post_msqr_i[idx] = 0.0;
		}

		outputThetaFile = new std::ofstream(thetafile.c_str());
		outputThetaFile->rdbuf()->pubsetbuf(0, 0);

		outputChiFile = new std::ofstream(chifile.c_str());
		outputChiFile->rdbuf()->pubsetbuf(0, 0);

		ntheta_samples_ = 0;
		nchi_samples_ = 0;
		current_iter_ = 1;
		fix_poppar = false; // default is to sample both the population parameters and characteristics
		fix_char = false;
	}
};

#endif /* GIBBSSAMPLER_HPP_ */

