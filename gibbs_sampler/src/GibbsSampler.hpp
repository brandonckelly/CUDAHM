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
			int nthin_chi=10, int nthin_theta=1);

	// fix the population parameters throughout the sampler?
	void FixPopPar(bool fix=true) { fix_poppar = fix; }
	void FixChar(bool fix=true) { fix_char = fix; }

	// perform a single iterations of the Gibbs Sampler
	virtual void Iterate();

	// run the MCMC sampler
	void Run();

	// print out useful information on the MCMC sampler results
	virtual void Report();

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

