/*
 * DustPopPar.hpp
 *
 *  Created on: Mar 21, 2014
 *      Author: brandonkelly
 */

#ifndef DUSTPOPPAR_HPP_
#define DUSTPOPPAR_HPP_

#include "parameters.cuh"

/*
 * Override PopulationPar's LogPrior method, since we do not want a uniform prior on theta.
 */

template <int mfeat, int pchi, int dtheta>
class DustPopPar: public PopulationPar<mfeat, pchi, dtheta>
{
	double LogPrior(hvector theta) {
		return 0.0;
	}
};

#endif /* DUSTPOPPAR_HPP_ */
