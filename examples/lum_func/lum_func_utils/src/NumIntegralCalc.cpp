/*
* NumIntegralCalc.cuh
*
*  Created on: March 11, 2015
*      Author: Janos M. Szalai-Gindl
*
*/
#ifndef NUMINTEGRALCALC_CUH_
#define NUMINTEGRALCALC_CUH_

#define _USE_MATH_DEFINES // for pi constant
#include <math.h>

#include <time.h>
#include <cubature.h>

#include "UIncGamma.cpp"

struct integrand_params {
	double beta; double lScale; double uScale;
};

// Cubature usage.
int lumIntegrandWithErf(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval)
{
	/* Parameters */
	double beta = ((double *)fdata)[0];
	double lScale = ((double *)fdata)[1];
	double uScale = ((double *)fdata)[2];
	double rmax = ((double *)fdata)[3];
	double fluxLimit = ((double *)fdata)[4];
	double sigma0 = ((double *)fdata)[5];
	double sigCoef = ((double *)fdata)[6];
	/* Inputs */
	double lum = x[0];
	double r = x[1];
	fval[0] = 0.0;
	if (lum > 0.0)
	{
		/* Compute auxiliary variables */
		double F = lum / (4 * M_PI *r*r);
		double sigma = sqrt(pow(sigma0, 2.0) + pow(sigCoef * F, 2.0));
		double eta = 0.5*(1 + erf((F - fluxLimit) / (sigma*sqrt(2.0))));

		double logLumPDFValue = log(1 - exp(-lum / lScale)) + beta * (log(lum) - log(uScale)) - (lum / uScale);
		double lumPDFValue = exp(logLumPDFValue);
		/* Compute the output value */
		fval[0] = eta * r * r * lumPDFValue;
	}
	return 0; // success
}

class NumIntegralCalc
{
public:
	NumIntegralCalc(UIncGamma& uIncGamma, double rmax, double fluxLimit, double sigma0, double sigCoef, double erfLimit) : uIncGamma(uIncGamma), rmax(rmax), fluxLimit(fluxLimit),
		sigma0(sigma0), sigCoef(sigCoef)
	{
		double upperLimit = determineUpperLimit(fluxLimit, sigma0, sigCoef, erfLimit);
		lumMax = upperLimit * rmax * rmax;
	}

	double determineLumFuncNormCnst(double beta, double lScale, double uScale)
	{
		double coef = 1.0;
		if ((beta >= -1.0001) && (beta <= -0.9999))
		{
			coef = coef * exp(-log(uScale * log(1.0 + uScale / lScale)));
		}
		else
		{
			coef = coef / (uScale *	tgamma(beta + 1.0) * (1.0 - (1.0 / pow(1.0 + (uScale / lScale), beta + 1.0))));
		}
		return coef;
	}

	double calculateLumFuncAndDistCommonCDF(double lumMax, double beta, double l, double u, double lumFuncNormCnst)
	{
		double result = lumFuncNormCnst * u;
		double temp = tgamma(beta + 1.0) - uIncGamma.computeValue(beta + 1.0, lumMax / u);
		temp += (uIncGamma.computeValue(beta + 1.0, lumMax * (1 / u + 1 / l)) - tgamma(beta + 1.0)) / pow(1 + u / l, beta + 1.0);
		result *= temp;
		return result;
	}

	double determineUpperLimit(double fluxLimit, double sigma0, double sigCoef, double erfLimit)
	{
		double temp = 1.0 - 2.0 * pow(erfLimit * sigCoef, 2.0);
		double result = (4.0 * M_PI) / temp;
		result *= (fluxLimit + sqrt(2.0) * erfLimit * sqrt(temp * pow(sigma0, 2.0) + pow(fluxLimit * sigCoef, 2.0)));
		return result;
	}

	double calculateIntegral(struct integrand_params parameters)
	{
		//clock_t begin = clock();
		//std::cout << "Elapsed time for parameters: beta: " << parameters.beta << " lScale: " << parameters.lScale << " uScale: " << parameters.uScale << "..." << std::endl;
		double result = 1.0;
		if (fluxLimit > 0)
		{
			double reqRelError = 5e-6;			
			double params[8] = { parameters.beta, parameters.lScale, parameters.uScale, rmax, fluxLimit, sigma0, sigCoef, lumMax };
			// Calculate integral with luminosity over (0.0, lumMax):
			double xmin[2] = { 0.0, 0.0 }, xmax[2] = { lumMax, rmax }, val, err;
			hcubature(1, lumIntegrandWithErf, &params, 2, xmin, xmax, 0, 0, reqRelError, ERROR_INDIVIDUAL, &val, &err);
			double lumFuncNormCnst = determineLumFuncNormCnst(parameters.beta, parameters.lScale, parameters.uScale);
			double coef = (lumFuncNormCnst * 3.0) / (rmax * rmax * rmax);
			val *= coef;
			// Calculate luminosity and distance common CDF:
			double val2 = calculateLumFuncAndDistCommonCDF(lumMax, parameters.beta, parameters.lScale, parameters.uScale, lumFuncNormCnst);
			result = 1.0 + val - val2;
		}
		//clock_t end = clock();
		//double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		//std::cout << "...      " << elapsed_secs << " sec" << std::endl;
		return result;
	}
private:
	UIncGamma& uIncGamma;
	double rmax;
	double fluxLimit;
	double sigma0;
	double sigCoef;
	double lumMax;
};

#endif /* NUMINTEGRALCALC_CUH_ */