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
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

#include "UIncGamma.cpp"

struct integrand_params {
	double beta; double lScale; double uScale;
};

struct innerIntegrand_params {
	double beta; double lScale; double uScale; double r; double fluxLimit; double sigma0; double sigCoef;
};

struct outerIntegrand_params {
	double beta; double lScale; double uScale; double fluxLimit; double sigma0; double sigCoef; double reqRelError; double upperLimit;
};

// Cubature usage. It was the first good solution.
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

double innerIntegrand(double lum, void * params)
{
	struct innerIntegrand_params * parameters = (struct innerIntegrand_params *) params;
	double beta = (parameters->beta);
	double uScale = (parameters->uScale);
	double lScale = (parameters->lScale);
	double r = (parameters->r);
	double fluxLimit = (parameters->fluxLimit);
	double sigma0 = (parameters->sigma0);
	double sigCoef = (parameters->sigCoef);

	/* Compute auxiliary variables */
	double F = lum / (4 * M_PI *r*r);
	double sigma = sqrt(pow(sigma0, 2.0) + pow(sigCoef * F, 2.0));
	double eta = 0.5*(1 + erf((F - fluxLimit) / (sigma*sqrt(2.0))));

	double logLumPDFValue = log(1 - exp(-lum / lScale)) + beta * (log(lum) - log(uScale)) - (lum / uScale);
	double lumPDFValue = exp(logLumPDFValue);
	return eta * lumPDFValue;
}

double outerIntegrand(double r, void * params)
{
	struct outerIntegrand_params * parameters = (struct outerIntegrand_params *) params;
	
	gsl_integration_workspace * w
		= gsl_integration_workspace_alloc(1000);
	double innerResult, error;
	gsl_function innerIntegrandGSLFunc;
	struct innerIntegrand_params innerParams = { (parameters->beta), (parameters->lScale), (parameters->uScale), r, (parameters->fluxLimit), (parameters->sigma0), (parameters->sigCoef) };
	innerIntegrandGSLFunc.function = &innerIntegrand;
	innerIntegrandGSLFunc.params = &innerParams;

	double upperLimit = (parameters->upperLimit);
	double upperLimitForCurrentR = upperLimit * r * r;
	double reqRelError = (parameters->reqRelError);
	gsl_integration_qag(&innerIntegrandGSLFunc, 0.0, upperLimitForCurrentR, 0, reqRelError, 1000, 6, w, &innerResult, &error);
	gsl_integration_workspace_free(w);

	return  r * r * innerResult;
}

// Cubature usage: inner integral depends of outer. It doestn't work.
//int lumIntegrandWithErf(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval)
//{
//	/* Parameters */
//	double beta = ((double *)fdata)[0];
//	double lScale = ((double *)fdata)[1];
//	double uScale = ((double *)fdata)[2];
//	double rmax = ((double *)fdata)[3];
//	double fluxLimit = ((double *)fdata)[4];
//	double sigma0 = ((double *)fdata)[5];
//	double sigCoef = ((double *)fdata)[6];
//	double upperLimit = ((double *)fdata)[7];
//	/* Inputs */
//	double lum = x[0];
//	double r = x[1];
//	double upperLimitForCurrentR = upperLimit * r * r;
//	fval[0] = 0.0;
//	if ((lum > 0.0) && (lum <= upperLimitForCurrentR))
//	{
//		/* Compute auxiliary variables */
//		double F = lum / (4 * M_PI *r*r);
//		double sigma = sqrt(pow(sigma0, 2.0) + pow(sigCoef * F, 2.0));
//		double eta = 0.5*(1 + erf((F - fluxLimit) / (sigma*sqrt(2.0))));
//
//		double logLumPDFValue = log(1 - exp(-lum / lScale)) + beta * (log(lum) - log(uScale)) - (lum / uScale);
//		double lumPDFValue = exp(logLumPDFValue);
//		/* Compute the output value */
//		fval[0] = eta * r * r * lumPDFValue;
//	}
//	return 0; // success
//}

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

	double calculateIntPart(double beta, double l, double u, double rmax, double fluxLimit, double sigma0, double sigCoef, double upperLimit)
	{
		double tag1 = tgamma(beta + 1.0) * (1 - (1 / pow(1 + u / l, beta + 1.0)));
		double tag2 = (tgamma(beta + 2.5) / (pow(rmax, 3.0) * pow(upperLimit, 1.5))) * (1 / (pow(1 + u / l, beta + 1.0) * pow(1 / u + 1 / l, 1.5)) - pow(u, 1.5));
		double tag3 = -uIncGamma.computeValue(beta + 1.0, (pow(rmax, 2.0) * upperLimit) / u);
		double tag4 = uIncGamma.computeValue(beta + 1.0, pow(rmax, 2.0) * upperLimit * (1 / u + 1 / l)) / pow(1 + u / l, beta + 1.0);
		double tag5 = (uIncGamma.computeValue(beta + 2.5, (pow(rmax, 2.0) * upperLimit) / u) * pow(u, 1.5)) / (pow(rmax, 3.0) * pow(upperLimit, 1.5));
		double tag6 = -uIncGamma.computeValue(beta + 2.5, pow(rmax, 2.0) * upperLimit * (1 / u + 1 / l)) / (pow(rmax, 3.0) * pow(upperLimit, 1.5) * pow(1 + u / l, beta + 1.0) * pow(1 / u + 1 / l, 1.5));
		double norm = determineLumFuncNormCnst(beta, l, u) * u;
		double result = norm * (tag1 + tag2 + tag3 + tag4 + tag5 + tag6);
		return result;
	}

	//double calculateIntegral(struct integrand_params parameters)
	//{
	//	double result = 1.0;
	//	if (fluxLimit > 0)
	//	{
	//		double reqRelError = 1e-6;
	//		double erfLimit = 6.0;
	//		double upperLimit = determineUpperLimit(fluxLimit, sigma0, sigCoef, erfLimit);
	//		
	//		gsl_set_error_handler_off();

	//		gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
	//		gsl_function outerIntegrandGSLFunc;
	//		struct outerIntegrand_params outerparams = { parameters.beta, parameters.lScale, parameters.uScale, fluxLimit, sigma0, sigCoef, reqRelError, upperLimit };
	//		outerIntegrandGSLFunc.function = &outerIntegrand;
	//		outerIntegrandGSLFunc.params = &outerparams;

	//		double val, err;
	//		gsl_integration_qag(&outerIntegrandGSLFunc, 0.0, rmax, 0, reqRelError, 1000, 6, w, &val, &err);

	//		double coef = (determineLumFuncNormCnst(parameters.beta, parameters.lScale, parameters.uScale) * 3.0) / (rmax * rmax * rmax);
	//		val *= coef;
	//		double val2 = calculateIntPart(parameters.beta, parameters.lScale, parameters.uScale, rmax, fluxLimit, sigma0, sigCoef, upperLimit);
	//		result = 1.0 + val - val2;

	//		gsl_integration_workspace_free(w);
	//	}
	//	return result;
	//}

	double calculateIntegral(struct integrand_params parameters)
	{
		//clock_t begin = clock();
		//std::cout << "Elapsed time for parameters: beta: " << parameters.beta << " lScale: " << parameters.lScale << " uScale: " << parameters.uScale << "..." << std::endl;
		double result = 1.0;
		if (fluxLimit > 0)
		{
			double reqRelError = 5e-6;
			// The following works but it is a rough estimate:
			//double lumMax = fluxLimit * 4.0 * M_PI * rmax * rmax;
			
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