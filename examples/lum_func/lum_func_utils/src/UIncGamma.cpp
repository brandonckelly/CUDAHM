/*
* UIncGamma.cuh
*
*  Created on: March 11, 2015
*      Author: Janos M. Szalai-Gindl
*
* This upper incomplete gamma implementation is based on the article:
*	Gautschi, Walter. "A computational procedure for incomplete gamma functions."
*	ACM Transactions on Mathematical Software (TOMS) 5.4 (1979): 466-481.
*
*/

#ifndef UINCGAMMA_CUH_
#define UINCGAMMA_CUH_

#include <math.h>

// Euler–Mascheroni constant
const double EulerMascheroniConst = 0.57721566490153286060;

class UIncGamma
{
public:
	UIncGamma() : KummerLim(20), gStarLim(20), ExpIntLim(15), CntnedFracLim(30)
	{
	}

	UIncGamma(int KummerLim, int gStarLim, int ExpIntLim, int CntnedFracLim) : KummerLim(KummerLim),
		gStarLim(gStarLim), ExpIntLim(ExpIntLim), CntnedFracLim(CntnedFracLim)
	{
	}

	double computeValue(double a, double x)
	{
		double result = 0.0;
		if ((x >= 0) && (x <= 1.5) && (a >= -0.5) && (a <= alphaStar(x))){
			if (a == 0.0)
			{
				result = computeInRegion1Spec(x);
			}
			else
			{
				result = computeInRegion1(a, x);
			}
		}
		else if ((x >= 0) && (x <= 1.5) && (a <= -0.5))
		{
			result = computeInRegion2(a, x);
		}
		else if ((x > 1.5) && (a <= alphaStar(x)))
		{
			result = computeInRegion3(a, x);
		}
		else if (a > alphaStar(x))
		{
			result = computeInRegion4(a, x);
		}
		return result;
	}
private:
	// The limit for the Kummer's function computation
	int KummerLim;
	// The limit for the g* function computation
	int gStarLim;
	// The limit for Taylor series of exponential integral E_1(x) computation
	int ExpIntLim;
	// The limit for the Legendre's continued fraction computation
	int CntnedFracLim;

	double alphaStar(double x)
	{
		double result;
		if (x >= 0.25)
		{
			result = x + 0.25;
		}
		else
		{
			result = log(0.5) / log(x);
		}
		return result;
	}

	double computeKummerFunc(double a, double b, double z)
	{
		double aProd = 1.0;
		double bProd = 1.0;
		double fact = 1.0;
		double result = 1.0;
		for (int idx = 0; idx < KummerLim + 1; idx++)
		{
			aProd *= (a + idx);
			bProd *= (b + idx);
			fact *= (idx + 1.0);
			result += (aProd * pow(z, idx + 1)) / (bProd * fact);
		}
		return result;
	}

	double computeGammaStar(double a, double x)
	{
		double result = (exp(-x) * computeKummerFunc(1.0, a + 1, x)) / tgamma(a + 1.0);
		return result;
	}

	double computeInRegion1(double a, double x)
	{
		double result = tgamma(a) * (1 - pow(x, a) * computeGammaStar(a, x));
		return result;
	}

	double computeInRegion1Spec(double x)
	{
		// When the a=0 the upper incomplete gamma function is approximated
		// by Taylor series of exponential integral E_1(x)
		double frac = 1.0;
		double sum = 0.0;
		for (int idx = 1; idx < ExpIntLim + 1; idx++)
		{
			sum += (pow(-x, idx) / (idx * frac));
			frac *= (idx + 1);
		}
		double result = -EulerMascheroniConst - log(x) - sum;
		return result;
	}

	double computeInRegion2(double a, double x)
	{
		int m = (int)(0.5 - a);
		double eps = a + (double)m; //  ==> -0.5 < eps <= 0.5
		double result = exp(x) * pow(x, -eps) * computeValue(eps, x);
		for (int idx = 1; idx < m + 1; idx++)
		{
			result = (1.0 / (idx - eps)) * (1 - x * result);
		}
		return exp(-x) * pow(x, a) * result;
	}

	double computeCntnedFrac(double a, double x)
	{
		double * aList = new double[CntnedFracLim + 1];
		double * bList = new double[CntnedFracLim + 1];
		for (int idx = 0; idx < CntnedFracLim + 1; idx++)
		{
			aList[idx] = 2.0 * idx + 1.0 - a;
			bList[idx] = idx * (a - idx);
		}
		bList[0] = 1.0;
		double result = 0.0;
		for (int idx = CntnedFracLim; idx >= 0; idx--)
		{
			result = bList[idx] / (x + aList[idx] + result);
		}
		delete[] aList;
		delete[] bList;
		return result;
	}

	double computeInRegion3(double a, double x)
	{
		double result = exp(-x) * pow(x, a) * computeCntnedFrac(a, x);
		return result;
	}

	double lowerCaseGStar(double a, double x)
	{
		double result = 0.0;
		for (int idx = 0; idx < gStarLim + 1; idx++)
		{
			result += pow(x, idx) / tgamma(a + idx + 1.0);
		}
		result *= pow(x, a) * exp(-x);
		return result;
	}

	double computeInRegion4(double a, double x)
	{
		double result = tgamma(a) * (1 - lowerCaseGStar(a, x));
		return result;
	}
};

#endif /* UINCGAMMA_CUH_ */