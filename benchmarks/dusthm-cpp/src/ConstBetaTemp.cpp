//
//  ConstBetaTemp.cpp
//  dusthm-cpp
//
//  Created by Brandon Kelly on 4/30/14.
//  Copyright (c) 2014 Brandon C. Kelly. All rights reserved.
//

#include <stdio.h>
#include <ConstBetaTemp.hpp>

// frequencies correspond to {500, 350, 250, 170, 70} microns, the Herschel bands
const double nu[5] = {5.99584916e11, 8.56549880e11, 1.19916983e12, 1.87370286e12, 2.99792458e12};
const double nu_ref = 2.3e11;  // 230 GHz

// physical constants, cgs
const double clight = 2.99792458e10;
const double hplanck = 6.6260755e-27;
const double kboltz = 1.380658e-16;

// Compute the model dust SED, a modified blackbody
double modified_blackbody(double freq, double C, double beta, double T) {
    double sed = 2.0 * hplanck * freq * freq * freq / (clight * clight) / (exp(hplanck * freq / (kboltz * T)) - 1.0);
    sed *= C * pow(freq / nu_ref, beta);
    
    return sed;
}