CUDAHM
======

Routines for using CUDA to accelerate Bayesian inference of Hierarchical Models using Markov Chain Monte Carlo with GPUs.

Description
-----------

`CUDAHM` enables one to easily and rapidly construct an MCMC sampler for a three-level hierarchical model, requiring the user to supply only a minimimal amount of CUDA code. `CUDAHM` assumes that a set of measurements are available for a sample of objects, and that these measurements are related to an unobserved set of characteristics for each object. For example, the measurements could be the spectral energy distributions of a sample of galaxies, and the unknown characteristics could be the physical quantities of the galaxies, such as mass, distance, age, etc. The measured spectral energy distributions depend on the unknown physical quantities, which enables one to derive their values from the measurements. The characteristics are also assumed to be independently and identically sampled from a parent population with unknown parameters (e.g., a Normal distribution with unkown mean and variance). `CUDAHM` enables one to simultaneously sample the values of the characteristics and the parameters of their parent population from their joint posterior probability distribution.

CUDAHM using a Metropolis-within-Gibbs sampler. Each iteration of the MCMC sampler performs the following steps:

1. For each object, update the characteristics given the measured data and the current value of the parent population parameters.
2. Update the parent population parameters given the current values of the characteristics.

Both steps are done using the *Robust adaptive Metropolis algorithm with coerced acceptance rate* (Vihola 2012, Statistics and Computing, 22, pp 997-1008) using a multivariate normal proposal distribution and a target acceptance rate of 0.4. Step 1 is done in parallel for each objects on the GPU, while the proposals for Step 2 are generated on the CPU, but the calculations are performed on the GPU.

Code Structure
--------------

There are three main classes in CUDAHM:

* `DataAugmentation`- This class controls the updates of the set of characteristics.
* `PopulationPar`- This class controls the updates of the parent population parameters.
* `GibbsSampler`- This class runs the MCMC sampler.

The simplest use case involves only instantiating the `GibbsSampler` class, since this will internally construct a `DataAugmentation` and `PopulationPar` object. However, if one wants to subclass the `DataAugmentation` or `PopulationPar` classes then these pointers to the instances of these classes must be provided to the `GibbsSampler` constructore. In general this is only needed if one want to override the default methods for setting the initial values, or if one want to override the default prior on the parent population parameters (an uninformative uniform distribution).

There are two functions that the use must provide: a function that compute the logarithm of the probability density of the measurements given the characteristics for each object in the sample, and a function that computes the logarithm of the probability density of the characteristics given the parent population parameters. These functions live on the GPU and must be written in CUDA. The file `cudahm_blueprint.cu` under the `cudahm` directory contains a blueprint with documentation that the user may use when constructing their own MCMC sampler. In addition, the following examples provide additional guidance:

* `normnorm`: A simple model where the measurements for each object are assumed to be drawn from 3-dimensional normal distribution with known covariance matrix and unknown mean vectors, and the unknown means are themselves drawn from a 3-dimensional normal distribution with unknown mean vector and known covariance matrix. The MCMC sampler provides samples of the unknown mean vectors for each object in the sample and for the parent population mean.
* `dusthm`: A more realistic model where the measurements for each object in the sample are a 5-dimensional vector drawn from a normal distribution with known diagonal covariance matrix. The 5-dimensional mean vector for each obect is a nonlinear function of three unknown characteristics. The parent population for the characteristics is assumed to be a 3-dimensional Student's t distribution with unknown mean and scale matrix. In this example both the `DataAugmentation` and `PopulationPar` class are subclassed in order to override their default values for setting the initial values of the parameters and for the prior on the parameters of the parent population.

Benchmarks
----------

In progress.

Installation
------------

In progress. For now probably the easiest thing is to just use the NVIDIA Nsight IDE that comes with CUDA to build the code. Once you build the code we recommend you run the unit tests under the `tests` directory. Note that you will need to build the code in seperate compilation mode.

Dependencies
------------

`CUDAHM` depends on the following libraries:

* `boost` C++ Libraries
* `CUDA` (at least v5.0)

In order to use `CUDAHM` you will need a NVIDIA GPU of CUDA compute capability 2.0 or higher.
