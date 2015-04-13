"""
Created 2014-07-15 by Janos Szalai-Gindl and Tamas Budavari;
"""
import argparse as argp
import numpy as np
from SimLFUtil import SimLFUtil

# Set up a SimLFUtil instance:
parser = argp.ArgumentParser()
parser.add_argument("gamma", help="The gamma parameter of 'Break-By-1 Truncated Power Law'. Only gamma <= 0 currently supported!", type=float)
parser.add_argument("scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("sdev", help="The standard deviation of Gaussian noise which is added to flux data", type=float)
parser.add_argument("--lower_scale", default = 1., help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("--n_samp", default = 10000000, help="The number of required samples", type=int)
parser.add_argument("--r_max", default = 0.0, help="The maximal distance", type=float)
parser.add_argument("--n_bins", default = 100, help="The number of bins for plotting histograms", type=int)
parser.add_argument("--flux_limit", default = 0.0, help="The flux limit: this is a threshold below which the fluxes are dropped from sample", type=float)

#For example:
#gamma = -1.2
#scale = 100.
#sdev = 1e-10
args = parser.parse_args()

tmp_r_max = args.r_max
if(tmp_r_max==0.0):
  if(args.flux_limit != 0.0):
    tmp_r_max = np.sqrt(np.exp(32.0)/(4.0 * np.pi * args.flux_limit))
  else:
    tmp_r_max = 4000.0

args.r_max = tmp_r_max
print "The maximal distance:", args.r_max

utl = SimLFUtil(args)

# Running sampling, plotting, writing
utl.batch_process()