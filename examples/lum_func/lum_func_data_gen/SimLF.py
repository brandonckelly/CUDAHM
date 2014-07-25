"""
Created 2014-07-15 by Janos Szalai-Gindl and Tamas Budavari;
"""
import argparse as argp
from SimLFUtil import SimLFUtil

# Set up a SimLFUtil instance:
parser = argp.ArgumentParser()
parser.add_argument("gamma", help="The gamma parameter of 'Break-By-1 Truncated Power Law'. Only gamma <= 0 currently supported!", type=float)
parser.add_argument("scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("sdev", help="The standard deviation of Gaussian noise which is added to flux data", type=float)
parser.add_argument("--lower_scale", default = 1., help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("--n_samp", default = 10000000, help="The number of required samples", type=int)
parser.add_argument("--r_max", default = 4000, help="The maximal distance", type=int)
parser.add_argument("--thr_coef", default = 5, help=("The threshold coefficient. This value is "
"a multiplication factor of the threshold as the standard deviation is also ((threshold) = (coefficient) x (standard deviation))."), type=int)
parser.add_argument("--n_bins", default = 100, help="The number of bins for plotting histograms", type=int)

#For example:
#gamma = -1.2
#scale = 100.
#sdev = 1e-10
args = parser.parse_args()

utl = SimLFUtil(args)

# Running sampling, plotting, writing
utl.batch_process()