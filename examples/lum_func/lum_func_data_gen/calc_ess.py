#executing e.g. python calc_ess.py lumfunc_thetas_2.dat
#output: n: 10000 , ESS_beta: 5014.78484634 , ESS_l: 2427.30073284 , ESS_u: 5001.69166914
import argparse as argp
import datetime as dt
import numpy as np
from AutoCorrUtil import AutoCorrUtil

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("--until", default = 0, help="This is the maximal k for which the autocorrelation is computed.", type = int)

args = parser.parse_args()
file = args.file
until = args.until

util = AutoCorrUtil()

t0 = dt.datetime.today()

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

if until == 0:
  until = theta_data.shape[0]-1

ess_beta = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 0)
ess_l = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 1)
ess_u = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 2)

print "n:", theta_data.shape[0], ", ESS_beta:", ess_beta, ", ESS_l:", ess_l, ", ESS_u:", ess_u

t1 = dt.datetime.today()
print 'Elapsed time of computation:', t1-t0