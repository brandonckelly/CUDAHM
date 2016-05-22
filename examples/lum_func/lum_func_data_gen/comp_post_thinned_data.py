#executing e.g. python comp_post_thinned_data.py lumfunc_thetas_2.dat 0.25
import argparse as argp
import numpy as np
from AutoCorrUtil import AutoCorrUtil

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("thinning_rate", help="This is the holding rate of the input theta data file", type = float)
parser.add_argument("--until", default = 0, help="This is the maximal k for which the autocorrelation is computed.", type = int)

args = parser.parse_args()
file = args.file
thinning_rate = args.thinning_rate
until = args.until

util = AutoCorrUtil()

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

nthin_theta = int(1.0/thinning_rate)

n_thinned_data = int(theta_data.shape[0]*thinning_rate)

thinned_theta_data = np.zeros((n_thinned_data,3))

thinned_idx = 0
for iter_idx in range(0,theta_data.shape[0]):
  if (((iter_idx % nthin_theta) == 0) and (thinned_idx < n_thinned_data)):
    thinned_theta_data[thinned_idx] = theta_data[iter_idx]
    thinned_idx += 1

if until == 0:
  until = thinned_theta_data.shape[0]-1

#print thinned_theta_data
#ess_corrmx_beta = util.effectiveSampleSizeBasedOnCorrMX(thinned_theta_data.shape[0],thinned_theta_data, 0)
#print "ESS_corrmx_beta:",ess_corrmx_beta

m_data = np.mean(thinned_theta_data, axis = 0)[0]
var_data = np.var(thinned_theta_data, axis = 0)[0]

autocorr_beta_lag1 = util.autocorr(1,thinned_theta_data.shape[0],thinned_theta_data,m_data,var_data,0)
autocorr_l_lag1 = util.autocorr(1,thinned_theta_data.shape[0],thinned_theta_data,m_data,var_data,1)
autocorr_u_lag1 = util.autocorr(1,thinned_theta_data.shape[0],thinned_theta_data,m_data,var_data,2)

print "ESS_beta_lag1:", (thinned_theta_data.shape[0]*(float(1-autocorr_beta_lag1)/float(1+autocorr_beta_lag1)))
print "ESS_l_lag1:", (thinned_theta_data.shape[0]*(float(1-autocorr_l_lag1)/float(1+autocorr_l_lag1)))
print "ESS_u_lag1:", (thinned_theta_data.shape[0]*(float(1-autocorr_u_lag1)/float(1+autocorr_u_lag1)))

ess_beta = util.effectiveSampleSizePymc(until, thinned_theta_data.shape[0], thinned_theta_data, 0)
ess_l = util.effectiveSampleSizePymc(until, thinned_theta_data.shape[0], thinned_theta_data, 1)
ess_u = util.effectiveSampleSizePymc(until, thinned_theta_data.shape[0], thinned_theta_data, 2)

print "thinned n:", n_thinned_data, ", ESS_beta:", ess_beta, ", ESS_l:", ess_l, ", ESS_u:", ess_u