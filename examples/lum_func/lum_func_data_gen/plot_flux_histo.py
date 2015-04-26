# executing e.g. python plot_flux_histo.py fluxes_cnt_10000.dat filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_10000.dat lumfunc_chi_summary.dat

import argparse as argp
import numpy as np
import datetime as dt
from scipy.stats import norm
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("real_flux_file", help="The file name of real flux data file.", type = str)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("estimated_flux_file", help="The file name of estimated flux data file.", type = str)
parser.add_argument("--sigma0", default = 1.0, help="The sigma0: basic of standard deviation.", type = float)

args = parser.parse_args()
real_flux_file = args.real_flux_file
noisy_flux_file = args.noisy_flux_file
estimated_flux_file = args.estimated_flux_file
sigma0 = args.sigma0

# Wider margins to allow for larger labels; may need to adjust left:
rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125

# Optionally make default line width thicker:
#rc('lines', linewidth=2.0) # doesn't affect frame lines

rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)

rc('savefig', dpi=150)  # mpl's default dpi is 100
rc('axes.formatter', limits=(-4,4))

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

t0 = dt.datetime.today()

real_flux_data=np.loadtxt(real_flux_file)
noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
noisy_flux_data_0=noisy_flux_data[:,0]
estimated_flux_data=np.loadtxt(estimated_flux_file,delimiter=' ',usecols=(0,1))
estimated_flux_data_0=estimated_flux_data[:,0]

real_filt_diff_data = []

for idx in range(0, len(noisy_flux_data_0)):
    actual_sdev = np.sqrt((sigma0)**2 + (0.01*noisy_flux_data_0[idx])**2)
#    real_filt_diff_data.append(np.absolute((real_flux_data[idx]-noisy_flux_data_0[idx])/actual_sdev))
    real_filt_diff_data.append((real_flux_data[idx]-noisy_flux_data_0[idx])/actual_sdev)

real_est_diff_data = []

for idx in range(0, len(estimated_flux_data_0)):
    actual_sdev = np.sqrt((sigma0)**2 + (0.01*estimated_flux_data_0[idx])**2)
#    real_est_diff_data.append(np.absolute((real_flux_data[idx]-estimated_flux_data_0[idx])/actual_sdev))
    real_est_diff_data.append((real_flux_data[idx]-estimated_flux_data_0[idx])/actual_sdev)

x = np.array(real_est_diff_data, float)

xlin = np.linspace(x.min(), x.max(), 300)

fig, ax = subplots()
tit = r'Histogram of real-noisy-estimated fluxes'
ax.set_title(tit)

ax.hist(real_filt_diff_data, bins=xlin, label='real-noisy flux diff', log=False, normed=True, color = 'b')
ax.hist(real_est_diff_data, bins=xlin, label='real-estim flux diff', log=False, normed=True, color = 'g')

mu = 0.0
sig = 1.0
pdf_norm = norm(loc = mu, scale = sig).pdf(xlin)
ax.plot(xlin, pdf_norm, 'r')

ax.legend(loc=0)
savefig('flux_histo.png')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of flux difference histogram:', t1-t0