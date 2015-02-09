# executing e.g. python plot_lumfunc_w_fl_histo.py -1.5 50000000000.0 5000000000000.0 filtered_fluxes_w_G_noise_mu_0.0_sig_1.0_fl_5.0_cnt_10000.dat dists_cnt_10000.dat 10000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0
import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
import datetime as dt
from scipy import stats

#ion()

t0 = dt.datetime.today()

parser = argp.ArgumentParser()
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("noisy_flux_file", help="The file name of noisy flux data file.", type = str)
parser.add_argument("dist_file", help="The file name of distance data file.", type = str)
parser.add_argument("obj_num", help="The object number of MCMC method", type=int)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)

args = parser.parse_args()
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
noisy_flux_file = args.noisy_flux_file
dist_file = args.dist_file
obj_num = args.obj_num
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor

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

noisy_flux_data=np.loadtxt(noisy_flux_file,delimiter=' ',usecols=(0,1))
dist_data=np.loadtxt(dist_file)
lum_data = []

for idx in range(0, noisy_flux_data.shape[0]):
    lum_data.append(noisy_flux_data[idx][0] * 4.0 * np.pi * (dist_data[idx]**2))

print lum_data[2]

#figsize=(10, 10)
fig_log = figure(figsize=(15.75, 10))
xlabel('$x$')
ylabel('$p(x)$')
tit = r'Obj. num.: %d;' % (obj_num)
fig_log.suptitle(tit, fontsize=18, fontweight='bold')

xlog = np.logspace(8, 13, 300)

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
lbl_0 = 'BB1 (%5.2f,%5.2e,%5.2e)' % (beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlog)
figure(fig_log.number)
loglog(xlog, pdf_0, 'r-', linewidth=2, label=lbl_0, zorder=3)

#lbins_lums = np.logspace(np.log10(lum_data.min()),np.log10(lum_data.max()),n_bins+1)
figure(fig_log.number)
hist(lum_data, bins=xlog, label='flux', log=True, normed=True)

legend(loc=3)
savefig('lumfunc_w_fl_histo.png')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0