# executing e.g. python plot_lums_dens.py lums_cnt_10000.dat -1.5 1.0 100.0 -1.4933 1.0032 95.0654
import argparse as argp
import numpy as np
from matplotlib.pyplot import *
from bb1truncpl import BB1TruncPL

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of luminosities.", type = str)
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("mean_beta", help="The mean of beta parameter value of MCMC method", type=float)
parser.add_argument("mean_lower_scale", help="The mean of lower scale of MCMC method", type=float)
parser.add_argument("mean_upper_scale", help="The mean of upper scale of MCMC method", type=float)

args = parser.parse_args()
file = args.file
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
mean_beta = args.mean_beta
mean_lower_scale = args.mean_lower_scale
mean_upper_scale = args.mean_upper_scale

lums_data=np.loadtxt(file)
#lbins_lums = np.logspace(np.log10(lums_data.min()),np.log10(lums_data.max()),11)

lbins_lums = np.logspace(-3,3,20)

bb1_0 = BB1TruncPL(beta,lower_scale,upper_scale)
xlog = np.logspace(-3, 3, 300)
pdf_0 = bb1_0.pdf(xlog)
lbl_0 = 'BB1 (%5.4f,%5.4f,%5.4f)' % (beta,lower_scale,upper_scale)

#print "pdf_0 at 1.4166: ", bb1_0.pdf(1.4166)

bb1_1 = BB1TruncPL(mean_beta,mean_lower_scale,mean_upper_scale)
pdf_1 = bb1_1.pdf(xlog)
lbl_1 = 'Mean BB1 (%5.4f,%5.4f,%5.4f)' % (mean_beta,mean_lower_scale,mean_upper_scale)

#print "pdf_1 at 1.4166: ", bb1_1.pdf(1.4166)

fig_log = figure()
loglog(xlog, pdf_0, 'r-', linewidth=2, label=lbl_0, zorder=2)
loglog(xlog, pdf_1, 'm-', linewidth=2, label=lbl_1, zorder=2)
hist(lums_data, bins=lbins_lums, color=(0.0,0.5,0.0), label='lum data', log=True, normed=True)

legend(loc=3)
savefig('lums_dens.png',dpi=120)