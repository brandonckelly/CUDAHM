# executing e.g. python _plot_lumfunc_histo.py -1.32 45000000000.0 6000000000000.0 fluxes_cnt2_1500000.dat dists_cnt2_1500000.dat 1500000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0

import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
import datetime as dt
from scipy import stats
from scipy.special import erf
from scipy.integrate import quad

#ion()

t0 = dt.datetime.today()

parser = argp.ArgumentParser()
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("flux_file", help="The file name of flux data file.", type = str)
parser.add_argument("dist_file", help="The file name of distance data file.", type = str)
parser.add_argument("obj_num", help="The object number of MCMC method", type=int)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--T", default = 5.0, help="The flux limit", type=float)
parser.add_argument("--rmax", default = 4000.0, help="The maximal distance", type=float)
parser.add_argument("--mu", default = 0.934197, help="The value of mu(theta)", type=float)

args = parser.parse_args()
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
flux_file = args.flux_file
dist_file = args.dist_file
obj_num = args.obj_num
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
T = args.T
rmax = args.rmax
mu = args.mu

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

flux_data=np.loadtxt(flux_file)
dist_data=np.loadtxt(dist_file)
lum_data = []

for idx in range(0, flux_data.shape[0]):
    lum_data.append(flux_data[idx] * 4.0 * np.pi * (dist_data[idx]**2))
#    lum_data.append(flux_data[idx] * 4.0 * np.pi * (dist_data[idx]**2) * np.power(4000.0/dist_data[idx], 3) * np.power(5.0/flux_data[idx], 1.5))

print lum_data[2]

def integrand(r, lum):
    x = lum/(4*np.pi*r**2)
    sigma0 = 1.0
    res = erf((T-x)/np.sqrt(2*(sigma0**2+(0.01*x)**2)))*r**2
#    res = np.power(4000.0/r, 3) * np.power(5.0/x, 1.5)
    return res

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
vals = []
for lumIdx in range(0, xlog.shape[0]):
    lum = xlog[lumIdx]
    I, abserr = quad(integrand, 0, rmax, args=(lum))
    selPdf = 0.5 - (1.5/(rmax**3)) * I
#    selPdf = 0.5 + 0.5 * erf(2.30008*(10.0**(-9.0))*lum - 1.39005)
    vals.append((selPdf*pdf_0[lumIdx])/mu)

print xlog[200], ' ', pdf_0[200]
#pdf_1 = np.multiply(pdf_0,0.956104)
figure(fig_log.number)
ax = fig_log.add_subplot(1,1,1) # one row, one column, first plot

ax.loglog(xlog, vals, 'r-', linewidth=2, label=lbl_0, zorder=3)

ax.axvline(x = 4*np.pi*T*rmax**2, color='r')

#lbins_lums = np.logspace(np.log10(lum_data.min()),np.log10(lum_data.max()),n_bins+1)
#figure(fig_log.number)
ax.hist(lum_data, bins=xlog, label='flux', log=True, normed=True)

legend(loc=3)
savefig('_lumfunc_histo.png')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0