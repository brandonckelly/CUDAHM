# executing e.g. python _plot_lumfunc_histo_new.py -1.5 50000000000.0 5000000000000.0 fluxes_cnt_100000.dat lums_cnt_100000.dat dists_cnt_100000.dat 100000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --rmax 1121041.72243 --mu 0.000536477 --xlog_min 9.5 --xlog_max 13.0

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
parser.add_argument("lum_file", help="The file name of flux data file.", type = str)
parser.add_argument("dist_file", help="The file name of distance data file.", type = str)
parser.add_argument("obj_num", help="The object number of MCMC method", type=int)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--T", default = 5.0, help="The flux limit", type=float)
parser.add_argument("--rmax", default = 4000.0, help="The maximal distance", type=float)
parser.add_argument("--mu", default = 0.934197, help="The value of mu(theta)", type=float)
parser.add_argument("--xlog_min", default = 8.0, help="The log of x-axis minimum", type=float)
parser.add_argument("--xlog_max", default = 13.0, help="The log of x-axis maximum", type=float)

args = parser.parse_args()
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
flux_file = args.flux_file
lum_file = args.lum_file
dist_file = args.dist_file
obj_num = args.obj_num
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
T = args.T
rmax = args.rmax
mu = args.mu
xlog_min = args.xlog_min
xlog_max = args.xlog_max

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
lum_data=np.loadtxt(lum_file)

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
xlabel(r'$L$')
ylabel(r'$\phi(L|\theta)$ and $\phi^{T}(L|\theta)$')
tit = r'Difference between lumfunc and lumfunc with sel. eff. (Obj. num.: %d)' % (obj_num)
fig_log.suptitle(tit, fontsize=18, fontweight='bold')

xlog = np.logspace(xlog_min, xlog_max, 300)

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
lbl_0 = r'$\phi(L|\theta)$ (%5.2f,%5.2e,%5.2e)' % (beta, lower_scale, upper_scale)
lbl_1 = r'$\phi^{T}(L|\theta)$ (%5.2f,%5.2e,%5.2e)' % (beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlog)
valsWSelEff = []
for lumIdx in range(0, xlog.shape[0]):
    lum = xlog[lumIdx]
    I, abserr = quad(integrand, 0, rmax, args=(lum))
    selPdf = 0.5 - (1.5/(rmax**3)) * I
#    selPdf = 0.5 + 0.5 * erf(2.30008*(10.0**(-9.0))*lum - 1.39005)
    valsWSelEff.append((selPdf*pdf_0[lumIdx])/mu)

print xlog[200], ' ', pdf_0[200]
#pdf_1 = np.multiply(pdf_0,0.956104)
figure(fig_log.number)
ax = fig_log.add_subplot(1,1,1) # one row, one column, first plot

xlim([10**xlog_min,10**xlog_max])
ax.loglog(xlog, pdf_0, 'b--', linewidth=2, label=lbl_0, zorder=3)
ax.loglog(xlog, valsWSelEff, 'r-', linewidth=2, label=lbl_1, zorder=3)

lbl_2 = r'$4*\pi*T*r_{max}^2$'
ax.axvline(x = 4*np.pi*T*rmax**2, color='black', linewidth=3, label=lbl_2)

#lbins_lums = np.logspace(np.log10(lum_data.min()),np.log10(lum_data.max()),n_bins+1)
#figure(fig_log.number)
ax.hist(lum_data, bins=xlog, label='luminosity sample data', log=True, normed=True)

ax2 = fig_log.add_subplot(1,1,1) # one row, one column, first plot
ax2.fill_between(xlog, pdf_0, valsWSelEff, facecolor='cyan', alpha=0.5)

legend(loc=0)
savefig('_lumfunc_histo.png')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0