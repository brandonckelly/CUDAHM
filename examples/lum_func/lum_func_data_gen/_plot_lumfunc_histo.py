# executing e.g. python _plot_lumfunc_histo.py -1.5 50000000000.0 5000000000000.0 fluxes_cnt_100000.dat lums_cnt_100000.dat dists_cnt_100000.dat 100000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --rmax 1121041.72243 --mu 0.000536477 --xlog_min 10.0 --xlog_max 14.0 --pdf_format False

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
parser.add_argument("--resolution", default = 300, help="The resolution of x-axis", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)
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
resolution = args.resolution
pdf_format = eval(args.pdf_format)
execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)

flux_data=np.loadtxt(flux_file)
dist_data=np.loadtxt(dist_file)

lum_data = []

for idx in range(0, flux_data.shape[0]):
    lum_data.append(flux_data[idx] * 4.0 * np.pi * (dist_data[idx]**2))

fig0, ax0 =  subplots()

# the following for only computation (not displaying)
xlog = np.logspace(xlog_min, xlog_max, resolution)
hist, bins, patches = ax0.hist(lum_data, bins=xlog, label='luminosity sample data', log=True, normed=True, color=(1.0,1.0,1.0), edgecolor=(1.0,1.0,1.0), zorder=1)

log10_of_lum_data = []

for idx in range(0, flux_data.shape[0]):
    log10_of_lum_data.append(np.log10(flux_data[idx] * 4.0 * np.pi * (dist_data[idx]**2)))

def integrand(r, lum):
    x = lum/(4*np.pi*r**2)
    sigma0 = 1.0
    res = erf((T-x)/np.sqrt(2*(sigma0**2+(0.01*x)**2)))*r**2
    return res

def transformScientificNotationToLaTeXCode(floatNum, format = '%5.2e'):
    sciNotationPython = format % floatNum
    return sciNotationPython.replace('e+','\cdot 10^{')+'}'

fig_log, ax =  subplots()
xlabel(r'$\log_{10}(L)$')
ylabel(r'$\log_{10}(\phi(L ; \theta))$ and $\log_{10}(\phi^{T}(L ; C, r, \theta))$')

xlin = np.linspace(10**xlog_min, 10**xlog_max, resolution)
log10_of_xlin = np.log10(xlin)

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)

lower_scale_sci_notation = transformScientificNotationToLaTeXCode(lower_scale,'%5.0e')
upper_scale_sci_notation = transformScientificNotationToLaTeXCode(upper_scale,'%5.0e')

lbl_0 = r'$\log_{10}(\phi(L ; \theta=(%5.1f,%s,%s))$)' % (beta, lower_scale_sci_notation, upper_scale_sci_notation)
lbl_1 = r'$\log_{10}(\phi^{T}(L ; C, r, \theta=(%5.1f,%s,%s))$)' % (beta, lower_scale_sci_notation, upper_scale_sci_notation)
pdf_0 = bb1_0.pdf(xlin)
log10_of_pdf_0 = np.log10(pdf_0)
log10_of_valsWSelEff = []
for lumIdx in range(0, xlin.shape[0]):
    lum = xlin[lumIdx]
    I, abserr = quad(integrand, 0, rmax, args=(lum))
    selPdf = 0.5 - (1.5/(rmax**3)) * I
    log10_of_valsWSelEff.append(np.log10((selPdf*pdf_0[lumIdx])/mu))

ax.plot(log10_of_xlin, log10_of_pdf_0, 'g-', linewidth=2, label=lbl_0, zorder=3)
ax.plot(log10_of_xlin, log10_of_valsWSelEff, 'r-', linewidth=2, label=lbl_1, zorder=3)

min_hist_not_zero = hist[0]
for x in hist:
  if(x < min_hist_not_zero and x != 0.0):
    min_hist_not_zero = x
bottom_of_hist_height = min(log10_of_pdf_0.min(), min(log10_of_valsWSelEff), min_hist_not_zero)
top_of_y_axis = max(log10_of_pdf_0.max(), max(log10_of_valsWSelEff))

log10_of_hist_height = [np.log10(x) - bottom_of_hist_height if x!=0.0 else 0.0 for x in hist]
log10_of_bins = [np.log10(x) for x in bins]

ax.bar(log10_of_bins[0:299], log10_of_hist_height, width = 0.8 * (log10_of_bins[1] - log10_of_bins[0]), bottom = bottom_of_hist_height, label='luminosity sample data', color=(0.5,0.5,1.0), edgecolor=(0.5,0.5,1.0))

ax.set_xlim([xlog_min,xlog_max])
ax.set_ylim([bottom_of_hist_height, top_of_y_axis])

legend(loc=3)
if(pdf_format):
  savefig('_lumfunc_histo.pdf', format='pdf')
else:
  savefig('_lumfunc_histo.png')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0