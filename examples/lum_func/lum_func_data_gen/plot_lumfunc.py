# executing e.g. python plot_lumfunc.py -1.5 5.0 5.0 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0

import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
import datetime as dt

parser = argp.ArgumentParser()
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--xlog_min", default = 8.0, help="The log of x-axis minimum", type=float)
parser.add_argument("--xlog_max", default = 13.0, help="The log of x-axis maximum", type=float)
parser.add_argument("--resolution", default = 5000, help="The resolution of x-axis", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
beta = args.beta
lower_scale = args.lower_scale * args.lower_scale_factor
upper_scale = args.upper_scale * args.upper_scale_factor
xlog_min = args.xlog_min
xlog_max = args.xlog_max
resolution = args.resolution
pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")

if(pdf_format!=True):
  rc('savefig', dpi=100)

t0 = dt.datetime.today()

xlin = np.linspace(10**xlog_min, 10**xlog_max, resolution)
log10_of_xlin = np.log10(xlin)

bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlin)
log10_of_pdf_0 = np.log10(pdf_0)

fig, ax = subplots()

ax.plot(log10_of_xlin, log10_of_pdf_0, 'r-', linewidth=1.0, zorder=1)

ax.text(8.8, -10.2, r'$\sim (\phi_{0}(\theta)/(l\cdot u^{\beta}))\cdot L^{\beta+1}$',zorder=1)

ax.axvline(x = np.log10(lower_scale), color='blue', linewidth=1.0, linestyle='-', zorder=2)

ax.axvline(x = np.log10(upper_scale), color='blue', linewidth=1.0, linestyle='-', zorder=2)

#ax.annotate(r'$\sim \frac{\phi_{0}(\theta)}{l\cdot u^{\beta}}\cdot L^{\beta+1}$', xy=(9, -10.5), xytext=(10, -10.5),
#            arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=2.0), zorder=1)

if(pdf_format):
  savefig('lumfunc_loglog.pdf', format='pdf')
else:
  savefig('lumfunc_loglog.png')

close() # it closes the previous plot to avoid memory leak

fig, ax = subplots()

ax.plot(xlin, log10_of_pdf_0, 'r-', linewidth=1.0, zorder=2)

if(pdf_format):
  savefig('lumfunc_linlog.pdf', format='pdf')
else:
  savefig('lumfunc_linlog.png')

close() # it closes the previous plot to avoid memory leak

fig, ax = subplots()

ax.plot(log10_of_xlin, pdf_0, 'r-', linewidth=1.0, zorder=2)

if(pdf_format):
  savefig('lumfunc_loglin.pdf', format='pdf')
else:
  savefig('lumfunc_loglin.png')

close() # it closes the previous plot to avoid memory leak

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function:', t1-t0