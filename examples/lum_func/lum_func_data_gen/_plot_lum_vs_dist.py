# executing e.g. python _plot_lum_vs_dist.py lums_cnt_100000.dat dists_cnt_100000.dat --T 5.0 --rmax 1121041.72243 --pdf_format False

import argparse as argp
import numpy as np
from matplotlib.pyplot import *
import datetime as dt

#ion()

t0 = dt.datetime.today()

parser = argp.ArgumentParser()
parser.add_argument("lum_file", help="The file name of luminosity data file.", type = str)
parser.add_argument("dist_file", help="The file name of distance data file.", type = str)
parser.add_argument("--T", default = 5.0, help="The flux limit", type=float)
parser.add_argument("--rmax", default = 4000.0, help="The maximal distance", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
lum_file = args.lum_file
dist_file = args.dist_file
T = args.T
rmax = args.rmax
pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)

lum_data=np.loadtxt(lum_file)
dist_data=np.loadtxt(dist_file)

lumVals = []
for lum in lum_data:
    lumVals.append(np.log(lum))

limitLumVals = []
distRange = range(1, int(rmax))
for dist in distRange:
    limitLumVals.append(np.log((dist**2)*T*4*np.pi))

fig_log, ax =  subplots()

xlabel(r'$r$')
ylabel(r'$\log(L)$')

#lbl_1 = 'distance vs log-luminosity'
ax.scatter(dist_data, lumVals, c='r', marker='.', linewidth=0, alpha=0.015)
lbl_2 = r'$\log(4\pi T r^2)$'
ax.plot(distRange, limitLumVals, zorder=3, linewidth=4, c='b', label=lbl_2)

ax.set_xlim([0, rmax])
ax.set_ylim([22, 33])
#ax.set_yscale('log')

legend(loc=0)
if(pdf_format):
  savefig('_lum_vs_dist.pdf', format='pdf')
else:
  savefig('_lum_vs_dist.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0