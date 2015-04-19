# executing e.g. python _plot_lum_vs_dist.py lums_cnt_100000.dat dists_cnt_100000.dat --T 5.0 --rmax 1121041.72243

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

args = parser.parse_args()
lum_file = args.lum_file
dist_file = args.dist_file
T = args.T
rmax = args.rmax

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

lum_data=np.loadtxt(lum_file)
dist_data=np.loadtxt(dist_file)

lumVals = []
for lum in lum_data:
    lumVals.append(np.log(lum))

limitLumVals = []
distRange = range(1, int(rmax))
for dist in distRange:
    limitLumVals.append(np.log((dist**2)*T*4*np.pi))

fig_log = figure(figsize=(15.75, 10))
ax = fig_log.add_subplot(1,1,1) # one row, one column, first plot
xlabel(r'$r$')
ylabel(r'$\log(L)$')

#lbl_1 = 'distance vs log-luminosity'
ax.scatter(dist_data, lumVals, c='r', marker='.', linewidth=0, alpha=0.2)
lbl_2 = r'$\log(4\pi T r^2)$'
ax.plot(distRange, limitLumVals, zorder=3, linewidth=4, c='b', label=lbl_2)

ax.set_xlim([0, rmax])
ax.set_ylim([22, 33])
#ax.set_yscale('log')

legend(loc=0)
savefig('_lum_vs_dist.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with noisy flux data:', t1-t0