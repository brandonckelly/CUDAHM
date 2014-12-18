# executing e.g. python plot_autocorr_funcs.py lumfunc_thetas.dat autocorr_b-1.2_l1.0_u100.0_init_b-1.3_l5.0_u110.0_ 100
import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("prefix", help="The prefix for created output files.", type = str)
parser.add_argument("until", help="This is the maximal k for which the autocorrelation is computed.", type = str)

args = parser.parse_args()
file = args.file
prefix = args.prefix
until = int(args.until)

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

def autocorr(k, n, data, idx):
    m_data = np.mean(data, axis = 0)[idx]
    var_data = np.var(data[:n-k], axis = 0)[idx]
    cov_data = 0.0
    for t in range(0, n-k):
        cov_data += float(data[t][idx] - m_data)*float(data[t+k][idx] - m_data)
    cov_data/=float(n)
    return cov_data / var_data

def autocorrfunc(until, n, data, idx):
    lst = []
    for k in range(0,until):
        lst.append(autocorr(k, n, data, idx))
    return lst

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

autocorrfn_beta = autocorrfunc(until, theta_data.shape[0],theta_data,0)
autocorrfn_l = autocorrfunc(until, theta_data.shape[0],theta_data,1)
autocorrfn_u = autocorrfunc(until, theta_data.shape[0],theta_data,2)

tit_beta = r'$\beta$'
tit_lowerscale = 'lower scale'
tit_upperscale = 'upper scale'
lbl_k = 'k'
lbl_autocorr = 'kth order autocorrelation'

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(0, len(autocorrfn_beta)),autocorrfn_beta, marker = ".", linewidth=0.01)

ax.set_xlabel(lbl_k)
ax.set_ylabel(lbl_autocorr)

fig.suptitle(tit_beta, fontsize=18, fontweight='bold')

savefig(prefix + 'beta.png',dpi=120)

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(0, len(autocorrfn_l)),autocorrfn_l, marker = ".", linewidth=0.01)

ax.set_xlabel(lbl_k)
ax.set_ylabel(lbl_autocorr)

fig.suptitle(tit_lowerscale, fontsize=18, fontweight='bold')

savefig(prefix + 'lowerscale.png',dpi=120)

fig = figure(figsize=(15.75, 10))
ax = fig.add_subplot(1,1,1) # one row, one column, first plot
ax.scatter(range(0, len(autocorrfn_u)),autocorrfn_u, marker = ".", linewidth=0.01)

ax.set_xlabel(lbl_k)
ax.set_ylabel(lbl_autocorr)

fig.suptitle(tit_upperscale, fontsize=18, fontweight='bold')

savefig(prefix + 'upperscale.png',dpi=120)