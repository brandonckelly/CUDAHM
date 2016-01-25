# executing e.g. python plot_autocorr_funcs.py lumfunc_thetas_2.dat autocorr_ 100
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

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.86, right=.9, left=.38)

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

def setAxesProperties(ax,lbl_k,lbl_autocorr,autocorrfn,tit):
    ax.axhline(color='r')
    ax.set_xlabel(lbl_k)
    ax.set_ylabel(lbl_autocorr)
    ax.set_xlim([0.0,len(autocorrfn)])
    ax.set_ylim([-0.6,0.6])
    ax.xaxis.set_ticks([0,until/2,until])
    ax.set_title(tit)
	
theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

autocorrfn_beta = autocorrfunc(until, theta_data.shape[0],theta_data,0)
autocorrfn_l = autocorrfunc(until, theta_data.shape[0],theta_data,1)
autocorrfn_u = autocorrfunc(until, theta_data.shape[0],theta_data,2)

tit_beta = r'$\beta$'
tit_lowerscale = 'lower scale'
tit_upperscale = 'upper scale'
lbl_k = r'$k$'
lbl_autocorr = r'$k$th order autocorr.'

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_beta)),autocorrfn_beta, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,lbl_autocorr,autocorrfn_beta,tit_beta)
savefig(prefix + 'beta.pdf', format='pdf')

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_l)),autocorrfn_l, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,lbl_autocorr,autocorrfn_l,tit_lowerscale)
savefig(prefix + 'lowerscale.pdf', format='pdf')

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_u)),autocorrfn_u, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,lbl_autocorr,autocorrfn_u,tit_upperscale)
savefig(prefix + 'upperscale.pdf', format='pdf')