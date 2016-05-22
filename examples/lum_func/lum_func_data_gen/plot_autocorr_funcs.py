#executing e.g. python plot_autocorr_funcs.py lumfunc_thetas_2.dat autocorr_ 100 --pdf_format False
import argparse as argp
import numpy as np
from matplotlib.pyplot import *
from AutoCorrUtil import AutoCorrUtil

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("prefix", help="The prefix for created output files.", type = str)
parser.add_argument("until", help="This is the maximal k for which the autocorrelation is computed.", type = str)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
prefix = args.prefix
until = int(args.until)
pdf_format = eval(args.pdf_format)

util = AutoCorrUtil()

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.85, right=.85, left=.3)
if(pdf_format!=True):
  rc('savefig', dpi=100)

def setAxesProperties(ax,lbl_k,autocorrfn,tit):
    ax.axhline(color='r')
    ax.set_xlabel(lbl_k)
#    ax.set_ylabel(lbl_autocorr)
    ax.set_xlim([1.0,len(autocorrfn)])
    ax.set_ylim([-0.05,0.05])
    ax.xaxis.set_ticks([0,until/2,until])
    ax.set_title(tit)
	
theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

autocorrfn_beta = util.autocorrfunc(until, theta_data.shape[0],theta_data,0)
autocorrfn_l = util.autocorrfunc(until, theta_data.shape[0],theta_data,1)
autocorrfn_u = util.autocorrfunc(until, theta_data.shape[0],theta_data,2)

ess_beta = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 0)
ess_l = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 1)
ess_u = util.effectiveSampleSize(until, theta_data.shape[0], theta_data, 2)

print "n:", theta_data.shape[0], ", ESS_beta:", ess_beta, ", ESS_l:", ess_l, ", ESS_u:", ess_u

tit_beta = r'$\beta$'
tit_lowerscale = r'$l$'
tit_upperscale = r'$u$'
lbl_k = r'$k$'
#lbl_autocorr = r'$k$th order autocorr.'

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_beta)),autocorrfn_beta, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,autocorrfn_beta,tit_beta)
if(pdf_format):
  savefig(prefix + 'beta.pdf', format='pdf')
else:
  savefig(prefix + 'beta.png')

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_l)),autocorrfn_l, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,autocorrfn_l,tit_lowerscale)
if(pdf_format):
  savefig(prefix + 'lowerscale.pdf', format='pdf')
else:
  savefig(prefix + 'lowerscale.png')

fig, ax = subplots()
ax.scatter(range(0, len(autocorrfn_u)),autocorrfn_u, marker = ".", linewidth=0.01)
setAxesProperties(ax,lbl_k,autocorrfn_u,tit_upperscale)
if(pdf_format):
  savefig(prefix + 'upperscale.pdf', format='pdf')
else:
  savefig(prefix + 'upperscale.png')