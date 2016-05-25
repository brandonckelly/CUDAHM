# executing e.g. python comp_geweke.py lumfunc_thetas_25_06_2016.dat --burn_in_end 2000000 --intervals 100 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *
from pymc.diagnostics import geweke

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("--burn_in_end", default = 0, help="The end postition of the candidate burn-in period. The default value 0 indicates the entire content of the source file is included for computation.", type=int)
parser.add_argument("--intervals", default = 20, help="The number of segments.", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
burn_in_end = args.burn_in_end
intervals = args.intervals
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.85, right=.85, left=.3)

def determineMagnitude(floatNum):
    sciNotationPython = '%e' % floatNum
    mantissa, exp = sciNotationPython.split('e+')
    return 10.0**int(exp),int(exp)

def setXAxisProperties(ax, lbl_iter, max_x_pos, max_x_pos_mag):
    ax.set_xlabel(lbl_iter)
    ax.xaxis.set_ticks([0.0,0.5*max_x_pos,max_x_pos])
    ax.xaxis.set_ticklabels([0.0,0.5*max_x_pos/max_x_pos_mag,max_x_pos/max_x_pos_mag])
    ax.set_xlim([0.0,max_x_pos])

# This method is inspired by the method geweke_plot from https://github.com/pymc-devs/pymc/blob/master/pymc/Matplot.py
def plotZScores(zscores,ylabel):
  xdata, ydata = np.transpose(zscores)
  fig, ax = subplots()
  ax.scatter(xdata.tolist(),ydata.tolist(), color='b', marker = ".", edgecolors='b', linewidth=1, zorder=2)
  max_x_pos_mag, max_x_pos_exp = determineMagnitude(xdata.max())
  lbl = r'First iter. ($\times 10^{%d}$)' % max_x_pos_exp
  setXAxisProperties(ax, lbl, xdata.max(), max_x_pos_mag)
  ax.set_ylabel(ylabel)
  ax.axhline(y = 0, color='black', linewidth=1, linestyle='-', zorder=1)
  ax.axhline(y = 2, color='red', linewidth=2, linestyle='--', zorder=1)
  ax.axhline(y = -2, color='red', linewidth=2, linestyle='--', zorder=1)
  ax.set_xlim([-0.02*xdata.max(),1.02*xdata.max()])
  if 1.02*np.abs(ydata).max() < 2.5:
    ax.set_ylim([-2.5,2.5])
  else:
    ax.set_ylim([-1.02*np.abs(ydata).max(),1.02*np.abs(ydata).max()])

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

theta_data_beta=theta_data[burn_in_end:,0]
theta_data_lower=theta_data[burn_in_end:,1]
theta_data_upper=theta_data[burn_in_end:,2]

beta_z_scores = geweke(theta_data_beta,last=.5,intervals=intervals)
lower_z_scores = geweke(theta_data_lower,last=.5,intervals=intervals)
upper_z_scores = geweke(theta_data_upper,last=.5,intervals=intervals)

print "Z-scores for beta"
for beta_z_score in beta_z_scores:
  print beta_z_score

print "Z-scores for lower scale"
for lower_z_score in lower_z_scores:
  print lower_z_score

print "Z-scores for upper scale"
for upper_z_score in upper_z_scores:
  print upper_z_score

if(pdf_format):
  plotZScores(beta_z_scores,r"Z-score for $\beta$")
  savefig('beta_z_score.pdf', format='pdf')
  plotZScores(lower_z_scores,r"Z-score for $l$")
  savefig('lower_z_score.pdf', format='pdf')
  plotZScores(upper_z_scores,r"Z-score for $u$")
  savefig('upper_z_score.pdf', format='pdf')
else:
  plotZScores(beta_z_scores,r"Z-score for $\beta$")
  savefig('beta_z_score.png')
  plotZScores(lower_z_scores,r"Z-score for $l$")
  savefig('lower_z_score.png')
  plotZScores(upper_z_scores,r"Z-score for $u$")
  savefig('upper_z_score.png')

t1 = dt.datetime.today()
print 'Elapsed time of computation:', t1-t0