# executing e.g. python plot_thetas_funcs.py lumfunc_thetas_2.dat _ (--lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --nthin_theta 150 --pdf_format False)import argparse as argp
import numpy as np
from matplotlib.pyplot import *

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("prefix", help="The prefic for created output files.", type = str)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--nthin_theta", default = 1.0, help="The thinning factor for theta", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
prefix = args.prefix
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
nthin_theta = args.nthin_theta
pdf_format = eval(args.pdf_format)

execfile("rc_settings.py")
rc('figure', figsize=(1.9, 1.9))
rc('figure.subplot', bottom=.275, top=.86, right=.9, left=.38)
if(pdf_format!=True):
  rc('savefig', dpi=100)
theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))
theta_data_beta=theta_data[:,0]
theta_data_l=theta_data[:,1]
theta_data_u=theta_data[:,2]

theta_data_l *= lower_scale_factor
theta_data_u *= upper_scale_factor

color_list = range(0, theta_data.shape[0])

for idx in range(0, theta_data.shape[0]):
    blue_rate = idx/float(theta_data.shape[0])
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    color_list[idx]=(red_rate*1.0,0.0,blue_rate*1.0)

def determineMagnitude(floatNum):
    sciNotationPython = '%e' % floatNum
    mantissa, exp = sciNotationPython.split('e+')
    return 10.0**int(exp),int(exp)

def setXAxisProperties(ax, lbl_iter, max_x_pos, max_x_pos_mag):
    ax.set_xlabel(lbl_iter)
    ax.xaxis.set_ticks([0.0,0.5*max_x_pos,max_x_pos])
    ax.xaxis.set_ticklabels([0.0,0.5*max_x_pos/max_x_pos_mag,max_x_pos/max_x_pos_mag])
    ax.set_xlim([0.0,max_x_pos])

def setYAxisProperties(ax, ylbl, data):
    ax.set_ylabel(ylbl)
    data_min = data.min()
    data_yrange = data.max() - data_min
    data_ticks = [data_min, data_min + data_yrange * 1.0/3.0, data_min + data_yrange * 2.0/3.0, data_min + data_yrange]
    ax.yaxis.set_ticks(data_ticks)
    return data_ticks

max_x_pos = len(theta_data_beta) * nthin_theta
max_x_pos_mag, max_x_pos_exp = determineMagnitude(max_x_pos)
max_lowerscale = theta_data_l.max()
max_lowerscale_mag, max_lowerscale_exp = determineMagnitude(max_lowerscale)
max_upperscale = theta_data_u.max()
max_upperscale_mag, max_upperscale_exp = determineMagnitude(max_upperscale)

lbl_beta = r'$\beta$'
lbl_lowerscale = r'lower scale ($\times 10^{%d}$)' % max_lowerscale_exp
lbl_upperscale = r'upper scale ($\times 10^{%d}$)' % max_upperscale_exp
lbl_iter= r'Iterations ($\times 10^{%d}$)' % max_x_pos_exp

fig, ax = subplots()
xrange = np.arange(1, len(theta_data_beta) + 1) * nthin_theta
ax.scatter(xrange,theta_data_beta, c=color_list, marker = ".", linewidth=0, alpha=0.05)
setXAxisProperties(ax, lbl_iter, max_x_pos, max_x_pos_mag)
beta_ticks = setYAxisProperties(ax, lbl_beta, theta_data_beta)
ax.yaxis.set_ticklabels(['%.2f' % y for y in beta_ticks])
if(pdf_format):
  savefig('beta.pdf', format='pdf')
else:
  savefig('beta.png')

fig, ax = subplots()
xrange = np.arange(1, len(theta_data_u) + 1) * nthin_theta
ax.scatter(xrange,theta_data_u, c=color_list, marker = ".", linewidth=0, alpha=0.05)
setXAxisProperties(ax, lbl_iter, max_x_pos, max_x_pos_mag)
upperscale_ticks = setYAxisProperties(ax, lbl_upperscale, theta_data_u)
ax.yaxis.set_ticklabels(['%.2f' % (y/max_upperscale_mag) for y in upperscale_ticks])
if(pdf_format):
  savefig('upperscale.pdf', format='pdf')
else:
  savefig('upperscale.png')

fig, ax = subplots()
xrange = np.arange(1, len(theta_data_l) + 1) * nthin_theta
ax.scatter(xrange,theta_data_l, c=color_list, marker = ".", linewidth=0, alpha=0.05)
setXAxisProperties(ax, lbl_iter, max_x_pos, max_x_pos_mag)
lowerscale_ticks = setYAxisProperties(ax, lbl_lowerscale, theta_data_l)
ax.yaxis.set_ticklabels(['%.2f' % (y/max_lowerscale_mag) for y in lowerscale_ticks])
if(pdf_format):
  savefig('lowerscale.pdf', format='pdf')
else:
  savefig('lowerscale.png')

fig, ax = subplots()
ax.scatter(theta_data_beta,theta_data_l, c=color_list, marker = ".", linewidth=0, alpha=0.05)
#ax.set_xlabel(lbl_beta)
#ax.set_ylabel(lbl_lowerscale)
ax.xaxis.set_ticks([0.0,0.5*max_x_pos,max_x_pos])
ax.set_xlim([0.0,max_x_pos])
if(pdf_format):
  savefig('beta_lowerscale.pdf', format='pdf')
else:
  savefig('beta_lowerscale.png')

fig, ax = subplots()
ax.scatter(theta_data_beta,theta_data_u, c=color_list, marker = ".", linewidth=0, alpha=0.05)
#ax.set_xlabel(lbl_beta)
#ax.set_ylabel(lbl_upperscale)
ax.xaxis.set_ticks([0.0,0.5*max_x_pos,max_x_pos])
ax.set_xlim([0.0,max_x_pos])
if(pdf_format):
  savefig('beta_upperscale.pdf', format='pdf')
else:
  savefig('beta_upperscale.png')

fig, ax = subplots()
ax.scatter(theta_data_l,theta_data_u, c=color_list, marker = ".", linewidth=0, alpha=0.05)
#ax.set_xlabel(lbl_lowerscale)
#ax.set_ylabel(lbl_upperscale)
ax.xaxis.set_ticks([0.0,0.5*max_x_pos,max_x_pos])
ax.set_xlim([0.0,max_x_pos])
if(pdf_format):
  savefig('lowerscale_upperscale.pdf', format='pdf')
else:
  savefig('lowerscale_upperscale.png')