# executing e.g. python plot_pairs_plot.py lumfunc_thetas_2.dat _ --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --pdf_format False
import argparse as argp
import datetime as dt
import numpy as np
from matplotlib.pyplot import *
import corner
#import pandas as pd
##import seaborn as sns

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("prefix", help="The prefic for created output files.", type = str)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
file = args.file
prefix = args.prefix
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
pdf_format = eval(args.pdf_format)

t0 = dt.datetime.today()

execfile("rc_settings.py")

def determineMagnitude(floatNum):
    sciNotationPython = '%e' % floatNum
    mantissa, exp = sciNotationPython.split('e+')
    return 10.0**int(exp),int(exp)

#def setAxisProperties(ax, data):
#    data_min = data.min()
#    data_range = data.max() - data_min
#    data_ticks = [data_min, data_min + data_range * 1.0/3.0, data_min + data_range * 2.0/3.0, data_min + data_range]
#    return data_ticks

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

max_lowerscale = theta_data[:,1].max()*lower_scale_factor
max_lowerscale_mag, max_lowerscale_exp = determineMagnitude(max_lowerscale)
max_upperscale = theta_data[:,2].max()*upper_scale_factor
max_upperscale_mag, max_upperscale_exp = determineMagnitude(max_upperscale)

lbl_lowerscale = r'$l$ ($\times 10^{%d}$)' % max_lowerscale_exp
lbl_upperscale = r'$u$ ($\times 10^{%d}$)' % max_upperscale_exp
figure = corner.corner(theta_data, labels=[r'$\beta$', lbl_lowerscale, lbl_upperscale], color='b', bins=50, hist_kwargs={'color':'b','edgecolor':'b','zorder':5})

#theta_data[:,1] = theta_data[:,1]*lower_scale_factor
#theta_data[:,2] = theta_data[:,2]*upper_scale_factor

#max_x_pos_mag, max_x_pos_exp = determineMagnitude(max_x_pos)
#max_lowerscale = theta_data[:,1].max()
#max_lowerscale_mag, max_lowerscale_exp = determineMagnitude(max_lowerscale)
#max_upperscale = theta_data[:,2].max()
#max_upperscale_mag, max_upperscale_exp = determineMagnitude(max_upperscale)

##sns.set(style="ticks", rc={'font.size':10,'font.family':'serif','font.serif':['Computer Modern Roman'],'axes.labelsize':12,'axes.formatter.limits':(-4,4),'xtick.major.pad':8,'xtick.labelsize':12,'ytick.major.pad':8,'ytick.labelsize':12,'savefig.dpi':150,'text.usetex':True,'figure.figsize':(5, 5)})

#lbl_upperscale = r'$u$ ($\times 10^{%d}$)' % max_upperscale_exp
#df = pd.DataFrame(data=theta_data,columns=(r'$\beta$',r'$l$',lbl_upperscale))
#axes = pd.tools.plotting.scatter_matrix(df, edgecolors='none', alpha=0.01, color = 'b', hist_kwds={'bins':100,'color':'b','edgecolor':'b','zorder':5})

#for x in range(0,3):
#  for y in range(0,3):
###    axes[x][y].spines['bottom'].set_visible(False)
###    axes[x][y].spines['top'].set_visible(False)
###    axes[x][y].spines['left'].set_visible(False)
###    axes[x][y].spines['right'].set_visible(False)
###    axes[x][y].tick_params(bottom='off',top='off',left='off',right='off')
#    axes[x][y].set_axis_bgcolor('white')
#beta_ticks = setAxisProperties(ax, theta_data[:,0])
#axes[2][0].xaxis.set_ticks(beta_ticks)
#beta_tick_labels = ['%.2f' % y for y in beta_ticks]
#axes[2][0].xaxis.set_ticklabels([r'$%s$' % y for y in beta_tick_labels])
#axes[2][0].set_xlim([beta_ticks[0],beta_ticks[-1]])

#axes[0][0].yaxis.set_ticks(beta_ticks)
#axes[0][0].yaxis.set_ticklabels([r'$%s$' % y for y in beta_tick_labels])
#axes[0][0].set_ylim([beta_ticks[0],beta_ticks[-1]])


#upperscale_ticks = setAxisProperties(ax, theta_data[:,2])
#axes[0][2].yaxis.set_ticks(upperscale_ticks)
#beta_tick_labels = ['%.2f' % y for y in upperscale_ticks]
#axes[0][2].yaxis.set_ticklabels(['%.2f' % (y/max_upperscale_mag) for y in upperscale_ticks])

#axes[2][2].xaxis.set_ticks(upperscale_ticks)
#axes[2][2].xaxis.set_ticklabels(['%.2f' % (y/max_upperscale_mag) for y in upperscale_ticks])

##g = sns.pairplot(df)

if(pdf_format):
  savefig('pair_plot.pdf', format='pdf')
##  g.savefig('pair_plot.pdf', format='pdf')
else:
  savefig('pair_plot.png')
##  g.savefig('pair_plot.png')
  
t1 = dt.datetime.today()
print 'Elapsed time of generating figures of error plots:', t1-t0