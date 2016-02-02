# executing e.g. python plot_lumfunc_w_thetas.py lumfunc_thetas_2.dat -1.5 50000000000.0 5000000000000.0 -1.41 4.0 5.8 -1.5564 7.3222 5.7207 1500000 1500000 100000 (--cov 1000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0 --pdf_format False --resolution 3000)
import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
from matplotlib.cbook import get_sample_data
from matplotlib import gridspec
import datetime as dt
from scipy import stats
import os

#ion()

t0 = dt.datetime.today()

parser = argp.ArgumentParser()
parser.add_argument("file", help="The file name of theta data file.", type = str)
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("init_beta", help="The initial beta parameter value of MCMC method", type=float)
parser.add_argument("init_lower_scale", help="The initial lower scale of MCMC method", type=float)
parser.add_argument("init_upper_scale", help="The initial upper scale of MCMC method", type=float)
parser.add_argument("maxlike_beta", help="The result beta parameter value from maximum likelihood estimation", type=float)
parser.add_argument("maxlike_lower_scale", help="The result lower scale from maximum likelihood estimation", type=float)
parser.add_argument("maxlike_upper_scale", help="The result upper scale from maximum likelihood estimation", type=float)
parser.add_argument("burnin_num", help="The iteration number of burn-in of MCMC method", type=int)
parser.add_argument("iter_num", help="The iteration number of MCMC method", type=int)
parser.add_argument("obj_num", help="The object number of MCMC method", type=int)
parser.add_argument("--cov", default = 10000, help="The value of this number determines, how many BB1 with samples of parameters theta will be plotted", type=int)
parser.add_argument("--lower_scale_factor", default = 1.0, help="The factor which scales up the lower scale samples", type=float)
parser.add_argument("--upper_scale_factor", default = 1.0, help="The factor which scales up the upper scale samples", type=float)
parser.add_argument("--xlog_min", default = 8.0, help="The log of x-axis minimum", type=float)
parser.add_argument("--xlog_max", default = 13.0, help="The log of x-axis maximum", type=float)
parser.add_argument("--resolution", default = 5000, help="The resolution of x-axis", type=int)
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)
args = parser.parse_args()
file = args.file
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
init_beta = args.init_beta
init_lower_scale = args.init_lower_scale
init_upper_scale = args.init_upper_scale
maxlike_beta = args.maxlike_beta
maxlike_lower_scale = args.maxlike_lower_scale * args.lower_scale_factor
maxlike_upper_scale = args.maxlike_upper_scale * args.upper_scale_factor
burnin_num = args.burnin_num
iter_num = args.iter_num
obj_num = args.obj_num
cov = args.cov 
lower_scale_factor = args.lower_scale_factor
upper_scale_factor = args.upper_scale_factor
xlog_min = args.xlog_min
xlog_max = args.xlog_max
resolution = args.resolution
pdf_format = eval(args.pdf_format)
execfile("rc_settings.py")
rc('figure.subplot', bottom=0.0, top=1.0, right=1.0, left=0.0)
if(pdf_format!=True):
  rc('savefig', dpi=100)

#theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',dtype=[('f0',np.float32),('f1',np.float32),('f2',np.float32)])

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

# Helper for plotting BB1 with samples of theta parameters:
def plot_figs(idx, xlin, log10_of_xlin, c, ax):
    smp_beta = theta_data[idx][0]
    smp_l =theta_data[idx][1] * lower_scale_factor
    smp_u = theta_data[idx][2] * upper_scale_factor
    bb1 = BB1TruncPL(smp_beta, smp_l, smp_u)
    pdf = bb1.pdf(xlin)
    log10_of_pdf = np.log10(pdf)
    #the green colored line belongs to the latest theta from iterations.
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    ax.plot(log10_of_xlin, log10_of_pdf, color=(red_rate*1.0,1.0,0.0), alpha=.01, linewidth=0.25, zorder=1)

def plot_figs_diffs(idx, xlin, log10_of_xlin, c, ax):
    smp_beta = theta_data[idx][0]
    smp_l =theta_data[idx][1] * lower_scale_factor
    smp_u = theta_data[idx][2] * upper_scale_factor
    bb1 = BB1TruncPL(smp_beta, smp_l, smp_u)
    pdf = bb1.pdf(xlin)
    log10_of_pdf = np.log10(pdf)
    #the green colored line belongs to the latest theta from iterations.
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    log10_of_diff_true_mcmc = log10_of_pdf_0 - log10_of_pdf
    ax.plot(log10_of_xlin, log10_of_diff_true_mcmc, color=(red_rate*1.0,1.0,0.0), alpha=.03, linewidth=0.25, zorder=1)

mean_vec = np.mean(theta_data, axis = 0)
std_vec = np.std(theta_data, axis = 0)
sem_vec = stats.sem(theta_data, axis = 0)

print 'Mean of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (mean_vec[0], mean_vec[1], mean_vec[2])
print 'Standard deviation of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (std_vec[0], std_vec[1], std_vec[2])
print 'Standard error of the mean of sample values of theta parameters: (%e,%e,%e)' % (sem_vec[0], sem_vec[1], sem_vec[2])

fig, ax = subplots()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(bottom='off',top='off',left='off',right='off')

xlin = np.linspace(10**xlog_min, 10**xlog_max, resolution)
log10_of_xlin = np.log10(xlin)

t1 = dt.datetime.today()
print 'Elapsed time of loading samples of theta parameters:', t1-t0

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlin)
log10_of_pdf_0 = np.log10(pdf_0)
ax.plot(log10_of_xlin, log10_of_pdf_0, 'r-', linewidth=0.6, zorder=3)

bb1_1 = BB1TruncPL(maxlike_beta, maxlike_lower_scale, maxlike_upper_scale)
pdf_1 = bb1_1.pdf(xlin)
log10_of_pdf_1 = np.log10(pdf_1)
ax.plot(log10_of_xlin, log10_of_pdf_1, 'b-', linewidth=0.6, zorder=2)

u_array = np.random.uniform(size=theta_data.shape[0])
accept_rate = cov/float(theta_data.shape[0])

cnt_accept = 0
for idx in range(0, theta_data.shape[0]):
	if u_array[idx] < accept_rate:
		plot_figs(idx, xlin, log10_of_xlin, 'b', ax)
		cnt_accept += 1

print 'Count of accepted array elements: %d' % cnt_accept

original_xticks1 = ax.get_xticks()
original_yticks1 = ax.get_yticks()

savefig('lumfunc_w_thetas_curve.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with samples of theta parameters:', t1-t0

close() # it closes the previous plot to avoid memory leak

execfile("rc_settings.py")
rc('figure.subplot', bottom=0.0, top=1.0, right=1.0, left=0.0)
fig, ax = subplots()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(bottom='off',top='off',left='off',right='off')

t0 = dt.datetime.today()

log10_of_diff_true_maxlike = log10_of_pdf_0 - log10_of_pdf_1
ax.plot(log10_of_xlin, log10_of_diff_true_maxlike, 'b-', linewidth=1.0, zorder=2)

u_array = np.random.uniform(size=theta_data.shape[0])
accept_rate = cov/float(theta_data.shape[0])

cnt_accept = 0
for idx in range(0, theta_data.shape[0]):
	if u_array[idx] < accept_rate:
		plot_figs_diffs(idx, xlin, log10_of_xlin, 'b', ax)
		cnt_accept += 1

print 'Count of accepted array elements: %d' % cnt_accept

original_xticks2 = ax.get_xticks()
original_yticks2 = ax.get_yticks()

savefig('lumfunc_w_thetas_diffs_curve.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function differences between with true parameters and with samples of theta parameters:', t1-t0

close() # it closes the previous plot to avoid memory leak

execfile("rc_settings.py")
rc('figure.subplot', bottom=.1, top=.97, right=.95, left=0.24)
rc('figure', figsize=(3.75, 5.625))
fig = figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
gs.update(hspace=0.1) # set the spacing between axes.
ax2 = subplot(gs[1])
ax = subplot(gs[0], sharex=ax2)

ax.set_ylabel(r'$\log_{10}(\phi(L ; \theta))$')
ax.tick_params(labelbottom='off')  # don't put tick labels at the bottom
ax2.set_ylabel(r'$\log_{10}(\phi(L ; \theta)) - log_{10}(\phi(L ; \theta_{true}))$')
ax2.set_xlabel(r'$\log_{10}(L)$')
tit = r'Luminosity density function'
#ax.set_title(tit))

currentdir = os.getcwd()
print currentdir

im = imread(get_sample_data(currentdir+'\\'+'lumfunc_w_thetas_curve.png'))
ax.imshow(im, extent=[original_xticks1[0],original_xticks1[-1],original_yticks1[0], original_yticks1[-1]], aspect="auto")

im = imread(get_sample_data(currentdir+'\\'+'lumfunc_w_thetas_diffs_curve.png'))
ax2.imshow(im, extent=[original_xticks2[0],original_xticks2[-1],original_yticks2[0], original_yticks2[-1]], aspect="auto")

lbl_0 = r'$\log_{10}(\phi(L ; \theta_{true}))$'
lbl_1 = r'$\log_{10}(\phi(L ; \theta_{maxlike}))$'

#The following 'custom legend' based on http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
#Get artists and labels for legend and chose which ones to display
handles, labels = ax.get_legend_handles_labels()
display = tuple(range(5))

#Create custom artists
trueFLArtist = Line2D((0,1),(0,0), color='r', linestyle='-', linewidth=0.5)
maxlikeFLArtist = Line2D((0,1),(0,0), color='b', linestyle='-', linewidth=0.5)

#Create legend from custom artist/label lists
ax.legend([handle for i,handle in enumerate(handles) if i in display]+[trueFLArtist,maxlikeFLArtist],
          [label for i,label in enumerate(labels) if i in display]+[lbl_0, lbl_1], loc=3)

#lbl_3 = r'$\log_{10}(\phi(L ; \theta_{maxlike})) - log_{10}(\phi(L ; \theta_{true}))$'
		  
#The following 'custom legend' based on http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
#Get artists and labels for legend and chose which ones to display
#handles, labels = ax2.get_legend_handles_labels()
#display = tuple(range(5))

#Create custom artists
#trueMaxlikeDiffFLArtist = Line2D((0,1),(0,0), color='b', linestyle='-', linewidth=0.5)

#Create legend from custom artist/label lists
#ax2.legend([handle for i,handle in enumerate(handles) if i in display]+[trueMaxlikeDiffFLArtist],
#          [label for i,label in enumerate(labels) if i in display]+[lbl_3], loc=1) # upper right

if(pdf_format):
  savefig('lumfunc_w_thetas.pdf', format='pdf')
else:
  savefig('lumfunc_w_thetas.png')