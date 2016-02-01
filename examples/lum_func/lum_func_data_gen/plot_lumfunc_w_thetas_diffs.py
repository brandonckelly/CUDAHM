# executing e.g. python plot_lumfunc_w_thetas_diffs.py lumfunc_thetas_2.dat -1.5 50000000000.0 5000000000000.0 -1.41 4.0 5.8 -1.5564 7.3222 5.7207 1500000 1500000 100000 (--cov 1000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0)
import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
from matplotlib.cbook import get_sample_data
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
parser.add_argument("--resolution", default = 300, help="The resolution of x-axis", type=int)
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
rc('figure', figsize=(5, 2.5))
if(pdf_format!=True):
  rc('savefig', dpi=100)

#theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',dtype=[('f0',np.float32),('f1',np.float32),('f2',np.float32)])

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

mean_vec = np.mean(theta_data, axis = 0)
std_vec = np.std(theta_data, axis = 0)
sem_vec = stats.sem(theta_data, axis = 0)

print 'Mean of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (mean_vec[0], mean_vec[1], mean_vec[2])
print 'Standard deviation of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (std_vec[0], std_vec[1], std_vec[2])
print 'Standard error of the mean of sample values of theta parameters: (%e,%e,%e)' % (sem_vec[0], sem_vec[1], sem_vec[2])

#med_vec = np.median(theta_data, axis = 0)
#print 'Median of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (med_vec[0], med_vec[1], med_vec[2])

fig, ax = subplots()
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(bottom='off',top='off',left='off',right='off')

xlin = np.linspace(10**xlog_min, 10**xlog_max, resolution)
log10_of_xlin = np.log10(xlin)

# Helper for plotting BB1 with samples of theta parameters:
def plot_figs(idx, xlin, log10_of_xlin, c):
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

t1 = dt.datetime.today()
print 'Elapsed time of loading samples of theta parameters:', t1-t0

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlin)
log10_of_pdf_0 = np.log10(pdf_0)

bb1_1 = BB1TruncPL(maxlike_beta, maxlike_lower_scale, maxlike_upper_scale)
pdf_1 = bb1_1.pdf(xlin)
log10_of_pdf_1 = np.log10(pdf_1)
log10_of_diff_true_maxlike = log10_of_pdf_0 - log10_of_pdf_1
ax.plot(log10_of_xlin, log10_of_diff_true_maxlike, 'b-', linewidth=0.5, zorder=2)

#med_beta = med_vec[0]
#med_l = med_vec[1] * lower_scale_factor
#med_u = med_vec[2] * upper_scale_factor
#bb1_2 = BB1TruncPL(med_beta, med_l, med_u)
#lbl_2 = r'$\log_{10}(\phi(L ; \theta_{postmed})) - log_{10}(\phi(L ; \theta_{true}))$'
#pdf_2 = bb1_2.pdf(xlin)
#log10_of_pdf_2 = np.log10(pdf_2)
#log10_of_diff_post_med = log10_of_pdf_0 - log10_of_pdf_2
#ax.plot(log10_of_xlin, log10_of_diff_post_med, 'darkgreen', linestyle='solid', linewidth=0.5, label=lbl_2, zorder=2)

u_array = np.random.uniform(size=theta_data.shape[0])
accept_rate = cov/float(theta_data.shape[0])

cnt_accept = 0
for idx in range(0, theta_data.shape[0]):
	if u_array[idx] < accept_rate:
		plot_figs(idx, xlin, log10_of_xlin, 'b')
		cnt_accept += 1

print 'Count of accepted array elements: %d' % cnt_accept

original_xticks = ax.get_xticks()
original_yticks = ax.get_yticks()

savefig('lumfunc_w_thetas_diffs_curve.png')

t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function differences between with true parameters and with samples of theta parameters:', t1-t0

execfile("rc_settings.py")
rc('figure.subplot', bottom=.195, top=.85, right=.95, left=0.175)
rc('figure', figsize=(5, 2.5))
fig, ax = subplots()
ax.set_xlabel(r'$\log_{10}(L)$')
ax.set_ylabel(r'$\log_{10}(\phi(L ; \theta)) - log_{10}(\phi(L ; \theta_{true}))$')

currentdir = os.getcwd()
im = imread(get_sample_data(currentdir+'\\'+'lumfunc_w_thetas_diffs_curve.png'))
ax.imshow(im, extent=[original_xticks[0],original_xticks[-1],original_yticks[0], original_yticks[-1]], aspect="auto")

lbl_1 = r'$\log_{10}(\phi(L ; \theta_{maxlike})) - log_{10}(\phi(L ; \theta_{true}))$'

#The following 'custom legend' based on http://stackoverflow.com/questions/13303928/how-to-make-custom-legend-in-matplotlib
#Get artists and labels for legend and chose which ones to display
handles, labels = ax.get_legend_handles_labels()
display = tuple(range(5))

#Create custom artists
trueMaxlikeDiffFLArtist = Line2D((0,1),(0,0), color='b', linestyle='-', linewidth=0.5)

#Create legend from custom artist/label lists
ax.legend([handle for i,handle in enumerate(handles) if i in display]+[trueMaxlikeDiffFLArtist],
          [label for i,label in enumerate(labels) if i in display]+[lbl_1], loc=1) # upper right

if(pdf_format):
  savefig('lumfunc_w_thetas_diffs.pdf', format='pdf')
else:
  savefig('lumfunc_w_thetas_diffs.png')