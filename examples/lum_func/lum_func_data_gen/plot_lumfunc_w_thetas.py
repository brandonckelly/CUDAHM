# executing e.g. python plot_lumfunc_w_thetas.py lumfunc_thetas_2.dat -1.5 50000000000.0 5000000000000.0 -1.41 4.0 5.8 -1.5564 7.3222 5.7207 1500000 1500000 100000 (--cov 1000 --lower_scale_factor 10000000000.0 --upper_scale_factor 1000000000000.0)
import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
import datetime as dt
from scipy import stats

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

execfile("rc_settings.py")

#theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',dtype=[('f0',np.float32),('f1',np.float32),('f2',np.float32)])

theta_data=np.loadtxt(file,delimiter=' ',usecols=(0,1,2))

mean_vec = np.mean(theta_data, axis = 0)
std_vec = np.std(theta_data, axis = 0)
sem_vec = stats.sem(theta_data, axis = 0)

print 'Mean of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (mean_vec[0], mean_vec[1], mean_vec[2])
print 'Standard deviation of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (std_vec[0], std_vec[1], std_vec[2])
print 'Standard error of the mean of sample values of theta parameters: (%e,%e,%e)' % (sem_vec[0], sem_vec[1], sem_vec[2])

fig, ax = subplots()
ax.set_xlabel(r'$L$')
ax.set_ylabel(r'$\phi(L ; \theta)$')
tit = r'Luminosity density function'
#ax.set_title(tit)

xlog = np.logspace(11, 13, 300)

# Helper for plotting BB1 with samples of theta parameters:
def plot_figs(idx, xlog, c):
    smp_beta = theta_data[idx][0]
    smp_l =theta_data[idx][1] * lower_scale_factor
    smp_u = theta_data[idx][2] * upper_scale_factor
    bb1 = BB1TruncPL(smp_beta, smp_l, smp_u)
    pdf = bb1.pdf(xlog)
    #the green colored line belongs to the latest theta from iterations.
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    ax.loglog(xlog, pdf, color=(red_rate*1.0,1.0,0.0), alpha=.01, linewidth=0.5, zorder=1)

t1 = dt.datetime.today()
print 'Elapsed time of loading samples of theta parameters:', t1-t0

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
lbl_0 = r'$\phi(L ; \theta_{true})$'
pdf_0 = bb1_0.pdf(xlog)
ax.loglog(xlog, pdf_0, 'r-', linewidth=1.0, label=lbl_0, zorder=3)

bb1_1 = BB1TruncPL(maxlike_beta, maxlike_lower_scale, maxlike_upper_scale)
lbl_1 = r'$\phi(L ; \theta_{maxlike})$'
pdf_1 = bb1_1.pdf(xlog)
ax.loglog(xlog, pdf_1, 'b-', linewidth=1.0, label=lbl_1, zorder=2)

u_array = np.random.uniform(size=theta_data.shape[0])
accept_rate = cov/float(theta_data.shape[0])

cnt_accept = 0
for idx in range(0, theta_data.shape[0]):
	if u_array[idx] < accept_rate:
		plot_figs(idx, xlog, 'b')
		cnt_accept += 1

print 'Count of accepted array elements: %d' % cnt_accept

ax.legend(loc=3)  # lower left
savefig('lumfunc_w_thetas.pdf', format='pdf')
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with samples of theta parameters:', t1-t0