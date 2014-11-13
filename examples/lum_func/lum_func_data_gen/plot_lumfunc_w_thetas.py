# executing e.g. python plot_lumfunc_w_thetas.py -1.5 1.0 100.0 -0.5 50.001 60.0001 500000 1000 (--cov 1000)
import argparse as argp
from bb1truncpl import BB1TruncPL
import numpy as np
from matplotlib.pyplot import *
import datetime as dt
from scipy import stats

#ion()

t0 = dt.datetime.today()

parser = argp.ArgumentParser()
parser.add_argument("beta", help="The beta parameter of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("lower_scale", help="The lower scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("upper_scale", help="The upper scale of 'Break-By-1 Truncated Power Law'", type=float)
parser.add_argument("init_beta", help="The initial beta parameter value of MCMC method", type=float)
parser.add_argument("init_lower_scale", help="The initial lower scale of MCMC method", type=float)
parser.add_argument("init_upper_scale", help="The initial upper scale of MCMC method", type=float)
parser.add_argument("iter_num", help="The iteration number of MCMC method", type=int)
parser.add_argument("obj_num", help="The object number of MCMC method", type=int)
parser.add_argument("--cov", default = 10000, help="The value of this number determines, how many BB1 with samples of parameters theta will be plotted", type=int)

args = parser.parse_args()
beta = args.beta
lower_scale = args.lower_scale
upper_scale = args.upper_scale
init_beta = args.init_beta
init_lower_scale = args.init_lower_scale
init_upper_scale = args.init_upper_scale
iter_num = args.iter_num
obj_num = args.obj_num
cov = args.cov 

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


#theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',dtype=[('f0',np.float32),('f1',np.float32),('f2',np.float32)])

theta_data=np.loadtxt('lumfunc_thetas.dat',delimiter=' ',usecols=(0,1,2))

mean_vec = np.mean(theta_data, axis = 0)
std_vec = np.std(theta_data, axis = 0)
sem_vec = stats.sem(theta_data, axis = 0)

print 'Mean of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (mean_vec[0], mean_vec[1], mean_vec[2])
print 'Standard deviation of samples of theta parameters: (%5.4f,%5.4f,%5.4f)' % (std_vec[0], std_vec[1], std_vec[2])
print 'Standard error of the mean of sample values of theta parameters: (%e,%e,%e)' % (sem_vec[0], sem_vec[1], sem_vec[2])

#figsize=(10, 10)
fig_log = figure(figsize=(15.75, 10))
xlabel('$x$')
ylabel('$p(x)$')
tit = r'Init. values of $\theta$: (%5.4f,%5.4f,%5.4f); Iter. num.: %d; Obj. num.: %d;\\Means of $\theta$: (%5.4f,%5.4f,%5.4f); St. deviations of $\theta$: (%5.4f,%5.4f,%5.4f); St. error: (%e,%e,%e)' % (init_beta, init_lower_scale, init_upper_scale, iter_num, obj_num, mean_vec[0], mean_vec[1], mean_vec[2], std_vec[0], std_vec[1], std_vec[2], sem_vec[0], sem_vec[1], sem_vec[2])
fig_log.suptitle(tit, fontsize=18, fontweight='bold')

xlog = np.logspace(-3, 3, 300)
#xlin = np.linspace(0.001, 300., 500)

# Helper for plotting BB1 with samples of theta parameters:
def plot_figs(idx, xlog, c):
    smp_beta = theta_data[idx][0]
    smp_l =theta_data[idx][1]
    smp_u = theta_data[idx][2]
    bb1 = BB1TruncPL(smp_beta, smp_l, smp_u)
    pdf = bb1.pdf(xlog)
    figure(fig_log.number)
    #alpha=.01, linewidth=0.5
    #the green colored line belongs to the latest theta from iterations.
    red_rate = (1.0 - idx/float(theta_data.shape[0]))
    loglog(xlog, pdf, color=(red_rate*1.0,1.0,0.0), alpha=.01, linewidth=0.5, zorder=1)

t1 = dt.datetime.today()
print 'Elapsed time of loading samples of theta parameters:', t1-t0

t0 = dt.datetime.today()
bb1_0 = BB1TruncPL(beta, lower_scale, upper_scale)
lbl_0 = 'BB1 (%5.2f,%5.2f,%5.2f)' % (beta, lower_scale, upper_scale)
pdf_0 = bb1_0.pdf(xlog)
figure(fig_log.number)
loglog(xlog, pdf_0, 'r-', linewidth=2, label=lbl_0, zorder=3)

bb1_1 = BB1TruncPL(mean_vec[0], mean_vec[1], mean_vec[2])
lbl_1 = 'Mean BB1 (%5.2f,%5.2f,%5.2f)' % (mean_vec[0], mean_vec[1], mean_vec[2])
pdf_1 = bb1_1.pdf(xlog)
figure(fig_log.number)
loglog(xlog, pdf_1, 'm-', linewidth=2, label=lbl_1, zorder=2)

u_array = np.random.uniform(size=theta_data.shape[0])
accept_rate = cov/float(theta_data.shape[0])

cnt_accept = 0
for idx in range(0, theta_data.shape[0]):
	if u_array[idx] < accept_rate:
		plot_figs(idx, xlog, 'b')
		cnt_accept += 1

print 'Count of accepted array elements: %d' % cnt_accept
#plot_figs(theta_data[19][0], theta_data[19][1], theta_data[19][2], xlog, 'm')

legend(loc=3)
savefig('lumfunc_w_thetas.png',dpi=120)
t1 = dt.datetime.today()
print 'Elapsed time of generating figure of luminosity density function with samples of theta parameters:', t1-t0