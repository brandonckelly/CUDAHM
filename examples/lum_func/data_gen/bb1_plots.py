"""
Demonstrate the break-by-one power law distribution.

Created 2014-07-15 by Tom Loredo; based on powerlaw_plots.py
"""

from scipy import *
from scipy import stats
from scipy.special import gamma

from matplotlib.pyplot import *

# from myplot import tex_on, FigLRAxes

from bb1truncpl import BB1TruncPL

ion()

# From myplot.py:
rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125
rc('font', size=14)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=18)
rc('xtick.major', pad=8)
rc('xtick', labelsize=14)
rc('ytick.major', pad=8)
rc('ytick', labelsize=14)
rc('savefig', dpi=150)
rc('axes.formatter', limits=(-4,4))
# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


class FigLRAxes:
    """
    A figure with two ordinate axes (left and right) sharing a common
    abscissa axis.
    
    In matplotlib lingo, this is a two-scale plot using twinx().
    """

    def __init__(self, figsize=(8,6), l=0.15, r=0.85):
        self.fig = figure(figsize=figsize)
        # Left and right axes:
        self.leftax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=l, right=r)
        self.rightax = self.leftax.twinx()
        # Use thicker frame lines.
        self.leftax.patch.set_lw(1.25)  # thicker frame lines
        self.rightax.patch.set_lw(1.25)  # thicker frame lines
        # Leave with the left axes as current.
        self.fig.sca(self.leftax)

    def left(self):
        self.fig.sca(self.leftax)
        return self.leftax

    def right(self):
        self.fig.sca(self.rightax)
        return self.rightax


# Setup two figures:

# 1:  log p vs log x
fig_log = figure()
xlabel('$x$')
ylabel('$p(x)$')

# 2:  x*log p vs log x
fig_slogx = FigLRAxes()  # semilogx, x*p(x) vs. log(x)
fig_slogx.left()
xlabel('$x$')
ylabel(r'$x \times p(x)$')
fig_slogx.right()
ylabel('Slope')

alpha = .2  # gamma dist'n shape parameter; PL index is gamma = alpha - 1
scale = 100.
xlog = logspace(-3, 3, 300)
xlin = linspace(0.001, 300., 500)


# First plot gamma dist'n:
gd = stats.gamma(alpha, scale=scale)
figure(fig_log.number)
pdf = gd.pdf(xlog)
loglog(xlog, pdf, 'b-', lw=2, label='gamma 0.2-1')
figure(fig_slogx.fig.number)
fig_slogx.left()
semilogx(xlog, xlog*pdf, 'b-', lw=2, label='gamma 0.2-1')
savefig('_t1.png',dpi=120)


# Helper for plotting BB1 instances:

def plot_figs(bpl, xlog, c, lbl):

    pdf = bpl.pdf(xlog)
    slope = bpl.log_slope(xlog)

    figure(fig_log.number)
    loglog(xlog, pdf, c+'-', lw=2, label=lbl)

    figure(fig_slogx.fig.number)
    fig_slogx.left()
    semilogx(xlog, xlog*pdf, c+'-', lw=2, label=lbl)

    fig_slogx.right()
    semilogx(xlog, slope, c+':', lw=1)


# Middle power law range is x_l to scale.
x_l = .1


# BB1, gamma = -.8 (valid range for gamma dist'n):
bb1 = BB1TruncPL(-.8, x_l, scale)
lbl = 'BB1 %3.1f' % -.8
plot_figs(bb1, xlog, 'r', lbl)

# BB1, gamma = -1.2 (gamma dist'n would be improper):
bb1 = BB1TruncPL(-1.2, x_l, scale)
lbl = 'BB1 %3.1f' % -1.2
plot_figs(bb1, xlog, 'g', lbl)

# BB1, gamma = -1:
bb1 = BB1TruncPL(-1., x_l, scale)
lbl = 'BB1 %3.1f' % -1
plot_figs(bb1, xlog, 'm', lbl)

figure(fig_log.number)
legend(loc=0)
figure(fig_slogx.fig.number)
fig_slogx.left()
legend(loc=1, framealpha=.8)  # 0=best, 1=UR, 2=UL...
fig_slogx.right()
ylim(-2, 1)
yticks([-2, -1, 0, 1])

savefig('_t2.png',dpi=120)
