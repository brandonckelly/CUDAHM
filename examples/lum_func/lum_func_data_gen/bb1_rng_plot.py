"""
Test the sampler for the BB1TruncPL class.

Created 2014-07-15 by Tom Loredo from sbpl_rng.py
"""

from scipy import *
from matplotlib.pyplot import *

from bb1truncpl import BB1TruncPL

ion()


# Helper function for any RNG:

def test_rng(bb1, pdf, n_samp, n_bins, n_in=5, log_log=False):
    """
    Test a pseudo-random number generator `sampler` by generating samples
    and comparing their histogram with predictions from the density function
    `pdf`, both graphically and via chi-squared.
    """
    if False:
        samps = array([ bb1.sample() for i in range(n_samp) ]) 
    else:
        samps, _ = bb1.sample_array(n_samp)
        samps = samps[:n_samp]
    #samps += random.normal(0,0.01,samps.size) # Gauss noise [Tamas]
    # Linear axis case, best for signed samples.
    if not log_log:
        cts, bins, patches = hist(samps, bins=n_bins, log=False)
        l, u = bins[0], bins[-1]
        xvals = linspace(l, u, n_bins+1 + n_bins*n_in)
        # Plot predicted number in bin subintervals.
        pdfs = pdf(xvals)
        diffs = diff(xvals)
        # *** should shift xvals .5 sub-bin size
        plot(xvals[:-1], n_samp*pdfs[:-1]*diffs*(n_in+1), 'g-', lw=2)
        # Plot predicted number in bins.
        pdxns = empty(n_bins)
        pdfs = n_samp*pdfs
        chi = 0.
        for i in range(n_bins):
            j = i*(n_in + 1)
            k = j + n_in + 1
            vals = 0.5*(pdfs[j:k] + pdfs[j+1:k+1])*diffs[j:k]
            pdxns[i] = vals.sum()
            chi += (cts[i] - pdxns[i])**2/pdxns[i]
        centers = bins[:-1] + .5*diff(bins)
        # plot(centers, pdxns, 'ro', mew=0)
        errorbar(centers, pdxns, sqrt(pdxns), fmt='ro', mew=0)
        l, u = ylim()
        ylim(0, u)
        s = r'$\chi^2_{%i} = %f$' % (n_bins-1, chi)
        text(.1, .8, s, fontsize=14, transform=gca().transAxes)

    # Log-log axes, for positive-valued samples.
    else:
        mask = samps > 0.
        lsamps = log10(samps[mask])
        l, u = samps[mask].min(), samps[mask].max()
        lbins = logspace(log10(l), log10(u), n_bins+1)
        cts, bins, patches = hist(samps[mask], bins=lbins, log=True)
        gca().set_xscale("log")
        xvals = logspace(log10(l), log10(u), n_bins+1 + n_bins*n_in)
        # Plot predicted number in bin subintervals.
        pdfs = pdf(xvals)
        diffs = diff(xvals)
        loglog(xvals[:-1], n_samp*pdfs[:-1]*diffs*(n_in+1), 'g-', lw=2)
        pdxns = empty(n_bins)
        pdfs = n_samp*pdfs
        chi = 0.
        for i in range(n_bins):
            j = i*(n_in + 1)
            k = j + n_in + 1
            vals = 0.5*(pdfs[j:k] + pdfs[j+1:k+1])*diffs[j:k]
            pdxns[i] = vals.sum()
            chi += (cts[i] - pdxns[i])**2/pdxns[i]
        centers = bins[:-1] + .5*diff(bins)
        # plot(centers, pdxns, 'ro', mew=0)
        errorbar(centers, pdxns, sqrt(pdxns), fmt='ro', mew=0)
        yl, yu = ylim()
        ylim(.5*pdxns.min(), 1.5*pdxns.max())
        s = r'$\chi^2_{%i} = %f$' % (n_bins-1, chi)
        text(.1, .8, s, fontsize=14, transform=gca().transAxes)
    # save 
    savefig('_r3.png',dpi=120)


# Set up a BB1 instance:
scale = 100.
#alpha = -.8
gamma = -1.2
bb1 = BB1TruncPL(gamma, 1., scale)
lbl = 'BB1 %3.1f' % gamma

# Check sampler:
figure()
test_rng(bb1, bb1.pdf, 1000000, 20, 10, True)
ns = 10000
ntry = 0
for i in range(ns):
    x, n = bb1.sample(return_n=True)
    ntry += n
print ('BB1 sampler efficiency:', ns/float(ntry))
