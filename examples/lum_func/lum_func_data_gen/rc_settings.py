from matplotlib.pyplot import *

# Wider margins to allow for larger labels; may need to adjust left:
rc('figure.subplot', bottom=.125, top=.95, right=.95, left=0.175)  # left=0.125

# Optionally make default line width thicker:
#rc('lines', linewidth=2.0) # doesn't affect frame lines

rc('font', size=24)  # default for labels (not axis labels)
rc('font', family='serif')  # default for labels (not axis labels)
rc('axes', labelsize=32)
rc('xtick.major', pad=8)
rc('xtick', labelsize=24)
rc('ytick.major', pad=8)
rc('ytick', labelsize=24)

rc('savefig', dpi=150)  # mpl's default dpi is 100
rc('axes.formatter', limits=(-4,4))

# Use TeX labels with CMR font:
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

# Use square shape
rc('figure', figsize=(10, 10))