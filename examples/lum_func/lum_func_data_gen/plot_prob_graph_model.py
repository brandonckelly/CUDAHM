# Using Daft python package (http://daft-pgm.org/)
# Executing e.g.: python plot_prob_graph_model.py --pdf_format False

import daft
import argparse as argp
from matplotlib import rc

parser = argp.ArgumentParser()
parser.add_argument("--pdf_format", default = 'True', help="Would you like pdf format and high resolution for the figure output(s)?", type=str)

args = parser.parse_args()
pdf_format = eval(args.pdf_format)
execfile("rc_settings.py")
if(pdf_format!=True):
  rc('savefig', dpi=100)

# Wider margins to allow for larger labels; may need to adjust left:
#rc('figure.subplot', bottom=.125, top=.95, right=.95)  # left=0.125

# Optionally make default line width thicker:
#rc('lines', linewidth=2.0) # doesn't affect frame lines

#rc('font', size=14)  # default for labels (not axis labels)
#rc('font', family='serif')  # default for labels (not axis labels)
#rc('axes', labelsize=18)
#rc('xtick.major', pad=8)
#rc('xtick', labelsize=14)
#rc('ytick.major', pad=8)
#rc('ytick', labelsize=14)

#rc('savefig', dpi=150)  # mpl's default dpi is 100
#rc('axes.formatter', limits=(-4,4))

# Use TeX labels with CMR font:
#rc('text', usetex=True)
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

pgm = daft.PGM([5.0, 3.5], origin=[0.5, 0.0])

pgm.add_node(daft.Node("shape_param", r"$\beta$", 1.5, 2.7))
pgm.add_node(daft.Node("upper_scale", r"$u$", 1.5, 1.7))
pgm.add_node(daft.Node("lower_scale", r"$l$", 1.5, 0.7))

pgm.add_node(daft.Node("luminosity", r"$L_i$", 3.0, 2.1))
pgm.add_node(daft.Node("distance", r"$r_i$", 3.0, 1.3))

pgm.add_node(daft.Node("true_flux", r"$F_i$", 3.75, 1.7))

pgm.add_node(daft.Node("noisy_flux", r"$D_i$", 4.5, 1.7, observed=True))

pgm.add_edge("shape_param", "luminosity")
pgm.add_edge("upper_scale", "luminosity")
pgm.add_edge("lower_scale", "luminosity")

pgm.add_edge("luminosity", "true_flux")
pgm.add_edge("distance", "true_flux")

pgm.add_edge("true_flux", "noisy_flux")

pgm.add_plate(daft.Plate([1.0, 0.2, 1.0, 3.2], label=r"$\theta$", shift=-0.1, label_offset=[45,5]))
pgm.add_plate(daft.Plate([2.5, 0.8, 2.5, 1.8], label=r"$N$", shift=-0.1, label_offset=[125,5]))

# Render and save.
pgm.render()

if(pdf_format):
  pgm.figure.savefig("prob_graph_model.pdf", format='pdf')
else:
  pgm.figure.savefig("prob_graph_model.png")