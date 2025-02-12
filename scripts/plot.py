#!/usr/bin/python3

# example for GPU vs CPU comparison:
# ./plot.py -g -s seaborn-v0_8-bright -t "HeCBench, GPU vs CPU speedup" -c hecbench_GPU.csv -b hecbench_CPU.csv

#### CONFIG USED BY CHIPSTAR PAPER:


# ./plot.py -v -r --color '#b7cce9' -m 0.8 -s seaborn-v0_8-pastel -t "HeCBench, Intel Arc750, HIP vs SYCL speedup" -c test_20_strict_hip_oclBE_arc_after.csv -b test_20_strict_sycl_oclBE_arc_after.csv -o hecbench_intel_arc750_hip_vs_sycl.pdf

# ./plot.py -v -r --color '#b7cce9' -s seaborn-v0_8-pastel -t "PowerVR BXE-4-32 GPU vs StarFive JH7110 CPU" -b CPU_3_x_with_abort_strictmath.csv -c GPU_3_x_with_abort_strictmath.csv -o hecbench_visionfive2_gpu_vs_cpu.pdf


from optparse import OptionParser
import matplotlib.pyplot as plt
import csv

import math
outlier_factor = 30

def geomean(xs):
        return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

parser = OptionParser(description="Takes as input two CSV files produced by HeCBench's runner script (autohecbench.py), and produces a matplotlib chart with data from one CSV normalized to other CSV. Note that since hecbench.py produces CSV with timings, the input files are internally swapped to produce relative speedups (A = 3x faster B) rather than relative timing (A = 0.33 of B's time).")
parser.add_option("-b", "--input-file-baseline", dest="input_base",
                  help="CSV file with baseline data", metavar="PATH")
parser.add_option("-c", "--input-file-compared", dest="input_comp",
                  help="CSV file with compared data", metavar="PATH")
parser.add_option("-o", "--output-file", dest="output", default=None, metavar="PATH",
                  help="if specified, write output to this file (SVG,PDF,..) otherwise show chart on screen")

parser.add_option("-e", "--errorbars", dest="errbars", default=False, action="store_true",
                  help="draw error bars, default = don't draw")

parser.add_option("-g", "--geomean", dest="geomean", default=False, action="store_true",
                  help="draw geometric mean, default = don't draw")

parser.add_option("--color", dest="color", default=None,
                  help="Adjust chart bar color (optional, default = None)", metavar="STRING")

parser.add_option("--ecolor", dest="ecolor", default=None,
                  help="Adjust chart bar error color (optional, default = None)", metavar="STRING")


parser.add_option("-m", "--bottom", dest="bottom", default=0.0,  type="float",
                  help="Adjust chart bottom (optional, default = 0.0)", metavar="FLOAT")

parser.add_option("-r", "--refline", dest="refline", default=False, action="store_true",
                  help="draw dotted line @ y=1.0, default = don't draw")

parser.add_option("-s", "--style", dest="style", default=None,
                  help="chart style (optional, default = None)", metavar="STRING")

parser.add_option("-t", "--title", dest="title", default=None,
                  help="chart title (optional)", metavar="TITLE")

parser.add_option("-x", "--xlabel", dest="xlabel", default=None,
                  help="X axis label (optional)", metavar="XLABEL")
parser.add_option("-y", "--ylabel", dest="ylabel", default=None,
                  help="Y axis label (optional)", metavar="YLABEL")
parser.add_option("-z", "--zlabel", dest="zlabel", default=None,
                  help="Label embedded into chart (optional)", metavar="ZLABEL")

parser.add_option("-v", "--bar-values", dest="bar_values", default=True, action="store_false",
                  help="do not draw values of each bar, default = draw")
parser.add_option("--log-scale", dest="log_scale", default=True, action="store_false",
                  help="plot y-axis on a logarithmic scale, default = True")

parser.add_option("--bar-labels", dest="bar_labels", default=True, action="store_false",
                  help="place speedup labels at the top of each bar, default = True")

(options, args) = parser.parse_args()
if (not options.input_comp) or (not options.input_base):
	parser.error("both input files must be specified")

Baseline = {}
Compared = {}

# baseline and compared are switched here, because we want the
# graph of speedup (3x faster) which is inverse of time (0.33 time),
# and the data in CSV are recorded times
f = open(options.input_comp,'r')
reader = csv.reader(f, delimiter = ',')
for row in reader:
	K = row[0]
	if K.endswith('-hip'):
		K = K[:-4]
	if K.endswith('-cuda'):
		K = K[:-5]
	if K.endswith('-sycl'):
		K = K[:-5]
	Baseline[K] = {
	'min': float(row[1]),
	'mean': float(row[2]),
	'stddev': float(row[3]),
	'var': float(row[4]) }
f.close()

f = open(options.input_base,'r')
reader = csv.reader(f, delimiter = ',')
for row in reader:
	K = row[0]
	if K.endswith('-hip'):
		K = K[:-4]
	if K.endswith('-cuda'):
		K = K[:-5]
	if K.endswith('-sycl'):
		K = K[:-5]
	Compared[K] = {
	'min': float(row[1]),
	'mean': float(row[2]),
	'stddev': float(row[3]),
	'var': float(row[4]) }
f.close()

bench_names = []
mins = []
means = []
stddevs = []
#vars = []

for K in Baseline.keys():
	if not K in Compared.keys():
		print("skipping bench: ", K)
		continue
	
	Comp = max(Compared[K]['min'] / Baseline[K]['min'], Baseline[K]['min'] / Compared[K]['min'])
	if Comp > outlier_factor:
		print(f"Outlier, exceeding factor {outlier_factor}:", K, " compared: ", Compared[K]['min'], " baseline: ", Baseline[K]['min'], " compared/base: ", Comp)
		continue
	bench_names.append(K)
	mins.append(Compared[K]['min'] / Baseline[K]['min'])
	means.append(Compared[K]['mean'] / Baseline[K]['mean'])
	stddevs.append(Compared[K]['stddev'] / Baseline[K]['mean'])

zipped_data = zip(bench_names, mins, means, stddevs)
sorted_data = sorted(zipped_data, key = lambda x: x[1])
sorted_bench_names, sorted_mins, sorted_means, sorted_stddevs = zip(*sorted_data)

errs = sorted_stddevs
if not options.errbars:
	errs = None

if options.style:
	plt.style.use(options.style)

# Adjust figure size for journal quality (typical column width)
plt.figure(figsize=(8, 6), dpi=300)  # Higher DPI for print quality

bar_width = 0.68
bar1 = plt.bar(sorted_bench_names, [x-float(options.bottom) for x in sorted_mins], bar_width, align='center',
               yerr=errs, color=options.color, ecolor=options.ecolor,
               label=options.zlabel, bottom=float(options.bottom) )

if options.xlabel:
	plt.xlabel(options.xlabel, fontsize=9)
if options.ylabel:
	plt.ylabel(options.ylabel, fontsize=9)
if options.title:
	plt.title(options.title, fontsize=10, pad=10)

# More conservative margins
plt.margins(x=0.01, y=0.15)

# Refined subplot parameters
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.25, top=0.92)

# Clean, readable axis labels
plt.xticks(rotation=45, fontsize=8, ha='right')

if options.refline:
	plt.axhline(1.0, ls='dotted', color='gray', alpha=0.5)
if options.geomean:
	g = geomean(sorted_mins)
	s = f"Geomean = {g:.2f}"
	plt.text(0.02, 0.95, s, transform=plt.gca().transAxes, 
			 verticalalignment='top', horizontalalignment='left',
			 fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
	
	# Add the number of samples being plotted
	n_samples = len(sorted_mins)
	n_text = f"N = {n_samples}"
	plt.text(0.02, 0.90, n_text, transform=plt.gca().transAxes, 
			 verticalalignment='top', horizontalalignment='left',
			 fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Only show one set of bar labels to reduce clutter
if options.bar_labels:
	plt.bar_label(bar1, fmt='%.2f', label_type='edge', 
				 rotation=45, fontsize=7,  # Rotated labels
				 padding=5)  # Increased padding
elif options.bar_values:  # Don't show both types of labels
	plt.bar_label(bar1, fmt='%.2f', label_type='edge', 
				 rotation=45, fontsize=7,
				 padding=5)  # Increased padding

if options.log_scale:
	plt.yscale('log')

plt.tight_layout()

if options.output:
	plt.savefig(options.output, bbox_inches='tight', dpi=300)  # High DPI for print
else:
	plt.show()

