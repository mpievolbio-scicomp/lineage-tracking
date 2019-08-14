import sys, os, glob, csv, re
import string, math, numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.patches as patches
import argparse
from scipy.special import erf
from copy import deepcopy
import matplotlib.gridspec as gridspec
import itertools

# add custom modules to path
sys.path.insert(0,'../modules/')

# import custom modules
import config
import local_matplotlibrc
import lineage.file_parser as file_parser
from lineage.plot_utils import * # contains muller and bar chart class
from lineage.tree_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-dpi", default = 600, type = int, help = 'resolution (dots per inch)')
parser.add_argument("-width", default = 3.4, type = float, help = 'figure width (inches)')
parser.add_argument("-height", default = 2.55, type = float, help = 'figure height (inches)')
parser.add_argument("-fontsize", default = 7, type = int, help = 'fontsize (pt)')
parser.add_argument("--stretched",dest='stretched', action='store_true',default=True,
                    help='display barcoding intervals as 100 generations long')

args = parser.parse_args()

dpi = args.dpi
fig_width = args.width
fig_height = args.height
fontsize = args.fontsize
stretched = args.stretched

matplotlib.rcParams['xtick.labelsize'] = fontsize
matplotlib.rcParams['ytick.labelsize'] = fontsize
matplotlib.rcParams['font.size'] = fontsize + 1

# display barcoding as length_of_barcoding_generations longs
if stretched:
	length_of_barcoding = 100
else:
	length_of_barcoding = 10

#time to add to 10 generations
additional_length_of_barcoding = length_of_barcoding - 10

for population in config.populations:

	if population == 'C1':
		subdir = 'main'
	else:
		subdir = 'other'

	max_barcode = config.max_barcode[population]

	timepoints, lineage_data, counts = file_parser.get_data(population, config.barcode_data_root_directory)
	times = timepoints[0]

	for epoch in range(1,max_barcode):
		times[epoch*11:]+= additional_length_of_barcoding

	freq_cutoff = 10**-3
	lineage_list = []
	population_tree = {}
	lineage_dict = {}
	for epoch in range(0,len(lineage_data)):
		for parental_bcd in lineage_data[epoch].keys():
			for bcd, lineage in lineage_data[epoch][parental_bcd].iteritems():
				if any(lineage.freqs > freq_cutoff):
					lineage_list.append(lineage.ID)
					insert_into_tree(lineage.ID,population_tree)
					lineage_dict.update({lineage.ID:lineage})


	#check tree doesn't have major flaws
	for key in population_tree.keys():
		ancestral_barcode_options = ["_".join(key.split("_")[0:l]) for l in range (1,len(key.split("_")))]
		parents = [p_ID for p_ID in lineage_list if p_ID in ancestral_barcode_options]
		if len(parents) > 0:
			print "Error!"

	fig_muller,ax_muller = plt.subplots(figsize = (fig_width,fig_height))

	muller_diagram = MullerDiagram(fig_muller,ax_muller,times)

	max_freqs = compile_max_freqs(population_tree.keys(),lineage_dict)
	sorted_ids = [x for y,x in reversed(sorted(zip(max_freqs,population_tree.keys())))]

	assign_muller_lower(sorted_ids,lineage_dict,population_tree,
		baseline = numpy.zeros(len(times)),
			range_available = numpy.ones(len(times)))

	number_of_lineages_with_children = len([key for key in population_tree.keys() if len(population_tree[key].keys())>0])
	if 'C1' in population:
		start_index = 7
		small_start_index = 0
		small_alpha = 0.3
	else:
		start_index = 8 # 0, 5, 7
		small_start_index = 10
		small_alpha = 0.3
	large_color_cycler = ColorCycler(number_of_lineages_with_children,start_index = start_index)
	small_color_cycler = ColorCycler(40,start_index = small_start_index)

	no_barcodes = 1
	for ID in lineage_list:
		if ID in population_tree.keys():
			alpha = 1.
			cycler =  large_color_cycler
		else:
			alpha = max(small_alpha,0.8-0.1*len(ID.split("_")))
			cycler = small_color_cycler
		lineage_dict[ID].color = cycler.get_new_color(alpha = alpha)
		muller_diagram.place(lineage_dict[ID])

	for no_barcodes in range(1,max_barcode):

		rectangle_right = (110 + additional_length_of_barcoding)*no_barcodes
		if stretched:
			bar_alpha = 0.3
		else:
			bar_alpha = 0.7
		if no_barcodes > 0:
			ax_muller.add_patch(patches.Rectangle(
	        	(rectangle_right - length_of_barcoding, -0.1), length_of_barcoding, 1.1,
	        	facecolor= '0.3',
	        	alpha = bar_alpha,
	        	edgecolor = "none"))

	ax_muller.set_xlabel('Generations')
	ax_muller.set_ylabel('Barcodes')

	xticks = []
	xticklabels = []

	xtick_dt = 110+additional_length_of_barcoding
	for i in range(0,max_barcode+1):
		if i == 0:
			xticks.extend([xtick_dt*i,xtick_dt*i+100])
		else:
			xticks.extend([xtick_dt*i,xtick_dt*i+100])
		xticklabels.extend(['%d.0'%(i+1),'%d.100'%(i+1)])
	ax_muller.set_xticks(xticks)
	ax_muller.set_xticklabels(xticklabels)
	ax_muller.set_xlim([0,xtick_dt*max_barcode-length_of_barcoding])

	i = 0
	for tick in ax_muller.get_xticklabels():
   		tick.set_rotation(90)
   		if stretched:
   			tick.set_horizontalalignment('center')
   		else:
	   		if i % 2 == 0:
	   			tick.set_horizontalalignment('left')
	   		else:
	   			tick.set_horizontalalignment('right')
	   		i+=1
	if stretched:
		insert_string = "_stretched"
	else:
		insert_string =""
	
	muller_diagram.save(config.figure_directory+'%s/%s_raw_muller%s.png' % (subdir,population,insert_string),dpi = dpi)

