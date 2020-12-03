import os,sys,inspect

# add modules to path
parent_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
barcode_data_root_directory = parent_directory + '/data/barcode_frequencies/'
error_model_directory = parent_directory + '/data/error_model/'
lineage_fitness_estimate_directory = parent_directory + '/data/relative_fitnesses/'
lineage_flag_directory = parent_directory + '/data/flags/'
clone_data_directory = parent_directory + '/data/accepted_clones/'
figure_directory = parent_directory + '/figures/'
wgs_data_directory = parent_directory + '/data/wgs/'

# population settings
populations = [
#                'C1',
#                'D1'
#                'C1test',
               'LTtest',
#                'LTtest_10epochs',
              ]

max_barcode = {'C1':9,
               'D1':10,
               'C1test':9,
               'LTtest':2,
               'LTtest_10epochs': 1,
              }

pop_labels = {'C1':'YPD','D1':'YPA'}
pop_colors = {'C1':'#232F52', 'D1':'orange'}
highlighted_barcodes = {'C1':['TCTAAGCGTATTGGTC_ATCCAGCGCTTGGACG','TTTGGCAACCTGGGTG_TATTGCAGTTATGCAG',
						'TACAGGGGATGAAGCT_GTTGATTGGACGGATG','TAGTGACTTAGACCTG_CTTATCAACGGTGCTA'],
					 'D1':['TCCATTGAGAACAACT','CGCGGTGGAACGGAGG_CGCAACATGTAAACTT','TCTAGCGCGGCCGAAT','CCACGCGCGGTACGTC']}

