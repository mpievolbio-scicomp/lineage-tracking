import sys, os, glob, csv, re, subprocess
import string, math, numpy
from copy import deepcopy

# add custom modules to path
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, modules_path)

# import custom modules
import config
import lineage.file_parser as file_parser
from lineage.fitness_estimator import *
from lineage.inference_params import INTERVALS_PER_EPOCH


for population in config.populations:

    barcode_lists = {}

    #read data
    timepoints, data, counts = file_parser.get_data(population, config.barcode_data_root_directory)
    max_barcode = config.max_barcode[population]
    

    #read kappas
    kappas = file_parser.read_kappas_from_file(config.error_model_directory+population+'-kappas.tsv')

    #loop through the different environments
    environments = ['barcoding', 'evolution']
    if inference_params.BARCODING_INTERVALS_PER_EPOCH == 0:
        environments = ['evolution']
    for environment in environments:

        #read empirical null distribution and q-values
        empirical_null, t_statistic_95_percent_cutoff = file_parser.read_empirical_null_from_file(
                                                                                config.error_model_directory
                                                                                +population
                                                                                +'-empirical_null_%s.tsv'%environment)
        qvals = file_parser.read_q_values_from_file(config.error_model_directory+population+'-q_values_%s.tsv'%environment)

        #initialize fitness estimator object
        fitness_estimator = FitnessEstimator(counts, kappas, 
                                                qvals = qvals,
                                                t_statistic_95_percent_cutoff = t_statistic_95_percent_cutoff,
                                                empirical_null = empirical_null
                                            )

        barcoding = (environment == 'barcoding')

        #estimate relative fitnesses
        #ensure all lineage fitnesses are zero to start with
        for dataset in range(0,max_barcode):
            for parent_id in data[dataset].keys():
                for bcd, lineage in data[dataset][parent_id].items():
                    lineage.relative_fitness = np.zeros(len(lineage.freqs)//INTERVALS_PER_EPOCH)
                    lineage.relative_fitness_CI = [(0,0) for i in range(0,len(lineage.relative_fitness))]
        fitness_estimator.population_fitness = numpy.zeros(len(fitness_estimator.population_fitness))
        for target_epoch in range(0,max_barcode-barcoding):
            sys.stderr.write("Target epoch is %d. \n" % target_epoch)
            
            converged = False
            it = 0

            while not converged and it < 40:
                
                if it > 0:
                    sys.stderr.write("\tre-estimating... (it = %d)\n" % (it + 1))
                it +=1

                converged = True

                for using_barcode in range(0,target_epoch+1):

                    begin, end = fitness_estimator.get_interval_endpoints(target_epoch, barcoding = barcoding)
                    max_mean_fitness =deepcopy(fitness_estimator.population_fitness[begin:end])

                    for parent_id in data[using_barcode].keys():
                        for bcd, lineage in data[using_barcode][parent_id].items():
#                             if bcd == "TACAGGGGATGAAGCT":
#                                 breakpoint()
                            fitness_estimator.update_lineage_fitness(lineage, target_epoch, barcoding = barcoding)
                    
                    fitness_estimator.update_mean_fitness(data[using_barcode], target_epoch, barcoding = barcoding)

                    if fitness_estimator.population_fitness[end-1] > max_mean_fitness[-1] + 10**-6:
                        converged = False
                    else:
                        fitness_estimator.population_fitness[begin:end] = max_mean_fitness

        sys.stderr.write("\n writing fitnesses to file\n")
        
        barcode_lists.update({environment:[]})

        for using_barcode in range(1,max_barcode+1):
            with open(config.lineage_fitness_estimate_directory+'%s-BC%d_%s_fitnesses.csv' % 
                                                    (population,using_barcode,environment), 'w') as csvfile:
                out_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                out_writer.writerow(['BC'] + ['Epoch%d' % (d+1) for d in range(using_barcode-1,10)])

                for parent_id in data[using_barcode - 1].keys():
                    for bcd, lineage in data[using_barcode - 1][parent_id].items():
                        if any(lineage.relative_fitness > 0):
                            barcode_lists[environment].append(lineage.ID)

                            row = [lineage.ID]
                            fitnesses = numpy.zeros(3*len(lineage.relative_fitness))
                            fitnesses[::3] = lineage.relative_fitness
                            fitnesses[1::3] = [lower[0] for lower in lineage.relative_fitness_CI]
                            fitnesses[2::3] = [upper[1] for upper in lineage.relative_fitness_CI]
                            row.extend(fitnesses.tolist())
                            out_writer.writerow(row)
    sys.stderr.write('A total of %d barcodes in the __evolution__ environment have been found.\n'
                                    %(len(barcode_lists['evolution'])))    
#     sys.stderr.write('A total of %d barcodes in the __barcoding__ environment have been found.\n' %(len(barcode_lists['barcoding'])))
#     num_in_both = sum([1 for ID in barcode_lists['barcoding'] if ID in barcode_lists['evolution'] ])
#     sys.stderr.write('\t   %d are selected in __  both  __ environments.\n'%(num_in_both))    
