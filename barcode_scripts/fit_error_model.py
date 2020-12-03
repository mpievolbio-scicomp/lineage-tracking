import sys, os, glob, csv, re
import string, math, numpy
import os

# add custom modules to path
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, modules_path)

#custom modules
import config
import lineage.inference_params as inference_params
import lineage.file_parser as file_parser

# look only at lineages within specified size range
drifting_lineage_size_range = inference_params.drifting_lineage_size_range

INTERVALS_PER_EPOCH = inference_params.INTERVALS_PER_EPOCH
EVOLUTION_INTERVALS_PER_EPOCH = inference_params.EVOLUTION_INTERVALS_PER_EPOCH
NUMBER_OF_EPOCHS = inference_params.NUMBER_OF_EPOCHS

for population in config.populations:

    timepoints, data, counts = file_parser.get_data(population, config.barcode_data_root_directory, as_matrix = True)
    count_data = [data[i]*counts[INTERVALS_PER_EPOCH*i:] for i in range(0,len(data))]

    max_barcode = config.max_barcode[population]
    # sort count data on min size in first epoch
    arrinds= [numpy.argsort(numpy.min(count_data[i].T[:INTERVALS_PER_EPOCH],axis = 0)) for i in range(0,len(data))]
    sorted_mins = [numpy.min(count_data[i].T[:INTERVALS_PER_EPOCH],axis = 0)[arrinds[i][::-1]] for i in range(0,len(data))]
    sorted_count_data = [count_data[i][arrinds[i][::-1]] for i in range(0,len(data))]

    sys.stderr.write('\tfitting kappas...\n')

    kappas = numpy.zeros(len(timepoints[0])-1)
    kappa_errors = numpy.zeros(len(timepoints[0])-1)
    ns = numpy.zeros(len(timepoints[0])-1)
    
    # In the original code, max_barcode is identical to the number of epochs. In
    # our case, we have multiple epochs, but only one barcode.
    for timepoint in range(0,max_barcode*INTERVALS_PER_EPOCH-1):
#     for timepoint in range(0,NUMBER_OF_EPOCHS*INTERVALS_PER_EPOCH-1):
        
        epoch = timepoint // INTERVALS_PER_EPOCH
        t = timepoint % INTERVALS_PER_EPOCH
    
        counts_before, counts_after = sorted_count_data[epoch].T[t],sorted_count_data[epoch].T[t+1]
        counts_before = numpy.asarray(counts_before,dtype = float)
        counts_after = numpy.asarray(counts_after,dtype = float)

        before_filter = numpy.logical_and(counts_before <= drifting_lineage_size_range[1], counts_before >= drifting_lineage_size_range[0])
        counts_after = counts_after[before_filter]
        counts_before = counts_before[before_filter]
        scaled_counts_before = counts_before * counts[INTERVALS_PER_EPOCH*epoch+t+1]/counts[INTERVALS_PER_EPOCH*epoch+t]
        n = len(counts_before)
              
        kappas[INTERVALS_PER_EPOCH*epoch + t] = 2* ((numpy.sqrt(counts_after) - numpy.sqrt(scaled_counts_before))**2).sum()/n
        kappa_errors[INTERVALS_PER_EPOCH*epoch+t] =kappas[INTERVALS_PER_EPOCH*epoch + t]* numpy.sqrt(2./n)
        ns[INTERVALS_PER_EPOCH*epoch+t] = n
    # Fix zeroes as these will cause DivideByZeros later.
    zero_kappas =  kappas==0.0
    kappas[zero_kappas] = 100.0
    kappa_errors[zero_kappas] = 10.0
    kappa_CI_lower = kappas - 2 *kappa_errors
    kappa_CI_upper = kappas + 2* kappa_errors

    with open(config.error_model_directory+'%s-kappas.tsv' % (population), 'w') as csvfile:
        out_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        out_writer.writerow(timepoints[0][1:])
        out_writer.writerow(["{0:.2f}".format(x) for x in kappas])
        out_writer.writerow(["{0:.2f}".format(x) for x in kappa_CI_lower])
        out_writer.writerow(["{0:.2f}".format(x) for x in kappa_CI_upper])
        out_writer.writerow(["{0:.2f}".format(x) for x in ns])

    with open(config.error_model_directory+'%s-kappas-barcoding.tsv' % (population), 'w') as csvfile:
        out_writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        out_writer.writerow(timepoints[0][INTERVALS_PER_EPOCH::INTERVALS_PER_EPOCH])
        out_writer.writerow(["{0:.2f}".format(x) for x in kappas[EVOLUTION_INTERVALS_PER_EPOCH::INTERVALS_PER_EPOCH]])
        out_writer.writerow(["{0:.2f}".format(x) for x in kappa_CI_lower[EVOLUTION_INTERVALS_PER_EPOCH::INTERVALS_PER_EPOCH]])
        out_writer.writerow(["{0:.2f}".format(x) for x in ns[EVOLUTION_INTERVALS_PER_EPOCH::INTERVALS_PER_EPOCH]])