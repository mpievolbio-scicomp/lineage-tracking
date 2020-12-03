import numpy as np

# fitness estimation settings
threshold_lineage_size = 1 # do not estimate fitness for lineages below this size
size_range_mean = 10
drifting_lineage_size_range = [size_range_mean - 2.*np.sqrt(size_range_mean),size_range_mean+2.*np.sqrt(size_range_mean)] # for construction of null model
FDR = 0.05

max_fitness = {'barcoding':5.,'evolution':0.5}
num_fitnesses = 1000

barcoding_fitness_grid = np.concatenate((np.linspace(-max_fitness['barcoding'],0.0,num_fitnesses+1),np.linspace(max_fitness['barcoding']/num_fitnesses,max_fitness['barcoding'],num_fitnesses)),axis = 0)
evolution_fitness_grid = np.concatenate((np.linspace(-max_fitness['evolution'],0.0,num_fitnesses+1),np.linspace(max_fitness['evolution']/num_fitnesses,max_fitness['evolution'],num_fitnesses)),axis = 0)

fitness_grid = {'barcoding':barcoding_fitness_grid,'evolution':evolution_fitness_grid}

# assign a number of generations per interval
scale_fitness_per_interval = {'barcoding' : 1., 'evolution':10.}

EVOLUTION_INTERVALS_PER_EPOCH = 5
BARCODING_INTERVALS_PER_EPOCH = 0

INTERVALS_PER_EPOCH = BARCODING_INTERVALS_PER_EPOCH + EVOLUTION_INTERVALS_PER_EPOCH
NUMBER_OF_EPOCHS = 2