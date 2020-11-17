import unittest
import sys
import os
import pandas
import numpy
import pytest
import shlex
from collections import namedtuple
from subprocess import Popen

# add custom modules to path
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, modules_path)

test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'barcode_scripts'))
sys.path.insert(0, test_path)

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
reference_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reference_data'))

import config

class BarcodeScriptsTest(unittest.TestCase):
    def setUp(self):
        pass

    @pytest.mark.dependency()
    def test_fit_error_model(self):
        """ Run 'fit_error_model.py' on reference data and compare to reference results."""
        # This runs the script.
        import fit_error_model

        # Read generated data.
        error_model_path = config.error_model_directory
        C1_kappas_evolution = pandas.read_csv(os.path.join(error_model_path, 'C1-kappas.tsv'), sep='\t')
        C1_kappas_barcoding = pandas.read_csv(os.path.join(error_model_path, 'C1-kappas-barcoding.tsv'), sep='\t')
        D1_kappas_evolution = pandas.read_csv(os.path.join(error_model_path, 'D1-kappas.tsv'), sep='\t')
        D1_kappas_barcoding = pandas.read_csv(os.path.join(error_model_path, 'D1-kappas-barcoding.tsv'), sep='\t')

        # Read reference data.
        error_model_reference_path = error_model_path.replace('data', 'reference_data')
        C1_reference_kappas_evolution = pandas.read_csv(os.path.join(error_model_reference_path, 'C1-kappas.tsv'), sep='\t')
        C1_reference_kappas_barcoding = pandas.read_csv(os.path.join(error_model_reference_path, 'C1-kappas-barcoding.tsv'), sep='\t')
        D1_reference_kappas_evolution = pandas.read_csv(os.path.join(error_model_reference_path, 'D1-kappas.tsv'), sep='\t')
        D1_reference_kappas_barcoding = pandas.read_csv(os.path.join(error_model_reference_path, 'D1-kappas-barcoding.tsv'), sep='\t')

        # Compare. We convert to numpy arrays and take the norm of the difference which should be 0.
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_kappas_evolution.to_numpy() - C1_kappas_evolution.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_kappas_evolution.to_numpy() - D1_kappas_evolution.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_kappas_barcoding.to_numpy() - C1_kappas_barcoding.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_kappas_barcoding.to_numpy() - D1_kappas_barcoding.to_numpy()), 0.0)

    @pytest.mark.dependency(depends=['test_fit_error_model'])
    def test_compile_null_distribution(self):
        """ Run 'compile_null_distribution.py' on reference data and compare to reference results."""
        # This runs the script.
        import compile_null_distribution

        # Read generated data.
        error_model_path = config.error_model_directory
        C1_empirical_null_evolution = numpy.loadtxt(os.path.join(error_model_path, 'C1-empirical_null_evolution.tsv'))
        C1_empirical_null_barcoding = numpy.loadtxt(os.path.join(error_model_path, 'C1-empirical_null_barcoding.tsv'))
        D1_empirical_null_evolution = numpy.loadtxt(os.path.join(error_model_path, 'D1-empirical_null_evolution.tsv'))
        D1_empirical_null_barcoding = numpy.loadtxt(os.path.join(error_model_path, 'D1-empirical_null_barcoding.tsv'))
        C1_q_evolution = numpy.loadtxt(os.path.join(error_model_path, 'C1-q_values_evolution.tsv'), max_rows=1)
        C1_q_barcoding = numpy.loadtxt(os.path.join(error_model_path, 'C1-q_values_barcoding.tsv'), max_rows=1)
        D1_q_evolution = numpy.loadtxt(os.path.join(error_model_path, 'D1-q_values_evolution.tsv'), max_rows=1)
        D1_q_barcoding = numpy.loadtxt(os.path.join(error_model_path, 'D1-q_values_barcoding.tsv'), max_rows=1)

        # Read reference data.
        error_model_reference_path = error_model_path.replace('data', 'reference_data')
        C1_reference_empirical_null_evolution = numpy.loadtxt(os.path.join(error_model_reference_path, 'C1-empirical_null_evolution.tsv'))
        C1_reference_empirical_null_barcoding = numpy.loadtxt(os.path.join(error_model_reference_path, 'C1-empirical_null_barcoding.tsv'))
        D1_reference_empirical_null_evolution = numpy.loadtxt(os.path.join(error_model_reference_path, 'D1-empirical_null_evolution.tsv'))
        D1_reference_empirical_null_barcoding = numpy.loadtxt(os.path.join(error_model_reference_path, 'D1-empirical_null_barcoding.tsv'))
        C1_reference_q_evolution = numpy.loadtxt(os.path.join(error_model_reference_path, 'C1-q_values_evolution.tsv'), max_rows=1)
        C1_reference_q_barcoding = numpy.loadtxt(os.path.join(error_model_reference_path, 'C1-q_values_barcoding.tsv'), max_rows=1)
        D1_reference_q_evolution = numpy.loadtxt(os.path.join(error_model_reference_path, 'D1-q_values_evolution.tsv'), max_rows=1)
        D1_reference_q_barcoding = numpy.loadtxt(os.path.join(error_model_reference_path, 'D1-q_values_barcoding.tsv'), max_rows=1)

        # Compare. We convert to numpy arrays and take the norm of the difference which should be 0.
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_empirical_null_evolution - C1_empirical_null_evolution), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_empirical_null_barcoding - C1_empirical_null_barcoding), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_empirical_null_evolution - D1_empirical_null_evolution), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_empirical_null_barcoding - D1_empirical_null_barcoding), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_q_evolution - C1_q_evolution), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_reference_q_barcoding - C1_q_barcoding), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_q_evolution - D1_q_evolution), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_reference_q_barcoding - D1_q_barcoding), 0.0)

    @pytest.mark.dependency(depends=['test_compile_null_distribution'])
    def test_estimate_relative_fitnesses(self):
        """Run 'estimate_relative_fitness.py' on reference data and compare to reference results."""
        # This runs the script.
        import estimate_relative_fitnesses

        # Set paths to data.
        lineage_fitness_path = config.lineage_fitness_estimate_directory
        reference_lineage_fitness_path = lineage_fitness_path.replace('reference_data', 'data')

        # C1 barcode fitness
        C1_BC1_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC1_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC2_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC2_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC3_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC3_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC4_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC4_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC5_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC5_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC6_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC6_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC7_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC7_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC8_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC8_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)

        reference_C1_BC1_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC1_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC2_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC2_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC3_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC3_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC4_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC4_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC5_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC5_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC6_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC6_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC7_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC7_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC8_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC8_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)

        # Set index to barcode column and remove the barcode column.
        C1_BC1_barcoding_fitnesses.index = C1_BC1_barcoding_fitnesses[0]
        C1_BC2_barcoding_fitnesses.index = C1_BC2_barcoding_fitnesses[0]
        C1_BC3_barcoding_fitnesses.index = C1_BC3_barcoding_fitnesses[0]
        C1_BC4_barcoding_fitnesses.index = C1_BC4_barcoding_fitnesses[0]
        C1_BC5_barcoding_fitnesses.index = C1_BC5_barcoding_fitnesses[0]
        C1_BC6_barcoding_fitnesses.index = C1_BC6_barcoding_fitnesses[0]
        C1_BC7_barcoding_fitnesses.index = C1_BC7_barcoding_fitnesses[0]
        C1_BC8_barcoding_fitnesses.index = C1_BC8_barcoding_fitnesses[0]

        reference_C1_BC1_barcoding_fitnesses.index = reference_C1_BC1_barcoding_fitnesses[0]
        reference_C1_BC2_barcoding_fitnesses.index = reference_C1_BC2_barcoding_fitnesses[0]
        reference_C1_BC3_barcoding_fitnesses.index = reference_C1_BC3_barcoding_fitnesses[0]
        reference_C1_BC4_barcoding_fitnesses.index = reference_C1_BC4_barcoding_fitnesses[0]
        reference_C1_BC5_barcoding_fitnesses.index = reference_C1_BC5_barcoding_fitnesses[0]
        reference_C1_BC6_barcoding_fitnesses.index = reference_C1_BC6_barcoding_fitnesses[0]
        reference_C1_BC7_barcoding_fitnesses.index = reference_C1_BC7_barcoding_fitnesses[0]
        reference_C1_BC8_barcoding_fitnesses.index = reference_C1_BC8_barcoding_fitnesses[0]

        del C1_BC1_barcoding_fitnesses[0]
        del C1_BC2_barcoding_fitnesses[0]
        del C1_BC3_barcoding_fitnesses[0]
        del C1_BC4_barcoding_fitnesses[0]
        del C1_BC5_barcoding_fitnesses[0]
        del C1_BC6_barcoding_fitnesses[0]
        del C1_BC7_barcoding_fitnesses[0]
        del C1_BC8_barcoding_fitnesses[0]

        del reference_C1_BC1_barcoding_fitnesses[0]
        del reference_C1_BC2_barcoding_fitnesses[0]
        del reference_C1_BC3_barcoding_fitnesses[0]
        del reference_C1_BC4_barcoding_fitnesses[0]
        del reference_C1_BC5_barcoding_fitnesses[0]
        del reference_C1_BC6_barcoding_fitnesses[0]
        del reference_C1_BC7_barcoding_fitnesses[0]
        del reference_C1_BC8_barcoding_fitnesses[0]

        # Sort according to index.
        C1_BC1_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC2_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC3_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC4_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC5_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC6_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC7_barcoding_fitnesses.sort_index(inplace=True)
        C1_BC8_barcoding_fitnesses.sort_index(inplace=True)

        reference_C1_BC1_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC2_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC3_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC4_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC5_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC6_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC7_barcoding_fitnesses.sort_index(inplace=True)
        reference_C1_BC8_barcoding_fitnesses.sort_index(inplace=True)

        self.assertAlmostEqual(numpy.linalg.norm(C1_BC1_barcoding_fitnesses.to_numpy() - reference_C1_BC1_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC2_barcoding_fitnesses.to_numpy() - reference_C1_BC2_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC3_barcoding_fitnesses.to_numpy() - reference_C1_BC3_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC4_barcoding_fitnesses.to_numpy() - reference_C1_BC4_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC5_barcoding_fitnesses.to_numpy() - reference_C1_BC5_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC6_barcoding_fitnesses.to_numpy() - reference_C1_BC6_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC7_barcoding_fitnesses.to_numpy() - reference_C1_BC7_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC8_barcoding_fitnesses.to_numpy() - reference_C1_BC8_barcoding_fitnesses.to_numpy()), 0.0)

        # C1 evolution fitness
        C1_BC1_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC1_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC2_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC2_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC3_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC3_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC4_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC4_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC5_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC5_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC6_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC6_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC7_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC7_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        C1_BC8_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'C1-BC8_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)

        reference_C1_BC1_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC1_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC2_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC2_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC3_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC3_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC4_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC4_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC5_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC5_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC6_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC6_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC7_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC7_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_C1_BC8_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'C1-BC8_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)

        # Set index to barcode column and remove the barcode column.
        C1_BC1_evolution_fitnesses.index = C1_BC1_evolution_fitnesses[0]
        C1_BC2_evolution_fitnesses.index = C1_BC2_evolution_fitnesses[0]
        C1_BC3_evolution_fitnesses.index = C1_BC3_evolution_fitnesses[0]
        C1_BC4_evolution_fitnesses.index = C1_BC4_evolution_fitnesses[0]
        C1_BC5_evolution_fitnesses.index = C1_BC5_evolution_fitnesses[0]
        C1_BC6_evolution_fitnesses.index = C1_BC6_evolution_fitnesses[0]
        C1_BC7_evolution_fitnesses.index = C1_BC7_evolution_fitnesses[0]
        C1_BC8_evolution_fitnesses.index = C1_BC8_evolution_fitnesses[0]

        reference_C1_BC1_evolution_fitnesses.index = reference_C1_BC1_evolution_fitnesses[0]
        reference_C1_BC2_evolution_fitnesses.index = reference_C1_BC2_evolution_fitnesses[0]
        reference_C1_BC3_evolution_fitnesses.index = reference_C1_BC3_evolution_fitnesses[0]
        reference_C1_BC4_evolution_fitnesses.index = reference_C1_BC4_evolution_fitnesses[0]
        reference_C1_BC5_evolution_fitnesses.index = reference_C1_BC5_evolution_fitnesses[0]
        reference_C1_BC6_evolution_fitnesses.index = reference_C1_BC6_evolution_fitnesses[0]
        reference_C1_BC7_evolution_fitnesses.index = reference_C1_BC7_evolution_fitnesses[0]
        reference_C1_BC8_evolution_fitnesses.index = reference_C1_BC8_evolution_fitnesses[0]

        del C1_BC1_evolution_fitnesses[0]
        del C1_BC2_evolution_fitnesses[0]
        del C1_BC3_evolution_fitnesses[0]
        del C1_BC4_evolution_fitnesses[0]
        del C1_BC5_evolution_fitnesses[0]
        del C1_BC6_evolution_fitnesses[0]
        del C1_BC7_evolution_fitnesses[0]
        del C1_BC8_evolution_fitnesses[0]

        del reference_C1_BC1_evolution_fitnesses[0]
        del reference_C1_BC2_evolution_fitnesses[0]
        del reference_C1_BC3_evolution_fitnesses[0]
        del reference_C1_BC4_evolution_fitnesses[0]
        del reference_C1_BC5_evolution_fitnesses[0]
        del reference_C1_BC6_evolution_fitnesses[0]
        del reference_C1_BC7_evolution_fitnesses[0]
        del reference_C1_BC8_evolution_fitnesses[0]

        # Sort according to index.
        C1_BC1_evolution_fitnesses.sort_index(inplace=True)
        C1_BC2_evolution_fitnesses.sort_index(inplace=True)
        C1_BC3_evolution_fitnesses.sort_index(inplace=True)
        C1_BC4_evolution_fitnesses.sort_index(inplace=True)
        C1_BC5_evolution_fitnesses.sort_index(inplace=True)
        C1_BC6_evolution_fitnesses.sort_index(inplace=True)
        C1_BC7_evolution_fitnesses.sort_index(inplace=True)
        C1_BC8_evolution_fitnesses.sort_index(inplace=True)

        reference_C1_BC1_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC2_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC3_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC4_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC5_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC6_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC7_evolution_fitnesses.sort_index(inplace=True)
        reference_C1_BC8_evolution_fitnesses.sort_index(inplace=True)

        self.assertAlmostEqual(numpy.linalg.norm(C1_BC1_evolution_fitnesses.to_numpy() - reference_C1_BC1_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC2_evolution_fitnesses.to_numpy() - reference_C1_BC2_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC3_evolution_fitnesses.to_numpy() - reference_C1_BC3_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC4_evolution_fitnesses.to_numpy() - reference_C1_BC4_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC5_evolution_fitnesses.to_numpy() - reference_C1_BC5_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC6_evolution_fitnesses.to_numpy() - reference_C1_BC6_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC7_evolution_fitnesses.to_numpy() - reference_C1_BC7_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(C1_BC8_evolution_fitnesses.to_numpy() - reference_C1_BC8_evolution_fitnesses.to_numpy()), 0.0)

        # D1 barcode fitness
        D1_BC1_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC1_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC2_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC2_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC3_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC3_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC4_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC4_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC5_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC5_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC6_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC6_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC7_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC7_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC8_barcoding_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC8_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None)

        reference_D1_BC1_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC1_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC2_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC2_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC3_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC3_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC4_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC4_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC5_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC5_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC6_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC6_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC7_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC7_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC8_barcoding_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC8_barcoding_fitnesses.csv'), skiprows=1, sep='\t', header=None,)

        # Set index to barcode column and remove the barcode column.
        D1_BC1_barcoding_fitnesses.index = D1_BC1_barcoding_fitnesses[0]
        D1_BC2_barcoding_fitnesses.index = D1_BC2_barcoding_fitnesses[0]
        D1_BC3_barcoding_fitnesses.index = D1_BC3_barcoding_fitnesses[0]
        D1_BC4_barcoding_fitnesses.index = D1_BC4_barcoding_fitnesses[0]
        D1_BC5_barcoding_fitnesses.index = D1_BC5_barcoding_fitnesses[0]
        D1_BC6_barcoding_fitnesses.index = D1_BC6_barcoding_fitnesses[0]
        D1_BC7_barcoding_fitnesses.index = D1_BC7_barcoding_fitnesses[0]
        D1_BC8_barcoding_fitnesses.index = D1_BC8_barcoding_fitnesses[0]

        reference_D1_BC1_barcoding_fitnesses.index = reference_D1_BC1_barcoding_fitnesses[0]
        reference_D1_BC2_barcoding_fitnesses.index = reference_D1_BC2_barcoding_fitnesses[0]
        reference_D1_BC3_barcoding_fitnesses.index = reference_D1_BC3_barcoding_fitnesses[0]
        reference_D1_BC4_barcoding_fitnesses.index = reference_D1_BC4_barcoding_fitnesses[0]
        reference_D1_BC5_barcoding_fitnesses.index = reference_D1_BC5_barcoding_fitnesses[0]
        reference_D1_BC6_barcoding_fitnesses.index = reference_D1_BC6_barcoding_fitnesses[0]
        reference_D1_BC7_barcoding_fitnesses.index = reference_D1_BC7_barcoding_fitnesses[0]
        reference_D1_BC8_barcoding_fitnesses.index = reference_D1_BC8_barcoding_fitnesses[0]

        del D1_BC1_barcoding_fitnesses[0]
        del D1_BC2_barcoding_fitnesses[0]
        del D1_BC3_barcoding_fitnesses[0]
        del D1_BC4_barcoding_fitnesses[0]
        del D1_BC5_barcoding_fitnesses[0]
        del D1_BC6_barcoding_fitnesses[0]
        del D1_BC7_barcoding_fitnesses[0]
        del D1_BC8_barcoding_fitnesses[0]

        del reference_D1_BC1_barcoding_fitnesses[0]
        del reference_D1_BC2_barcoding_fitnesses[0]
        del reference_D1_BC3_barcoding_fitnesses[0]
        del reference_D1_BC4_barcoding_fitnesses[0]
        del reference_D1_BC5_barcoding_fitnesses[0]
        del reference_D1_BC6_barcoding_fitnesses[0]
        del reference_D1_BC7_barcoding_fitnesses[0]
        del reference_D1_BC8_barcoding_fitnesses[0]

        # Sort according to index.
        D1_BC1_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC2_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC3_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC4_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC5_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC6_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC7_barcoding_fitnesses.sort_index(inplace=True)
        D1_BC8_barcoding_fitnesses.sort_index(inplace=True)

        reference_D1_BC1_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC2_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC3_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC4_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC5_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC6_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC7_barcoding_fitnesses.sort_index(inplace=True)
        reference_D1_BC8_barcoding_fitnesses.sort_index(inplace=True)

        self.assertAlmostEqual(numpy.linalg.norm(D1_BC1_barcoding_fitnesses.to_numpy() - reference_D1_BC1_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC2_barcoding_fitnesses.to_numpy() - reference_D1_BC2_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC3_barcoding_fitnesses.to_numpy() - reference_D1_BC3_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC4_barcoding_fitnesses.to_numpy() - reference_D1_BC4_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC5_barcoding_fitnesses.to_numpy() - reference_D1_BC5_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC6_barcoding_fitnesses.to_numpy() - reference_D1_BC6_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC7_barcoding_fitnesses.to_numpy() - reference_D1_BC7_barcoding_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC8_barcoding_fitnesses.to_numpy() - reference_D1_BC8_barcoding_fitnesses.to_numpy()), 0.0)

        # D1 evolution fitness
        D1_BC1_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC1_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC2_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC2_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC3_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC3_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC4_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC4_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC5_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC5_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC6_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC6_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC7_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC7_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)
        D1_BC8_evolution_fitnesses = pandas.read_csv(os.path.join(lineage_fitness_path, 'D1-BC8_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None)

        reference_D1_BC1_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC1_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC2_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC2_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC3_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC3_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC4_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC4_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC5_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC5_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC6_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC6_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC7_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC7_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)
        reference_D1_BC8_evolution_fitnesses = pandas.read_csv(os.path.join(reference_lineage_fitness_path, 'D1-BC8_evolution_fitnesses.csv'), skiprows=1, sep='\t', header=None,)

        # Set index to barcode column and remove the barcode column.
        D1_BC1_evolution_fitnesses.index = D1_BC1_evolution_fitnesses[0]
        D1_BC2_evolution_fitnesses.index = D1_BC2_evolution_fitnesses[0]
        D1_BC3_evolution_fitnesses.index = D1_BC3_evolution_fitnesses[0]
        D1_BC4_evolution_fitnesses.index = D1_BC4_evolution_fitnesses[0]
        D1_BC5_evolution_fitnesses.index = D1_BC5_evolution_fitnesses[0]
        D1_BC6_evolution_fitnesses.index = D1_BC6_evolution_fitnesses[0]
        D1_BC7_evolution_fitnesses.index = D1_BC7_evolution_fitnesses[0]
        D1_BC8_evolution_fitnesses.index = D1_BC8_evolution_fitnesses[0]

        reference_D1_BC1_evolution_fitnesses.index = reference_D1_BC1_evolution_fitnesses[0]
        reference_D1_BC2_evolution_fitnesses.index = reference_D1_BC2_evolution_fitnesses[0]
        reference_D1_BC3_evolution_fitnesses.index = reference_D1_BC3_evolution_fitnesses[0]
        reference_D1_BC4_evolution_fitnesses.index = reference_D1_BC4_evolution_fitnesses[0]
        reference_D1_BC5_evolution_fitnesses.index = reference_D1_BC5_evolution_fitnesses[0]
        reference_D1_BC6_evolution_fitnesses.index = reference_D1_BC6_evolution_fitnesses[0]
        reference_D1_BC7_evolution_fitnesses.index = reference_D1_BC7_evolution_fitnesses[0]
        reference_D1_BC8_evolution_fitnesses.index = reference_D1_BC8_evolution_fitnesses[0]

        del D1_BC1_evolution_fitnesses[0]
        del D1_BC2_evolution_fitnesses[0]
        del D1_BC3_evolution_fitnesses[0]
        del D1_BC4_evolution_fitnesses[0]
        del D1_BC5_evolution_fitnesses[0]
        del D1_BC6_evolution_fitnesses[0]
        del D1_BC7_evolution_fitnesses[0]
        del D1_BC8_evolution_fitnesses[0]

        del reference_D1_BC1_evolution_fitnesses[0]
        del reference_D1_BC2_evolution_fitnesses[0]
        del reference_D1_BC3_evolution_fitnesses[0]
        del reference_D1_BC4_evolution_fitnesses[0]
        del reference_D1_BC5_evolution_fitnesses[0]
        del reference_D1_BC6_evolution_fitnesses[0]
        del reference_D1_BC7_evolution_fitnesses[0]
        del reference_D1_BC8_evolution_fitnesses[0]

        # Sort according to index.
        D1_BC1_evolution_fitnesses.sort_index(inplace=True)
        D1_BC2_evolution_fitnesses.sort_index(inplace=True)
        D1_BC3_evolution_fitnesses.sort_index(inplace=True)
        D1_BC4_evolution_fitnesses.sort_index(inplace=True)
        D1_BC5_evolution_fitnesses.sort_index(inplace=True)
        D1_BC6_evolution_fitnesses.sort_index(inplace=True)
        D1_BC7_evolution_fitnesses.sort_index(inplace=True)
        D1_BC8_evolution_fitnesses.sort_index(inplace=True)

        reference_D1_BC1_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC2_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC3_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC4_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC5_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC6_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC7_evolution_fitnesses.sort_index(inplace=True)
        reference_D1_BC8_evolution_fitnesses.sort_index(inplace=True)

        self.assertAlmostEqual(numpy.linalg.norm(D1_BC1_evolution_fitnesses.to_numpy() - reference_D1_BC1_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC2_evolution_fitnesses.to_numpy() - reference_D1_BC2_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC3_evolution_fitnesses.to_numpy() - reference_D1_BC3_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC4_evolution_fitnesses.to_numpy() - reference_D1_BC4_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC5_evolution_fitnesses.to_numpy() - reference_D1_BC5_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC6_evolution_fitnesses.to_numpy() - reference_D1_BC6_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC7_evolution_fitnesses.to_numpy() - reference_D1_BC7_evolution_fitnesses.to_numpy()), 0.0)
        self.assertAlmostEqual(numpy.linalg.norm(D1_BC8_evolution_fitnesses.to_numpy() - reference_D1_BC8_evolution_fitnesses.to_numpy()), 0.0)

    def test_compile_clone_list(self):
        """ Run the 'compile_clone_list.sh' script on test data and compare to reference results."""
        command = " ".join([os.path.join(test_path, 'compile_clone_list.sh'),'C1'])
        command_tokens = shlex.split(command)
        proc = Popen(command_tokens, shell=False)
        proc.wait()

        for i in range(1,10):
            data_fname =  os.path.join(data_path, 'flags', 'C1-BC{0:d}_flags.tsv'.format(i))
            data = pandas.read_csv(data_fname, sep='\t', header=0, index_col=0)
            reference_data_fname =  os.path.join(reference_data_path, 'flags', 'C1-BC{0:d}_flags.tsv'.format(i))
            reference_data = pandas.read_csv(reference_data_fname, sep='\t', header=0, index_col=0)

            data.sort_index(inplace=True)
            reference_data.sort_index(inplace=True)

            self.assertDictEqual(data.to_dict(), reference_data.to_dict())

    def test_calculate_clone_frequencies(self):
        """Run 'calculate_clone_frequencies.py' on test data and compare to reference results."""
        # This runs the script.
        import calculate_clone_frequencies

        # Set paths
        data_path = config.clone_data_directory
        reference_data_path = data_path.replace("data", "reference_data")

        fnames = [
            'C1-clone_frequencies.tsv',
            'D1-clone_frequencies.tsv',
        ]

        for fname in fnames:
            data_fpath = os.path.join(data_path, fname)
            ref_data_fpath = os.path.join(reference_data_path, fname)
            if not os.path.exists(data_fpath):
                raise IOError("Expected file does not exist: {}".format(data_fpath))

            data = pandas.read_csv(data_fpath, sep="\t", index_col=[0,1], header=0,)
            ref_data = pandas.read_csv(ref_data_fpath, sep="\t", index_col=[0,1], header=0,)

            # Sort
            data.sort_index(level=1, axis=0, inplace=True)
            ref_data.sort_index(level=1, axis=0, inplace=True)

            # Check.
            self.assertDictEqual(data.to_dict(),
                                 ref_data.to_dict(),
                                 )

        # Check clone list.
        clone_lists = [
            'C1_clone_list.tsv',
            'D1_clone_list.tsv'
        ]

        for clone_list in clone_lists:
            data_fpath = os.path.join(data_path, clone_list)
            ref_data_fpath = os.path.join(reference_data_path, clone_list)
            if not os.path.exists(data_fpath):
                raise IOError("Expected file does not exist: {}".format(data_fpath))

            data = pandas.read_csv(data_fpath, sep="\t", index_col=0, header=None, )
            ref_data = pandas.read_csv(ref_data_fpath, sep="\t", index_col=0, header=None, )

            # Sort
            data.sort_index(level=1, axis=0, inplace=True)
            ref_data.sort_index(level=1, axis=0, inplace=True)

            # Check.
            self.assertDictEqual(data.to_dict(),
                                 ref_data.to_dict(),
                                 )

    def test_measure_evolution_fitness(self):
        """ Run 'measure_evolution_fitness.py' on test data and compare to reference data."""

        import measure_evolution_fitness

        # Construct mock input arguments.
        Args = namedtuple('args', ['pop_name',])
        args = Args('C1')
        measure_evolution_fitness.main(args)

        args = Args('D1')
        measure_evolution_fitness.main(args)

        # Set paths
        data_path = config.clone_data_directory
        reference_data_path = data_path.replace("data", "reference_data")

        fnames = [
            'C1-evolution_fitnesses.tsv',
            'D1-evolution_fitnesses.tsv',
        ]

        for fname in fnames:
            data_fpath = os.path.join(data_path, fname)
            ref_data_fpath = os.path.join(reference_data_path, fname)
            if not os.path.exists(data_fpath):
                raise IOError("Expected file does not exist: {}".format(data_fpath))

            data = pandas.read_csv(data_fpath, sep="\t", index_col=0, header=0, )
            ref_data = pandas.read_csv(ref_data_fpath, sep="\t", index_col=0, header=0, )

            # Sort
            data.sort_index(inplace=True)
            ref_data.sort_index(inplace=True)

            # Check. Round-off errors make this imprecise.
            self.assertAlmostEqual(numpy.linalg.norm(data.to_numpy() - ref_data.to_numpy()), 0.0, delta=0.1)


    def test_measure_barcoding_fitness(self):
        """ Run 'measure_barcodingevolution_fitness.py' on test data and compare to reference data."""

        import measure_barcoding_fitness

        # Construct mock input arguments.
        Args = namedtuple('args', ['pop_name',])

        args = Args('C1')
        measure_barcoding_fitness.main(args)

        args = Args('D1')
        measure_barcoding_fitness.main(args)

        # Set paths
        data_path = config.clone_data_directory
        reference_data_path = data_path.replace("data", "reference_data")

        fnames = [
            'C1-barcoding_fitnesses.tsv',
            'D1-barcoding_fitnesses.tsv',
        ]

        for fname in fnames:
            data_fpath = os.path.join(data_path, fname)
            ref_data_fpath = os.path.join(reference_data_path, fname)
            if not os.path.exists(data_fpath):
                raise IOError("Expected file does not exist: {}".format(data_fpath))

            data = pandas.read_csv(data_fpath, sep="\t", index_col=0, header=0, )
            ref_data = pandas.read_csv(ref_data_fpath, sep="\t", index_col=0, header=0, )

            # Sort
            data.sort_index(inplace=True)
            ref_data.sort_index(inplace=True)

            # Check. Round-off errors make this imprecise.
            self.assertAlmostEqual(numpy.linalg.norm(data.to_numpy() - ref_data.to_numpy()), 0.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
