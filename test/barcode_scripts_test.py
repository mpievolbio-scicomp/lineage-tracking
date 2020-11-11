import unittest
import sys
import os
import pandas
import numpy

# add custom modules to path
modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, modules_path)

test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'barcode_scripts'))
sys.path.insert(0, test_path)

import config

class BarcodeScriptsTest(unittest.TestCase):
    def setUp(self):
        pass

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

        # Read error model data.
if __name__ == '__main__':
    unittest.main()
