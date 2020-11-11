import unittest
import sys

sys.path.insert(0,'../..')
sys.path.insert(0,'../..')


class ErrorModelTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_reference(self):
        """ Run 'fit_error_model.py' on reference data and compare to reference results."""
        from barcode_scripts import fit_error_model


if __name__ == '__main__':
    unittest.main()
