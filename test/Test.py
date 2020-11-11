import unittest
import sys

from barcode_scripts_test import BarcodeScriptsTest

def suite():
    suites = [
               unittest.makeSuite(BarcodeScriptsTest),
             ]

    return unittest.TestSuite(suites)

# Run the top level suite and return a success status code. This enables running an automated git-bisect.
if __name__=="__main__":

    result = unittest.TextTestRunner(verbosity=2).run(suite())

    if result.wasSuccessful():
        print('---> OK <---')
        sys.exit(0)

    sys.exit(1)

