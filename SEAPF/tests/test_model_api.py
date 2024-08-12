# run with: python -m unittest SEAPF/tests/test_model_api.py

import unittest
from sklearn.utils.estimator_checks import check_estimator
from ..Model import Model


class ValidateSkLearnEstimator(unittest.TestCase):
    def test_sklearn_estimator(self):
        estimator = check_estimator(Model())
        print(estimator)

if __name__ == '__main__':
    unittest.main()