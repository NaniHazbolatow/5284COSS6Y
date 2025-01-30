import unittest
from unittest.mock import patch

import numpy as np

from CAStockModel.model.analysis.signal_function import _calc_signal_function_value, det_phase



class TestSignalFunction(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([[0, 1, -1, 0, 1],
                              [1, 0, -1, 0, 0],
                              [1, 0, 1, 0, 1],
                              [0, 0, 1, 0, 1],
                              [1, 0, -1, 0, 1]])
        self.clusters = np.array([[[False, True, True, False, False],
                                   [False, False, True, False, False],
                                   [False, False, True, False, False],
                                   [False, False, True, False, False],
                                   [False, False, True, False, False]],
                                  [[False, False, False, False, True],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False]],
                                  [[False, False, False, False, False],
                                   [True, False, False, False, False],
                                   [True, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False]],
                                  [[False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, True],
                                   [False, False, False, False, True],
                                   [False, False, False, False, True]],
                                  [[False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [False, False, False, False, False],
                                   [True, False, False, False, False]]])

    def test_calc_signal_function_value(self):
        filtered_grid = self.grid[np.any(self.clusters[:-1], axis=0)]
        test_solution = _calc_signal_function_value(filtered_grid)
        self.assertEqual(test_solution, 0.5, msg="Signal calculation function does not work properly")

    def test_det_phase(self):
        self.assertEqual(det_phase(0.901), 1)
        self.assertEqual(det_phase(-0.901), -1)
        self.assertEqual(det_phase(0.899), 0)
        self.assertEqual(det_phase(-0.899), 0)