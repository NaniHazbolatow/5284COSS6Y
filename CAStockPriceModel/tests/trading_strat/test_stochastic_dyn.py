import unittest
from unittest.mock import patch

import numpy as np

from CAStockModel.model.trading_strat.stochastic_dyn import (_calculate_price_info, _calculate_price_reference_moments,
                                                             _categorise_traders_by_cluster_size,
                                                             _decide_trader_strategy, _update_traders)


class TestTradingStrategyStocDyn(unittest.TestCase):

    def setUp(self):
        self.grid = np.array([[0, 1, -1,  0, 1],
                              [1, 0, -1,  0, 0],
                              [1, 0,  1,  0, 1],
                              [0, 0,  1,  0, 1],
                              [1, 0, -1,  0, 1]])
        self.trader_decision_sensitivity = np.array([[0, 0, 0, 0, 0],
                                                     [0.5, 0.5, 0.5, 0.5, 0.5],
                                                     [0.75, 0.75, 0.75, 0.75, 0.75],
                                                     [2, 2, 2, 2, 2],
                                                     [5, 5, 5, 5, 5]])
        self.clusters = np.array([[[ False,  True,  True, False, False],
                                    [False, False,  True, False, False],
                                    [False, False,  True, False, False],
                                    [False, False,  True, False, False],
                                    [False, False,  True, False, False]],
                                   [[False, False, False, False,  True],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False]],
                                   [[False, False, False, False, False],
                                    [ True, False, False, False, False],
                                    [ True, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False]],
                                   [[False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False,  True],
                                    [False, False, False, False,  True],
                                    [False, False, False, False,  True]],
                                   [[False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [False, False, False, False, False],
                                    [ True, False, False, False, False]]])
        self.delay_categories = np.array([[0, 0, 0, 0, 3],
                                          [2, 0, 0, 0, 0],
                                          [2, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1],
                                          [3, 0, 0, 0, 1]])
        self.trader_following_coupling_strat = np.array([[False, True, True, False, True],
                                                         [True, False, True, False, False],
                                                         [True, False, True, False, True],
                                                         [False, False, False, False, False],
                                                         [False, False, False, False, False]])
        self.trader_coupling_strat_spin_dir = np.vstack((np.ones((3, 5)), np.ones((2, 5)) * -1))
        self.trader_following_coupling_strat_alt = np.array([[False, True, True, False, True],
                                                             [True, False, True, False, False],
                                                             [True, False, True, False, True],
                                                             [False, False, True, False, True],
                                                             [False, False, True, False, True]])

        self.trader_coupling_strat_spin_dir_alt = np.array([[1., 1., 1., 1., 1.],
                                                            [1., 1., 1., 1., 1.],
                                                            [1., 1., 1., 1., -1.],
                                                            [1., 1., 1., 1., -1.],
                                                            [-1., 1., 1., 1., -1.]])
        self.avg_price = np.array([48, 46, 44, 42])
        self.A = 1.8
        self.a = 2 * 1.8
        self.h = 0
        self.raw_price_index = np.arange(51)
        self.delayed_price_indices = np.array([[46, 47, 48, 49, 50],
                                               [44, 45, 46, 47, 48],
                                               [42, 43, 44, 45, 46],
                                               [40, 41, 42, 43, 44]])
        self.std_price = np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
        self.time_delay = 2
        self.alpha = 5

    def test_calculate_price_info_normal(self):
        test_output = _calculate_price_info(self.raw_price_index, self.time_delay, self.alpha)

        np.testing.assert_array_equal(self.delayed_price_indices, test_output,
                                      err_msg="The Price Info calculation is not correct.")

    def test_calculate_price_info_assert(self):
        with self.assertRaises(ValueError):
            _calculate_price_info(np.arange(10), 7, 3)
            _calculate_price_info(self.raw_price_index, -1, self.time_delay)
            _calculate_price_info(self.raw_price_index, self.time_delay, -5)

    def test_calculate_price_reference_moments_normal(self):
        test_mean, test_std = _calculate_price_reference_moments(self.delayed_price_indices)

        np.testing.assert_array_equal(self.avg_price, test_mean,
                                      err_msg="The calculated average price is not correct.")
        np.testing.assert_array_equal(self.std_price, test_std,
                                      err_msg="The calculated standard deviation is not correct.")

    def test_calculate_price_reference_moments_assert(self):
        input_price_idx = np.array([[46, 47, 48, 49, 50],
                                    [44, 45, 46, 47, 48],
                                    [42, 43, 44, 45, 46],
                                    [40, 41, 42, 43, 44],
                                    [38, 39, 40, 41, 42]])

        with self.assertRaises(ValueError):
            _calculate_price_reference_moments(input_price_idx)

    def test_categorise_traders_by_cluster_size_normal(self):
        test_result = _categorise_traders_by_cluster_size(self.grid, self.clusters)

        np.testing.assert_array_equal(self.delay_categories, test_result,
                                      err_msg="Trader Categorisation in the general case is in not correct.")

    def test_categorise_traders_by_cluster_size_short_input(self):
        clusters = np.array([[[ False, True, True, False, False],
                               [False, False, True, False, False],
                               [False, False, True, False, False],
                               [False, False, True, False, False],
                               [False, False, True, False, False]],
                              [[False, False, False, False, False],
                               [True, False, False, False, False],
                               [True, False, False, False, False],
                               [False, False, False, False, False],
                               [False, False, False, False, False]],
                              [[False, False, False, False, False],
                               [False, False, False, False, False],
                               [False, False, False, False,  True],
                               [False, False, False, False,  True],
                               [False, False, False, False,  True]]])
        grid = np.array([[0, 1, -1, 0, 0],
                         [1, 0, -1, 0, 0],
                         [1, 0,  1, 0, 1],
                         [0, 0,  1, 0, 1],
                         [0, 0, -1, 0, 1]])
        test_result = _categorise_traders_by_cluster_size(grid, clusters)

        np.testing.assert_array_equal(np.zeros_like(grid), test_result,
                                      err_msg="Trader Categorisation in the short case is in not correct.")

    def test_decide_trader_strategy_normal(self):
        strat_decisions, coupling_strat_result = _decide_trader_strategy(self.grid,
                                                                         self.delayed_price_indices,
                                                                         self.delay_categories,
                                                                         self.trader_decision_sensitivity,
                                                                         self.avg_price,
                                                                         self.std_price)

        np.testing.assert_array_equal(strat_decisions, self.trader_following_coupling_strat,
                                      err_msg="The trader decisions on the strategies are not correct.")
        np.testing.assert_array_equal(coupling_strat_result, self.trader_coupling_strat_spin_dir,
                                      err_msg="The traders coupling strategy is not correct.")

    def test_decide_trader_strategy_alt(self):
        strat_decisions, coupling_strat_result = _decide_trader_strategy(self.grid,
                                                                         self.delayed_price_indices,
                                                                         self.delay_categories,
                                                                         self.trader_decision_sensitivity,
                                                                         np.array([30, 70, 44, 42]),
                                                                         self.std_price)

        np.testing.assert_array_equal(strat_decisions, self.trader_following_coupling_strat_alt,
                                      err_msg="The trader decisions on the strategies are not correct.")
        np.testing.assert_array_equal(coupling_strat_result, self.trader_coupling_strat_spin_dir_alt,
                                      err_msg="The traders coupling strategy is not correct.")

    @patch("CAStockModel.model.trading_strat.stochastic_dyn.apply_stochastic_dynamics")
    def test_update_traders_normal(self, mock_apply_stochastic_dynamics):
        mock_apply_stochastic_dynamics.return_value = np.ones_like(self.grid) * 5

        grid_copy = self.grid.copy()

        _update_traders(grid_copy,
                        self.clusters,
                        self.trader_following_coupling_strat,
                        self.trader_coupling_strat_spin_dir,
                        self.A,
                        self.a,
                        self.h)

        np.testing.assert_array_equal(grid_copy, np.array([[5., 1,  1, 5., 1],
                                                           [1, 5., 1, 5., 5.],
                                                           [1, 5., 1, 5., 1],
                                                           [5., 5., 5., 5., 5.],
                                                           [5., 5., 5., 5., 5.]]))

    @patch("CAStockModel.model.trading_strat.stochastic_dyn.apply_stochastic_dynamics")
    def test_update_traders_alt(self, mock_apply_stochastic_dynamics):
        mock_apply_stochastic_dynamics.return_value = np.ones_like(self.grid) * 5

        grid_copy = self.grid.copy()

        _update_traders(grid_copy,
                        self.clusters,
                        self.trader_following_coupling_strat_alt,
                        self.trader_coupling_strat_spin_dir_alt,
                        self.A,
                        self.a,
                        self.h)

        np.testing.assert_array_equal(grid_copy, np.array([[5, 1, 1, 5,  1],
                                                           [1, 5, 1, 5,  5],
                                                           [1, 5, 1, 5, -1],
                                                           [5, 5, 1, 5, -1],
                                                           [5, 5, 1, 5, -1]]))


if __name__ == "__main__":
    unittest.main()
