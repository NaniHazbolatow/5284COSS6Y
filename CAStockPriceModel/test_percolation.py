import unittest
import numpy as np
from utils.utility_elements import (
    initialise_market_grid,
    find_clusters,
    calculate_log_return,
    convert_time
)
from model.percolation_dyn import (
    apply_percolation_dyn,
    _create_neighbours,
    _create_masks,
    _count_neighbours,
    _calculate_probabilities,
    _create_effective_filters,
    _create_new_grid
)
from model.stochastic_dyn import (
    apply_stochastic_dynamics,
    _simulate_connections,
    _calculate_buying_probs,
    _cluster_value_upgrade
)


class TestUtilityElements(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)  # Ensure reproducibility
        self.test_shape = (4, 4)
        self.test_ratio = 0.5
        
    def test_initialise_market_grid(self):
        grid = initialise_market_grid(self.test_shape, self.test_ratio)
        
        # Test shape
        self.assertEqual(grid.shape, self.test_shape)
        
        # Test ratio of active cells
        active_cells = np.count_nonzero(grid)
        expected_active_cells = int(self.test_ratio * self.test_shape[0] * self.test_shape[1])
        self.assertEqual(active_cells, expected_active_cells)
        
        # Test values are only -1, 0, or 1
        unique_values = np.unique(grid)
        self.assertTrue(all(val in [-1, 0, 1] for val in unique_values))

    def test_find_clusters(self):
        test_grid = np.array([
            [1, 1, 0, -1],
            [1, 0, 0, -1],
            [0, 0, 1, -1],
            [0, 1, 1, 0]
        ])
        
        clusters = find_clusters(test_grid)
        
        # Test we get the right number of clusters
        self.assertEqual(len(clusters), 3)  # Should find 3 distinct clusters
        
        # Test each cluster is a boolean array
        for cluster in clusters:
            self.assertEqual(cluster.dtype, np.bool_)
            self.assertEqual(cluster.shape, test_grid.shape)

    def test_calculate_log_return(self):
        test_grid = np.array([
            [1, 1, 0],
            [0, 0, -1],
            [1, 0, -1]
        ])
        test_clusters = np.array([
            [[True, True, False],
             [False, False, False],
             [False, False, False]],
            [[False, False, False],
             [False, False, True],
             [False, False, True]]
        ])
        beta = 9.0  # 3^2 * 3^2
        
        log_return = calculate_log_return(test_grid, test_clusters, beta)
        
        # Manual calculation: (2*2 + 2*(-2)) / 9 = 0
        self.assertEqual(log_return, 0.0)

    def test_convert_time(self):
        test_cases = [
            (3661, "1 h 1 m 1"),      # 1 hour, 1 minute, 1 second
            (7200, "2 h 0 m 0"),      # 2 hours exactly
            (59, "0 h 0 m 59"),       # Less than 1 minute
            (3600, "1 h 0 m 0")       # 1 hour exactly
        ]
        
        for input_time, expected in test_cases:
            result = convert_time(input_time)
            self.assertEqual(result, expected)


class TestPercolationDynamics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.test_grid = np.array([
            [0, 1, 0, -1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [-1, 0, 0, 1]
        ])
        
    def test_create_neighbours(self):
        neighbours = _create_neighbours(self.test_grid)
        
        # Test shape (4 directions x grid_height x grid_width)
        self.assertEqual(neighbours.shape, (4, 4, 4))
        
        # Test specific neighbor values
        # Testing top neighbor of position (1,1)
        self.assertEqual(neighbours[0, 1, 1], self.test_grid[0, 1])
        # Testing bottom neighbor of position (1,1)
        self.assertEqual(neighbours[1, 1, 1], self.test_grid[2, 1])

    def test_create_masks(self):
        neighbours = _create_neighbours(self.test_grid)
        mask_1, mask_3, mask_4 = _create_masks(self.test_grid, neighbours)
        
        # Test shapes
        self.assertEqual(mask_1.shape, self.test_grid.shape)
        self.assertEqual(mask_3.shape, self.test_grid.shape)
        self.assertEqual(mask_4.shape, self.test_grid.shape)
        
        # Test types
        self.assertTrue(np.issubdtype(mask_1.dtype, np.bool_))
        self.assertTrue(np.issubdtype(mask_3.dtype, np.bool_))
        self.assertTrue(np.issubdtype(mask_4.dtype, np.bool_))
        
        # Test mutual exclusivity
        overlap = mask_1 & mask_3 & mask_4
        self.assertFalse(np.any(overlap))

    def test_count_neighbours(self):
        neighbours = _create_neighbours(self.test_grid)
        n_inactive, n_active = _count_neighbours(neighbours)
        
        # Test shapes
        self.assertEqual(n_inactive.shape, self.test_grid.shape)
        self.assertEqual(n_active.shape, self.test_grid.shape)
        
        # Test sum of inactive and active equals total neighbors
        total_neighbors = n_inactive + n_active
        self.assertTrue(np.all(total_neighbors <= 4))  # Max 4 neighbors per cell

    def test_calculate_probabilities(self):
        n_inactive = np.array([[2, 1], [3, 0]])
        n_active = np.array([[2, 3], [1, 4]])
        p_e, p_d, p_h = 0.1, 0.2, 0.3
        
        prob_1, prob_3, prob_4 = _calculate_probabilities(n_inactive, n_active, p_e, p_d, p_h)
        
        # Test probability bounds
        self.assertTrue(0 <= prob_1 <= 1)
        self.assertTrue(np.all((0 <= prob_3) & (prob_3 <= 1)))
        self.assertTrue(np.all((0 <= prob_4) & (prob_4 <= 1)))


class TestStochasticDynamics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.test_grid = np.array([
            [1, 1, -1],
            [-1, 1, 0],
            [0, -1, 1]
        ])
        self.test_cluster = np.array([
            [True, True, True],
            [True, True, False],
            [False, True, True]
        ])
        
    def test_simulate_connections(self):
        n = 5
        A = 1.6
        a = 2 * A
        h = 0.5
        
        A_ij, h_i = _simulate_connections(n, A, a, h)
        
        # Test shapes
        self.assertEqual(A_ij.shape, (n, n))
        self.assertEqual(h_i.shape, (n,))
        
        # Test bounds
        self.assertTrue(np.all(A_ij >= -3.2) and np.all(A_ij <= 3.2))  # A + a bounds
        self.assertTrue(np.all(h_i >= -0.5) and np.all(h_i <= 0.5))    # h bounds

    def test_calculate_buying_probs(self):
        n = 3
        cluster_elements = np.array([1, -1, 1])
        A_ij = np.ones((n, n))
        h_i = np.zeros(n)
        
        probs = _calculate_buying_probs(n, cluster_elements, A_ij, h_i)
        
        # Test shape
        self.assertEqual(probs.shape, (n,))
        
        # Test probability bounds
        self.assertTrue(np.all((0 <= probs) & (probs <= 1)))

    def test_cluster_value_upgrade(self):
        test_probs = np.array([0.2, 0.8, 0.5])
        result = _cluster_value_upgrade(test_probs)
        
        # Test shape
        self.assertEqual(result.shape, test_probs.shape)
        
        # Test values are only -1 or 1
        self.assertTrue(np.all(np.abs(result) == 1))


if __name__ == '__main__':
    unittest.main()