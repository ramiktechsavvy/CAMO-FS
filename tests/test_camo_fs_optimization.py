import numpy as np
import unittest

from camo_fs.optimization.eci import eci_from_rank_vectors
from camo_fs.optimization.indicators import hypervolume_3d
from camo_fs.optimization.nsga2 import (
    crowding_distance,
    dominates,
    mutation_probabilities,
    non_dominated_sort,
    weighted_distance_selection,
)
from camo_fs.optimization.objectives import objective_values, validate_chromosome


class TestCamoFSOptimization(unittest.TestCase):
  def test_chromosome_validity_and_empty_rejection(self):
    chromosome = np.zeros(261, dtype=np.uint8)
    chromosome[12] = 1
    self.assertEqual(validate_chromosome(chromosome).sum(), 1)
    with self.assertRaises(ValueError):
        validate_chromosome(np.zeros(261, dtype=np.uint8))
    with self.assertRaises(ValueError):
        validate_chromosome(np.array([0, 2, 1]))


  def test_shap_guided_mutation_probabilities(self):
    chromosome = np.array([1, 0], dtype=np.uint8)
    priors = np.array([0.8, 0.8])
    np.testing.assert_allclose(mutation_probabilities(chromosome, priors), [0.006, 0.024])


  def test_objective_calculation(self):
    np.testing.assert_allclose(objective_values(0.8, 87, 0.6), [0.2, 87 / 261, 0.4])


  def test_dominance_relation(self):
    assert dominates(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.2, 0.4]))
    assert not dominates(np.array([0.1, 0.3, 0.3]), np.array([0.2, 0.2, 0.4]))
    assert not dominates(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))


  def test_non_dominated_sorting(self):
    values = np.array([[0.1, 0.5, 0.5], [0.2, 0.6, 0.6], [0.5, 0.1, 0.5], [0.4, 0.4, 0.4]])
    fronts = non_dominated_sort(values)
    assert set(fronts[0]) == {0, 2, 3}
    assert fronts[1] == [1]


  def test_crowding_distance_preserves_extremes(self):
    values = np.array([[0, 1], [0.5, 0.5], [1, 0]], dtype=float)
    distances = crowding_distance(values, [0, 1, 2])
    assert np.isinf(distances[0]) and np.isinf(distances[2])
    assert distances[1] > 0


  def test_eci_subset_restriction(self):
    ranks = [np.array([1, 2, 3]), np.array([2, 1, 3]), np.array([1, 3, 2])]
    self.assertAlmostEqual(eci_from_rank_vectors(ranks, [0, 1, 2]), 5 / 9)
    self.assertAlmostEqual(eci_from_rank_vectors(ranks, [0, 1]), 1 / 3)
    self.assertAlmostEqual(eci_from_rank_vectors(ranks, [0, 2]), 1.0)
    self.assertEqual(eci_from_rank_vectors(ranks, [1]), 1.0)


  def test_weighted_final_selection_and_hypervolume(self):
    objectives = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.2, 0.2, 0.2]])
    selected, normalized, distances = weighted_distance_selection(objectives)
    self.assertEqual(selected, 2)
    self.assertEqual(normalized.shape, (3, 3))
    self.assertLess(distances[2], distances[0])
    self.assertAlmostEqual(hypervolume_3d(np.array([[0.5, 0.5, 0.5]])), 0.125)


if __name__ == "__main__":
    unittest.main()

