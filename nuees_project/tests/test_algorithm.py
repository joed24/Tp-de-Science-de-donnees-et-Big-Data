import unittest

import numpy as np

from src.dynamic_clouds import DynamicClouds, make_demo_dataset


class TestDynamicClouds(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_demo_dataset(seed=123, n_per_cluster=30)

    def test_all_representations(self):
        for mode in DynamicClouds.MODES:
            model = DynamicClouds(k=4, mode=mode, q=3, seed=123, max_iter=30)
            model.fit(self.X)
            self.assertEqual(len(model.labels_), len(self.X))
            self.assertEqual(len(model.prototypes), 4)
            self.assertTrue(np.isfinite(model.history_[-1]))
            self.assertTrue(np.isfinite(model.predict(self.X)).all())

    def test_kmeans_case_has_one_point_per_class(self):
        model = DynamicClouds(k=4, mode="point", seed=123)
        model.fit(self.X)
        self.assertTrue(all(hasattr(p, "center") for p in model.prototypes))
        self.assertEqual(len(model.prototypes), 4)


if __name__ == "__main__":
    unittest.main()
