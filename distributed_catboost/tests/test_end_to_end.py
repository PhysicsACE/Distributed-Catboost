import os
import shutil
import tempfile
import unittest

import numpy as np
import ray
from distributed_catboost.catboost import catboost
from ray.exceptions import RayActorError, RayTaskError
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from scipy.sparse import csr_matrix
from distributed_catboost import RayPool, predict, train, RayShardingMode
from xgboost_ray.callback import DistributedCallback
from xgboost_ray import RayParams, RayDMatrix


class CatboostRayEndToEndTest(unittest.TestCase):

    def setUp(self) -> None:
        repeat = 8
        self.x = np.array(
            [
                [1, 0, 0, 0],  # Feature 0 -> Label 0
                [0, 1, 0, 0],  # Feature 1 -> Label 1
                [0, 0, 1, 1],  # Feature 2+3 -> Label 2
                [0, 0, 1, 0],  # Feature 2+!3 -> Label 3
            ]
            * repeat
        )
        self.y = np.array([0, 1, 2, 3] * repeat)

        self.params = {}

    def tearDown(self) -> None:
        if ray.is_initialized:
            ray.shutdown()

    def testSingleTraining(self):
        """Test that XGBoost learns to predict full matrix"""
        dtrain = RayDMatrix(self.x, self.y)
        # bst = catboost.train(dtrain, self.params, num_boost_round=2)
        dtrain2 = RayDMatrix(self.x, self.y)

        bst2 = train(self.params, dtrain2, num_boost_round=2, ray_params=RayParams(num_actors=2))

        x_mat = RayDMatrix(self.x)
        pred_y = bst2.predict(x_mat)
        self.assertSequenceEqual(list(self.y), list(pred_y))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))