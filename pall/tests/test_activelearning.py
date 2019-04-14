import unittest
import warnings
import numpy as np
from sklearn.svm import SVC

import pall.querystrategies as QS
from pall.activelearning import ActiveLearningModel


def disable_DeprecationWarning(fn):
    def _wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return fn(*args, **kwargs)
    return _wrapped


class ActiveLearningModelTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(311)
        self.clf = SVC(gamma="auto", probability=True)
        self.kwargs = {'U_proportion': 0.4}
        self.train_X = np.array([[0.5, 1.5], [0.5, 3], [2, 3],
                                 [2, 1.5], [3, 1], [3, 3]])
        self.train_y = np.array([0, 0, 0, 1, 1, 1])
        self.test_X = np.array([[1, 2], [2, 1]])
        self.test_y = np.array([0, 1])
        self.ndraws = int(np.ceil(0.4 * self.train_X.shape[0]))

    def tearDown(self):
        del self.clf

    def test_random(self):
        qs = QS.Random()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    @disable_DeprecationWarning
    def test_minmax(self):
        qs_kwargs = {'metric': 'mahalanobis'}
        qs = QS.MinMax(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(al.query_strategy.distance_metric, 'mahalanobis')

    def test_simple_margin(self):
        qs = QS.SimpleMargin()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_margin(self):
        qs = QS.Margin()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_entropy(self):
        qs_kwargs = {'model_change': False}
        qs = QS.Entropy(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lc(self):
        qs_kwargs = {'model_change': False}
        qs = QS.LeastConfidence(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lcb(self):
        qs_kwargs = {'model_change': False}
        qs = QS.LeastConfidenceBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_lcb2(self):
        qs_kwargs = {'model_change': False}
        qs = QS.LeastConfidenceDynamicBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_entropy_mc(self):
        qs_kwargs = {'model_change': True}
        qs = QS.Entropy(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lc_mc(self):
        qs_kwargs = {'model_change': True}
        qs = QS.LeastConfidence(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lcb_mc(self):
        qs_kwargs = {'model_change': True}
        qs = QS.LeastConfidenceBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    def test_lcb2_mc(self):
        qs_kwargs = {'model_change': True}
        qs = QS.LeastConfidenceDynamicBias(**qs_kwargs)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.model_change, True)

    @disable_DeprecationWarning
    def test_id(self):
        qs1 = QS.LeastConfidence(model_change=False)
        qs2 = QS.Density()
        qs = QS.CombinedSampler(qs1, qs2, beta=3)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))
        self.assertEqual(qs.beta, 3)

    @disable_DeprecationWarning
    def test_distdiv(self):
        qs1 = QS.LeastConfidence(model_change=False)
        qs2 = QS.MinMax(metric="euclidean")
        qs = QS.DistDivSampler(qs1, qs2, lam=1)
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y)
        self.assertEqual(scores.shape, (self.ndraws,))

    def test_limit_draws(self):
        qs = QS.Random()
        al = ActiveLearningModel(self.clf, qs, **self.kwargs)
        scores, choices = al.run(self.train_X, self.test_X,
                                 self.train_y, self.test_y,
                                 ndraws=self.ndraws-1)
        self.assertEqual(scores.shape, (self.ndraws-1,))


if __name__ == '__main__':
    unittest.main()
