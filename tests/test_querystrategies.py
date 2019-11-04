import unittest
import warnings
import numpy as np

import yall.querystrategies as QS
from yall.activelearning import Data

from packaging import version
from sklearn import __version__
sklearn_ver = version.parse(__version__)


def disable_DeprecationWarning(fn):
    def _wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return fn(*args, **kwargs)
    return _wrapped


class QueryStrategyTest(unittest.TestCase):
    class DummyClassifier(object):

        def decision_function(self, X):
            return np.array([0.3, -0.3, 0.5, -0.5, 0.5])

        def predict_proba(self, X):
            return np.array([[0.1, 0.9],
                             [0.3, 0.7],
                             [0.5, 0.5],
                             [0.7, 0.3],
                             [0.9, 0.1]])

    def setUp(self):
        np.random.seed(420)
        XU = np.linspace(1, 15, 15).reshape(5, 3)
        yU = np.array([0, 1, 0, 1, 0])
        XL = np.linspace(9, 1, 9).reshape(3, 3)
        yL = np.array([1, 0, 1])
        self.unlabeled = Data(X=XU, y=yU)
        self.labeled = Data(X=XL, y=yL)
        self.clf = self.DummyClassifier()
        self.args = [self.unlabeled, self.labeled, self.clf]

    def tearDown(self):
        del self.unlabeled
        del self.labeled
        del self.clf

    def test_random(self):
        qs = QS.Random()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 1)

    def test_simple_margin(self):
        qs = QS.SimpleMargin()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 0)

    def test_margin(self):
        qs = QS.Margin()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 2)

    def test_entropy(self):
        qs = QS.Entropy()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 2)

    def test_lc(self):
        qs = QS.LeastConfidence()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 2)

    def test_lcb(self):
        qs = QS.LeastConfidenceBias()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 2)

    def test_lcb2(self):
        qs = QS.LeastConfidenceDynamicBias()
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 2)

    @disable_DeprecationWarning
    def test_d2c(self):
        qs_kwargs = {'metric': 'euclidean'}
        qs = QS.DistanceToCenter(**qs_kwargs)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 4)

    @disable_DeprecationWarning
    def test_density(self):
        qs_kwargs = {'metric': 'euclidean'}
        qs = QS.Density(**qs_kwargs)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 0)

    @disable_DeprecationWarning
    def test_density_nan(self):
        XU = np.linspace(1, 15, 15).reshape(5, 3)
        XU[3] = np.array([0., 0., 0.])
        yU = np.array([0, 1, 0, 1, 0])
        XL = np.linspace(9, 1, 9).reshape(3, 3)
        yL = np.array([1, 0, 1])
        unlabeled = Data(X=XU, y=yU)
        labeled = Data(X=XL, y=yL)
        clf = self.DummyClassifier()
        args = [unlabeled, labeled, clf]
        qs_kwargs = {'metric': 'cosine'}
        qs = QS.Density(**qs_kwargs)
        with self.assertRaises(ValueError,
                               msg="Distances contain NaN values. Check that input vectors != 0."):  # noqa
            qs.query(*args)

    @disable_DeprecationWarning
    def test_minmax_metric(self):
        qs = QS.MinMax()
        self.assertEqual(qs.distance_metric, 'euclidean')

    @disable_DeprecationWarning
    def test_minmax_query_euc(self):
        qs_kwargs = {'metric': 'euclidean'}
        qs = QS.MinMax(**qs_kwargs)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 4)

    @disable_DeprecationWarning
    def test_minmax_query_mah(self):
        qs_kwargs = {'metric': 'mahalanobis'}
        qs = QS.MinMax(**qs_kwargs)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        # A difference in sklearn versions cause
        # a different result to be computed.
        if sklearn_ver == version.parse("0.19.0"):
            self.assertEqual(choice, 0)
        else:
            self.assertEqual(choice, 4)

    @disable_DeprecationWarning
    def test_minmax_query_mah_singular(self):
        """Singular matrix"""
        qs_kwargs = {'metric': 'mahalanobis'}
        XU = np.array([[5, 5, 5, 5, 5]] * 5)
        XL = np.array([[3, 3, 3, 3, 3]] * 3)
        yU = np.array([0, 1, 0, 1, 0])
        yL = np.array([1, 0, 1])
        U = Data(X=XU, y=yU)
        L = Data(X=XL, y=yL)
        args = [U, L, self.clf]
        qs = QS.MinMax(**qs_kwargs)
        scores = qs.score(*args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 0)

    @disable_DeprecationWarning
    def test_combined(self):
        qs1 = QS.Entropy()
        qs2 = QS.MinMax()
        beta = 1
        qs = QS.CombinedSampler(qs1=qs1, qs2=qs2, beta=beta,
                                choice_metric=np.argmax)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 3)

    @disable_DeprecationWarning
    def test_combined_dynamic_beta(self):
        qs1 = QS.Entropy()
        qs2 = QS.MinMax()
        beta = 'dynamic'
        qs = QS.CombinedSampler(qs1=qs1, qs2=qs2, beta=beta,
                                choice_metric=np.argmax)
        self.assertEqual(qs.beta, 'dynamic')
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 3)

    @disable_DeprecationWarning
    def test_distdiv(self):
        qs1 = QS.Entropy()
        qs2 = QS.MinMax()
        lam = 0.5  # Gives equal weight to qs1 and qs2, like beta=1 above.
        qs = QS.DistDivSampler(qs1=qs1, qs2=qs2, lam=lam,
                               choice_metric=np.argmax)
        scores = qs.score(*self.args)
        self.assertEqual(scores.shape, self.unlabeled.y.shape)
        self.assertNotIn(np.NaN, scores)
        choice = qs.choose(scores)
        self.assertEqual(choice, 3)
