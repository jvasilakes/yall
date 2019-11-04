import os
import unittest
import datetime
import warnings
import json
import numpy as np

from collections import namedtuple
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import yall.querystrategies as QS
from yall.activelearning import ActiveLearningModel
from yall.utils import compute_alc, plot_learning_curve
from yall.datasets import load_dexter, load_spect, load_spectf


# Set the logfile name for all dataset tests.
dt = datetime.datetime.now()
dt = dt.strftime("%Y-%m-%dT%H:%M")
module_path = os.path.dirname(__file__)
LOGDIR = os.path.join(module_path, "logs", f"log_regression_{dt}")
os.makedirs(LOGDIR)


SplitData = namedtuple("SplitData", ["name", "train_X", "test_X",
                                     "train_y", "test_y"])


def disable_DeprecationWarning(fn):
    def _wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return fn(*args, **kwargs)
    return _wrapped


class ActiveLearningTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ActiveLearningTest, self).__init__(*args, **kwargs)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            raise AttributeError("No dataset specified.")
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @disable_DeprecationWarning
    def run_test(self, qs, eval_metric="accuracy"):
        al = ActiveLearningModel(self.clf, qs, **self.kwargs,
                                 eval_metric=eval_metric)
        scores, choices = al.run(self.data.train_X, self.data.test_X,
                                 self.data.train_y, self.data.test_y)
        alc = compute_alc(scores, normalize=True)
        L_init = self.data.train_X.shape[0] - scores.shape[0]
        L_end = self.data.train_X.shape[0]

        # Save the scores and learning curve plots.
        outdir = os.path.join(LOGDIR, self.data.name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        score_file = os.path.join(outdir, "scores.jsonl")
        with open(score_file, 'a') as outF:
            outjson = {"query_strategy": str(qs),
                       "dataset": self.data.name,
                       "draws": (L_init, L_end),
                       "alc": f"{alc:.4f}",
                       "max_accuracy": f"{np.max(scores):.4f}"}
            json.dump(outjson, outF)
            outF.write('\n')
        plt_title = f"{str(qs)} : {self.data.name}"
        fig_file = os.path.join(outdir, f"{'_'.join(str(qs).split())}.png")
        plot_learning_curve(scores, L_init, L_end, eval_metric=eval_metric,
                            title=plt_title, saveto=fig_file)

    @disable_DeprecationWarning
    def test_random(self):
        qs = QS.Random()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_simple_margin(self):
        qs = QS.SimpleMargin()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_margin(self):
        qs = QS.Margin()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_entropy(self):
        qs = QS.Entropy()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_leastconfidence(self):
        qs = QS.LeastConfidence()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_lcb(self):
        qs = QS.LeastConfidenceBias()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_lcb2(self):
        qs = QS.LeastConfidenceDynamicBias()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_d2c(self):
        qs = QS.DistanceToCenter()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_density(self):
        qs = QS.Density()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_minmax(self):
        qs = QS.MinMax()
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_combined_even(self):
        ent = QS.Entropy()
        mm = QS.MinMax()
        qs = QS.CombinedSampler(qs1=ent, qs2=mm, beta=1)
        self.run_test(qs)

    @disable_DeprecationWarning
    def test_combined_dyn(self):
        ent = QS.Entropy()
        mm = QS.MinMax()
        qs = QS.CombinedSampler(qs1=ent, qs2=mm, beta="dynamic")
        self.run_test(qs)

    @ disable_DeprecationWarning
    def test_distdiv_even(self):
        ent = QS.Entropy()
        mm = QS.MinMax()
        qs = QS.DistDivSampler(qs1=ent, qs2=mm, lam=0.5)
        self.run_test(qs)

    @ disable_DeprecationWarning
    def test_distdiv_dyn(self):
        ent = QS.Entropy()
        mm = QS.MinMax()
        qs = QS.DistDivSampler(qs1=ent, qs2=mm, lam="dynamic")
        self.run_test(qs)


class TestIris(ActiveLearningTest):

    def setUp(self):
        self.random_state = 7
        np.random.seed(self.random_state)

        self.data = SplitData("iris", *self.prep_iris())

        self.clf = SVC(kernel="rbf", gamma=0.10, C=1.0,
                       random_state=self.random_state,
                       probability=True)
        self.kwargs = {"U_proportion": 0.97,
                       "random_state": self.random_state}

    def tearDown(self):
        del self.clf

    def prep_iris(self):
        iris = load_iris()
        X, y = shuffle(iris.data, iris.target,
                       random_state=self.random_state)
        split = train_test_split(X, y, test_size=0.2,
                                 random_state=self.random_state)
        return split


class TestBreastCancer(ActiveLearningTest):

    def setUp(self):
        self.random_state = 7
        np.random.seed(self.random_state)

        self.data = SplitData("breast_cancer", *self.prep_breast_cancer())

        self.clf = LR(penalty="l2", C=1.0, solver="liblinear",
                      multi_class="auto", random_state=self.random_state)
        self.kwargs = {"U_proportion": 0.98,
                       "random_state": self.random_state}

    def tearDown(self):
        del self.clf

    def prep_breast_cancer(self):
        bc = load_breast_cancer()
        X, y = shuffle(bc.data, bc.target,
                       random_state=self.random_state)
        split = train_test_split(X, y, test_size=0.1,
                                 random_state=self.random_state)
        return split


class TestDexter(ActiveLearningTest):

    def setUp(self):
        self.random_state = 7
        np.random.seed(self.random_state)

        self.data = SplitData("dexter", *self.prep_dexter())

        self.clf = LR(penalty="l2", C=1.0, solver="liblinear",
                      multi_class="auto", random_state=self.random_state)
        self.kwargs = {"U_proportion": 0.98,
                       "random_state": self.random_state}

    def tearDown(self):
        del self.clf

    def prep_dexter(self):
        dex = load_dexter()
        X, y = shuffle(dex.data, dex.target,
                       random_state=self.random_state)
        split = train_test_split(X, y, test_size=0.1,
                                 random_state=self.random_state)
        return split


class TestSPECT(ActiveLearningTest):

    def setUp(self):
        self.random_state = 7
        np.random.seed(self.random_state)

        self.data = SplitData("SPECT", *self.prep_spect())

        self.clf = SVC(kernel="linear", gamma=0.10, C=1.0,
                       random_state=self.random_state,
                       probability=True)
        self.kwargs = {"U_proportion": 0.96,
                       "random_state": self.random_state}

    def tearDown(self):
        del self.clf

    def prep_spect(self):
        spect = load_spect()
        X, y = shuffle(spect.data, spect.target,
                       random_state=self.random_state)
        split = train_test_split(X, y, test_size=0.1,
                                 random_state=self.random_state)
        return split


class TestSPECTF(ActiveLearningTest):

    def setUp(self):
        self.random_state = 10
        np.random.seed(self.random_state)

        self.data = SplitData("SPECTF", *self.prep_spectf())

        self.clf = SVC(kernel="linear", gamma=0.10, C=1.0,
                       random_state=self.random_state,
                       probability=True)
        self.kwargs = {"U_proportion": 0.96,
                       "random_state": self.random_state}

    def tearDown(self):
        del self.clf

    def prep_spectf(self):
        spectf = load_spectf()
        X, y = shuffle(spectf.data, spectf.target,
                       random_state=self.random_state)
        split = train_test_split(X, y, test_size=0.1,
                                 random_state=self.random_state)
        return split
