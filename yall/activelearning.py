import numpy as np
from sklearn import model_selection, metrics

from .containers import Data
from .querystrategies import QueryStrategy, SimpleMargin
from .initializations import LDSCentrality as LDS


class ActiveLearningModel(object):

    def __init__(self, classifier, query_strategy, eval_metric="auc",
                 U_proportion=0.9, init_L="random", random_state=None):
        '''
        :param sklearn.base.BaseEstimator classifier: Classifier to
                                                      build the model.
        :param QueryStrategy query_strategy: QueryStrategy instance to use.
        :param str eval_metric: One of "auc", "accuracy".
        :param float U_proportion: proportion of training data to be assigned
                                   the unlabeled set.
        :param str init_L: How to initialize L: "random" or "LDS".
        :param int random_state: Sets the random_state parameter
                                 of train_test_split.
        '''
        self.__check_args(classifier, query_strategy, U_proportion)
        self.classifier = classifier
        self.query_strategy = query_strategy
        self.eval_metric = eval_metric
        self.U_proportion = U_proportion
        self.init_L = init_L
        self.random_state = random_state
        self.L = Data()  # Labeled data.
        self.U = Data()  # Unlabeled data.
        self.T = Data()  # Test data.
        self.classes = None

    def __check_args(self, classifier, query_strategy, U_proportion):
        if not isinstance(query_strategy, QueryStrategy):
            raise ValueError("query_strategy must be an instance of QueryStrategy.")  # noqa
        if not 0 < U_proportion < 1:
            raise ValueError("U_proportion must be in range (0,1) exclusive. Got {}."  # noqa
                              .format(U_proportion))
        if isinstance(query_strategy, SimpleMargin):
            if not hasattr(classifier, "decision_function"):
                raise ValueError("{} compatible only with discriminative models."  # noqa
                                 .format(str(query_strategy)))

    def _random_init(self, X, y, U_size):
        """
        Initialize the labeled set at random.

        :param np.array X: feature matrix
        :param np.array y: label vector
        :param int U_size: The number of samples to keep unlabeled.
        :returns tuple of labeled X, unlabeled X, labeled y, unlabeled y
        :rtype tuple(np.array, np.array, np.array, np.array)
        """
        split = model_selection.train_test_split(X, y, test_size=U_size,
                                                 random_state=self.random_state)  # noqa
        return split

    def _LDS_init(self, X, y, U_size):
        """
        Initialize the labeled set using local density score (LDS) sampling.

        :param np.array X: feature matrix
        :param np.array y: label vector
        :param int U_size: The number of samples to keep unlabeled.
        :returns tuple of labeled X, unlabeled X, labeled y, unlabeled y
        :rtype tuple(np.array, np.array, np.array, np.array)
        """
        k = 10
        idxs = LDS(k=k, threshold="auto").find_centers(X, y)
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[idxs] = True
        Lx = X[mask, ]
        Ux = X[np.logical_not(mask), ]
        Ly = y[mask]
        Uy = y[np.logical_not(mask), ]
        return Lx, Ux, Ly, Uy

    def prepare_data(self, train_X, test_X, train_y, test_y):
        '''
        Splits data into unlabeled, labeled, and test sets
        according to self.U_proportion.

        :param np.array train_X: Training data features.
        :param np.array test_X: Test data features.
        :param np.array train_y: Training data labels.
        :param np.array test_y: Test data labels.
        '''
        U_size = int(np.ceil(self.U_proportion * train_X.shape[0]))
        if not 0 < U_size < train_X.shape[0]:
            raise ValueError("U_proportion must result in non-empty labeled and unlabeled sets.")  # noqa
        if train_X.shape[0] - U_size <= 1:
            raise ValueError("U_proportion must result in a labeled set with > 1 members.")  # noqa
        if self.init_L == "random":
            split = self._random_init(train_X, train_y, U_size)
        elif self.init_L == "LDS":
            split = self._LDS_init(train_X, train_y, U_size)

        self.L.X, self.U.X, self.L.y, self.U.y = split
        self.T.X = test_X
        self.T.y = test_y

    def update_labels(self):
        '''
        Gets the chosen index from the query strategy,
        adds the corresponding data point to L and removes
        it from U. Logs which instance is picked from U.

        :returns: chosen x and y, for use with partial_train()
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        '''
        index = self.query_strategy.query(self.U, self.L, self.classifier)
        chosen_x = self.U.X[index]
        chosen_y = np.array([self.U.y[index]])
        self.L.y = np.append(self.L.y, chosen_y, axis=0)
        self.L.X = np.vstack((self.L.X, chosen_x))
        self.U.X = np.delete(self.U.X, index, axis=0)
        self.U.y = np.delete(self.U.y, index, axis=0)
        return chosen_x.reshape(1, -1), chosen_y

    def train(self):
        '''
        Trains the classifier on L.
        '''
        self.classifier.fit(self.L.X, self.L.y)

    def partial_train(self, new_x, new_y):
        '''
        Given a subset of training examples, calls partial_fit.

        :param numpy.ndarray new_x: Feature array.
        :param numpy.ndarray new_y: Label array.
        '''
        if self.classes is None:
            self.classes = np.unique(self.U.y)
        self.classifier.partial_fit(new_x, new_y, classes=self.classes)

    def score(self):
        '''
        Computes the performance of the current classifier according
        to self.eval_metric.

        :returns: performance score
        :rtype: float
        '''
        if self.eval_metric == "auc":
            try:  # If the classifier is probabilistic.
                res = self.classifier.predict_proba(self.T.X)[:, 1]
            except AttributeError:
                res = self.classifier.decision_function(self.T.X)
            score = metrics.roc_auc_score(self.T.y, res)
        elif self.eval_metric == "accuracy":
            res = self.classifier.predict(self.T.X)
            score = metrics.accuracy_score(self.T.y, res)
        else:
            raise AttributeError("Metric '{}' is not supported."
                                 .format(self.eval_metric))
        return score

    def _get_choice_order(self, ndraws):
        """
        Finds the members of the labeled set in the order
        in which they were chosen by the query strategy.

        :param int ndraws: The number of draws made.
        :returns: labeled X and labeled y according to their choice order
        :rtype: dict({'X': np.array, 'y': np.array})
        """
        mask = np.ones(self.L.y.shape, dtype=bool)
        L_0_index = self.L.y.shape[0] - ndraws
        mask[:L_0_index] = False
        choice_order = {'X': self.L.X[mask], 'y': self.L.y[mask]}
        return choice_order

    def run(self, train_X, test_X, train_y, test_y, ndraws=None, verbose=0):
        '''
        Run the active learning model. Saves AUC scores for
        each sampling iteration.

        :param np.array train_X: Training data features.
        :param np.array test_X: Test data features.
        :param np.array train_y: Training data labels.
        :param np.array test_y: Test data labels.
        :param int ndraws: Number of times to query the unlabeled set.
                            If None, query entire unlabeled set.
        :param int verbose: If > 0, print information.
        :returns: AUC scores for each sampling iteration.
        :rtype: numpy.ndarray(shape=(ndraws, ))
        '''
        # Populate L, U, and T
        self.prepare_data(train_X, test_X, train_y, test_y)
        if ndraws is None:
            ndraws = self.U.X.shape[0]
        scores = np.zeros(ndraws, dtype=np.float32)
        for i in range(ndraws):
            if verbose > 0:
                print(f"{i}\r", end='')
            self.train()
            auc = self.score()
            scores[i] = auc
            self.update_labels()
        choice_order = self._get_choice_order(ndraws)
        return scores, choice_order
