# Y'ALL: Yet another Active Learning Library

[![Build Status](https://travis-ci.org/jvasilakes/yall.svg?branch=master)](https://travis-ci.org/jvasilakes/yall)
[![Coverage Status](https://coveralls.io/repos/github/jvasilakes/yall/badge.svg?branch=master)](https://coveralls.io/github/jvasilakes/yall?branch=master)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![license MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jvasilakes/yall/blob/master/LICENSE)

<https://yall.readthedocs.io/>

## Getting Started

### Prerequisites
* [Python 3](www.python.org/downloads)
* [NumPy](www.numpy.org)
* [SciPy](www.scipy.org)
* [scikit-learn](www.scikit-learn.org)

### Installing
Clone or download this repository and run
```
python setup.py install
```

#### Supported query strategies
- Random Sampling (passive learning)
- Uncertainty Sampling
  * Entropy Sampling
  * Least Confidence
  * Least Confidence with Bias
  * Least Confidence with Dynamic Bias
  * Margin Sampling
  * Simple Margin Sampling
- Representative Sampling
  * Density Sampling
  * Distance to Center
  * MinMax Sampling
- Combined Sampling Methods
  * Beta-weighted Combined Sampling
  * Lambda-weighted Combined Sampling

## A motivating example

Active learning can often discover a subset of the full data set that generalizes well
to the test set. For example, we consider the Iris data set:

```python
import numpy as np

from yall import ActiveLearningModel
from yall.querystrategies import Margin
from yall.utils import plot_learning_curve

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.base import clone

np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
lr = LR(solver="liblinear", multi_class="auto").fit(train_X, train_y)
print(lr.score(test_X, test_y))
```

Output: 
```
0.967
```
Using the full data set, logistic regression acheives an accuracy of 0.967 on the test data.

```python
alm = ActiveLearningModel(clone(lr), Margin(), eval_metric="accuracy",
                          U_proportion=0.95, random_state=0)
accuracies, choices = alm.run(train_X, test_X, train_y, test_y)
plot_learning_curve(accuracies, 0, len(accuracies), eval_metric="accuracy")
```

Output:
![](docs/images/learning_curve.png?raw=true)

From the learning curve we see that only the first 25 or so data points
are required to acheive perfect 1.0 accuracy on the test data.

```python
lr_small = clone(lr)
lr_small = lr_small.fit(alm.L.X[:25, ], alm.L.y[:25])
print(lr_small.score(test_X, test_y))
```

Output:
```
1.0
```

## Running Tests
From the project home directory run
```
py.test --cov=yall/ tests/
```

## Authors
* **Jake Vasilakes** - jvasilakes@gmail.com

## Acknowledgements
This project grew out of a study of active learning methods for biomedical text classification. The paper associated with this study can be found at [https://doi.org/10.1093/jamiaopen/ooy021](https://doi.org/10.1093/jamiaopen/ooy021).
