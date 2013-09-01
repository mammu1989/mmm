"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""
print(__doc__)

import numpy as np
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(10 * rng.rand(1000, 1), axis=0)
y = np.sin(X).ravel()
#y[::5] += 3 * (0.5 - rng.rand(16))
y = np.sin(X).ravel() + np.random.normal(0, .3, 1000)

# Fit regression model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

clf_1 = BayesianRidge()
clf_2 = RandomForestRegressor(max_depth=5)
clf_1.fit(np.vander(X.ravel(), 5), y)
clf_2.fit(X, y)
# Predict
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
y_1 = clf_1.predict(np.vander(X_test.ravel(), 5))
y_2 = clf_2.predict(X_test)

# Plot the results
import pylab as pl

pl.figure()
pl.scatter(X, y, c="k", label="data")
pl.plot(X_test, y_1, c="g", label="max_depth=5", linewidth=2)
pl.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression")
pl.legend()
pl.show()
