
__all__ = ["CategoryLearner", "RewardLearner", "SoftmaxTempModel", "softmax_cv_trial_metrics"]

import numpy as np
from einops import rearrange
from fastcore.script import call_parse
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import BayesianRidge, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


class CategoryLearner:
    "Online category learner using LogisticRegression with L2 (C tuned via CV)."
    def __init__(self):
        self.values = None
        self.C = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CategoryLearner":
        self.values = np.zeros((len(x), 2))
        self.values[:] = 0.5

        cv_est = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', solver='lbfgs', max_iter=200)
        cv_est.fit(x, y)
        self.C = cv_est.C_[0]

        est = LogisticRegression(C=self.C, penalty='l2', solver='lbfgs', warm_start=True, max_iter=200)

        for t in range(len(x)):
            if t > 0 and len(np.unique(y[:t])) >= 2:
                est.fit(x[:t], y[:t])
                self.values[t] = est.predict_proba(x[t:t+1])[0]
        return self

class RewardLearner:
    "Online reward learner using BayesianRidge."
    def __init__(self):
        self.values = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RewardLearner":
        nt, no, _ = x.shape
        self.values = np.zeros((nt, no))

        est = BayesianRidge()

        for t in range(nt):
            if t > 0:
                self.values[t] = est.predict(x[t])

            train_x = rearrange(x[:t+1], 't o f -> (t o) f')
            train_y = rearrange(y[:t+1], 't o -> (t o)')
            est.fit(train_x, train_y)

        return self
    
class SoftmaxTempModel:
    "Softmax model with temperature and optional position biases."
    def __init__(self, temp_init=1.0, bias_init=0.0):
        self.temp_init, self.bias_init = temp_init, bias_init
        self.temp = temp_init
        self.biases = None

    def _nll(self, params, X, y):
        log_temp = params[0]
        temp = np.exp(log_temp)
        
        # only use biases if they were provided (for 3-choice tasks)
        if len(params) > 1:
            # specifically for 3-choice tasks in odd-one-out
            biases = np.array([params[1], 0.0, params[2]])
            probs = softmax((X + biases) / temp, axis=1)
        else:
            probs = softmax(X / temp, axis=1)

        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))

    def fit(self, X, y):
        n_choices = X.shape[1]
        if n_choices == 3:
            # use biases for 3-choice tasks (odd-one-out)
            x0 = [np.log(self.temp_init), self.bias_init, self.bias_init]
            bounds = [(-4.6, 4.6), (-10, 10), (-10, 10)]
        else:
            # no biases for other tasks
            x0 = [np.log(self.temp_init)]
            bounds = [(-4.6, 4.6)]

        res = minimize(self._nll, x0, args=(X, y), method='L-BFGS-B', bounds=bounds)
        self.temp = np.exp(res.x[0])
        if n_choices == 3:
            self.biases = np.array([res.x[1], 0.0, res.x[2]])
        return self

    def predict_proba(self, X):
        if self.biases is not None:
            return softmax((X + self.biases) / self.temp, axis=1)
        return softmax(X / self.temp, axis=1)

    def nll(self, X, y):
        probs = self.predict_proba(X)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))


def softmax_cv_trial_metrics(X, y, n_folds=5):
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_classes < 2 or counts.min() < 2: return None, None
    n_folds = min(n_folds, counts.min())
    if n_folds < 2: return None, None

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1234)
    trial_nlls = np.full(len(y), np.nan)
    trial_correct = np.full(len(y), np.nan)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2: continue

        model = SoftmaxTempModel().fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)

        trial_nlls[test_idx] = -np.log(probs[np.arange(len(y_test)), y_test])
        trial_correct[test_idx] = (probs.argmax(axis=1) == y_test).astype(float)

    return trial_nlls, trial_correct


@call_parse
def main():
    n_trials, n_feats, n_opts = 60, 512, 2

    x_cat = np.random.randn(n_trials, n_feats)
    y_cat = np.random.randint(0, 2, n_trials)
    cat_learner = CategoryLearner().fit(x_cat, y_cat)
    assert cat_learner.values.shape == (n_trials, 2)
    assert np.allclose(cat_learner.values.sum(axis=1), 1)

    x_rew = np.random.randn(n_trials, n_opts, n_feats)
    y_rew = np.random.rand(n_trials, n_opts)
    rew_learner = RewardLearner().fit(x_rew, y_rew)
    assert rew_learner.values.shape == (n_trials, n_opts)

