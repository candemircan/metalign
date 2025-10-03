"""
simple linear cognitive models
"""
__all__ = ["CategoryLearner", "RewardLearner"]

import numpy as np
from einops import rearrange
from fastcore.script import call_parse
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge, LogisticRegression


class CategoryLearner:
    def __init__(self,
                 est=LogisticRegression(max_iter=4000), # Sklearn-compatible model with a `predict_proba` method.
                ):
        "agent that models a category-learning task using a linear model."
        self.est = est
        self.x,self.y,self.values = None,None,None
        self.mean,self.std = 0,1

    def _predict(self, t:int):
        "make predictions for a given trial `t`."
        x_test = ((self.x[t] - self.mean) / self.std).reshape(1, -1)
        self.values[t] = self.est.predict_proba(x_test)

    def _learn(self, t:int):
        "fit model on observations up to trial `t`."
        y_train = self.y[:t+1]
        if len(np.unique(y_train)) < 2: return # only refit if both classes have been seen

        self.est = clone(self.est)
        x_train = self.x[:t+1]

        self.mean = x_train.mean(axis=0)
        self.std = x_train.std(axis=0)
        self.std[self.std == 0] = 1 # avoid division by zero

        x_train_scaled = (x_train - self.mean) / self.std
        self.est.fit(x_train_scaled, y_train)

    def fit(self,
            x:np.ndarray, # observations shaped (n_trials, n_features)
            y:np.ndarray, # categories shaped (n_trials,)
           ) -> "CategoryLearner":
        "Sequentially fit the model to the task."
        self.x,self.y = x,y
        self.values = np.zeros((self.x.shape[0], 2))

        # init with pseudo-observations to allow prediction from trial 0
        self.est.fit(np.zeros((2, self.x.shape[1])), np.array([0, 1]))

        for t in range(self.x.shape[0]):
            self._predict(t)
            self._learn(t)
        return self


class RewardLearner:
    def __init__(self,
                 est=BayesianRidge(), # sklearn-compatible model with a `predict` method.
                ):
        "agent that models a reward-guided task using a linear model."
        self.est = est
        self.x,self.y,self.values = None,None,None

    def _get_training_data(self, t:int):
        "collapse history of observations and rewards into 2D/1D arrays for training."
        x_history = self.x[:t+1]
        y_history = self.y[:t+1]
        x_train = rearrange(x_history, 't o f -> (t o) f')
        y_train = rearrange(y_history, 't o -> (t o)')
        return x_train, y_train

    def _predict(self, x_test:np.ndarray, t:int):
        "Predict reward for both options at trial `t`."
        if t > 0: self.values[t] = self.est.predict(x_test)

    def _learn(self, x_train:np.ndarray, y_train:np.ndarray):
        "fit model to given data."
        self.est = clone(self.est)
        self.est.fit(x_train, y_train)

    def fit(self,
            x:np.ndarray, # observations shaped (n_trials, n_options, n_features)
            y:np.ndarray, # rewards shaped (n_trials, n_options)
           ) -> "RewardLearner":
        "sequentially fit the model to the task."
        self.x,self.y = x,y
        n_trials, _, n_feats = self.x.shape
        self.values = np.zeros((n_trials, 2))

        mean,std = np.zeros(n_feats),np.ones(n_feats)

        for t in range(n_trials):
            x_test = self.x[t]
            self._predict((x_test - mean) / std, t)

            x_train,y_train = self._get_training_data(t)
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            std[std == 0] = 1 # avoid division by zero

            self._learn((x_train - mean) / std, y_train)
        return self


@call_parse
def main():
    # tests for CategoryLearner
    n_trials,n_feats = 20,2
    # create a simple, linearly separable dataset
    x_cat = np.random.randn(n_trials, n_feats)
    x_cat[:n_trials//2] += 3 # create a second cluster
    y_cat = np.array([0]*(n_trials//2) + [1]*(n_trials//2))
    
    cat_learner = CategoryLearner().fit(x_cat, y_cat)
    
    assert cat_learner.values.shape == (n_trials, 2)
    assert (cat_learner.values >= 0).all() and (cat_learner.values <= 1).all()
    assert np.allclose(cat_learner.values.sum(axis=1), 1)
    # check that scaling parameters were updated
    assert not np.allclose(cat_learner.mean, 0)
    assert not np.allclose(cat_learner.std, 1)

    # tests for RewardLearner
    n_trials,n_opts,n_feats = 20,2,3
    x_rew = np.random.randn(n_trials, n_opts, n_feats)
    # make one feature predictive of reward
    y_rew = np.zeros((n_trials, n_opts))
    y_rew[:, 0] = x_rew[:, 0, 1] * 2 + np.random.randn(n_trials) * 0.1 # option 0 reward depends on feature 1
    
    rew_learner = RewardLearner().fit(x_rew, y_rew)

    assert rew_learner.values.shape == (n_trials, n_opts)
    # first trial predictions should be zero
    assert (rew_learner.values[0] == 0).all()
    # subsequent trial predictions should not be all zero
    assert not (rew_learner.values[1:] == 0).all()