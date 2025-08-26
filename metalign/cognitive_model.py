__all__ = ["CategoryLearner"]


import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


class CategoryLearner:
    def __init__(
        self,
        estimator=LogisticRegression(max_iter=4000),  # Linear model to be used for the task. Defaults to `sklearn.linear_model.LogisticRegression`.
    ):
        """
        A class of agent that is used to model the category-learning learning task
        using a linear model of choosing.
        """
        self.estimator = estimator
        # below are place holders
        self.X = np.zeros(1)
        self.y = np.zeros(1)
        self.values = np.zeros(1)
        self.mean = 0
        self.std = 1

    def _predict(self, trial: int):
        """
        Make predictions for the observation for the given trial
        """

        # scale test
        X_test = self.X[trial] - self.mean
        X_test /= self.std
        X_test = X_test.reshape(1, -1)

        self.values[trial, :] = self.estimator.predict_proba(X_test)

    def _learn(self, trial: int):
        """
        Fit the model on observations up until the given trial.
        If that does not include observations belonging to both classes,
        use the pseudo-observations to make predictions
        """

        if 0 in self.y[: trial + 1] and 1 in self.y[: trial + 1]:
            self.estimator = clone(self.estimator)
            train_X = self.X[: trial + 1]

            # update scaling parameters
            self.mean = train_X.mean(axis=0)
            self.std = train_X.std(axis=0)
            self.std = np.where(self.std == 0, 1, self.std)

            train_X -= self.mean
            train_X /= self.std

            self.estimator.fit(train_X, self.y[: trial + 1])

    def fit(self, X: np.ndarray, y: np.ndarray):  # Observations  # Category
        """
        Fit the model to the task in a sequential manner like participants did the task.
        Also save the evolving weights into an array.
        """

        self.X = X
        self.y = y
        self.values = np.zeros((self.X.shape[0], 2))

        # give pseudo-observations so the model can make predictions
        self.estimator.fit(np.zeros((2, self.X.shape[1])), np.array([0, 1]))

        for trial in range(self.X.shape[0]):
            self._predict(trial)
            self._learn(trial)
