from __future__ import print_function

from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
import numpy as np


class TuneClassifier(object):
    """ This class tunes the hyperparameters for several common classifiers
    using GridSearchCV and AUC

    Parameters
    ----------
    df : pandas dataframe

    predicted : y column (discrete) who's values are being predicted

    Returns
    -------
    self : object
    """

    def __init__(self, df, predictedcol, testsize):
        self.df = df
        self.predictedcol = predictedcol

        # remove columns that are objects (ie not numbers)
        cols = (self.df.dtypes != object)
        self.df = self.df[cols[cols].index]

        # remove rows with any null values
        # self.df = self.df[~self.df.isnull().any(axis=1)]

        y = self.df[self.predictedcol]
        del self.df[self.predictedcol]

        X = self.df

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(
            X, y, test_size=testsize, random_state=0)

    def logitreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([("imputer", Imputer(
                                 axis=0)),
                             ("logit", LogisticRegression())])

        # Set the parameters by cross-validation
        parameters = {'logit__C': (np.logspace(-2, 2, 10)),
                      'imputer__strategy': ('mean', 'median', 'most_frequent')}

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(pipeline, parameters, cv=self.folds, scoring=score, n_jobs=self.cores)

            clf.fit(self.X_train, self.y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self.y_test, clf.predict(self.X_test)
            print(classification_report(y_true, y_pred))
            print()

    def treesreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([("imputer", Imputer(
                                axis=0)),
                             ("trees", DecisionTreeClassifier())])

        # Set the parameters by cross-validation
        parameters = {'trees__criterion': ["gini", "entropy"],
                    'imputer__strategy': ('mean', 'median', 'most_frequent')}

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(pipeline, parameters, cv=self.folds, scoring=score, n_jobs=self.cores)
            clf.fit(self.X_train, self.y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self.y_test, clf.predict(self.X_test)
            print(classification_report(y_true, y_pred))
            print()


    def extratreesreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([("imputer", Imputer(
                                axis=0)),
                             ("extra", ExtraTreeClassifier())])

        # Set the parameters by cross-validation
        parameters = {'extra__criterion': ["gini", "entropy"],
                    'imputer__strategy': ('mean', 'median', 'most_frequent')}

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(pipeline, parameters, cv=self.folds, scoring=score, n_jobs=self.cores)
            clf.fit(self.X_train, self.y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_estimator_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() / 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = self.y_test, clf.predict(self.X_test)
            print(classification_report(y_true, y_pred))
            print()