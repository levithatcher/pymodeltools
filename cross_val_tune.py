from __future__ import print_function

from sklearn import cross_validation
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

    def __init__(self, df, predictedcol):
        self.df = df
        self.predictedcol = predictedcol

        # remove columns that are objects (ie not numbers)
        cols = (self.df.dtypes != object)
        self.df = self.df[cols[cols].index]

        # remove rows with any null values
        self.df = self.df[~self.df.isnull().any(axis=1)]

        y = self.df[self.predictedcol]
        del self.df[self.predictedcol]

        X = self.df

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(
            X, y, test_size=0.5, random_state=0)

    def logitreport(self, folds, cores):
        self.folds=folds
        self.cores=cores

        # Set the parameters by cross-validation
        tuned_parameters = [{'C': (np.logspace(-2, 2, 10))}]

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=self.folds, scoring=score, n_jobs=self.cores)
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
        self.folds=folds
        self.cores=cores

        # Set the parameters by cross-validation
        tuned_parameters = [{'criterion': ["gini","entropy"]}]

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=self.folds, scoring=score, n_jobs=self.cores)
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
        self.folds=folds
        self.cores=cores

        # Set the parameters by cross-validation
        tuned_parameters = [{'criterion': ["gini","entropy"]}]

        scores = ['roc_auc']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(ExtraTreeClassifier(), tuned_parameters, cv=self.folds, scoring=score, n_jobs=self.cores)
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