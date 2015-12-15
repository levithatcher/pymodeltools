import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import pandas as pd
import sys


class OverallRank(object):

    """ This class splits a dataframe into .25 test and .75 train and
    for the ExtraTreesClassifier orders the features by importance

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

        y = self.df[self.predictedcol]
        del self.df[self.predictedcol]

        X = self.df
        #X = preprocessing.scale(X) # may improve AUC

        # split data into training and test
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=42)

        pipeline = Pipeline([("imputer", Imputer(strategy='mean', axis=0)),
                             # Todo: get list working with feature selection (how to grab names after selected?)
                             # ("feature_selection", SelectFromModel(
                             #     LinearSVC(), threshold="median")),
                             ("extratrees", ExtraTreesClassifier(n_estimators=250, random_state=0))
                             ])

        # Train model--note that as of 10/08/2015 only ExtraTrees (of sklearn) can show feature importances this way
        pipeline.fit(X_train, y_train)

        # Calculate feature importances
        self.importances = pipeline.named_steps['extratrees'].feature_importances_
        self.std = np.std([tree.feature_importances_ for tree in pipeline.named_steps['extratrees'].estimators_], axis=0)

        self.indices = np.argsort(self.importances)[::-1]

        self.colnames = list(self.df) # get list of column names

        self.namelist = [self.colnames[i] for i in self.indices]

    def printit(self):
        """
        Print list of ranked feature importances

        Parameters
        ----------
        None

        Returns
        -------
        self : object
        No variables
        Printed list of columns in X and their ranked predictive power on y.
        """

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(len(list(self.df))):
            print("%d. feature %s (%f)" % (f + 1, self.namelist[f], self.importances[self.indices[f]]))

    def plotit(self):
        """
        Plot figure showing ranked feature importances

        Parameters
        ----------
        None

        Returns
        -------
        self : object
        No variables
        Pyplot figure showing ranking of feature importances.
        Saved figure.
        """
        print(self.namelist)
        print(self.colnames)

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(self.colnames)), self.importances[self.indices],
               color="r", yerr=self.std[self.indices], align="center")
        plt.xticks(range(len(self.colnames)), self.namelist, rotation=90)
        plt.xlim([-1, len(self.colnames)])
        plt.tight_layout()
        plt.savefig('FeatureImportances.png')
        plt.show()