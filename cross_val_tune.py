from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
import numpy as np
from pymodeltools import plotutilities


class TuneModel(object):
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

    def __init__(self, df, predictedcol, testsize, type, scores):
        self.df = df
        self.predictedcol = predictedcol
        self.type = type
        # Todo: make scores property, such that people are warned that they have to pass array
        self.scores = scores

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

    def linearreport(self, folds, cores, plotit, save):
        self.folds = folds
        self.cores = cores

        if self.type == 'class':

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                 # Todo: get Logistic feature selection working (test on IU)
                                ("randlogit", RandomizedLogisticRegression()),
                                ("logit", LogisticRegression())
                                 ])

            # Set the parameters by cross-validation
            self.parameters = {'logit__C': (np.logspace(-2, 2, 10)),
                              'imputer__strategy': ('mean', 'median', 'most_frequent')}

        elif self.type == 'regress':

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                 # Todo: get Logistic feature selection working (test on IU)
                                #("randlogit", RandomizedLogisticRegression()),
                                ("regress", LinearRegression())
                                 ])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__normalize': (False, True)}

        # Fit/Predict and run report
        TuneModel.clfreport(self)

        #Todo: make plotting symmetric between regress/class
        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=save)

    def treesreport(self, folds, cores, plotit, save):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                     axis=0)),
                                 ("feature_selection", SelectFromModel(
                                     LinearSVC(), threshold="median")),
                                 ("trees", DecisionTreeClassifier())])

            # Set the parameters by cross-validation
            self.parameters = {'trees__criterion': ["gini", "entropy"],
                          'trees__class_weight': ["balanced"],
                          'imputer__strategy': ('mean', 'median', 'most_frequent')}

        elif self.type == "regress":

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", DecisionTreeRegressor())
                     ])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__splitter': ('best','random')}

        # Fit/Predict and run report
        TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=save)

    def extratreesreport(self, folds, cores, plotit, save):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("extra", ExtraTreeClassifier())])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'extra__criterion': ["gini", "entropy"],
                          'extra__class_weight': ["balanced"]
                          }

        elif self.type == "regress":

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", ExtraTreeRegressor())
                     ])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent')}

        # Fit/Predict and run report
        TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=save)

    def randomforestreport(self, folds, cores, plotit, save):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("randforest", RandomForestClassifier())])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                        'randforest__criterion': ["gini", "entropy"],
                        'randforest__n_estimators': [10,50,100,250,500],
                        'randforest__bootstrap': [True, False],
                        'randforest__class_weight': ["balanced", "balanced_subsample"]}

        elif self.type == "regress":

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", RandomForestRegressor())
                     ])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__n_estimators': (10,50,100),
                          'regress__bootstrap': (True, False)}

        # Fit/Predict and run report
        TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=save)

    def svm(self, folds, cores, plotit, save):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("lsvc", LinearSVC())])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'lsvc__C': (np.logspace(-2, 2, 10))}

        elif self.type == "regress":

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                  # Todo: get Logistic feature selection working (test on IU)
                                  #("randlogit", RandomizedLogisticRegression()),
                                 ("lsvr", LinearSVR())])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'lsvr__C': (np.logspace(-2, 2, 10))}

        # Fit/Predict and run report
        TuneModel.clfreport(self)

        # todo: fix plt.show() stopping program execution
        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=save)

    def clfreport(self):

        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(self.pipeline, self.parameters, cv=self.folds, scoring=score, n_jobs=self.cores)

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
            self.y_true, self.y_pred = self.y_test, clf.predict(self.X_test)

            print("Date, True, Pred, True-Pred")
            for i in range(0,10):
                print(str(self.X_test.iloc[i,0]).zfill(2) + "-" + str(self.X_test.iloc[i,0]).zfill(2),
                      self.y_true.iloc[i], self.y_pred[i], self.y_true.iloc[i] - self.y_pred[i])

            if self.type == 'class':
                print(classification_report(self.y_true, self.y_pred))
            print()