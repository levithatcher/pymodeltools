from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression, RandomizedLogisticRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVC, LinearSVR
import numpy as np
from pymodeltools import plotutilities, modelutilities


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

    def __init__(self, df, predictedcol, testsize, modeltype, impute, scores):
        self.df = df
        self.predictedcol = predictedcol
        self.modeltype = modeltype
        # Todo: make scores property, such that people are warned that they have to pass array
        self.scores = scores
        self.impute = impute

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

        # Get split for viz purposes
        #self.X_testfinal = X[:100]
        #self.y_testfinal = y[:100]

    def linearreport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == 'class':

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                 # Todo: get Logistic feature selection working (test on IU)
                                ("randlogit", RandomizedLogisticRegression()),
                                ("logit", LogisticRegression())
                                 ])

            algorithm = "Logistic_Regression"

            baseparam = "{'logit__C': (np.logspace(-2, 2, 10))"

        elif self.modeltype == 'regress':

            algorithm = LinearRegression()

            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #                      # Todo: get Logistic feature selection working (test on IU)
            #                     #("randlogit", RandomizedLogisticRegression()),
            #                     ("regress", LinearRegression())

            baseparam = {'regress__normalize': (False, True)}

            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)



        #Todo: make plotting symmetric between regress/class
        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, self.X_test, save=saveim, title=algorithm)

    def ridgereport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == 'class':

            self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                      # Todo: get Logistic feature selection working
                                      ("randlogit", RandomizedLogisticRegression()),
                                      ("logit", LogisticRegression())
                                        ])

            algorithm = "Logistic_Regression"

            baseparam = "{'logit__C': (np.logspace(-2, 2, 10))"

        elif self.modeltype == 'regress':

            algorithm = Ridge()

            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #                      # Todo: get Logistic feature selection working (test on IU)
            #                     #("randlogit", RandomizedLogisticRegression()),
            #                     ("regress", LinearRegression())

            baseparam = {'regress__alpha': (np.logspace(-2, 2, 10)),
                         'regress__normalize': (True, False)}

            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)

        #Todo: make plotting symmetric between regress/class
        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, self.X_test, save=saveim, title=algorithm)

    def treereport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                     axis=0)),
                                 ("feature_selection", SelectFromModel(
                                     LinearSVC(), threshold="median")),
                                 ("trees", DecisionTreeClassifier())])

            title = "Decision_Tree_Classifier"

            #algorithm = DecisionTreeClassifier()

            #self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)

            # Set the parameters by cross-validation
            self.parameters = {'trees__criterion': ["gini", "entropy"],
                               'trees__class_weight': ["balanced"],
                               'imputer__strategy': ('mean', 'median', 'most_frequent')}

        elif self.modeltype == "regress":

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #          # Todo: get Logistic feature selection working (test on IU)
            #         #("randlogit", RandomizedLogisticRegression()),
            #         ("regress", DecisionTreeRegressor())
            #          ])

            algorithm = DecisionTreeRegressor()

            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)

            baseparam = {'regress__splitter': ('best','random')}

            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_truefinal, save=saveim, title=algorithm)

    def randomforestreport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("randforest", RandomForestClassifier())])

            title = "Random_Forest_Classifier"

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                               'randforest__criterion': ["gini", "entropy"],
                               'randforest__n_estimators': [10,50,100,250,500],
                               'randforest__bootstrap': [True, False],
                               'randforest__class_weight': ["balanced", "balanced_subsample"]}

        elif self.modeltype == "regress":

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #          #("feature_selection", RFE(
            #          #   RandomForestRegressor(), 6)),
            #          ("regress", RandomForestRegressor())])



            algorithm = RandomForestRegressor()
            #
            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)
            #
            baseparam = {'regress__n_estimators': [10,50,100,250,500],
                               'regress__bootstrap': [True, False]}
            #
            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #          # Todo: get Logistic feature selection working (test on IU)
            #         #("randlogit", RandomizedLogisticRegression()),
            #         ("regress", RandomForestRegressor())
            #          ])

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, self.X_test,
                                                     save=saveim, title="RandomForestRegressor with holiday and precip data")

    def gradboostreport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == "class":

            pass
            #todo: set up classifier gridsearch

        elif self.modeltype == "regress":

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #          #("feature_selection", RFE(
            #          #   RandomForestRegressor(), 6)),
            #          ("regress", RandomForestRegressor())])



            algorithm = GradientBoostingRegressor()
            #
            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)
            #
            baseparam = {'regress__loss': ['ls','lad'],
                         #'regress__learning_rate': [0.05, 0.1],
                         'regress__n_estimators': [75,100,250,350,500]}

            #
            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #          # Todo: get Logistic feature selection working (test on IU)
            #         #("randlogit", RandomizedLogisticRegression()),
            #         ("regress", RandomForestRegressor())
            #          ])

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)

        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, self.X_test,
                                                     save=saveim, title="RandomForestRegressor with holiday and precip data")


    def svmreport(self, folds, cores, plotit, saveim):
        self.folds = folds
        self.cores = cores

        if self.modeltype == "class":

            self.pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("lsvc", LinearSVC())])

            # Set the parameters by cross-validation
            self.parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'lsvc__C': (np.logspace(-2, 2, 10))}

            title = "Linear_SVC"

        elif self.modeltype == "regress":

            # self.pipeline = Pipeline([("imputer", Imputer(axis=0)),
            #                       # Todo: get Logistic feature selection working (test on IU)
            #                       #("randlogit", RandomizedLogisticRegression()),
            #                      ("lsvr", LinearSVR())])

            algorithm = LinearSVR()

            self.pipeline = modelutilities.buildclfpipeline(algorithm, self.impute)

            baseparam = {'regress__C': (np.logspace(-2, 2, 10))}

            self.parameters = modelutilities.buildgridparameters(baseparam, self.impute)

        # Fit/Predict, run report, and return fit
        return TuneModel.clfreport(self)

        # todo: fix plt.show() stopping program execution
        # Plot prediction against truth
        if plotit: plotutilities.plottmplfillbetween(self.y_pred, self.y_true, save=saveim, title=algorithm)

    def clfreport(self, printsample=False):

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

            if printsample:
                print("Date, True, Pred, True-Pred")
                for i in range(0,15):
                    print('%s %.0f  %.0f  %.0f' %

                          (str(self.X_test.iloc[i,0]).zfill(2) + "-" + str(self.X_test.iloc[i,2]).zfill(2),
                          self.y_true.iloc[i], self.y_pred[i], self.y_true.iloc[i] - self.y_pred[i]))

            if self.modeltype == 'class':
                print(classification_report(self.y_true, self.y_pred))
            print()

        return clf