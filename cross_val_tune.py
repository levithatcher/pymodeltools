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
import matplotlib.pyplot as plt

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

    def __init__(self, df, predictedcol, testsize, type):
        self.df = df
        self.predictedcol = predictedcol
        self.type = type

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

    def linearreport(self, folds, cores, plotit):
        self.folds = folds
        self.cores = cores

        if self.type == 'class':

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                 # Todo: get Logistic feature selection working (test on IU)
                                ("randlogit", RandomizedLogisticRegression()),
                                ("logit", LogisticRegression())
                                 ])

                    # Set the parameters by cross-validation
            parameters = {'logit__C': (np.logspace(-2, 2, 10)),
                          'imputer__strategy': ('mean', 'median', 'most_frequent')}

            scores = ['roc_auc']

        elif self.type == 'regress':

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                 # Todo: get Logistic feature selection working (test on IU)
                                #("randlogit", RandomizedLogisticRegression()),
                                ("regress", LinearRegression())
                                 ])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__normalize': (False, True)}

            scores = ['r2'] #'r2']

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

            print("True, Pred")
            for i in range(0,10):
                print(y_true.iloc[i], y_pred[i], y_true.iloc[i] - y_pred[i])

            if self.type == 'class':
                print(classification_report(y_true, y_pred))
            print()

            if plotit == "true":
                x = list(range(0,len(y_pred)))
                fig, ax = plt.subplots(1)
                ax.plot(x, y_true, linewidth=2, color='k')
                ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
                ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
                plt.axis('tight')
                plt.xlim([0, len(y_pred)-1])
                ax.set_ylabel('Taxi min per day')
                ax.set_xlabel('Day of year')
                ax.set_title('Linear Regression Prediction')
                plt.savefig("TaxiMinPerDay_LinearRegression.png", bbox_inches='tight')
                #plt.show()


    def treesreport(self, folds, cores, plotit):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            pipeline = Pipeline([("imputer", Imputer(
                                     axis=0)),
                                 ("feature_selection", SelectFromModel(
                                     LinearSVC(), threshold="median")),
                                 ("trees", DecisionTreeClassifier())])

            # Set the parameters by cross-validation
            parameters = {'trees__criterion': ["gini", "entropy"],
                          'trees__class_weight': ["balanced"],
                          'imputer__strategy': ('mean', 'median', 'most_frequent')}

            scores = ['roc_auc']

        elif self.type == "regress":

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", DecisionTreeRegressor())
                     ])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__splitter': ('best','random')}

            scores = ['r2']

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

            print("True, Pred")
            for i in range(0,10):
                print(y_true.iloc[i], y_pred[i], y_true.iloc[i] - y_pred[i])

            if self.type == 'class':
                print(classification_report(y_true, y_pred))
            print()
            
            if plotit == "true":
                x = list(range(0,len(y_pred)))
                fig, ax = plt.subplots(1)
                ax.plot(x, y_true, linewidth=2, color='k')
                ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
                ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
                plt.axis('tight')
                plt.xlim([0, len(y_pred)-1])
                ax.set_ylabel('Taxi min per day')
                ax.set_xlabel('Day of year')
                ax.set_title('Decision Trees Prediction')
                plt.savefig("TaxiMinPerDay_DecisionTrees.png", bbox_inches='tight')
                #plt.show()

    def extratreesreport(self, folds, cores, plotit):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("extra", ExtraTreeClassifier())])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'extra__criterion': ["gini", "entropy"],
                          'extra__class_weight': ["balanced"]
                          }

            scores = ['roc_auc']


        elif self.type == "regress":

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", ExtraTreeRegressor())
                     ])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent')}

            scores = ['r2']

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

            print("True, Pred")
            for i in range(0,10):
                print(y_true.iloc[i], y_pred[i], y_true.iloc[i] - y_pred[i])

            if self.type == 'class':
                print(classification_report(y_true, y_pred))
            print()
            
            if plotit == "true":
                x = list(range(0,len(y_pred)))
                fig, ax = plt.subplots(1)
                ax.plot(x, y_true, linewidth=2, color='k')
                ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
                ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
                plt.axis('tight')
                plt.xlim([0, len(y_pred)-1])
                ax.set_ylabel('Taxi min per day')
                ax.set_xlabel('Day of year')
                ax.set_title('Extra Tree Prediction')
                plt.savefig("TaxiMinPerDay_ExtraTree.png", bbox_inches='tight')
                #plt.show()

    def randomforestreport(self, folds, cores, plotit):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("randforest", RandomForestClassifier())])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                        'randforest__criterion': ["gini", "entropy"],
                        'randforest__n_estimators': [10,50,100,250,500],
                        'randforest__bootstrap': [True, False],
                        'randforest__class_weight': ["balanced", "balanced_subsample"]}

            scores = ['roc_auc']

        elif self.type == "regress":

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                     # Todo: get Logistic feature selection working (test on IU)
                    #("randlogit", RandomizedLogisticRegression()),
                    ("regress", RandomForestRegressor())
                     ])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'regress__n_estimators': (10,50,100),
                          'regress__bootstrap': (True, False)}

            scores = ['r2']

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

            print("True, Pred")
            for i in range(0,10):
                print(y_true.iloc[i], y_pred[i], y_true.iloc[i] - y_pred[i])

            if self.type == 'class':
                print(classification_report(y_true, y_pred))
            print()
            
            if plotit == "true":
                x = list(range(0,len(y_pred)))
                fig, ax = plt.subplots(1)
                ax.plot(x, y_true, linewidth=2, color='k')
                ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
                ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
                plt.axis('tight')
                plt.xlim([0, len(y_pred)-1])
                ax.set_ylabel('Taxi min per day')
                ax.set_xlabel('Day of year')
                ax.set_title('Random Forest Prediction')
                plt.savefig("TaxiMinPerDay_RandomForest.png", bbox_inches='tight')
                #plt.show()
    
    def svm(self, folds, cores, plotit):
        self.folds = folds
        self.cores = cores

        if self.type == "class":

            pipeline = Pipeline([("imputer", Imputer(
                                    axis=0)),
                                 ("feature_selection", SelectFromModel(
                                    LinearSVC(), threshold="median")),
                                 ("lsvc", LinearSVC())])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'lsvc__C': (np.logspace(-2, 2, 10))}

            scores = ['roc_auc']

        elif self.type == "regress":

            pipeline = Pipeline([("imputer", Imputer(axis=0)),
                                  # Todo: get Logistic feature selection working (test on IU)
                                  #("randlogit", RandomizedLogisticRegression()),
                                 ("lsvr", LinearSVR())])

            # Set the parameters by cross-validation
            parameters = {'imputer__strategy': ('mean', 'median', 'most_frequent'),
                          'lsvr__C': (np.logspace(-2, 2, 10))}

            scores = ['r2']

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

            print("True, Pred, True-Pred")
            for i in range(0,10):
                print(y_true.iloc[i], y_pred[i], y_true.iloc[i] - y_pred[i])

            if self.type == 'class':
                print(classification_report(y_true, y_pred))
            print()
            
            if plotit == "true":
                x = list(range(0,len(y_pred)))
                fig, ax = plt.subplots(1)
                ax.plot(x, y_true, linewidth=2, color='k')
                ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
                ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
                plt.axis('tight')
                plt.xlim([0, len(y_pred)-1])
                ax.set_ylabel('Taxi min per day')
                ax.set_xlabel('Day of year')
                ax.set_title('SVM Prediction')
                plt.savefig("TaxiMinPerDay_SVM.png", bbox_inches='tight')
                #plt.show()