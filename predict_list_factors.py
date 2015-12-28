from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
import numpy as np
import pandas as pd
import sys

class PredictListFactors(object):
    """ This class predicts a discrete value and determines importance of individual factors

    Parameters
    ----------
    df : pandas dataframe

    predicted : y column (discrete) who's values are being predicted

    Returns
    -------
    self : object
    """

    def __init__(self, df, predictedcol, idcol, testsplitcol):
        self.df = df
        self.predictedcol = predictedcol
        self.idcol = idcol

        # remove columns that are objects (ie not numbers)
        # todo: convert objects to discrete numbers
        cols = (self.df.dtypes != object)
        self.df = self.df[cols[cols].index]

        # remove id column from ML, but will use in display method below (in self.df)
        # Note that self.df is not updated here!
        collist = self.df.columns.tolist()
        collist.remove(idcol)
        df = self.df[collist]

        # Split the dataset using by conditional on testsplitcol
        self.y_train = df[self.predictedcol][df[testsplitcol] == 5]

        del df[self.predictedcol]

        self.X_train = df[df[testsplitcol] == 5]
        self.X_test = df[df[testsplitcol] != 5]

        #grab column names for ML
        self.names = list(df)

    # todo: use property so parameter accepts only certain imputation strategy and classifier
    def predictfactors(self):

        pipeline = Pipeline([("imputer", Imputer(strategy='mean', axis=0)),
                            ("logistic", LogisticRegression())
                             ])

        predict = pipeline.fit(self.X_train, self.y_train).predict(self.X_test)

        firstfactor=[]
        secondfactor=[]
        thirdfactor=[]

        res = np.array(pipeline.named_steps['logistic'].coef_ * self.X_test.iloc[[0]])
        threemaxindexes = np.array((-res).argsort().ravel())
        #[3 1 5 4 2 6 0]
        print(res)
        print(threemaxindexes)
        print(self.names)

        #sys.exit()
        for i in range(0,len(self.X_test)):
            res = np.array(pipeline.named_steps['logistic'].coef_ * self.X_test.iloc[[i]])

            threemaxindexes = np.array((-res).argsort().ravel())

            firstfactor.append(self.names[threemaxindexes[0]])
            secondfactor.append(self.names[threemaxindexes[1]])
            thirdfactor.append(self.names[threemaxindexes[2]])

        for i in range(0,len(self.X_test)):
            print([self.df[self.idcol][i], predict[i], firstfactor[i], secondfactor[i], thirdfactor[i]])