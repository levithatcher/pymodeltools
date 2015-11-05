import numpy as np
from sklearn import linear_model
from sklearn import cross_validation


class PersonalRank(object):

    """ This class splits a dataframe into (default .05) test and train and
    creates a model using LogisticClassifier, and orders importance of features in each prediction

    Parameters
    ----------
    df : pandas dataframe

    predicted : y (predicted) column who's values are being predicted

    testpercent : percent of rows in dataframe that are split into test set (default 0.10)

    Returns
    -------
    self : object
    """

    def __init__(self, df, predictedcol, testpercent=0.10):
        self.df = df
        self.predictedcol = predictedcol
        self.testpercent = testpercent

        y = self.df[self.predictedcol]

        del self.df[self.predictedcol]

        X = self.df
        #X = preprocessing.scale(X) # may improve AUC

        # Create list of X column names
        self.names = list(self.df)

        # split data into training and test
        X_train, self.X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=self.testpercent, random_state=42)

        self.logistic = linear_model.LogisticRegression()

        # Predict class (0 or 1) and probability
        self.y_pred = self.logistic.fit(X_train, y_train).predict(self.X_test)

    def printlist(self):

        print('Rank individual feature importances for predicted rows')

        for i in range(0,len(self.y_pred)):
            res = np.array(self.logistic.coef_ * self.X_test[i,:])
            topthreefeatures = np.array((-res).argsort()[:5].ravel())

            #if self.y_pred[i] == 1:
            print(self.names[topthreefeatures[0]], self.names[topthreefeatures[1]], self.names[topthreefeatures[2]])