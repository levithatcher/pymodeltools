from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
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
import pandas as pd

def buildclfpipeline(algorithm, impute):
    if impute:
        pipestring = Pipeline([
                              ("imputer", Imputer(axis=0)),
                              ("regress", algorithm)
                              ])
    else:
        pipestring=algorithm

    return pipestring

def buildgridparameters(baseparam, impute):

    if impute:

        baseparam['imputer__strategy']=('mean', 'median', 'most_frequent')

        return baseparam

    elif not impute:
        # If we're not Imputing (and not using Pipeline), take regress__ out of dict
        for key, value in baseparam.items():   # iter on both keys and values
            if key.startswith('regress__'):
                baseparam[key.replace("regress__","")] = baseparam.pop(key)

        return baseparam

def create_future_X_test(start, end, periods=0, freq='D', holidays=False):
    """Creates a dataframe (Xtest) for predicting into future (past training dataset)

    Parameters:

        start: (string) Date df starts

        end: (string, optional) Date df ends

        periods: (int, optional) Number of periods after start

        freq: (string, optional) Unit of periods (ie, H for hour, D for day)

    Note: if 'end' argument is used, then periods and freq can't be"""

    if end:
        dates = pd.date_range(start=start, end=end)

    if periods:
        dates = pd.date_range(start=start, periods=periods, freq=freq)

    if freq == 'H' or freq == 'h':

        df = pd.DataFrame({'DayOfWeek': dates.dayofweek, 'DayOfMonth': dates.day,
                               'Month': dates.month, 'Hour': dates.hour,
                               'WeekofYear': dates.weekofyear}, index=dates)

    elif freq == 'D' or freq == 'd':

        df = pd.DataFrame({'DayOfWeek': dates.dayofweek, 'DayOfMonth': dates.day,
                               'Month': dates.month,
                               'WeekofYear': dates.weekofyear}, index=dates)

    # Todo: refactor this to be slowest changing first
    cols = ['DayOfWeek','DayOfMonth','WeekofYear','Month']

    return df[cols]