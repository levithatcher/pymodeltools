from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
import pandas as pd

def buildclfpipeline(algorithm, impute, modeltype):
    if impute and modeltype == 'regress':
        pipestring = Pipeline([
                              ("imputer", Imputer(axis=0)),
                              ("regress", algorithm)
                              ])
    elif not impute and modeltype == 'regress':
        pipestring=algorithm

    elif impute and modeltype == 'classify':
        pipestring = Pipeline([
                              ("imputer", Imputer(axis=0)),
                              ("classify", algorithm)
                              ])
    elif not impute and modeltype == 'classify':
        pipestring=algorithm

    return pipestring

def buildgridparameters(baseparam, impute, modeltype):

    if impute and modeltype == 'regress':

        baseparam['imputer__strategy']=('mean', 'median', 'most_frequent')

    elif not impute and modeltype == 'regress':
        # If we're not Imputing (and not using Pipeline), take regress__ out of dict
        for key, value in baseparam.items():   # iter on both keys and values
            if key.startswith('regress__'):
                baseparam[key.replace("regress__","")] = baseparam.pop(key)

    elif impute and modeltype == 'classify':

        baseparam['imputer__strategy']=('mean', 'median', 'most_frequent')

    elif not impute and modeltype == 'classify':
        # If we're not Imputing (and not using Pipeline), take regress__ out of dict
        for key, value in baseparam.items():   # iter on both keys and values
            if key.startswith('classify__'):
                baseparam[key.replace("classify__","")] = baseparam.pop(key)

    return baseparam

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

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