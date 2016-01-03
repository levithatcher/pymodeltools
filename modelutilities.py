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



