from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import string


class TweetSentExample(object):

    def __init__(self, df, predictedcol, textcol, testsize):

        self.df = df
        self.predictedcol = predictedcol
        self.textcol = textcol
        self.stemmer = PorterStemmer()

        # remove rows with any null values
        self.df = self.df[~self.df.isnull().any(axis=1)]

        y = self.df[self.predictedcol]
        X = self.df[self.textcol]

        # Split the dataset in two equal parts
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=testsize, random_state=0)

    def tokenlogitreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=self.tokenize)),
        ('logit', LogisticRegression())
        ])

        # Set the parameters by cross-validation
        parameters = {'tfidf__stop_words': ['english',None],
                      'tfidf__ngram_range': [(1,1),(1,2)]}
                        #'logit__C': (np.logspace(-2, 2, 10))}

        scores = ['roc_auc']

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
            print(classification_report(y_true, y_pred))
            print()

    def notokenlogitreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logit', LogisticRegression())
        ])

        # Set the parameters by cross-validation
        parameters = {'tfidf__stop_words': ['english',None],
                      'tfidf__ngram_range': [(1,1),(1,2)]}

        scores = ['roc_auc']

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
            print(classification_report(y_true, y_pred))
            print()

    def tokensvcreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=self.tokenize)),
        ('svc', LinearSVC())
        ])

        # Set the parameters by cross-validation
        parameters = {'tfidf__stop_words': ['english',None],
                      'tfidf__ngram_range': [(1,1),(1,2)]}


        scores = ['roc_auc']

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
            print(classification_report(y_true, y_pred))
            print()

    def notokensvcreport(self, folds, cores):
        self.folds = folds
        self.cores = cores

        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svc', LinearSVC())
        ])

        # Set the parameters by cross-validation
        parameters = {'tfidf__stop_words': ['english',None],
                      'tfidf__ngram_range': [(1,1),(1,2)]}

        scores = ['roc_auc']

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
            print(classification_report(y_true, y_pred))
            print()


    @staticmethod
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(self, text):
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = word_tokenize(text)
        stems = self.stem_tokens(tokens, self.stemmer)
        return stems