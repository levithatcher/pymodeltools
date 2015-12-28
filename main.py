import pandas as pd
from pymodeltools import TweetSentExample, OverallRank, PersonalRank, TuneClassifier, PredictListFactors
#import pyodbc

if __name__ == "__main__":

    df = pd.read_csv('Sentiment Analysis Dataset.csv', error_bad_lines=False, nrows=50000)

    o = TweetSentExample(df, predictedcol='Sentiment', textcol='SentimentText', testsize=.25)

    o.logitreport(folds=2, cores=5)

    #o.logitsimplevectreport(folds=2, cores=5)

    #o.logitsupersimplevectreport(folds=2, cores=5)

    #o.logit_simplengrams(folds=2, cores=5)




    #### CSV Example of how to use TuneClassifier and printreport for hyperparameter optimization
    #df = pd.read_csv('SalesOrderHeaderNULL.csv')

    #p = PredictListFactors(df, 'OnlineOrderFlag', 'SalesPersonID', 'ShipMethodID')

    #p.predictfactors()

    # p = TuneClassifier(df,'OnlineOrderFlag', testsize=.5)
    #
    # p.logitreport(folds=2,cores=6)
    #
    # p.treesreport(folds=2, cores=6)
    #
    # p.extratreesreport(folds=2, cores=6)
    #
    # p.randomforestreport(folds=2, cores=6)

    ## CSV example of how to use OverallRank class and methods printit, plotit
    ## General model feature importance
    #t = OverallRank(df,'OnlineOrderFlag')

    #t.printit()

    #t.plotit()

    #
    ## CSV example of how to use PersonalRank class and methods printlist
    ## Personalized (ie, individual row) feature importance
    #t2 = PersonalRank(df, 'RejectedQty',0.001)
    #
    #t2.printlist()

    ## AND corresponding example of overall feature importance ranking for SQL Connection
    # cnxn = pyodbc.connect(DRIVER='{SQL Server Native Client 11.0}', SERVER='localhost',\
    #                     Trusted_Connection='yes')
    #
    # tempsql = ("""SELECT [UnitPrice]
    #                     ,[LineTotal]
    #                     ,[ReceivedQty]
    #                     ,[RejectedQty]
    #                   FROM [AdventureWorks2012].[Purchasing].[PurchaseOrderDetail]
    #             """)
    #
    # df = pd.read_sql(tempsql, cnxn).astype(float)
    #
    # t = OverallRank(df,'RejectedQty') # note different col from csv example above
    #
    # t.printit()
    #
    # t.plotit()