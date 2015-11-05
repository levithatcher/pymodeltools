import pandas as pd
from pymodelaids import OverallRank, PersonalRank, TuneClassifier
#import pyodbc


if __name__ == "__main__":

    #### CSV Example of how to use TuneClassifier and printreport for hyperparameter optimization
    df = pd.read_csv('SalesOrderHeader.csv')

    p = TuneClassifier(df,'OnlineOrderFlag')

    p.logitreport(folds=2,cores=6)

    p.treesreport(folds=2,cores=6)

    p.extratreesreport(folds=2,cores=6)

    ## CSV example of how to use OverallRank class and methods printit, plotit
    ## General model feature importance
    # t = OverallRank(df,'RejectedQty')
    #
    # t.printit()
    #
    # #t.plotit()
    #
    ## CSV example of how to use PersonalRank class and methods printlist
    ## Personalized (ie, individual row) feature importance
    # t2 = PersonalRank(df, 'RejectedQty',0.001)
    #
    # t2.printlist()

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