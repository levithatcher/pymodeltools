import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dt

#todo: add parameters for labels
def plottmplfillbetween(y_pred, y_true, xlabel, save, title):

    title = str(title).replace("()","")

    xx=[]
    #xx = [datetime.strptime(str(xlabel.iloc[i,0]).zfill(2) + "-" + str(xlabel.iloc[i,2]).zfill(2) + '-14', '%m-%d-%y') for i in range(xlabel.shape[0])]
    #xx = ['2014-' + str(xlabel.iloc[i,0]).zfill(2) + "-" + str(xlabel.iloc[i,2]).zfill(2) for i in range(xlabel.shape[0])]
    for i in range(0,xlabel.shape[0]):
        xx.append(dt.datetime(2014,xlabel.iloc[i,0],xlabel.iloc[i,2]))
    print(xx)


    print(len(xx))
    print(len(y_pred))

    #x = list(range(0,len(y_pred)))
    fig, ax = plt.subplots(1)
    ax.plot(xx, y_true, linewidth=2, color='k')
    ax.fill_between(xx, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
    ax.fill_between(xx, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
    plt.axis('tight')
    plt.xlim([0, len(y_pred)-1])
    ax.set_ylabel('Drivers needed per day')
    ax.set_xlabel('Day of year')
    ax.set_title(title)

    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    if save:
        plt.savefig("TaxiMinPerDay_" + str(title) + ".png", bbox_inches='tight')
    else:
        plt.show(block=False)