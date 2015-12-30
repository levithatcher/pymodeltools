import matplotlib.pyplot as plt

#todo: add parameters for labels
def plottmplfillbetween(y_pred, y_true, save):

    x = list(range(0,len(y_pred)))
    fig, ax = plt.subplots(1)
    ax.plot(x, y_true, linewidth=2, color='k')
    ax.fill_between(x, y_true, y_pred, where=y_true>y_pred, interpolate=True, color='blue')
    ax.fill_between(x, y_true, y_pred, where=y_true<y_pred, interpolate=True, color='red')
    plt.axis('tight')
    plt.xlim([0, len(y_pred)-1])
    ax.set_ylabel('Drivers needed per day')
    ax.set_xlabel('Day of year')
    ax.set_title('Linear Regression Prediction')

    if save:
        plt.savefig("TaxiMinPerDay_LinearRegression.png", bbox_inches='tight')
    else:
        plt.show(block=False)