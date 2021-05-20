import pandas as pd 
import matplotlib.pyplot as plt
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima


df = pd.read_csv(config.TRAINING)
df = df['AdjClose'].dropna()
train, test = train_test_split(df, test_size=0.01, shuffle=False)


def data_stationary(timeseries):
    """
    Augmented Dickey-Fuller Test
    (H0): If FTR, the time series has a unit root it's Non-tationary.
    (H1): H0 Rejected; the time series doesn't have a unit root, meaning it's Stationary.
    P-value at 5% suggests we reject the H0 (stationary), o/w if p-value > 0.05 we FTR the H0 (non-stationary).
    """
    result = adfuller(df)
    if result[1] < 0.05:
        print(f'Reject H0, TimeSeries is Stationary: P-value = {result[1]}')
    else:
        print(f'FTR the H0, TimeSeries is Non-Stationary: P-value = {result[1]}')


def arimamodel(timeseries):
    """
    AR(p): Linear combination Lags of Y
    MA(q): Linear combination of Lagged forecast errors
    D(d): Number of differencing required to make the time series stationary
    """
    model = auto_arima(train, start_p=0, 
                       start_q=0, test='adf', 
                       d=None, max_p=3,
                       max_q=3, trace=True, 
                      )
    return model

def prediction(timeseries):
    global test
    test = pd.DataFrame(test)
    test['ARIMA'] = arimamodel(train).predict(len(test),index=test.index)
    return test


def pred_plot():
    """Potting Curde Oil Prediction"""
    plt.plot(train,label="Training")
    plt.plot(test,label="Test")
    plt.title('Curde Oil Prediction ARIMA(p=1,d=1,q=1) Model')
    plt.xlabel('Number of Days')
    plt.ylabel('Price (USD)')
    plt.plot(prediction(test), label='Prediction')
    plt.legend(loc = 'upper left')
    plt.savefig('../plots/Prediction.jpg')


if __name__ == '__main__':
    # Best model:  ARIMA(1,1,1)
    print(data_stationary(df)) 
    pred_plot()
    plt.show()
    print(f'MAE:{mean_absolute_error(test.AdjClose, test.ARIMA):.2f}%')