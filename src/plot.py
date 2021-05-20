from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import config
import statsmodels.api as sm


df = pd.read_csv(config.TRAINING)
df = df['AdjClose'].dropna()
train, test = train_test_split(df, test_size=0.025, shuffle=False)
log_ret = pd.read_csv(config.LOG_TRAINING)


def train_test_plot():
    """ Plotting Training/Testing Time Series"""
    plt.title('Crude Oil Time-Series')
    plt.plot(train,label="Training Set")
    plt.plot(test,label="Test Set")
    plt.xlabel('Number of Days')
    plt.ylabel('Price in USD')
    plt.legend(loc = 'upper left')
    plt.savefig('../plots/Time-Series.jpg')


def ACF():
    """ Plotting the Autocorrelation with Non-Stationary Time-Series"""
    sm.graphics.tsa.plot_acf(df.values.squeeze(), lags=40)
    plt.title('Crude Oil AdjClose Price Autocorrelation')
    plt.savefig('../plots/ACF_Nonstationary.jpg')


def ACF_log():
    """ Plotting the Autocorrelation with Log-Returns Time-Series"""
    sm.graphics.tsa.plot_acf(log_ret['AdjClose'].values.squeeze(), lags=40)
    plt.title('Crude Oil Log-Returns Autocorrelation')
    plt.savefig('../plots/ACF_Stationary.jpg')


if __name__ == '__main__':
    train_test_plot()
    ACF()
    ACF_log()
    plt.show()