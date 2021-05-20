import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import config


if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING)
    df = df['AdjClose']
    log_returns = np.log(df) - np.log(df.shift(1))
    log_returns = log_returns.dropna()
    log_returns.to_csv('../inputs/log_data.csv')