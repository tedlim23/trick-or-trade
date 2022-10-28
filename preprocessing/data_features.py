import os
from datetime import datetime
import pymssql
import ta
import pandas as pd

from pykrx import stock
from features import microstructure_features, ta_features
from tqdm import tqdm
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler


def get_ticker_list(date="20220601"):
    pykrx_kospi = stock.get_market_ticker_list(date, market="KOSPI")
    pykrx_kosdaq = stock.get_market_ticker_list(date, market="KOSDAQ")
    
    return pykrx_kospi, pykrx_kosdaq


def get_data(ticker="005930", fromdate="20000101", todate="20220923", adjusted=True, detail=False):
    ohlcv = stock.get_market_ohlcv(fromdate=fromdate, todate=todate, ticker=ticker, adjusted=adjusted)
    ohlcv.columns = ["open", "high", "low", "close", "volume"]
    tv = stock.get_market_trading_volume_by_date(fromdate=fromdate, todate=todate, ticker=ticker, detail=detail)
    if detail == False:
        tv.drop(["기타법인", "전체"], axis=1, inplace=True)
        tv.columns = ["institutional", "individual", "foreign"]
    else:
        tv.drop(["전체"], axis=1, inplace=True)
    df = pd.merge(ohlcv, tv, on="날짜", how="inner")
    df = df.astype("float")
    return df


def get_feature(df_, windows=[5, 10, 20]):
    df = df_.copy()
    df["transaction_amount"] = df["volume"] * df["close"]
    for i in windows:
        # 1. pct_change
        df[f"pct_change_close_{i}"] = df["close"].pct_change(i)
        df[f"pct_change_volume_{i}"] = df["volume"].pct_change(i)
        
        # 2. std
        df[f"std_close_{i}"] = df["close"].rolling(i).std()
        df[f"std_volume_{i}"] = df["volume"].rolling(i).std()
        
        # 3. Microstructure
        df[f"kyle_lambda_{i}"] = microstructure_features.get_bar_based_kyle_lambda(df["close"], df["volume"], i)
        df[f"amihud_lambda{i}"] = microstructure_features.get_bar_based_amihud_lambda(df["close"], df["transaction_amount"], i)
        df[f"hasbrouck_lambda{i}"] = microstructure_features.get_bar_based_hasbrouck_lambda(df["close"], df["transaction_amount"], i)
        df[f"bekker_parkinson_vol{i}"] = microstructure_features.get_bekker_parkinson_vol(df["high"], df["low"], i)
        df[f"corwin_schultz_estimator{i}"] = microstructure_features.get_corwin_schultz_estimator(df["high"], df["low"], i)
        
    # 4. Techincal Analysis
    df = ta_features.get_ta_features(df, windows=windows)

    return df


def convert_timeseries(df_: pd.DataFrame, window: int=5) -> pd.DataFrame:
    df = df_.copy()
    
    x = df.drop(columns=["target"])
    y = df["target"]
    dataset_x = []
    dataset_y = []
    for i in tqdm(range(window, len(df))):
        # x
        temp_x = x.iloc[i-window:i, :].values.tolist() # sliding window
        dataset_x.append(pd.DataFrame(temp_x))
        # y
        temp_y = y[i-1]
        dataset_y.append(temp_y)
    
    return dataset_x, dataset_y


# %% -------------------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# 차트 설정
# plt.rcParams["font.family"] = 'nanummyeongjo'
plt.rcParams["figure.figsize"] = (14,4)
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.grid"] = True

def plot_df(df):
    for i in df.columns:
        plt.figure(figsize=(10, 1))
        plt.hist(df[i])
        plt.title(i)
        plt.show()

