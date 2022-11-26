import os
import numpy as np
import pandas as pd

from features import data_features_new as feat, feature_selection
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


DATA_PATH = './data'  # data csv file 이 있는 경로 기입
DATA_FILE_LIST = os.listdir(DATA_PATH)


def concat_data(data_path: list) -> pd.DataFrame:
    """
    분산 저장된 분봉 데이터를 순차적으로 로드하여, DataFrame 으로 합침

    Args:
        data_path (list): data file들 명칭 리스트

    Returns:
        pd.DataFrame: _description_
    """
    df_tmp = pd.DataFrame()
    for path in tqdm(data_path):
        if path.endswith('.csv'):
            file_path = os.path.join(DATA_PATH, path)
            df_tmp = pd.concat([df_tmp, pd.read_csv(file_path)])
    return df_tmp



def cal_iqr(series: pd.Series) -> pd.Series:
    """
    windsering, IQR 함수
    1) 4분위 상단, 하단의 폭을 IQR로 산출
    2) 4분위 상단, 하단의 IQR의 1.5배 범위 이내로 값을 제한

    Args:
        series (pd.Series): 시계열 data

    Returns:
        pd.Series: IQR 범위로 제한된 시계열
    """
    tmp = series.copy()
    
    # 데이터의 4분위중 1, 4분위값을 탐색
    Q1 = tmp.quantile(0.25)
    Q3 = tmp.quantile(0.75)
    
    # 1, 4분위값의 차이, Inter Quantile Range IQR을 산출
    iqr = Q3 - Q1
    
    # 이상치의 제한값을 iqr의 1.5배로 설정
    outlier_step = 1.5 * iqr
    
    # 이상치의 제한값을 넘어서는 이상치들을 제한값으로 한정
    lower_bound = tmp < Q1 - outlier_step
    upper_bound = tmp > Q3 + outlier_step
    tmp[lower_bound] = Q1 - outlier_step
    tmp[upper_bound] = Q3 + outlier_step
    
    return tmp


def result_dataframe(clf, X_train, X_test, y_train, y_test, features=[]):
    """_summary_

    Args:
        clf (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        features (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    # dataset
    if features:
        X_train = X_train[features]
        X_test = X_test[features]
    
    # result
    result = []
    for x, y in zip([X_train, X_test], [y_train, y_test]):
        y_pred = clf.predict(x)
        y_true = y.values
        
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_pred)

        result.append([acc, pre, rec, f1, roc])
        
    df_result = pd.DataFrame(result)
    df_result.index = ["train", "test"]
    df_result.columns = ["accuracy", "precision", "recall", "f1", "roc"]
    
    return df_result


def get_feature(df_raw, freq, days=[1, 3, 5, 10], is_classifier=True):
    df_total_feat = pd.DataFrame()
    df_total_target = pd.DataFrame()
    ticker_list = df_raw["market"].unique().tolist()
    pick_cols = ['candle_date_time_utc', 'market', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
    
    # 각 코인별로 10분봉으로 변환하여 Feature 생성
    for ticker in tqdm(ticker_list):
        resmpl_df = pd.DataFrame()
        df_ticker = df_raw.loc[df_raw["market"]==ticker, pick_cols].copy()

        df_ticker.columns = ['time', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        df_ticker.sort_values("time", ascending=True, inplace=True)
        df_ticker.set_index(pd.to_datetime(df_ticker['time']), inplace=True)

        # 10분봉으로 upsampling
        resmpl_df['open'] = df_ticker['open'].resample(freq, label='right', closed='right').ohlc()['open']
        resmpl_df['high'] = df_ticker['high'].resample(freq, label='right', closed='right').max()
        resmpl_df['low'] = df_ticker['low'].resample(freq, label='right', closed='right').min()
        resmpl_df['close'] = df_ticker['close'].resample(freq, label='right', closed='right').ohlc()['close']
        resmpl_df['volume'] = df_ticker['volume'].resample(freq, label='right', closed='right').sum()
        resmpl_df['ticker'] = ticker
        
        # 기본 피처 생성
        df_feature = feat.get_feature(resmpl_df)
        df_total_feat = pd.concat([df_total_feat, df_feature.reset_index()])
        
        # Target 설정
        df_target = pd.DataFrame()
        for i in days:
            df_target[f"target_close_{i}"] = (df_feature["close"].shift(-i) - df_feature["close"]).dropna()
        
        if is_classifier:
            df_target = df_target.applymap(lambda x: 1 if x > 0 else 0)
        
        df_target['ticker'] = ticker
        df_total_target = pd.concat([df_total_target, df_target])
    
    return df_total_target


def processing_outlier_iqr(df_total_feat):
    df_iqr_data = pd.DataFrame()
    ticker_list = df_total_feat["ticker"].unique().tolist()
    drop_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    
    for ticker in tqdm(ticker_list):
        tmp_ticker = df_total_feat.loc[df_total_feat["ticker"]==ticker].drop(columns=drop_cols).fillna(method='ffill').copy()
        
        tmp_feat_all = pd.DataFrame()  
        for col in tmp_ticker.columns:
            if tmp_ticker.shape[0] == (tmp_ticker[col] == 0).sum():
                print(col)
                print(tmp_ticker[col].describe())
                continue
            tmp_series = cal_iqr(tmp_ticker[col])
            tmp_feat_all = pd.concat([tmp_feat_all, tmp_series], axis=1)
            
        # tmp_feat_all['ticker'] = ticker
        print(ticker)
        print(tmp_feat_all.shape)
        df_iqr_data = pd.concat([df_iqr_data, tmp_feat_all], join='outer')
        
    df_iqr_data = pd.concat([df_total_feat[drop_cols], df_iqr_data], axis=1, join='outer')
    
    return df_iqr_data


def scale_data(df_iqr_data, n=5, window=20):
    df_scaled = pd.DataFrame()
    ticker_list = df_iqr_data["ticker"].unique().tolist()
    drop_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    
    for ticker in tqdm(ticker_list):
        df_ticker = df_iqr_data[df_iqr_data["ticker"] == ticker].copy().iloc[:-n]
        df_ticker = df_ticker.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        scaled = feature_selection.scaling_data(df_ticker.drop(columns=drop_cols), window=window, scale="rob")
        scaled.loc[:, drop_cols] = df_ticker[drop_cols].copy()
        df_scaled = pd.concat([df_scaled, scaled])
        
    return df_scaled


def rolling_cook_dataset(df: pd.DataFrame, ticker_list: list=None, n: int=5, window: int=20, is_classifier: bool=False):
    if ticker_list is None:
        ticker_list = df['ticker'].drop_duplicates().to_list()
    tmp_df = df.copy(deep=True)
    tmp_df['median'] = (tmp_df['high'] + tmp_df['low']) * 0.5
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'ticker', 'median']
    
    df_ticker_all, df_target_all = pd.DataFrame(), pd.DataFrame()
    last_no_samples = 0
    for ticker in tqdm(ticker_list):
        ticker_df = tmp_df[df['ticker']==ticker]
        sub_df = ticker_df.drop(columns=drop_cols)
        sub_arr = sub_df.values
        sub_cols = sub_df.columns
        tmp_target = ticker_df['median'].shift(-n) / ticker_df['median'] - 1.0
        if is_classifier:
            tmp_target = tmp_target.apply(lambda x: 1 if x > 0 else 0)  
        
        for i in range(len(sub_arr) - window + 1):
            df_subset = pd.DataFrame(sub_arr[i:i+window], columns=sub_cols)
            df_subset['no_samples'] = i + last_no_samples
            if last_no_samples > 0 :
                df_subset['no_samples'] += 1 # 임의의 한 코인 Data 재구성이 끝난 후, 새 코인이 시작될 때 샘플수 카운팅이 직전값과 동일하지 않게, 1 을 더해줌.
            df_subset['ticker'] = ticker
            df_ticker_all = pd.concat([df_ticker_all, df_subset], ignore_index=True)
        
        reset_target = tmp_target.iloc[window-1:].reset_index()
        reset_target['ticker'] = ticker
        df_target_all = pd.concat([df_target_all, reset_target], ignore_index=True)
        last_no_samples = df_ticker_all['no_samples'].iloc[-1]
        
    return df_ticker_all, df_target_all


def main(freq='10T'):
    # Data 로딩
    df_raw = concat_data(DATA_FILE_LIST)
    # Feature 생성
    df_total_feat = get_feature(df_raw, freq)
    # 이상치 제한, IQR
    df_iqr_data = processing_outlier_iqr(df_total_feat)
    # data 스케일링
    df_scaled = scale_data(df_iqr_data)
    # rolling dataset 생성
    df_roll_data, df_target = rolling_cook_dataset(df_scaled)
    print(df_roll_data['no_samples'].value_counts())
    # datset 저장
    df_roll_data.set_index('no_samples', inplace=True, drop=True).to_pickle(os.path.join(DATA_PATH, 'TSR_upbit_min_feature.pkl'))
    df_target.set_index('index', inplace=True, drop=True).to_pickle(os.path.join(DATA_PATH, 'TSR_upbit_min_label.pkl'))

    
if __name__ == "__main__":
    main()

    