import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import levene, ttest_ind

# 스케일링
def scaling_data(df_feature, window=20, scale="rob"):
    df_scaled = pd.DataFrame()
    df_feature_only = df_feature.drop(columns=["ticker"])
    
    # 기간 설정
    for i in range(0, len(df_feature), window):
        if i == len(df_feature) // window * window:
            df_window = df_feature_only.iloc[i:, :].copy()
        else:
            df_window = df_feature_only.iloc[i:i+window, :].copy()

        # 스케일링
        if scale == "rob":
            scaler = RobustScaler()
        elif scale == "std":
            scaler = StandardScaler()
        elif scale == "norm":
            scaler = Normalizer()
        elif scale == "mm":
            scaler = MinMaxScaler()
        df_feature_scaled = pd.DataFrame(scaler.fit_transform(df_window))
        df_scaled = pd.concat([df_scaled, df_feature_scaled])

    df_scaled.index = df_feature_only.index
    df_scaled.columns = df_feature_only.columns
    df_scaled["ticker"] = df_feature["ticker"]
    # df_scaled["target"] = df_feature["target"]

    return df_scaled


# 독립표본 T-Test
def dataset_ttest(df, scaling="rob", p_value=0.01):
    target = df["target"].values
    
    if scaling == "norm":
        norm = Normalizer()
        test_data = pd.DataFrame(norm.fit_transform(df.drop(columns=["target"])))
    elif scaling == "std":
        std = StandardScaler()
        test_data = pd.DataFrame(std.fit_transform(df.drop(columns=["target"])))
    elif scaling == "mm":
        mm = MinMaxScaler()
        test_data = pd.DataFrame(mm.fit_transform(df.drop(columns=["target"])))
    elif scaling == "rob":
        rob = RobustScaler()
        test_data = pd.DataFrame(rob.fit_transform(df.drop(columns=["target"])))
    else:
        test_data = df.drop(columns=["target"])

    result = []
    diff = []
    for i in range(len(df.columns)-1):
        # Leven's test
        test_col = test_data.columns[i]
        
        levene_p_value = levene(target, test_data[test_col])[1]
        # pvalue가 유의수준 0.05보다 크기 때문에 귀무가설 채택 -> 두 집단의 데이터는 등분산성을 만족
        if levene_p_value >= p_value:
            equal_var = True
        else:
            equal_var = False
            
        # T-test
        ttest_p_value = ttest_ind(target, test_data[test_col], equal_var=equal_var)[1]
        if ttest_p_value >= p_value:
            print(f"{df.columns[i]} : 주가 등락 값에 따른 {df.columns[i]}의 값은 통계적으로 유의한 차이가 없음")
            diff.append(df.columns[i])
        else:
            # pvalue가 유의수준 0.05보다 작기 때문에 귀무가설 기각 -> 등락변수에 따른 0번 변수의 값는 통계적으로 유의한 차이가 있음 -> 변수 채택
            print(f"{df.columns[i]} : 주가 등락 값에 따른 {df.columns[i]}의 값은 통계적으로 유의한 차이가 있음")
            result.append(df.columns[i])
    
    # col = [str(i) for i in range(len(df.columns)-1)]
    # diff = list(set(col) - set(result))
    return result, diff