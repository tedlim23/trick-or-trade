import pandas as pd

from str_df import NameTechnicalIndex as nti
from ta.momentum import (
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    RSIIndicator,
    StochasticOscillator,
    StochRSIIndicator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
)
from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    NegativeVolumeIndexIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    VolumeWeightedAveragePrice,
)
from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    DPOIndicator,
    EMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    PSARIndicator,
    SMAIndicator,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    WMAIndicator,
)
from ta.others import (
    CumulativeReturnIndicator,
    DailyLogReturnIndicator,
    DailyReturnIndicator,
)

def get_ta_features(df_:pd.DataFrame, 
                    fillna:bool=False, 
                    windows:list=[5, 10, 20],
                    config:list=None
) -> pd.DataFrame:
    df = df_.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    if config is not None:
        if len(config) == 0:
            raise ValueError('config should have components at least 1.')
        get_ta_selectively(df, config=config)
    else:
        get_all_features(df)

    return df


def get_all_features(df_:pd.DataFrame, 
                    fillna:bool=False, 
                    windows:list=[5, 10, 20],
) -> pd.DataFrame:
    df = df_.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Momentum Indicators ---------------------------------------------------------------------------------------------------
    # Relative Strength Index (RSI)
    df["momentum_rsi"] = RSIIndicator(
        close=close, window=14, fillna=fillna
    ).rsi()
    
    # Stochastic RSI
    stoch_rsi_indicator = StochRSIIndicator(
        close=close, window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df["momentum_stoch_rsi"] = stoch_rsi_indicator.stochrsi()
    df["momentum_stoch_rsi_k"] = stoch_rsi_indicator.stochrsi_k()
    df["momentum_stoch_rsi_d"] = stoch_rsi_indicator.stochrsi_d()
    
    # True strength index (TSI)
    df["momentum_tsi"] = TSIIndicator(
        close=close, window_slow=25, window_fast=13, fillna=fillna
    ).tsi()

    # Ultimate Oscillator
    df["momentum_uo"] = UltimateOscillator(
        high=high, low=low, close=close, window1=7, window2=14, window3=28, weight1=4, weight2=2, weight3=1, fillna=fillna
    ).ultimate_oscillator()
    
    # Stochastic Oscillator
    stoch_oscillator = StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3, fillna=fillna
    )
    df["momentum_stoch"] = stoch_oscillator.stoch()
    df["momentum_stoch_signal"] = stoch_oscillator.stoch_signal()
    
    # Williams %R
    df["momentum_wr"] = WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14, fillna=fillna
    ).williams_r()
    
    # Awesome Oscillator
    df["momentum_ao"] = AwesomeOscillatorIndicator(
        high=high, low=low, window1=5, window2=34, fillna=fillna
    ).awesome_oscillator()

    # Rate of Change (ROC)
    df["momentum_roc"] = ROCIndicator(
        close=close, window=12, fillna=fillna
    ).roc()

    # Percentage Price Oscillator (PPO)
    percentage_price_oscillator = PercentagePriceOscillator(
        close=close, window_slow=26, window_fast=12, window_sign=9, fillna=9
    )
    df["momentum_ppo"] = percentage_price_oscillator.ppo()
    df["momentum_ppo_signal"] = percentage_price_oscillator.ppo_signal()
    df["momentum_ppo_hist"] = percentage_price_oscillator.ppo_hist()

    # Percentage Volume Oscillator (PVO)
    percentage_volume_oscillator = PercentageVolumeOscillator(
        volume=volume, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df["momentum_pvo"] = percentage_volume_oscillator.pvo()
    df["momentum_pvo_signal"] = percentage_volume_oscillator.pvo_signal()
    df["momentum_pvo_hist"] = percentage_volume_oscillator.pvo_hist()

    # Kaufman’s Adaptive Moving Average (KAMA)
    df["momentum_kama"] = KAMAIndicator(
        close=close, window=10, pow1=2, pow2=30, fillna=fillna
    ).kama()

    # Volume Indicators ---------------------------------------------------------------------------------------------------
    # Accumulation/Distribution Index (ADI) -> Infinity value
    # df["volume_adi"] = AccDistIndexIndicator(
    #     high=high, low=low, close=close, volume=volume, fillna=fillna
    # ).acc_dist_index()

    # On-balance volume (OBV)
    df["volume_obv"] = OnBalanceVolumeIndicator(
        close=close, volume=volume, fillna=fillna
    ).on_balance_volume()
    
    # Chaikin Money Flow (CMF)
    df["volume_cmf"] = ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20, fillna=fillna
    ).chaikin_money_flow()
    
    # Force Index (FI)
    df["volume_fi"] = ForceIndexIndicator(
        close=close, volume=volume, window=13, fillna=fillna
    ).force_index()
    
    # Ease of movement (EoM, EMV)
    ease_of_movement_indicator = EaseOfMovementIndicator(
        high=high, low=low, volume=volume, window=14, fillna=fillna
    )
    df["volume_em"] = ease_of_movement_indicator.ease_of_movement()
    df["volume_sma_em"] = ease_of_movement_indicator.sma_ease_of_movement()

    # Volume-price trend (VPT)
    df["volume_vpt"] = VolumePriceTrendIndicator(
        close=close, volume=volume, fillna=fillna
    ).volume_price_trend()

    # Volume Weighted Average Price (VWAP)
    df["volume_vwap"] = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
    ).volume_weighted_average_price()
    
    # Money Flow Index (MFI)
    df["volume_mfi"] = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
    ).money_flow_index()
    
    # Negative Volume Index (NVI)
    df["volume_nvi"] = NegativeVolumeIndexIndicator(
        close=close, volume=volume, fillna=fillna
    ).negative_volume_index()
    
    # Volatility Indicators ---------------------------------------------------------------------------------------------------
    # Bollinger Bands
    bollinger_bands = BollingerBands(
        close=close, window=20, window_dev=2, fillna=fillna
    )
    df["volatility_bbh"] = bollinger_bands.bollinger_hband()
    df["volatility_bbhi"] = bollinger_bands.bollinger_hband_indicator()
    df["volatility_bbl"] = bollinger_bands.bollinger_lband()
    df["volatility_bbli"] = bollinger_bands.bollinger_lband_indicator()
    df["volatility_bbm"] = bollinger_bands.bollinger_mavg()
    df["volatility_bbp"] = bollinger_bands.bollinger_pband()
    df["volatility_bbw"] = bollinger_bands.bollinger_wband()
    
    # Donchian Channel
    donchian_channel = DonchianChannel(
        high=high, low=low, close=close, window=20, offset=0, fillna=0
    )
    df["volatility_dch"] = donchian_channel.donchian_channel_hband()
    df["volatility_dcl"] = donchian_channel.donchian_channel_lband()
    df["volatility_dcm"] = donchian_channel.donchian_channel_mband()
    df["volatility_dcp"] = donchian_channel.donchian_channel_pband()
    df["volatility_dcw"] = donchian_channel.donchian_channel_wband()
    
    # Keltner Channel -> Infinity value
    keltner_channel = KeltnerChannel(
        high=high, low=low, close=close, window=20, window_atr=10, fillna=fillna, original_version=True, multiplier=2
    )
    df["volatility_kch"] = keltner_channel.keltner_channel_hband()
    df["volatility_kchi"] = keltner_channel.keltner_channel_hband_indicator()
    df["volatility_kcl"] = keltner_channel.keltner_channel_lband()
    df["volatility_kcli"] = keltner_channel.keltner_channel_lband_indicator()
    df["volatility_kcm"] = keltner_channel.keltner_channel_mband()
    df["volatility_kcp"] = keltner_channel.keltner_channel_pband()
    df["volatility_kcw"] = keltner_channel.keltner_channel_wband()
    
    # Average True Range (ATR)
    df["volatility_atr"] = AverageTrueRange(
        high=high, low=low, close=close, window=14, fillna=fillna
    ).average_true_range()
    
    # Ulcer Index
    df["volatility_ui"] = UlcerIndex(
        close=close, window=14, fillna=fillna
    ).ulcer_index(    )
    
    # Trend Indicators ---------------------------------------------------------------------------------------------------
    # Moving Average Convergence Divergence (MACD)
    macd = MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df["trend_macd"] = macd.macd()
    df["trend_macd_diff"] = macd.macd_diff()
    df["trend_macd_signal"] = macd.macd_signal()

    for i in windows:
        # Simple Moving Average (SMA)
        df[f"trend_sma_{i}"] = SMAIndicator(
            close=close, window=i, fillna=fillna
        ).sma_indicator()
    
        # Exponential Moving Average (EMA)
        df[f"trend_ema_{i}"] = EMAIndicator(
            close=close, window=i, fillna=fillna
        ).ema_indicator()
        
        # Weighted Moving Average (WMA)
        df[f"trend_wma_{i}"] = WMAIndicator(
            close=close, window=i, fillna=fillna
        ).wma()

    # Vortex Indicator (VI)
    vortex_indicator = VortexIndicator(
        high=high, low=low, close=close, window=14, fillna=fillna
    )
    df["trend_vortex_ind_diff"] = vortex_indicator.vortex_indicator_diff()
    df["trend_vortex_ind_pos"] = vortex_indicator.vortex_indicator_pos()
    df["trend_vortex_ind_neg"] = vortex_indicator.vortex_indicator_neg()
 
    # Tripple Exponential Smoothed Moving Average (TRIX)
    df["trend_trix"] = TRIXIndicator(
        close=close, window=15, fillna=fillna
    ).trix()
 
    # Mass Index (MI)
    df["trend_mass_index"] = MassIndex(
        high=high, low=low, window_fast=9, window_slow=25, fillna=fillna
    ).mass_index()
    
    # Detrended Price Oscillator (DPO)
    df["trend_dpo"] = DPOIndicator(
        close=close, window=20, fillna=fillna
    ).dpo()
    
    # KST Oscillator (KST Signal)
    kst_indicator = KSTIndicator(
        close=close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=fillna
    )
    df["trend_kst"] = kst_indicator.kst()
    df["trend_kst_diff"] = kst_indicator.kst_diff()
    df["trend_kst_sig"] = kst_indicator.kst_sig()
 
    # Ichimoku Kinkō Hyō (Ichimoku)
    ichimoku_indicator = IchimokuIndicator(
        high=high, low=low, window1=9, window2=26, window3=52, visual=False, fillna=False
    )
    df["trend_ichimoku_a"] = ichimoku_indicator.ichimoku_a()
    df["trend_ichimoku_b"] = ichimoku_indicator.ichimoku_b()
    df["trend_ichimoku_base"] = ichimoku_indicator.ichimoku_base_line()
    df["trend_ichimoku_conv"] = ichimoku_indicator.ichimoku_conversion_line()
 
    # Schaff Trend Cycle (STC)
    df["trend_stc"] = STCIndicator(
        close=close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=fillna
    ).stc()
 
    # # Average Directional Movement Index (ADX)
    average_directional_movement_indicator = ADXIndicator(
        high=high, low=low, close=close, window=14, fillna=fillna
    )
    df["trend_adx"] = average_directional_movement_indicator.adx()
    df["trend_adx_pos"] = average_directional_movement_indicator.adx_pos()
    df["trend_adx_neg"] = average_directional_movement_indicator.adx_neg()
    
    # Commodity Channel Index (CCI)
    df["trend_cci"] = CCIIndicator(
        high=high, low=low, close=close, window=20, constant=0.015, fillna=fillna
    ).cci()
    
    # Aroon Indicator
    aroon_indicator = AroonIndicator(
        close=close, window=25, fillna=fillna
    )
    df["trend_aroon_up"] = aroon_indicator.aroon_up()
    df["trend_aroon_down"] = aroon_indicator.aroon_down()
    df["trend_aroon_ind"] = aroon_indicator.aroon_indicator()
 
    # Parabolic Stop and Reverse (Parabolic SAR)
    psar_indicator = PSARIndicator(
        high=high, low=low, close=close, step=0.02, max_step=0.2, fillna=fillna
    )
    df["trend_psar"] = psar_indicator.psar()
    df["trend_psar_up"] = psar_indicator.psar_up()
    df["trend_psar_up_indicator"] = psar_indicator.psar_up_indicator()
    df["trend_psar_down"] = psar_indicator.psar_down()
    df["trend_psar_down_indicator"] = psar_indicator.psar_down_indicator()
    df[["trend_psar_up", "trend_psar_down"]] = df[["trend_psar_up", "trend_psar_down"]].fillna(0) # psar 컬럼 null 채우기
    
    # Others Indicators ---------------------------------------------------------------------------------------------------
    # Daily Return (DR)
    df["daily_return"] = DailyReturnIndicator(
        close=close, fillna=fillna
    ).daily_return()
    
    # Daily Log Return (DLR)
    df["daily_log_return"] = DailyLogReturnIndicator(
        close=close, fillna=fillna
    ).daily_log_return()

    # Cumulative Return (CR)
    df["cumlative_return"] = CumulativeReturnIndicator(
        close=close, fillna=fillna
    ).cumulative_return()
    
    # Daily Maximum Return (DMR)
    df["daily_maximum_return"] = (df["high"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    return df


def get_ta_selectively(df_:pd.DataFrame, 
                    config:list,
                    fillna:bool=False, 
                    windows:list=[5, 10, 20],
) -> pd.DataFrame:
    df = df_.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Momentum Indicators ---------------------------------------------------------------------------------------------------
    # Relative Strength Index (RSI)
    if nti.RSI in config:
        df["momentum_rsi"] = RSIIndicator(
            close=close, window=14, fillna=fillna
        ).rsi()
    
    # Stochastic RSI
    if nti.sRSI in config:
        stoch_rsi_indicator = StochRSIIndicator(
            close=close, window=14, smooth1=3, smooth2=3, fillna=fillna
        )
        df["momentum_stoch_rsi"] = stoch_rsi_indicator.stochrsi()
        df["momentum_stoch_rsi_k"] = stoch_rsi_indicator.stochrsi_k()
        df["momentum_stoch_rsi_d"] = stoch_rsi_indicator.stochrsi_d()
    
    # True strength index (TSI)
    if nti.TSI in config:
        df["momentum_tsi"] = TSIIndicator(
            close=close, window_slow=25, window_fast=13, fillna=fillna
        ).tsi()

    # Ultimate Oscillator
    if nti.UO in config:
        df["momentum_uo"] = UltimateOscillator(
            high=high, low=low, close=close, window1=7, window2=14, window3=28, weight1=4, weight2=2, weight3=1, fillna=fillna
        ).ultimate_oscillator()
    
    # Stochastic Oscillator
    if nti.SO in config:
        stoch_oscillator = StochasticOscillator(
            high=high, low=low, close=close, window=14, smooth_window=3, fillna=fillna
        )
        df["momentum_stoch"] = stoch_oscillator.stoch()
        df["momentum_stoch_signal"] = stoch_oscillator.stoch_signal()
    
    # Williams %R
    if nti.wR in config:
        df["momentum_wr"] = WilliamsRIndicator(
            high=high, low=low, close=close, lbp=14, fillna=fillna
        ).williams_r()
    
    # Awesome Oscillator
    if nti.AO in config:
        df["momentum_ao"] = AwesomeOscillatorIndicator(
            high=high, low=low, window1=5, window2=34, fillna=fillna
        ).awesome_oscillator()

    # Rate of Change (ROC)
    if nti.ROC in config:
        df["momentum_roc"] = ROCIndicator(
            close=close, window=12, fillna=fillna
        ).roc()

    # Percentage Price Oscillator (PPO)
    if nti.PPO in config:
        percentage_price_oscillator = PercentagePriceOscillator(
            close=close, window_slow=26, window_fast=12, window_sign=9, fillna=9
        )
        df["momentum_ppo"] = percentage_price_oscillator.ppo()
        df["momentum_ppo_signal"] = percentage_price_oscillator.ppo_signal()
        df["momentum_ppo_hist"] = percentage_price_oscillator.ppo_hist()

    # Percentage Volume Oscillator (PVO)
    if nti.PVO in config:
        percentage_volume_oscillator = PercentageVolumeOscillator(
            volume=volume, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df["momentum_pvo"] = percentage_volume_oscillator.pvo()
        df["momentum_pvo_signal"] = percentage_volume_oscillator.pvo_signal()
        df["momentum_pvo_hist"] = percentage_volume_oscillator.pvo_hist()

    # Kaufman’s Adaptive Moving Average (KAMA)
    if nti.KAMA in config:
        df["momentum_kama"] = KAMAIndicator(
            close=close, window=10, pow1=2, pow2=30, fillna=fillna
        ).kama()

    # Volume Indicators ---------------------------------------------------------------------------------------------------
    # Accumulation/Distribution Index (ADI)
    if nti.ADI in config:
        df["volume_adi"] = AccDistIndexIndicator(
            high=high, low=low, close=close, volume=volume, fillna=fillna
        ).acc_dist_index()

    # On-balance volume (OBV)
    if nti.OBV in config:
        df["volume_obv"] = OnBalanceVolumeIndicator(
            close=close, volume=volume, fillna=fillna
        ).on_balance_volume()
    
    # Chaikin Money Flow (CMF)
    if nti.CMF in config:
        df["volume_cmf"] = ChaikinMoneyFlowIndicator(
            high=high, low=low, close=close, volume=volume, window=20, fillna=fillna
        ).chaikin_money_flow()
    
    # Force Index (FI)
    if nti.FI in config:
        df["volume_fi"] = ForceIndexIndicator(
            close=close, volume=volume, window=13, fillna=fillna
        ).force_index()
    
    # Ease of movement (EoM, EMV)
    if nti.EoM in config:
        ease_of_movement_indicator = EaseOfMovementIndicator(
            high=high, low=low, volume=volume, window=14, fillna=fillna
        )
        df[f"volume_em"] = ease_of_movement_indicator.ease_of_movement()
        df[f"volume_sma_em"] = ease_of_movement_indicator.sma_ease_of_movement()

    # Volume-price trend (VPT)
    if nti.VPT in config:
        df["volume_vpt"] = VolumePriceTrendIndicator(
            close=close, volume=volume, fillna=fillna
        ).volume_price_trend()

    # Volume Weighted Average Price (VWAP)
    if nti.VWAP in config:
        df["volume_vwap"] = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
        ).volume_weighted_average_price()
    
    # Money Flow Index (MFI)
    if nti.MFI in config:
        df["volume_mfi"] = MFIIndicator(
            high=high, low=low, close=close, volume=volume, window=14, fillna=fillna
        ).money_flow_index()
        
    # Negative Volume Index (NVI)
    if nti.NVI in config:
        df["volume_nvi"] = NegativeVolumeIndexIndicator(
            close=close, volume=volume, fillna=fillna
        ).negative_volume_index()
        
    # Volatility Indicators ---------------------------------------------------------------------------------------------------
    # Bollinger Bands
    if nti.BB in config:
        bollinger_bands = BollingerBands(
            close=close, window=20, window_dev=2, fillna=fillna
        )
        df["volatility_bbh"] = bollinger_bands.bollinger_hband()
        df["volatility_bbhi"] = bollinger_bands.bollinger_hband_indicator()
        df["volatility_bbl"] = bollinger_bands.bollinger_lband()
        df["volatility_bbli"] = bollinger_bands.bollinger_lband_indicator()
        df["volatility_bbm"] = bollinger_bands.bollinger_mavg()
        df["volatility_bbp"] = bollinger_bands.bollinger_pband()
        df["volatility_bbw"] = bollinger_bands.bollinger_wband()
    
    # Donchian Channel
    if nti.DC in config:
        donchian_channel = DonchianChannel(
            high=high, low=low, close=close, window=20, offset=0, fillna=0
        )
        df["volatility_dch"] = donchian_channel.donchian_channel_hband()
        df["volatility_dcl"] = donchian_channel.donchian_channel_lband()
        df["volatility_dcm"] = donchian_channel.donchian_channel_mband()
        df["volatility_dcp"] = donchian_channel.donchian_channel_pband()
        df["volatility_dcw"] = donchian_channel.donchian_channel_wband()
    
    # Keltner Channel -> Infinity value
    if nti.KC in config:
        keltner_channel = KeltnerChannel(
            high=high, low=low, close=close, window=20, window_atr=10, fillna=fillna, original_version=True, multiplier=2
        )
        df["volatility_kch"] = keltner_channel.keltner_channel_hband()
        df["volatility_kchi"] = keltner_channel.keltner_channel_hband_indicator()
        df["volatility_kcl"] = keltner_channel.keltner_channel_lband()
        df["volatility_kcli"] = keltner_channel.keltner_channel_lband_indicator()
        df["volatility_kcm"] = keltner_channel.keltner_channel_mband()
        df["volatility_kcp"] = keltner_channel.keltner_channel_pband()
        df["volatility_kcw"] = keltner_channel.keltner_channel_wband()
    
    # Average True Range (ATR)
    if nti.ATR in config:
        df["volatility_atr"] = AverageTrueRange(
            high=high, low=low, close=close, window=14, fillna=fillna
        ).average_true_range()
        
    # Ulcer Index
    if nti.UI in config:
        df["volatility_ui"] = UlcerIndex(
            close=close, window=14, fillna=fillna
        ).ulcer_index()
    
    # Trend Indicators ---------------------------------------------------------------------------------------------------
    # Moving Average Convergence Divergence (MACD)
    if nti.MACD in config:
        macd = MACD(
            close=close, window_slow=26, window_fast=12, window_sign=9, fillna=fillna
        )
        df["trend_macd"] = macd.macd()
        df["trend_macd_diff"] = macd.macd_diff()
        df["trend_macd_signal"] = macd.macd_signal()

    # Moving Average
    if nti.MA in config:
        for i in windows:
            # Simple Moving Average (SMA)
            df[f"trend_sma_{i}"] = SMAIndicator(
                close=close, window=i, fillna=fillna
            ).sma_indicator()
        
            # Exponential Moving Average (EMA)
            df[f"trend_ema_{i}"] = EMAIndicator(
                close=close, window=i, fillna=fillna
            ).ema_indicator()
            
            # Weighted Moving Average (WMA)
            df[f"trend_wma_{i}"] = WMAIndicator(
                close=close, window=i, fillna=fillna
            ).wma()

    # Vortex Indicator (VI)
    if nti.VI in config:
        vortex_indicator = VortexIndicator(
            high=high, low=low, close=close, window=14, fillna=fillna
        )
        df["trend_vortex_ind_diff"] = vortex_indicator.vortex_indicator_diff()
        df["trend_vortex_ind_pos"] = vortex_indicator.vortex_indicator_pos()
        df["trend_vortex_ind_neg"] = vortex_indicator.vortex_indicator_neg()
 
    # Tripple Exponential Smoothed Moving Average (TRIX)
    if nti.TRIX in config:
        df["trend_trix"] = TRIXIndicator(
            close=close, window=15, fillna=fillna
        ).trix()
 
    # Mass Index (MI)
    if nti.MI in config:
        df["trend_mass_index"] = MassIndex(
            high=high, low=low, window_fast=9, window_slow=25, fillna=fillna
        ).mass_index()
    
    # Detrended Price Oscillator (DPO)
    if nti.DPO in config:
        df["trend_dpo"] = DPOIndicator(
            close=close, window=20, fillna=fillna
        ).dpo()
    
    # KST Oscillator (KST Signal)
    if nti.KSTs in config:
        kst_indicator = KSTIndicator(
            close=close, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9, fillna=fillna
        )
        df["trend_kst"] = kst_indicator.kst()
        df["trend_kst_diff"] = kst_indicator.kst_diff()
        df["trend_kst_sig"] = kst_indicator.kst_sig()
    
    # Ichimoku Kinkō Hyō (Ichimoku)
    if nti.ICHIMOKU in config:
        ichimoku_indicator = IchimokuIndicator(
            high=high, low=low, window1=9, window2=26, window3=52, visual=False, fillna=False
        )
        df["trend_ichimoku_a"] = ichimoku_indicator.ichimoku_a()
        df["trend_ichimoku_b"] = ichimoku_indicator.ichimoku_b()
        df["trend_ichimoku_base"] = ichimoku_indicator.ichimoku_base_line()
        df["trend_ichimoku_conv"] = ichimoku_indicator.ichimoku_conversion_line()
    
    # Schaff Trend Cycle (STC)
    if nti.STC in config:
        df["trend_stc"] = STCIndicator(
            close=close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=fillna
        ).stc()
    
    # # Average Directional Movement Index (ADX)
    if nti.ADX in config:
        average_directional_movement_indicator = ADXIndicator(
            high=high, low=low, close=close, window=14, fillna=fillna
        )
        df["trend_adx"] = average_directional_movement_indicator.adx()
        df["trend_adx_pos"] = average_directional_movement_indicator.adx_pos()
        df["trend_adx_neg"] = average_directional_movement_indicator.adx_neg()
        
    # Commodity Channel Index (CCI)
    if nti.CCI in config:
        df["trend_cci"] = CCIIndicator(
            high=high, low=low, close=close, window=20, constant=0.015, fillna=fillna
        ).cci()
    
    # Aroon Indicator
    if nti.AI in config:
        aroon_indicator = AroonIndicator(
            close=close, window=25, fillna=fillna
        )
        df["trend_aroon_up"] = aroon_indicator.aroon_up()
        df["trend_aroon_down"] = aroon_indicator.aroon_down()
        df["trend_aroon_ind"] = aroon_indicator.aroon_indicator()
    
    # Parabolic Stop and Reverse (Parabolic SAR)
    if nti.PSAR in config:
        psar_indicator = PSARIndicator(
            high=high, low=low, close=close, step=0.02, max_step=0.2, fillna=fillna
        )
        df["trend_psar"] = psar_indicator.psar()
        df["trend_psar_up"] = psar_indicator.psar_up()
        df["trend_psar_up_indicator"] = psar_indicator.psar_up_indicator()
        df["trend_psar_down"] = psar_indicator.psar_down()
        df["trend_psar_down_indicator"] = psar_indicator.psar_down_indicator()
        df[["trend_psar_up", "trend_psar_down"]] = df[["trend_psar_up", "trend_psar_down"]].fillna(0) # psar 컬럼 null 채우기
    
    # Others Indicators ---------------------------------------------------------------------------------------------------
    # Daily Return (DR)
    if nti.DR in config:
        df["daily_return"] = DailyReturnIndicator(
            close=close, fillna=fillna
        ).daily_return()
        
    # Daily Log Return (DLR)
    if nti.DLR in config:
        df["daily_log_return"] = DailyLogReturnIndicator(
            close=close, fillna=fillna
        ).daily_log_return()

    # Cumulative Return (CR)
    if nti.CR in config:
        df["cumlative_return"] = CumulativeReturnIndicator(
            close=close, fillna=fillna
        ).cumulative_return()
    
    # Daily Maximum Return (DMR)
    if nti.DMR in config:
        df["daily_maximum_return"] = (df["high"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    return df