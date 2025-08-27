import logging
import importlib
import numpy as np
import pandas as pd
from ai_trading.utils.base import safe_to_datetime

logger = logging.getLogger(__name__)
MFI_PERIOD = 14

def prepare_indicators(df: pd.DataFrame, freq: str='daily') -> pd.DataFrame:
    ta = importlib.import_module('pandas_ta')
    df = df.copy()
    rename_map = {}
    variants = {'high': ['High', 'HIGH', 'H', 'h'], 'low': ['Low', 'LOW', 'L', 'l'], 'close': ['Close', 'CLOSE', 'C', 'c'], 'open': ['Open', 'OPEN', 'O', 'o'], 'volume': ['Volume', 'VOLUME', 'V', 'v']}
    for std, cols in variants.items():
        for col in cols:
            if col in df.columns:
                rename_map[col] = std
    if rename_map:
        df = df.rename(columns=rename_map)
    for col in ['high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame in prepare_indicators")
        df[col] = df[col].astype(float)
    if 'open' in df.columns:
        df['open'] = df['open'].astype(float)
    idx = safe_to_datetime(df.index, context='retrain index')
    if idx.empty:
        raise ValueError('Invalid date values in dataframe')
    df = df.sort_index()
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume']).astype(float)
    df.dropna(subset=['vwap'], inplace=True)
    df['rsi'] = ta.rsi(df['close'], length=14).astype('float64')
    df.dropna(subset=['rsi'], inplace=True)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).astype('float64')
    df.dropna(subset=['atr'], inplace=True)
    df['kc_lower'] = np.nan
    df['kc_mid'] = np.nan
    df['kc_upper'] = np.nan
    try:
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        df['kc_lower'] = kc.iloc[:, 0].astype(float)
        df['kc_mid'] = kc.iloc[:, 1].astype(float)
        df['kc_upper'] = kc.iloc[:, 2].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('KC indicator failed: %s', e)
    df['atr_band_upper'] = np.nan
    df['atr_band_lower'] = np.nan
    df['avg_vol_20'] = np.nan
    df['dow'] = np.nan
    df['atr_band_upper'] = (df['close'] + 1.5 * df['atr']).astype(float)
    df['atr_band_lower'] = (df['close'] - 1.5 * df['atr']).astype(float)
    df['avg_vol_20'] = df['volume'].rolling(20).mean().astype(float)
    if len(idx) == len(df):
        df['dow'] = idx.dayofweek.astype(float)
    df['macd'] = np.nan
    df['macds'] = np.nan
    try:
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9'].astype(float)
        df['macds'] = macd['MACDs_12_26_9'].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('MACD calculation failed: %s', e)
    df['bb_upper'] = np.nan
    df['bb_lower'] = np.nan
    df['bb_percent'] = np.nan
    try:
        bb = ta.bbands(df['close'], length=20)
        df['bb_upper'] = bb['BBU_20_2.0'].astype(float)
        df['bb_lower'] = bb['BBL_20_2.0'].astype(float)
        df['bb_percent'] = bb['BBP_20_2.0'].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('Bollinger Bands failed: %s', e)
    df['adx'] = np.nan
    df['dmp'] = np.nan
    df['dmn'] = np.nan
    try:
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx['ADX_14'].astype(float)
        df['dmp'] = adx['DMP_14'].astype(float)
        df['dmn'] = adx['DMN_14'].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('ADX calculation failed: %s', e)
    df['cci'] = np.nan
    try:
        if hasattr(ta, 'cci'):
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20).astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('CCI calculation failed: %s', e)
    try:
        mfi_vals = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=MFI_PERIOD).astype(float)
        df['mfi_14'] = mfi_vals
        df.dropna(subset=['mfi_14'], inplace=True)
    except (ValueError, TypeError) as e:
        logger.exception('MFI calculation failed: %s', e)
        df['mfi_14'] = np.nan
    df['tema'] = np.nan
    try:
        if hasattr(ta, 'tema'):
            df['tema'] = ta.tema(df['close'], length=10).astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('TEMA calculation failed: %s', e)
    df['willr'] = np.nan
    try:
        if hasattr(ta, 'willr'):
            df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14).astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('Williams %%R calculation failed: %s', e)
    df['psar_long'] = np.nan
    df['psar_short'] = np.nan
    try:
        psar = ta.psar(df['high'], df['low'], df['close'])
        df['psar_long'] = psar['PSARl_0.02_0.2'].astype(float)
        df['psar_short'] = psar['PSARs_0.02_0.2'].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('PSAR calculation failed: %s', e)
    df['ichimoku_conv'] = np.nan
    df['ichimoku_base'] = np.nan
    try:
        ich = ta.ichimoku(high=df['high'], low=df['low'], close=df['close'])
        conv = ich[0] if isinstance(ich, tuple) else ich.iloc[:, 0]
        base = ich[1] if isinstance(ich, tuple) else ich.iloc[:, 1]
        df['ichimoku_conv'] = (conv.iloc[:, 0] if hasattr(conv, 'iloc') else conv).astype(float)
        df['ichimoku_base'] = (base.iloc[:, 0] if hasattr(base, 'iloc') else base).astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('Ichimoku calculation failed: %s', e)
    df['stochrsi'] = np.nan
    try:
        st = ta.stochrsi(df['close'])
        df['stochrsi'] = st['STOCHRSIk_14_14_3_3'].astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('StochRSI calculation failed: %s', e)
    df['ret_5m'] = np.nan
    df['ret_1h'] = np.nan
    df['ret_d'] = np.nan
    df['ret_w'] = np.nan
    df['vol_norm'] = np.nan
    df['5m_vs_1h'] = np.nan
    df['vol_5m'] = np.nan
    df['vol_1h'] = np.nan
    df['vol_d'] = np.nan
    df['vol_w'] = np.nan
    df['vol_ratio'] = np.nan
    df['mom_agg'] = np.nan
    df['lag_close_1'] = np.nan
    df['lag_close_3'] = np.nan
    try:
        df['ret_5m'] = df['close'].pct_change(5, fill_method=None).astype(float)
        df['ret_1h'] = df['close'].pct_change(60, fill_method=None).astype(float)
        df['ret_d'] = df['close'].pct_change(390, fill_method=None).astype(float)
        df['ret_w'] = df['close'].pct_change(1950, fill_method=None).astype(float)
        df['vol_norm'] = (df['volume'].rolling(60).mean() / df['volume'].rolling(5).mean()).astype(float)
        df['5m_vs_1h'] = (df['ret_5m'] - df['ret_1h']).astype(float)
        df['vol_5m'] = df['close'].pct_change(fill_method=None).rolling(5).std().astype(float)
        df['vol_1h'] = df['close'].pct_change(fill_method=None).rolling(60).std().astype(float)
        df['vol_d'] = df['close'].pct_change(fill_method=None).rolling(390).std().astype(float)
        df['vol_w'] = df['close'].pct_change(fill_method=None).rolling(1950).std().astype(float)
        df['vol_ratio'] = (df['vol_5m'] / df['vol_1h']).astype(float)
        df['mom_agg'] = (df['ret_5m'] + df['ret_1h'] + df['ret_d']).astype(float)
        df['lag_close_1'] = df['close'].shift(1).astype(float)
        df['lag_close_3'] = df['close'].shift(3).astype(float)
    except (ValueError, TypeError) as e:
        logger.exception('Multi-timeframe features failed: %s', e)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    required = ['vwap', 'rsi', 'atr', 'macd', 'macds', 'ichimoku_conv', 'ichimoku_base', 'stochrsi']
    if freq == 'daily':
        df['sma_50'] = np.nan
        df['sma_200'] = np.nan
        try:
            if hasattr(ta, 'sma'):
                df['sma_50'] = ta.sma(df['close'], length=50).astype(float)
                df['sma_200'] = ta.sma(df['close'], length=200).astype(float)
        except (ValueError, TypeError) as e:
            logger.exception('SMA calculation failed: %s', e)
        required += ['sma_50', 'sma_200']
    df.dropna(subset=required, how='all', inplace=True)
    if freq != 'daily':
        df.reset_index(drop=True, inplace=True)
    return df

__all__ = ['prepare_indicators']
