"""
内存版本的风险指标计算器（不依赖CSV文件）

该模块实现了不依赖CSV文件的风险指标计算，直接从内存中的交易结果计算风险指标。
避免文件I/O操作，大幅提升性能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def calculate_risk_metrics_from_results(
    all_results: Dict[str, Dict],  # {date: {stock_code: {closing_price, volume, transactions}}}
    stock_weights: Optional[Dict[str, float]] = None,
    metric_name: str = 'risk_indicator_simple',
    min_volume_threshold: Optional[int] = None  # 最小成交量阈值，None表示不使用阈值（保持原有行为）
) -> float:
    """
    直接从内存中的交易结果计算风险指标（不依赖CSV文件）
    
    Args:
        all_results: 所有日期的交易结果，格式：
            {
                'date1': {
                    'stock_code1': {'closing_price': float, 'volume': int, 'transactions': [...]},
                    'stock_code2': {...},
                    ...
                },
                'date2': {...},
                ...
            }
        stock_weights: 股票权重字典 {stock_code: weight}
        metric_name: 风险指标名称
        
    Returns:
        风险指标值
    """
    if not all_results:
        return 0.0
    
    # 按日期排序
    sorted_dates = sorted(all_results.keys())
    
    # 构建每日市场指标
    daily_rows = []
    for date in sorted_dates:
        date_results = all_results[date]
        
        # 收集所有交易
        all_transactions = []
        stock_closing_prices = {}
        
        for stock_code, result in date_results.items():
            stock_closing_prices[stock_code] = result['closing_price']
            if 'transactions' in result:
                all_transactions.extend(result['transactions'])
        
        # 即使没有交易，也应该使用收盘价计算市场平均价格
        # 这对于baseline策略（hold，不交易）很重要
        df_trans = pd.DataFrame(all_transactions) if all_transactions else pd.DataFrame()
        
        # 计算市场平均价格
        # 如果有交易，使用VWAP；如果没有交易，使用收盘价
        total_volume = df_trans["executed_quantity"].sum() if not df_trans.empty else 0
        total_turnover = (df_trans["executed_price"] * df_trans["executed_quantity"]).sum() if not df_trans.empty else 0
        
        # 计算每只股票的VWAP（如果有交易）
        # 注意：如果交易量太小，VWAP可能不稳定，应该优先使用收盘价
        stock_vwap = {}
        
        if not df_trans.empty and 'stock_code' in df_trans.columns and total_volume > 0:
            for stock_code in df_trans['stock_code'].unique():
                stock_trans = df_trans[df_trans['stock_code'] == stock_code]
                stock_volume = stock_trans['executed_quantity'].sum()
                stock_turnover = (stock_trans['executed_price'] * stock_trans['executed_quantity']).sum()
                # 如果指定了最小成交量阈值，只有当成交量足够大时才使用VWAP
                # 否则使用收盘价（避免少量交易导致的价格波动放大）
                if min_volume_threshold is None or stock_volume > min_volume_threshold:
                    if stock_volume > 0:
                        stock_vwap[stock_code] = stock_turnover / stock_volume
        
        # 计算市场平均价格（优先使用VWAP+权重加权，但需要足够的交易量）
        market_avg_price = None
        
        if stock_vwap and stock_weights:
            vwap_list = []
            weight_list = []
            for stock_code, vwap_value in stock_vwap.items():
                if stock_code in stock_weights:
                    vwap_list.append(vwap_value)
                    weight_list.append(stock_weights[stock_code])
            
            if vwap_list and weight_list:
                total_weight = sum(weight_list)
                if total_weight > 0:
                    market_avg_price = sum(v * w for v, w in zip(vwap_list, weight_list)) / total_weight
        
        # 如果VWAP+权重不可用（交易量太小或没有交易），使用收盘价+权重
        if market_avg_price is None and stock_closing_prices:
            if stock_weights:
                weighted_prices = []
                total_weight = 0
                for stock_code, price in stock_closing_prices.items():
                    if stock_code in stock_weights:
                        weighted_prices.append(price * stock_weights[stock_code])
                        total_weight += stock_weights[stock_code]
                if total_weight > 0:
                    market_avg_price = sum(weighted_prices) / total_weight
            
            # 如果还是None，使用成交量加权
            if market_avg_price is None:
                total_vol = sum(result.get('volume', 0) for result in date_results.values())
                if total_vol > 0:
                    weighted_sum = sum(
                        result.get('closing_price', 0) * result.get('volume', 0)
                        for result in date_results.values()
                    )
                    market_avg_price = weighted_sum / total_vol
            
            # 最后使用简单平均
            if market_avg_price is None:
                prices = [p for p in stock_closing_prices.values() if p > 0]
                if prices:
                    market_avg_price = sum(prices) / len(prices)
        
        if market_avg_price is None or market_avg_price <= 0:
            continue
        
        daily_rows.append({
            'date': date,
            'market_avg_price': market_avg_price
        })
    
    if not daily_rows:
        return 0.0
    
    # 构建DataFrame
    daily_df = pd.DataFrame(daily_rows)
    
    # 计算价格指数
    if 'market_avg_price' in daily_df.columns:
        valid_prices = daily_df['market_avg_price'].dropna()
        valid_prices = valid_prices[valid_prices > 0]
        
        if not valid_prices.empty:
            base_price = valid_prices.iloc[0]
            daily_df['price_index'] = (daily_df['market_avg_price'] / base_price) * 100
            
            # 计算对数收益率
            price_index_positive = daily_df['price_index'].where(daily_df['price_index'] > 0)
            log_price_index = np.log(price_index_positive)
            daily_df['price_index_log_return'] = log_price_index.diff()
            
            # 计算风险指标（risk_indicator_simple）
            if metric_name == 'risk_indicator_simple':
                lambda_risk = 0.94
                pi_t = daily_df['price_index_log_return'].copy()
                
                if len(pi_t.dropna()) > 0:
                    # 简单预期：E_{t-1}[π_t] = π_{t-1}
                    expected_pi_simple = pi_t.shift(1)
                    forecast_error_simple = pi_t - expected_pi_simple
                    
                    # 递归计算风险指标：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
                    h_simple = pd.Series(index=pi_t.index, dtype=float)
                    e_squared_simple = forecast_error_simple ** 2
                    
                    for i in range(len(pi_t)):
                        current_idx = pi_t.index[i]
                        if pd.notna(e_squared_simple.loc[current_idx]):
                            if i == 0 or pd.isna(h_simple.loc[pi_t.index[i-1]]):
                                h_simple.loc[current_idx] = e_squared_simple.loc[current_idx]
                            else:
                                prev_idx = pi_t.index[i-1]
                                h_simple.loc[current_idx] = (
                                    lambda_risk * h_simple.loc[prev_idx] + 
                                    (1 - lambda_risk) * e_squared_simple.loc[prev_idx]
                                )
                    
                    daily_df['risk_indicator_simple'] = h_simple
                    
                    # 返回目标日期的风险指标值
                    if len(daily_df) > 0:
                        last_risk = daily_df['risk_indicator_simple'].iloc[-1]
                        if pd.notna(last_risk):
                            return float(last_risk)
    
    return 0.0

