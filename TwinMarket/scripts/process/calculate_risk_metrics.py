"""
TwinMarket 风险指标计算脚本

该脚本从 TwinMarket 模拟系统的日志数据中计算各种风险指标
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3


class RiskMetricsCalculator:
    """风险指标计算器"""
    
    def __init__(self, log_dir: str, stock_profile_path: Optional[str] = None):
        """
        初始化风险指标计算器
        
        Args:
            log_dir: 日志目录路径
            stock_profile_path: 股票配置文件路径（包含权重信息）
        """
        self.log_dir = Path(log_dir)
        self.trading_records_dir = self.log_dir / "trading_records"
        self.simulation_results_dir = self.log_dir / "simulation_results"
        
        # 加载股票权重信息（如果提供）
        self.stock_weights = {}
        if stock_profile_path and Path(stock_profile_path).exists():
            try:
                df_profile = pd.read_csv(stock_profile_path)
                if 'stock_id' in df_profile.columns and 'weight' in df_profile.columns:
                    self.stock_weights = dict(zip(df_profile['stock_id'], df_profile['weight']))
            except Exception as e:
                print(f"警告: 无法加载股票权重文件: {e}")
        
    def load_daily_summary(self, date: str) -> pd.DataFrame:
        """加载每日摘要数据"""
        summary_file = self.simulation_results_dir / date / f"daily_summary_{date}.csv"
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()
    
    def load_transactions(self, date: str) -> pd.DataFrame:
        """加载交易数据"""
        trans_file = self.simulation_results_dir / date / f"transactions_{date}.csv"
        if trans_file.exists():
            df = pd.read_csv(trans_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    
    def load_orders(self, date: str) -> Dict:
        """加载订单数据"""
        orders_file = self.trading_records_dir / f"{date}_orders.json"
        if orders_file.exists():
            with open(orders_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def calculate_market_metrics(
        self, dates: List[str]
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        计算市场级指标（按交易日聚合）
        如果某个交易日当天没有交易数据，使用上一个交易日的数据填充价格相关字段

        Returns:
            daily_df: 每日市场指标
            summary_df: 包含市场整体波动率、最大回撤的汇总
        """
        rows = []
        prev_row = None  # 保存上一个交易日的数据，用于填充没有交易数据的交易日

        for date in sorted(dates):
            df_trans = self.load_transactions(date)
            df_summary = self.load_daily_summary(date)
            
            # 如果当天没有交易数据，使用上一个交易日的数据填充价格相关字段
            if df_trans.empty:
                if prev_row is None:
                    # 如果第一个交易日就没有数据，跳过（因为没有前一个交易日的数据）
                    continue
                
                # 使用上一个交易日的数据填充价格相关字段
                row = {
                    "date": date,
                    "vwap": prev_row.get("vwap"),  # 使用上一个交易日的 VWAP
                    "market_avg_price": prev_row.get("market_avg_price"),  # 使用上一个交易日的市场平均价格
                    "market_avg_price_vwap_weighted": prev_row.get("market_avg_price_vwap_weighted"),  # 使用上一个交易日的 VWAP+权重
                    "total_volume": 0,  # 没有交易，成交量为0
                    "total_turnover": 0.0,  # 没有交易，成交额为0
                    "buy_volume_exec": 0,
                    "sell_volume_exec": 0,
                    "buy_sell_imbalance_pct_exec": 0.0,
                    "buy_volume_intent": 0,
                    "sell_volume_intent": 0,
                    "buy_sell_imbalance_pct": 0.0,
                    "unfilled_rate_pct": 0.0,
                    "large_order_ratio_pct": 0.0,
                }
                rows.append(row)
                prev_row = row  # 更新 prev_row，以便下一个交易日使用
                continue

            total_volume = df_trans["executed_quantity"].sum()
            total_turnover = (df_trans["executed_price"] * df_trans["executed_quantity"]).sum()
            vwap = total_turnover / total_volume if total_volume > 0 else None
            
            # 计算每只股票的 VWAP（用于权重加权 VWAP）
            stock_vwap = {}
            if not df_trans.empty and 'stock_code' in df_trans.columns:
                for stock_code in df_trans['stock_code'].unique():
                    stock_trans = df_trans[df_trans['stock_code'] == stock_code]
                    stock_volume = stock_trans['executed_quantity'].sum()
                    stock_turnover = (stock_trans['executed_price'] * stock_trans['executed_quantity']).sum()
                    if stock_volume > 0:
                        stock_vwap[stock_code] = stock_turnover / stock_volume
            
            # 计算市场平均价格，支持多种方法：
            # 方法1: VWAP + 权重加权（推荐）：结合实际交易价格和股票重要性
            # 方法2: 收盘价 + 权重加权：使用收盘价，更稳定
            # 方法3: 收盘价 + 成交量加权：反映交易活跃度
            # 方法4: 收盘价简单平均：所有股票等权重
            
            market_avg_price = None
            market_avg_price_vwap_weighted = None  # VWAP + 权重加权
            
            # 方法1: VWAP + 权重加权（如果可用）
            if stock_vwap and self.stock_weights:
                vwap_list = []
                weight_list = []
                for stock_code, vwap_value in stock_vwap.items():
                    if stock_code in self.stock_weights:
                        vwap_list.append(vwap_value)
                        weight_list.append(self.stock_weights[stock_code])
                
                if vwap_list and weight_list:
                    total_weight = sum(weight_list)
                    if total_weight > 0:
                        market_avg_price_vwap_weighted = sum(v * w for v, w in zip(vwap_list, weight_list)) / total_weight
                        market_avg_price = market_avg_price_vwap_weighted  # 优先使用 VWAP+权重
            
            # 方法2: 收盘价 + 权重加权（如果 VWAP+权重不可用）
            if market_avg_price is None and not df_summary.empty and 'closing_price' in df_summary.columns:
                if self.stock_weights and 'stock_code' in df_summary.columns:
                    weights = df_summary['stock_code'].map(self.stock_weights)
                    if weights.notna().any():
                        total_weight = weights.sum()
                        if total_weight > 0:
                            market_avg_price = (df_summary['closing_price'] * weights).sum() / total_weight
                
                # 方法3: 收盘价 + 成交量加权
                if market_avg_price is None and 'volume' in df_summary.columns:
                    total_volume_summary = df_summary['volume'].sum()
                    if total_volume_summary > 0:
                        market_avg_price = (df_summary['closing_price'] * df_summary['volume']).sum() / total_volume_summary
                
                # 方法4: 收盘价简单平均
                if market_avg_price is None:
                    market_avg_price = df_summary['closing_price'].mean()
            
            # 如果仍然无法计算价格，使用上一个交易日的数据
            if market_avg_price is None and prev_row is not None:
                market_avg_price = prev_row.get("market_avg_price")
                market_avg_price_vwap_weighted = prev_row.get("market_avg_price_vwap_weighted")
                if vwap is None:
                    vwap = prev_row.get("vwap")

            # 成交通道：买卖方向的成交量（匹配后总是平衡，仅保留参考）
            buy_volume_exec = df_trans[df_trans["direction"] == "buy"]["executed_quantity"].sum()
            sell_volume_exec = df_trans[df_trans["direction"] == "sell"]["executed_quantity"].sum()
            imbalance_pct_exec = (
                (buy_volume_exec - sell_volume_exec) / total_volume * 100 if total_volume > 0 else 0.0
            )

            # 订单意图：使用原始下单量衡量买卖失衡，避免撮合平衡导致的恒为0
            buy_volume_intent = df_trans[df_trans["direction"] == "buy"]["original_quantity"].sum()
            sell_volume_intent = df_trans[df_trans["direction"] == "sell"]["original_quantity"].sum()
            total_intent = buy_volume_intent + sell_volume_intent
            imbalance_pct_intent = (
                (buy_volume_intent - sell_volume_intent) / total_intent * 100 if total_intent > 0 else 0.0
            )

            total_original = df_trans["original_quantity"].sum()
            total_unfilled = df_trans["unfilled_quantity"].sum()
            unfilled_rate_pct = (
                (total_unfilled / total_original) * 100 if total_original > 0 else 0.0
            )

            avg_trade_size = df_trans["executed_quantity"].mean()
            large_threshold = avg_trade_size * 2 if pd.notna(avg_trade_size) else 0
            if large_threshold > 0:
                large_orders = df_trans[df_trans["executed_quantity"] >= large_threshold]
                large_order_ratio_pct = (
                    large_orders["executed_quantity"].sum() / total_volume * 100
                    if total_volume > 0
                    else 0.0
                )
            else:
                large_order_ratio_pct = 0.0

            row = {
                "date": date,
                "vwap": vwap,  # 整体市场 VWAP（所有股票混合）
                "market_avg_price": market_avg_price,  # 市场平均价格（优先使用 VWAP+权重，否则收盘价+权重）
                "market_avg_price_vwap_weighted": market_avg_price_vwap_weighted,  # VWAP+权重加权（如果可用）
                "total_volume": total_volume,
                "total_turnover": total_turnover,
                "buy_volume_exec": buy_volume_exec,
                "sell_volume_exec": sell_volume_exec,
                "buy_sell_imbalance_pct_exec": imbalance_pct_exec,
                "buy_volume_intent": buy_volume_intent,
                "sell_volume_intent": sell_volume_intent,
                "buy_sell_imbalance_pct": imbalance_pct_intent,
                "unfilled_rate_pct": unfilled_rate_pct,
                "large_order_ratio_pct": large_order_ratio_pct,
            }
            rows.append(row)
            prev_row = row  # 保存当前交易日的数据，供下一个交易日使用

        daily_df = pd.DataFrame(rows)
        if daily_df.empty:
            return daily_df, pd.DataFrame()

        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.sort_values("date")

        # 计算价格指数（以首日为基准=100）
        # 优先使用 market_avg_price，它按以下优先级选择：
        # 1. VWAP + 权重加权：结合实际交易价格（VWAP）和股票重要性（权重）
        # 2. 收盘价 + 权重加权：使用收盘价，更稳定
        # 3. 收盘价 + 成交量加权：反映交易活跃度
        # 4. 收盘价简单平均：所有股票等权重
        price_column = "market_avg_price" if "market_avg_price" in daily_df.columns else "vwap"
        
        if not daily_df.empty and price_column in daily_df.columns:
            # 找到第一个有效的价格值作为基准
            valid_prices = daily_df[price_column].dropna()
            valid_prices = valid_prices[valid_prices > 0]
            
            if not valid_prices.empty:
                base_price = valid_prices.iloc[0]
                # 计算价格指数：所有值相对于基准值归一化，基准值对应 100
                daily_df["price_index"] = (daily_df[price_column] / base_price) * 100
                # 对于无效的价格值，price_index 也会是 NaN（这是预期的）
            else:
                # 如果没有有效的价格值，price_index 全部设为 None
                daily_df["price_index"] = None
        else:
            daily_df["price_index"] = None

        # 市场波动率与最大回撤基于价格序列（使用与 price_index 相同的价格源）
        price_series = daily_df[price_column].dropna()
        # 每日市场收益率（%）
        daily_df["market_return_pct"] = daily_df[price_column].pct_change() * 100
        daily_df["market_abs_return_pct"] = daily_df["market_return_pct"].abs()
        
        # 计算价格指数的对数收益率：πt = log(Pt) - log(Pt-1)
        # 这是风险指标变换，用于衡量价格指数的对数变化率
        if "price_index" in daily_df.columns:
            # 确保价格指数都大于0（对数需要正数）
            price_index_positive = daily_df["price_index"].where(daily_df["price_index"] > 0)
            # 计算对数：log(Pt)
            log_price_index = np.log(price_index_positive)
            # 计算对数收益率：log(Pt) - log(Pt-1)
            daily_df["price_index_log_return"] = log_price_index.diff()
            # 也可以计算为百分比形式（乘以100）
            daily_df["price_index_log_return_pct"] = daily_df["price_index_log_return"] * 100
            
            # 计算风险归因指标（基于 Engle 1982 和 Bollerslev 1986）
            # 参数设置
            lambda_risk = 0.94  # RiskMetrics 标准值
            rolling_window = 5  # rolling mean 的窗口大小（可以根据需要调整）
            
            pi_t = daily_df["price_index_log_return"].copy()
            
            if len(pi_t.dropna()) > 0:
                # 规则1：E_{t-1}[π_t] = (1/k) * Σ_{i=1}^k π_{t-i} （rolling mean）
                # 在 t-1 时刻，使用 t-1 之前 k 个值的平均值作为对 t 时刻的预期
                expected_pi_rolling = pi_t.rolling(window=rolling_window, min_periods=1).mean().shift(1)
                # 预测误差：e_t = π_t - E_{t-1}[π_t]
                forecast_error_rolling = pi_t - expected_pi_rolling
                # 风险指标：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
                # 使用递归计算，初始化 h_0 = e_0^2（如果存在）
                h_rolling = pd.Series(index=pi_t.index, dtype=float)
                e_squared_rolling = forecast_error_rolling ** 2
                for i in range(len(pi_t)):
                    current_idx = pi_t.index[i]
                    if pd.notna(e_squared_rolling.loc[current_idx]):
                        if i == 0 or pd.isna(h_rolling.loc[pi_t.index[i-1]]):
                            # 初始化：h_0 = e_0^2
                            h_rolling.loc[current_idx] = e_squared_rolling.loc[current_idx]
                        else:
                            # 递归：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
                            prev_idx = pi_t.index[i-1]
                            h_rolling.loc[current_idx] = (
                                lambda_risk * h_rolling.loc[prev_idx] + 
                                (1 - lambda_risk) * e_squared_rolling.loc[prev_idx]
                            )
                
                # 规则2：E_{t-1}[π_t] = π_{t-1} （简单预期：今天就是明天）
                expected_pi_simple = pi_t.shift(1)
                # 预测误差：e_t = π_t - E_{t-1}[π_t]
                forecast_error_simple = pi_t - expected_pi_simple
                # 风险指标：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
                h_simple = pd.Series(index=pi_t.index, dtype=float)
                e_squared_simple = forecast_error_simple ** 2
                for i in range(len(pi_t)):
                    current_idx = pi_t.index[i]
                    if pd.notna(e_squared_simple.loc[current_idx]):
                        if i == 0 or pd.isna(h_simple.loc[pi_t.index[i-1]]):
                            # 初始化：h_0 = e_0^2
                            h_simple.loc[current_idx] = e_squared_simple.loc[current_idx]
                        else:
                            # 递归：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
                            prev_idx = pi_t.index[i-1]
                            h_simple.loc[current_idx] = (
                                lambda_risk * h_simple.loc[prev_idx] + 
                                (1 - lambda_risk) * e_squared_simple.loc[prev_idx]
                            )
                
                # 将结果合并到 daily_df
                daily_df["expected_pi_rolling"] = expected_pi_rolling
                daily_df["forecast_error_rolling"] = forecast_error_rolling
                daily_df["risk_indicator_rolling"] = h_rolling
                
                daily_df["expected_pi_simple"] = expected_pi_simple
                daily_df["forecast_error_simple"] = forecast_error_simple
                daily_df["risk_indicator_simple"] = h_simple

        if len(price_series) >= 2:
            returns = price_series.pct_change().dropna()
            market_vol = returns.std() * np.sqrt(252) if not returns.empty else np.nan
        else:
            market_vol = np.nan

        # 最大回撤
        max_dd = 0.0
        if not price_series.empty:
            peak = price_series.iloc[0]
            for price in price_series:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd

        summary_df = pd.DataFrame(
            [
                {
                    "market_volatility_ann": market_vol,
                    "market_max_drawdown_pct": max_dd * 100 if price_series.size else np.nan,
                }
            ]
        )

        return daily_df, summary_df
    
    def generate_risk_report(self, dates: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        生成市场风险报告
        
        Args:
            dates: 日期列表
            output_file: 输出文件路径（可选）
            
        Returns:
            市场指标 DataFrame
        """
        analysis_dir = Path(output_file).parent if output_file else None

        # 市场级指标
        df_market_daily, df_market_summary = self.calculate_market_metrics(dates)

        if output_file and analysis_dir:
            if not df_market_daily.empty:
                market_daily_path = analysis_dir / "market_metrics.csv"
                df_market_daily.to_csv(market_daily_path, index=False, encoding="utf-8-sig")
                print(f"市场每日指标已保存到: {market_daily_path}")
            if not df_market_summary.empty:
                market_summary_path = analysis_dir / "market_metrics_summary.csv"
                df_market_summary.to_csv(
                    market_summary_path, index=False, encoding="utf-8-sig"
                )
                print(f"市场汇总指标已保存到: {market_summary_path}")
        
        return df_market_daily


def main():
    """主函数 - 示例用法"""
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算 TwinMarket 风险指标')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs_100_nips_gpt-4o-mini_0.8',
        help='结果目录名称（在 results/ 下）'
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket',
        help='项目根目录路径'
    )
    args = parser.parse_args()
    
    # 设置日志目录（在 results/ 下）
    base_path = Path(args.base_path)
    log_dir = base_path / "results" / args.log_dir
    
    if not log_dir.exists():
        print(f"错误: 目录不存在: {log_dir}")
        print("请确保模拟结果已生成在 results/ 目录下")
        sys.exit(1)
    
    # 查找股票配置文件（用于权重加权）
    stock_profile_path = base_path / "data" / "stock_profile.csv"
    if not stock_profile_path.exists():
        stock_profile_path = None
        print("提示: 未找到 stock_profile.csv，将使用成交量加权或简单平均")
    else:
        print(f"使用股票权重文件: {stock_profile_path}")
    
    # 初始化计算器
    calculator = RiskMetricsCalculator(str(log_dir), str(stock_profile_path) if stock_profile_path else None)
    
    # 获取所有可用的日期
    simulation_results_dir = log_dir / "simulation_results"
    if not simulation_results_dir.exists():
        print(f"错误: simulation_results 目录不存在: {simulation_results_dir}")
        sys.exit(1)
    
    dates = [d.name for d in simulation_results_dir.iterdir() if d.is_dir() and d.name.startswith('2023-')]
    if not dates:
        print("错误: 未找到任何交易日数据")
        sys.exit(1)
    
    dates = sorted(dates)  # 使用所有可用日期
    
    print(f"分析日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"共 {len(dates)} 个交易日\n")
    
    # 创建 analysis 目录
    analysis_dir = log_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # 生成市场风险报告
    output_file = analysis_dir / "market_metrics.csv"
    df_market = calculator.generate_risk_report(dates, str(output_file))
    
    # 加载市场指标（由 generate_risk_report 写入）
    market_daily_path = analysis_dir / "market_metrics.csv"
    market_summary_path = analysis_dir / "market_metrics_summary.csv"
    if market_daily_path.exists():
        print(f"\n=== 市场每日指标 (market_metrics.csv) ===")
        print(pd.read_csv(market_daily_path).head())
    if market_summary_path.exists():
        print(f"\n=== 市场汇总指标 (market_metrics_summary.csv) ===")
        print(pd.read_csv(market_summary_path))

    # 显示报告摘要
    if not df_market.empty:
        print("\n=== 市场指标摘要 ===")
        print(df_market.describe())
        
        print("\n=== 市场每日指标 ===")
        print(df_market.to_string(index=False))


if __name__ == "__main__":
    main()

