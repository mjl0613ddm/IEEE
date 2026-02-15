#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinMarket Monte Carlo Shapley Value 计算脚本（纯内存，不访问数据库）

对 (user_id, date) 玩家计算风险指标 risk_indicator_simple 的 Shapley 归因。
- 直接读取 trading_records/*.json，不依赖数据库
- 使用 matching_engine_memory（纯内存撮合）和 calculate_risk_metrics_memory
- Baseline: hold（被 mask 的玩家不提交订单）
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# 项目根目录 (TwinMarket)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trader.matching_engine_memory import (
    convert_decisions_from_dict,
    process_trading_day_memory_no_csv,
)
from scripts.process.calculate_risk_metrics_memory import calculate_risk_metrics_from_results

# 已知股票列表（用于默认 last_prices）
STOCK_LIST = ['TLEI', 'MEI', 'CPEI', 'IEEI', 'REEI', 'TSEI', 'CGEI', 'TTEI', 'EREI', 'FSEI']
DEFAULT_PRICE = 10.0


def load_stock_weights(base_path: Path) -> Optional[Dict[str, float]]:
    """从 data/stock_profile.csv 加载股票权重 (stock_id -> weight)。"""
    path = base_path / "data" / "stock_profile.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if 'stock_id' in df.columns and 'weight' in df.columns:
            return dict(zip(df['stock_id'].astype(str), df['weight'].astype(float)))
    except Exception:
        pass
    return None


def load_decisions_by_date(
    result_dir: Path,
    start_date: str,
    target_date: str,
) -> Tuple[Dict[str, dict], List[str], List[Tuple[str, str]]]:
    """
    加载 trading_records 下 [start_date, target_date] 内的决策。
    返回:
        decisions_by_date: {date: {user_id: user_data}}
        sorted_dates: 排序后的日期列表
        active_players: [(user_id, date), ...] 有有效 stock_decisions 的 (user, date)
    """
    trading_records = result_dir / "trading_records"
    if not trading_records.exists():
        raise FileNotFoundError(f"trading_records 不存在: {trading_records}")

    decisions_by_date = {}
    active_players = []

    for p in trading_records.iterdir():
        if not p.is_file() or p.suffix != ".json" or "_orders" in p.name:
            continue
        date_str = p.stem
        if date_str < start_date or date_str > target_date:
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        decisions_by_date[date_str] = data
        for user_id, user_data in data.items():
            if not user_data or "stock_decisions" not in user_data:
                continue
            sd = user_data["stock_decisions"]
            if not sd:
                continue
            has_order = any(
                si.get("sub_orders") for si in sd.values()
            )
            if has_order:
                active_players.append((user_id, date_str))

    sorted_dates = sorted(decisions_by_date.keys())
    if not sorted_dates:
        raise ValueError(f"在 {start_date} 到 {target_date} 范围内没有找到决策文件")
    return decisions_by_date, sorted_dates, active_players


def get_initial_last_prices(
    result_dir: Path,
    base_path: Path,
    sorted_dates: List[str],
    decisions_by_date: Dict[str, dict],
) -> Dict[str, float]:
    """
    获取第一日之前的 last_prices（用于 process_trading_day_memory_no_csv）。
    优先: simulation_results 首日 daily_summary -> stock_profile 或默认。
    """
    # 收集所有出现过的 stock_code
    stock_codes = set()
    for data in decisions_by_date.values():
        for user_data in data.values():
            if not user_data or "stock_decisions" not in user_data:
                continue
            stock_codes.update(user_data["stock_decisions"].keys())
    if not stock_codes:
        stock_codes = set(STOCK_LIST)

    # 尝试 simulation_results 第一日的 daily_summary
    sim_dir = result_dir / "simulation_results"
    first_date = sorted_dates[0]
    if sim_dir.exists():
        day_dir = sim_dir / first_date
        summary_file = day_dir / f"daily_summary_{first_date}.csv"
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                if "stock_code" in df.columns and "closing_price" in df.columns:
                    last_prices = dict(zip(
                        df["stock_code"].astype(str),
                        df["closing_price"].astype(float),
                    ))
                    # 用前一日的收盘价作为“上一日收盘”；若只有一日则用当日收盘
                    for code in stock_codes:
                        if code not in last_prices:
                            last_prices[code] = DEFAULT_PRICE
                    return last_prices
            except Exception:
                pass

    # 默认：所有股票同一价格
    return {code: DEFAULT_PRICE for code in stock_codes}


def evaluate_coalition(
    coalition: Set[Tuple[str, str]],
    decisions_by_date: Dict[str, dict],
    sorted_dates: List[str],
    initial_last_prices: Dict[str, float],
    stock_weights: Optional[Dict[str, float]],
    metric_name: str,
    seed: int,
) -> float:
    """
    计算联盟 S 的价值 v(S)：按日顺序运行反事实，得到 target_date 的 risk 指标。
    """
    random.seed(seed)
    np.random.seed(seed)
    last_prices = dict(initial_last_prices)
    all_results = {}

    for date in sorted_dates:
        if date not in decisions_by_date:
            continue
        data = decisions_by_date[date]
        # 只保留 coalition 中 (user_id, date) 的用户的决策
        filtered = {
            uid: ud for uid, ud in data.items()
            if (uid, date) in coalition and ud and "stock_decisions" in ud
        }
        decisions_list = convert_decisions_from_dict(filtered)
        try:
            if not decisions_list:
                results = {
                    code: {
                        "closing_price": last_prices.get(code, DEFAULT_PRICE),
                        "volume": 0,
                        "transactions": [],
                    }
                    for code in last_prices
                }
            else:
                results = process_trading_day_memory_no_csv(
                    decisions_list, last_prices, date,
                )
        except Exception:
            results = {
                code: {
                    "closing_price": last_prices.get(code, DEFAULT_PRICE),
                    "volume": 0,
                    "transactions": [],
                }
                for code in last_prices
            }
        all_results[date] = results
        for code, res in results.items():
            last_prices[code] = res["closing_price"]

    return calculate_risk_metrics_from_results(
        all_results,
        stock_weights=stock_weights,
        metric_name=metric_name,
    )


def _run_permutation(args: Tuple) -> Dict[Tuple[str, str], float]:
    """单次排列的边际贡献（供多进程调用）。"""
    (
        perm,
        decisions_by_date,
        sorted_dates,
        initial_last_prices,
        stock_weights,
        metric_name,
        seed,
    ) = args
    contributions = {}
    coalition = set()
    v_prev = evaluate_coalition(
        coalition, decisions_by_date, sorted_dates,
        initial_last_prices, stock_weights, metric_name, seed,
    )
    for player in perm:
        coalition.add(player)
        v_curr = evaluate_coalition(
            coalition, decisions_by_date, sorted_dates,
            initial_last_prices, stock_weights, metric_name, seed + 1 + len(contributions),
        )
        contributions[player] = v_curr - v_prev
        v_prev = v_curr
    return contributions


def read_real_metric_from_analysis(result_dir: Path, target_date: str, metric_name: str) -> Optional[float]:
    """
    若存在 analysis/market_metrics.csv，直接读取 target_date 对应的 risk 值，避免重跑全联盟。
    纯文件读取，不访问数据库。
    """
    csv_path = result_dir / "analysis" / "market_metrics.csv"
    if not csv_path.exists() or metric_name not in ("risk_indicator_simple", "risk_indicator_rolling"):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "date" not in df.columns or metric_name not in df.columns:
            return None
        row = df[df["date"] == target_date]
        if row.empty:
            row = df.iloc[-1]  # 取最后一行
        else:
            row = row.iloc[0]
        val = row.get(metric_name)
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def run_monte_carlo_shapley(
    decisions_by_date: Dict[str, dict],
    sorted_dates: List[str],
    active_players: List[Tuple[str, str]],
    initial_last_prices: Dict[str, float],
    stock_weights: Optional[Dict[str, float]],
    metric_name: str,
    n_samples: int,
    seed: int,
    n_threads: int,
    result_dir: Optional[Path] = None,
    target_date: Optional[str] = None,
    use_cached_real_metric: bool = True,
) -> Tuple[Dict[Tuple[str, str], float], float, float]:
    """
    Monte Carlo Shapley：采样排列，平均边际贡献。
    返回: (shapley_per_player, baseline_metric, real_metric)
    不访问数据库，仅读取文件与内存计算。
    """
    baseline_metric = evaluate_coalition(
        set(), decisions_by_date, sorted_dates,
        initial_last_prices, stock_weights, metric_name, seed,
    )
    # 优先从 analysis/market_metrics.csv 读取 real_metric，避免重跑全联盟
    real_metric = None
    if use_cached_real_metric and result_dir and target_date:
        real_metric = read_real_metric_from_analysis(result_dir, target_date, metric_name)
    if real_metric is None:
        real_metric = evaluate_coalition(
            set(active_players), decisions_by_date, sorted_dates,
            initial_last_prices, stock_weights, metric_name, seed + 1,
        )

    n_workers = n_threads if n_threads > 0 else max(1, os.cpu_count() or 1)
    accum = {p: 0.0 for p in active_players}
    rng = random.Random(seed)

    chunk_size = max(1, n_samples // n_workers)
    tasks = []
    for _ in range(n_samples):
        perm = list(active_players)
        rng.shuffle(perm)
        tasks.append((
            perm,
            decisions_by_date,
            sorted_dates,
            initial_last_prices,
            stock_weights,
            metric_name,
            seed + hash(tuple(perm)) % (2 ** 32),
        ))

    if n_workers <= 1:
        for t in tqdm(tasks, desc="Shapley MC"):
            c = _run_permutation(t)
            for p, v in c.items():
                accum[p] += v
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_run_permutation, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Shapley MC"):
                c = fut.result()
                for p, v in c.items():
                    accum[p] += v

    n = float(n_samples)
    shapley_per_player = {p: accum[p] / n for p in active_players}
    return shapley_per_player, baseline_metric, real_metric


def build_matrix_and_labels(
    shapley_per_player: Dict[Tuple[str, str], float],
    user_ids: List[str],
    dates: List[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """从 (user_id, date) -> value 构建 (n_users, n_dates) 矩阵和 labels。"""
    user_idx = {u: i for i, u in enumerate(user_ids)}
    date_idx = {d: i for i, d in enumerate(dates)}
    matrix = np.zeros((len(user_ids), len(dates)), dtype=np.float64)
    for (uid, date), val in shapley_per_player.items():
        i = user_idx.get(uid)
        j = date_idx.get(date)
        if i is not None and j is not None:
            matrix[i, j] = val
    labels = {"user_ids": np.array(user_ids), "dates": np.array(dates)}
    return matrix, labels


def main():
    parser = argparse.ArgumentParser(
        description="TwinMarket Monte Carlo Shapley Value 计算",
    )
    parser.add_argument("--result_dir", type=str, required=True, help="结果目录（含 trading_records）")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--target_date", type=str, required=True, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--n_samples", type=int, default=1000, help="Monte Carlo 采样数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n_threads", type=int, default=0, help="并行进程数，0=CPU 核心数")
    parser.add_argument("--base_path", type=str, default=None, help="项目根目录，默认自动检测")
    parser.add_argument("--metric_name", type=str, default="risk_indicator_simple")
    parser.add_argument("--baseline_type", type=str, default="hold")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认 result_dir/shapley")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        print(f"错误: result_dir 不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    base_path = Path(args.base_path).resolve() if args.base_path else PROJECT_ROOT
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / "shapley"
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_weights = load_stock_weights(base_path)
    if stock_weights is None:
        print("提示: 未找到 data/stock_profile.csv，使用无权重加权")

    print("加载决策...")
    decisions_by_date, sorted_dates, active_players = load_decisions_by_date(
        result_dir, args.start_date, args.target_date,
    )
    # 日期限制在 [start_date, target_date]
    sorted_dates = [d for d in sorted_dates if args.start_date <= d <= args.target_date]
    n_users = len(set(p[0] for p in active_players))
    n_dates = len(sorted_dates)
    print(f"  日期数: {len(sorted_dates)}, 玩家数 (user,date): {len(active_players)}, 用户数: {n_users}")

    print("获取初始 last_prices...")
    initial_last_prices = get_initial_last_prices(
        result_dir, base_path, sorted_dates, decisions_by_date,
    )

    print("运行 Monte Carlo Shapley（纯内存，不访问数据库）...")
    shapley_per_player, baseline_metric, real_metric = run_monte_carlo_shapley(
        decisions_by_date,
        sorted_dates,
        active_players,
        initial_last_prices,
        stock_weights,
        args.metric_name,
        args.n_samples,
        args.seed,
        args.n_threads,
        result_dir=result_dir,
        target_date=args.target_date,
        use_cached_real_metric=True,
    )

    date_range = f"{args.start_date}_{args.target_date}"
    user_ids_sorted = sorted(set(p[0] for p in active_players))

    # 1) shapley_stats_*.json
    values = list(shapley_per_player.values())
    stats = {
        "metric_name": args.metric_name,
        "baseline_type": args.baseline_type,
        "n_samples": args.n_samples,
        "date_range": date_range,
        "n_users": n_users,
        "n_dates": n_dates,
        "shapley_stats": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        },
        "baseline_metric": baseline_metric,
        "real_metric": real_metric,
    }
    stats_path = output_dir / f"shapley_stats_{date_range}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  已写: {stats_path}")

    # 2) shapley_attribution_timeseries_*.csv
    rows = []
    for (uid, date), val in shapley_per_player.items():
        rows.append({
            "start_date": args.start_date,
            "target_date": args.target_date,
            "user_id": uid,
            "date": date,
            "shapley_value": val,
            "metric_name": args.metric_name,
            "baseline_metric": baseline_metric,
            "real_metric": real_metric,
        })
    csv_path = output_dir / f"shapley_attribution_timeseries_{date_range}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  已写: {csv_path}")

    # 3) shapley_matrix_*.npy, shapley_labels_*.npy
    matrix, labels = build_matrix_and_labels(
        shapley_per_player, user_ids_sorted, sorted_dates,
    )
    np.save(output_dir / f"shapley_matrix_{date_range}.npy", matrix)
    np.save(output_dir / f"shapley_labels_{date_range}.npy", labels, allow_pickle=True)
    print(f"  已写: shapley_matrix_{date_range}.npy (shape {matrix.shape})")
    print(f"  已写: shapley_labels_{date_range}.npy")

    # 4) baseline_risk_indicators_*.csv (按日 baseline/real，这里简化为每日用同一 real)
    baseline_rows = [{"date": d, "baseline_risk": baseline_metric, "real_risk": real_metric} for d in sorted_dates]
    baseline_csv = output_dir / f"baseline_risk_indicators_{date_range}.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_csv, index=False)
    print(f"  已写: {baseline_csv}")

    print("完成.")


if __name__ == "__main__":
    main()
