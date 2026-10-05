#!/usr/bin/env python3
"""Jev 训练数据导出示例。

从默认数据库（data/quant_trading.db）读取历史 Jev 决策，关联后续价格走势
生成带标签训练样本，按时间切分 train/val/test，导出 JSONL + CSV 到
``output/training_data/``，并打印统计信息。

运行方式:
    cd quant_trading_system
    python examples/export_jev_training.py
    # 可选：限定标的/时间范围
    python examples/export_jev_training.py --symbol 600519.SH --days 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jev.training_data import TrainingDataExporter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 Jev 决策训练数据")
    parser.add_argument("--symbol", default=None, help="只导出某标的，如 600519.SH")
    parser.add_argument("--strategy", default=None, help="只导出某策略信号，如 ma_cross")
    parser.add_argument("--start-date", default=None, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=5, help="后向观察交易日数")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="hold 判定阈值（绝对收益率）")
    parser.add_argument("--out", default="output/training_data",
                        help="导出目录")
    args = parser.parse_args()

    # 不传 price_data → 惰性在线拉取 K线；失败的标的自动跳过
    exporter = TrainingDataExporter(
        hold_threshold=args.threshold,
        forward_days=args.days,
    )

    result = exporter.export_all(
        output_dir=args.out,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        strategy=args.strategy,
    )

    stats = result["stats"]
    print("=" * 56)
    print(f"训练样本总数: {result['sample_count']}")
    dist = stats.get("label_distribution", {})
    for label in ("buy", "sell", "hold"):
        d = dist.get(label, {})
        print(f"  标签 {label:>4}: {d.get('count', 0):>5} 条 "
              f"({d.get('ratio', 0) * 100:.1f}%)")
    print(f"  平均未来收益: {stats.get('avg_future_return', 0):.4f}")
    print(f"  原始动作与最优标签一致率: {stats.get('accuracy', 0) * 100:.1f}%")
    print(f"  各标的样本数: {json.dumps(stats.get('by_symbol', {}), ensure_ascii=False)}")
    print("-" * 56)
    print("导出文件:")
    for name, path in result["file_paths"].items():
        print(f"  {name:>14}: {path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
