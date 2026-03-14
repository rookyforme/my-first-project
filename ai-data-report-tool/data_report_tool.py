#!/usr/bin/env python3
"""
数据报告工具（支持 Mac）

功能：
1. 读取 CSV / Excel（xlsx）文件
2. 自动清洗：删除空值、删除重复行、用四分位数法检测异常值
3. 自动统计：基础统计 + 按指定字段分组统计
4. 输出清洗后的 Excel：清洗后_原文件名.xlsx
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# 使用无界面后端，保证在终端环境也能稳定保存图表
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_chart_style() -> None:
    """配置图表样式与中文字体，尽量避免 Mac 环境中文乱码。"""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.unicode_minus"] = False

    # 自动选择系统可用中文字体
    preferred_fonts = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in preferred_fonts if name in available]
    if chosen:
        plt.rcParams["font.sans-serif"] = chosen

    # 如果系统字体仍缺少部分字形，忽略该类告警，不影响图片保存
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font\(s\).*")

def read_data(file_path: Path) -> pd.DataFrame:
    """读取 CSV 或 xlsx 文件。"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        # 常见编码容错：优先 utf-8，其次 utf-8-sig，最后 gb18030
        last_err: Optional[Exception] = None
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as err:
                last_err = err
        raise ValueError(f"CSV 编码无法识别，请确认文件编码。原始错误: {last_err}")
    if suffix == ".xlsx":
        return pd.read_excel(file_path)
    raise ValueError("仅支持 .csv 或 .xlsx 文件")


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    自动清洗数据并返回：
    - cleaned_df: 清洗后的数据
    - outliers_df: 检测到的异常值行（四分位数法）
    """
    # 1) 去重
    cleaned_df = df.drop_duplicates()

    # 2) 去空值（只要该行存在空值就删除）
    cleaned_df = cleaned_df.dropna()

    # 3) 异常值检测（仅针对数值列，四分位数法）
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        # 没有数值列时，无法做 IQR 检测
        empty_outliers = cleaned_df.iloc[0:0]
        return cleaned_df, empty_outliers

    # 使用向量化操作提高性能
    Q1 = cleaned_df[numeric_cols].quantile(0.25)
    Q3 = cleaned_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 检测每个数值列的异常值
    outlier_mask = ((cleaned_df[numeric_cols] < lower_bound) | (cleaned_df[numeric_cols] > upper_bound)).any(axis=1)

    outliers_df = cleaned_df[outlier_mask].copy()
    # 把异常值行从清洗结果中移除
    cleaned_df = cleaned_df[~outlier_mask]
    return cleaned_df, outliers_df


def basic_statistics(df: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """输出每列的基础统计：均值、中位数、最大值、最小值、缺失率。"""
    # 使用 pandas 内置函数生成基础统计信息
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    
    rows = []
    total_count = len(raw_df) if raw_df is not None else len(df)
    
    # 处理数值列
    if numeric_cols:
        numeric_stats = df[numeric_cols].agg(["mean", "median", "max", "min"]).T
        for col in numeric_cols:
            miss_base_series = raw_df[col] if (raw_df is not None and col in raw_df.columns) else df[col]
            missing_rate = (miss_base_series.isna().sum() / total_count) if total_count else 0.0
            rows.append({
                "字段": col,
                "均值": float(numeric_stats.loc[col, "mean"]) if not pd.isna(numeric_stats.loc[col, "mean"]) else None,
                "中位数": float(numeric_stats.loc[col, "median"]) if not pd.isna(numeric_stats.loc[col, "median"]) else None,
                "最大值": float(numeric_stats.loc[col, "max"]) if not pd.isna(numeric_stats.loc[col, "max"]) else None,
                "最小值": float(numeric_stats.loc[col, "min"]) if not pd.isna(numeric_stats.loc[col, "min"]) else None,
                "缺失率": round(float(missing_rate), 4),
            })
    
    # 处理非数值列
    for col in non_numeric_cols:
        miss_base_series = raw_df[col] if (raw_df is not None and col in raw_df.columns) else df[col]
        missing_rate = (miss_base_series.isna().sum() / total_count) if total_count else 0.0
        rows.append({
            "字段": col,
            "均值": None,
            "中位数": None,
            "最大值": None,
            "最小值": None,
            "缺失率": round(float(missing_rate), 4),
        })

    return pd.DataFrame(rows)


def grouped_statistics(
    df: pd.DataFrame, group_col: Optional[str], value_col: Optional[str]
) -> pd.DataFrame:
    """按指定字段分组统计指定数值字段。"""
    if not group_col or not value_col:
        return pd.DataFrame()

    if group_col not in df.columns:
        raise ValueError(f"分组字段不存在: {group_col}")
    if value_col not in df.columns:
        raise ValueError(f"统计字段不存在: {value_col}")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"统计字段必须是数值类型: {value_col}")

    grouped = (
        df.groupby(group_col)[value_col]
        .agg(["count", "sum", "mean", "median", "max", "min"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "count": "数量",
            "sum": "总和",
            "mean": "均值",
            "median": "中位数",
            "max": "最大值",
            "min": "最小值",
        }
    )
    return grouped


def save_result(
    source_path: Path,
    cleaned_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    basic_stats_df: pd.DataFrame,
    group_stats_df: pd.DataFrame,
) -> Tuple[Path, int]:
    """将结果写入 Excel，多 sheet 输出。"""
    prefix = f"清洗后_{source_path.stem}_"
    existing_indices: List[int] = []
    for p in source_path.parent.glob(f"{prefix}*.xlsx"):
        suffix = p.stem.replace(prefix, "", 1)
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    # 从 1 开始，若已有历史文件则在最大编号基础上 +1
    next_index = max(existing_indices, default=0) + 1
    output_name = f"{prefix}{next_index}.xlsx"
    output_path = source_path.with_name(output_name)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cleaned_df.to_excel(writer, sheet_name="清洗后数据", index=False)
        outliers_df.to_excel(writer, sheet_name="异常值数据", index=False)
        basic_stats_df.to_excel(writer, sheet_name="基础统计", index=False)
        if not group_stats_df.empty:
            group_stats_df.to_excel(writer, sheet_name="分组统计", index=False)

    return output_path, next_index


def save_charts(
    source_path: Path,
    cleaned_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    group_col: Optional[str],
    value_col: Optional[str],
    output_index: int,
) -> Path:
    """
    生成并保存图表到 charts 文件夹：
    1) 数值字段分布直方图
    2) 分组统计柱状图（若提供分组参数）
    3) 异常值标注散点图
    """
    configure_chart_style()

    charts_dir = source_path.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    def safe_filename(text: str) -> str:
        """将字段名转换为适合文件名的安全字符串。"""
        safe = re.sub(r'[\\\\/:*?"<>|\\s]+', "_", str(text)).strip("_")
        return safe or "field"

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    # 1) 数值型字段分布直方图（每个数值字段一张）
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df[col], kde=True, bins=20, color="#4C78A8")
        plt.title(f"{col} 分布直方图")
        plt.xlabel(col)
        plt.ylabel("频数")
        plt.tight_layout()
        hist_col = safe_filename(col)
        hist_path = charts_dir / f"{source_path.stem}_{output_index}_hist_{hist_col}.png"
        plt.savefig(hist_path, dpi=160)
        plt.close()

    # 2) 分组统计柱状图
    if group_col and value_col and group_col in cleaned_df.columns and value_col in cleaned_df.columns:
        if pd.api.types.is_numeric_dtype(cleaned_df[value_col]):
            grouped_sum = (
                cleaned_df.groupby(group_col, dropna=False)[value_col].sum().sort_values(ascending=False).reset_index()
            )
            # 限制显示的分组数量，避免图表过于拥挤
            if len(grouped_sum) > 20:
                grouped_sum = grouped_sum.head(20)
                title = f"{group_col} - {value_col} 分组销售额柱状图（前20）"
            else:
                title = f"{group_col} - {value_col} 分组销售额柱状图"
                
            plt.figure(figsize=(11, 6))
            sns.barplot(
                data=grouped_sum,
                x=group_col,
                y=value_col,
                hue=group_col,
                palette="Blues_d",
                dodge=False,
                legend=False,
            )
            plt.title(title)
            plt.xlabel(group_col)
            plt.ylabel(f"{value_col}（总和）")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            bar_group = safe_filename(group_col)
            bar_value = safe_filename(value_col)
            bar_path = charts_dir / f"{source_path.stem}_{output_index}_bar_{bar_group}_{bar_value}.png"
            plt.savefig(bar_path, dpi=160)
            plt.close()

    # 3) 异常值标注散点图（选择前两个数值字段）
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=cleaned_df, x=x_col, y=y_col, color="#4C78A8", label="正常数据", alpha=0.7)
        if not outliers_df.empty:
            sns.scatterplot(data=outliers_df, x=x_col, y=y_col, color="#E45756", label="异常值", s=60)
        plt.title(f"异常值标注散点图（{x_col} vs {y_col}）")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.tight_layout()
        scatter_path = charts_dir / f"{source_path.stem}_{output_index}_scatter_outliers.png"
        plt.savefig(scatter_path, dpi=160)
        plt.close()

    return charts_dir


def run(file_path: str, group_col: Optional[str], value_col: Optional[str]) -> None:
    """主流程。"""
    start_time = time.time()
    src = Path(file_path).expanduser().resolve()
    
    print(f"开始处理文件: {src}")
    raw_df = read_data(src)
    print(f"原始数据形状: {raw_df.shape}")
    
    cleaned_df, outliers_df = clean_data(raw_df)
    print(f"清洗后数据形状: {cleaned_df.shape}")
    print(f"检测到异常值: {len(outliers_df)} 行")
    
    # 缺失率基于原始数据口径计算；其他统计值基于清洗后数据
    basic_stats_df = basic_statistics(cleaned_df, raw_df=raw_df)
    group_stats_df = grouped_statistics(cleaned_df, group_col, value_col)
    
    output_path, output_index = save_result(src, cleaned_df, outliers_df, basic_stats_df, group_stats_df)
    charts_dir = save_charts(src, cleaned_df, outliers_df, group_col, value_col, output_index)
    
    end_time = time.time()
    print(f"处理完成，结果已保存: {output_path}")
    print(f"图表已保存到: {charts_dir}")
    print(f"处理时间: {end_time - start_time:.2f} 秒")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B端数据文件自动清洗与统计工具")
    parser.add_argument("--file", required=True, help="输入文件路径（支持 .csv/.xlsx）")
    parser.add_argument("--group-col", help="分组字段，例如：部门")
    parser.add_argument("--value-col", help="统计字段，例如：销售额（需为数值列）")
    return parser


if __name__ == "__main__":
    try:
        args = build_parser().parse_args()
        run(args.file, args.group_col, args.value_col)
    except FileNotFoundError as e:
        print(f"[文件错误] {e}")
    except ValueError as e:
        print(f"[参数/格式错误] {e}")
    except Exception as e:  # noqa: BLE001
        print(f"[未知错误] {e}")
