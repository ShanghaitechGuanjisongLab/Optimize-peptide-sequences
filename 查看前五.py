#!/usr/bin/env python3
"""仅显示当前乘积靶标评分齐全的前五名，可按乘积或 HTR1A 排序。只读数据库。"""

import math
import os
import argparse

import 小肽数据库 as db


TARGETS = db.PRODUCT_TARGETS


def 读取排名(按HTR1A=False):
    rows = []
    for tag, row in db.load_database().items():
        values = [row.get(f"iptm_{target.lower()}") for target in TARGETS]
        if any(value is None or not math.isfinite(value) for value in values):
            continue
        product = math.prod(values)
        rows.append((product, tag, row, values))

    if 按HTR1A:
        rows.sort(key=lambda item: (-item[2]["iptm_htr1a"], item[1]))
    else:
        rows.sort(key=lambda item: (-item[0], item[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-htr1a", action="store_true",
        help="在当前启用靶标评分齐全的记录中按 HTR1A ipTM 排序",
    )
    args = parser.parse_args()
    rows = 读取排名(args.by_htr1a)
    sort_description = "HTR1A ipTM" if args.by_htr1a else " × ".join(TARGETS) + " 乘积"
    print(f"数据库：{db.db_root()}")
    print(f"排序：{sort_description} | 靶标齐全记录：{len(rows)} 条")
    if not rows:
        print("没有符合当前排序条件的记录。")
        return

    print("\n排名  序列                         " + "  ".join(f"{靶标:>6}" for 靶标 in TARGETS))
    for rank, (product, tag, row, values) in enumerate(rows[:5], 1):
        peptide = row.get("peptide") or ""
        def format_score(value, width):
            return f"{value:>{width}.2f}" if value is not None else f"{'-':>{width}}"

        分量文本 = "  ".join(format_score(值, 6) for 值 in values)
        print(f"{rank:>2}    {peptide:<28} {分量文本}")


if __name__ == "__main__":
    if os.environ.get("PEPOPT_DB_ROOT"):
        print(f"使用 PEPOPT_DB_ROOT={os.environ['PEPOPT_DB_ROOT']}")
    main()
