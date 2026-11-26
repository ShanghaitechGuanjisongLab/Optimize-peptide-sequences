#!/usr/bin/env python3
"""显示三个榜：乘积前五（靶标齐全记录）与各靶标各自的前五（该靶标有分即参与）。只读数据库。"""

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


def 读取靶标排名(靶标):
    """单靶标榜：该靶标有合法分即参与排序，其余靶标缺分显示为 -。"""
    key = f"iptm_{靶标.lower()}"
    rows = []
    for tag, row in db.load_database().items():
        value = row.get(key)
        if value is None or not math.isfinite(value):
            continue
        values = [row.get(f"iptm_{t.lower()}") for t in TARGETS]
        rows.append((value, tag, row, values))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return rows


def 打印榜(rows, 排序描述, 计数描述):
    print(f"\n排序：{排序描述} | {计数描述}：{len(rows)} 条")
    if not rows:
        print("没有符合当前排序条件的记录。")
        return
    print("排名  序列                         " + "  ".join(f"{靶标:>6}" for 靶标 in TARGETS))
    for rank, (score, tag, row, values) in enumerate(rows[:5], 1):
        peptide = row.get("peptide") or ""
        def format_score(value, width):
            return f"{value:>{width}.2f}" if value is not None and math.isfinite(value) else f"{'-':>{width}}"

        分量文本 = "  ".join(format_score(值, 6) for 值 in values)
        print(f"{rank:>2}    {peptide:<28} {分量文本}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"数据库：{db.db_root()}")
    打印榜(读取排名(), " × ".join(TARGETS) + " 乘积", "靶标齐全记录")
    for 靶标 in TARGETS:
        打印榜(读取靶标排名(靶标), f"{靶标} ipTM", "该靶标有分记录")


if __name__ == "__main__":
    if os.environ.get("PEPOPT_DB_ROOT"):
        print(f"使用 PEPOPT_DB_ROOT={os.environ['PEPOPT_DB_ROOT']}")
    main()
