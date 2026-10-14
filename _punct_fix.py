#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 git 新增注释/文档字符串中的英文标点换成中文标点（仅动注释部分，不碰代码）。"""
import re, subprocess, sys, collections

ROOT = "/home_data/home/slst/v-jiamh/张天夫/Optimize-peptide-sequences"
APPLY = "--apply" in sys.argv


def added_lines():
    out = subprocess.check_output(["git", "diff", "HEAD~1", "-U0"], cwd=ROOT, text=True)
    added = collections.defaultdict(set)
    cur, new_ln = None, 0
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            cur = line[6:]
        elif line.startswith("@@"):
            new_ln = int(re.search(r"\+(\d+)", line).group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            added[cur].add(new_ln)
            new_ln += 1
        elif line.startswith("-") or line.startswith("---") or line.startswith("\\"):
            pass
        else:
            new_ln += 1
    # 未跟踪的新文件：全部行都算新增
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).split()
    for u in untracked:
        if u.endswith(".py") and u != "_punct_fix.py":
            with open(ROOT + "/" + u, encoding="utf-8") as f:
                n = f.read().count("\n") + 1
            added[u] |= set(range(1, n + 1))
    return added


PH = "\x00%d\x00"

PROTECT = [
    r'"""',                                              # 三引号标记
    r'(?<=[∈×^=<>+\-*/~])\([^()]*\)',                    # 算式括号 α^(1/n) t∈(b,U) P×(N-K+2)
    r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_.]*\([^()]*\)',  # 函数调用/公式记法 make_tag(var) ipTM(…)
    r'\[[^\[\]]*\]',                                     # 下标/列表字面量 [0] ['B','C','D']
    r"'[^']*'",                                          # 单引号字符串/标识符
    r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+',  # 点号限定名
]


def convert(text):
    spans = []
    s = text
    for pat in PROTECT:
        def rep(m):
            spans.append(m.group(0))
            return PH % (len(spans) - 1)
        s = re.sub(pat, rep, s)
    s = s.replace("...", "……")
    # 枚举顿号：仅 ≥3 项的纯西文/数字列举用顿号，二元组/分句逗号保持
    s = re.sub(
        r"\b[A-Za-z0-9_]+(?:,\s*[A-Za-z0-9_]+){2,}\b",
        lambda m: "、".join(p.strip() for p in m.group(0).split(",")),
        s,
    )
    s = s.replace(",", "，")
    s = re.sub(r"，\s*…", "…", s)
    s = s.replace(";", "；").replace(":", "：")
    s = s.replace("(", "（").replace(")", "）")
    s = s.replace("?", "？").replace("!", "！")
    s = re.sub(r'\\?"([^"\\]*)\\?"', '“\\1”', s)         # 双引号 → 中文引号
    s = re.sub(r"([，。；：？！、）”…])\s+", r"\1", s)     # 全角标点后不留空格
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+（", "（", s)     # 中文后多余空格
    for i, sp in enumerate(spans):
        s = s.replace(PH % i, sp)
    return s


def split_comment(line):
    in_s = in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return line[:i], line[i:]
    return line, ""


def main():
    added = added_lines()
    n_change = 0
    for fname in sorted(added):
        path = ROOT + "/" + fname
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines(keepends=True)
        in_doc = False
        out_lines = []
        for idx, line in enumerate(lines, 1):
            raw = line
            body = line.rstrip("\n")
            nq = body.count('"""')
            is_doc = in_doc or nq > 0
            if nq % 2 == 1:
                in_doc = not in_doc
            if idx in added[fname]:
                if is_doc:
                    new_body = convert(body)
                else:
                    code, com = split_comment(body)
                    new_body = code + (convert(com) if com else "")
                if new_body != body:
                    n_change += 1
                    print(f"--- {fname}:{idx}")
                    print(f"  - {body}")
                    print(f"  + {new_body}")
                    raw = new_body + ("\n" if line.endswith("\n") else "")
            out_lines.append(raw)
        if APPLY:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
    print(f"\n共 {n_change} 行变更 ({'已写入' if APPLY else '干跑，未写入'})")


main()
