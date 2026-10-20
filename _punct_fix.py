#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把注释/文档字符串中的英文标点换成中文标点（仅动注释部分，不碰代码）。
数学公式、代码引用等本来就该用英文标点的部分原样保留；不含中文的行不处理。"""
import re, sys, os

ROOT = "/home_data/home/slst/v-jiamh/张天夫/Optimize-peptide-sequences"
APPLY = "--apply" in sys.argv

# 全量处理所有代码文件的注释/文档字符串（不限于 git 新增行）
CODE_EXT = {".py", ".sh", ".sbatch"}
SKIP_DIRS = {".git", "__pycache__", "af3_local_results", "output_af3",
             "output_bo", "output_bo_htr1a-bin1"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def target_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] in CODE_EXT:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


PH = "\x00%d\x00"

# 保护段：数学公式、代码引用等本来就该用英文标点的部分原样保留
PROTECT = [
    r'"""',                                              # 三引号标记
    r'https?://[\w\-.~/?=&%#:+]+',                       # URL
    r'\\.',                                              # 转义序列 \d \n \\ 等
    r'!=',                                               # 不等运算符
    r'(?<=[+*}\])])\?',                                  # 正则/可选量词 +? *? {2,3}?
    r'(?<=\d):(?=\d)',                                   # 时间/比值 12:30 3:1
    r"'[^']*'",                                          # 单引号字符串/标识符
    r'"[^"\\\u3400-\u4dbf\u4e00-\u9fff]*"',              # 纯西文双引号（代码/文件名引用）
    r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_.]*\([^()]*\)',  # 函数调用/公式记法 make_tag(var) ipTM(…)
    r'(?<=[∈×^=<>+\-*/~])\([^()]*\)',                    # 算式括号 α^(1/n) t∈(b,U) P×(N-K+2)
    r'\([^()\u3400-\u4dbf\u4e00-\u9fff]*\)',             # 纯西文括号：数学/元组/西文插入语
    r'\[[^\[\]]*\]',                                     # 下标/列表字面量 [0] ['B','C','D']
    r'\{[^{}]*\}',                                       # 字典/格式符 {'a': 1} {x:.2f}
    r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+',  # 点号限定名
]


def convert(text):
    spans = []
    s = text

    def rep(m):
        spans.append(m.group(0))
        return PH % (len(spans) - 1)

    # 迭代到不动点：min(a, f(b)) 这类嵌套要逐层套上占位符
    for _ in range(5):
        n0 = len(spans)
        for pat in PROTECT:
            s = re.sub(pat, rep, s)
        if len(spans) == n0:
            break

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
    s = re.sub(r'"([^"]*)"', '“\\1”', s)                 # 剩余（含中文）双引号 → 中文引号
    s = re.sub(r"([，。；：？！、）”…])\s+", r"\1", s)     # 全角标点后不留空格
    s = re.sub(r"\s+([，。；：？！、）”…])", r"\1", s)     # 全角标点前不留空格
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+（", "（", s)     # 中文后多余空格
    for i in range(len(spans) - 1, -1, -1):              # 倒序还原，避免嵌套占位符泄漏
        s = s.replace(PH % i, spans[i])
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
    n_change = 0
    for path in target_files():
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines(keepends=True)
        except (UnicodeDecodeError, OSError) as e:
            print(f"跳过 {rel}: {e}")
            continue
        in_doc = False
        out_lines = []
        for idx, line in enumerate(lines, 1):
            raw = line
            body = line[:-1] if line.endswith("\n") else line
            # 单引号内的 """ 不参与文档字符串状态判断
            scan = re.sub(r"'[^']*'", "", body)
            nq = scan.count('"""')
            is_doc = in_doc or nq > 0
            if nq % 2 == 1:
                in_doc = not in_doc
            if is_doc:
                new_body = convert(body) if CJK.search(body) else body
            else:
                code, com = split_comment(body)
                new_body = code + (convert(com) if com and CJK.search(com) else com)
            if new_body != body:
                n_change += 1
                print(f"--- {rel}:{idx}")
                print(f"  - {body}")
                print(f"  + {new_body}")
                raw = new_body + ("\n" if line.endswith("\n") else "")
            out_lines.append(raw)
        if APPLY:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
    print(f"\n共 {n_change} 行变更 ({'已写入' if APPLY else '干跑，未写入'})")


main()
