#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注释规范修复：把注释/文档字符串/文档正文里的英文标点换成中文标点，并把被硬换行打断的句子接回一行。

只动注释与文档字符串，不碰代码；数学公式、代码引用、CLI 参数值（如 --targets HTR1A,BIN1）等本来就该用英文标点的部分原样保留，不含中文的行不处理。
用法：python3 _punct_fix.py [--apply] [--no-unwrap] [--diff]；不加 --apply 为干跑。
"""
import re, sys, os, io, ast, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
# 两个项目一起治：AF3 优化（本目录）与通用的任务中间层
ROOTS = [HERE, os.path.join(os.path.dirname(HERE), "任务中间层")]
APPLY = "--apply" in sys.argv
UNWRAP = "--no-unwrap" not in sys.argv
VERBOSE = "--diff" in sys.argv

# 全量处理所有注释/文档（不限于 git 新增行）
TEXT_EXT = {".py", ".sh", ".sbatch", ".md"}
SKIP_DIRS = {".git", "__pycache__", "af3_local_results", "output_af3",
             "output_bo", "output_bo_htr1a-bin1", "state", "bin", "_bundles"}
# 排除本工具自己：它的代码里满是引号/标点判定用的字面量，被自身规则转换会直接改坏语法
SKIP_FILES = {os.path.basename(__file__)}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def target_files():
    out = []
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1] in TEXT_EXT and fn not in SKIP_FILES:
                    out.append(os.path.join(dirpath, fn))
    return sorted(set(out))


PH = "\x00%d\x00"

# 保护段：数学公式、代码引用等本来就该用英文标点的部分原样保留
PROTECT = [
    r'"""',                                              # 三引号标记
    r'`[^`]*`',                                          # 行内代码引用 `--export=a,b,c`
    r'--?[A-Za-z][A-Za-z0-9_-]*(?:[= ][A-Za-z0-9_][A-Za-z0-9_./+*%-]*(?:,[A-Za-z0-9_.:/+*%-]+)*)?',  # CLI 选项及值 --targets HTR1A,BIN1（不吃尾随逗号与冒号）
    r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*=[A-Za-z0-9_./+*%-]+(?:,[A-Za-z0-9_./+*%-]+)*',  # 键值 PEPOPT_TARGETS=HTR1A,BIN1（同上）
    r'https?://[\w\-.~/?=&%#:+]+',                       # URL
    r'\\.',                                              # 转义序列 \d \n \\ 等
    r'!=',                                               # 不等运算符
    r'(?<=[+*}\])])\?',                                  # 正则/可选量词 +? *? {2,3}?
    r'(?<=\d):(?=\d)',                                   # 时间/比值 12:30 3:1
    r"'[^']*'",                                          # 单引号字符串/标识符
    r'"[^"\\\u3400-\u4dbf\u4e00-\u9fff]*"',              # 纯西文双引号（代码/文件名引用）
    r'(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_.]*\([^()\u3400-\u4dbf\u4e00-\u9fff]*\)',  # 函数记法 make_tag(var) ipTM(…)；括号内含中文时不保护（按规则转全角）
    r'(?<=[∈×^=<>+\-*/~])\([^()\u3400-\u4dbf\u4e00-\u9fff]*\)',  # 算式括号 α^(1/n) t∈(b,U)
    r'\([^()\u3400-\u4dbf\u4e00-\u9fff]*\)',             # 纯西文括号：数学/元组/西文插入语
    r'\[[^\[\]\u3400-\u4dbf\u4e00-\u9fff]*\]',            # 纯西文方括号：下标/列表字面量 [0] ['B','C','D']
    r'\{[^{}\u3400-\u4dbf\u4e00-\u9fff]*\}',             # 纯西文花括号：字典/格式符 {'a': 1} {x:.2f}
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
    # 仅清水平空白：用 \s+ 会把全角标点两侧的行末换行一并吞掉，导致多行文档字符串整段黏连（已实测踩到）。
    s = re.sub(r"([，。；：？！、）”…])[^\S\n]+", r"\1", s)     # 全角标点后不留空格
    s = re.sub(r"[^\S\n]+([，。；：？！、）”…])", r"\1", s)     # 全角标点前不留空格
    s = re.sub(r"(?<=[\u4e00-\u9fff])[^\S\n]+（", "（", s)     # 中文后多余空格
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


# ===================== 断行修复（一句话内部不得断行） =====================

FULLWIDTH = "，。；：？！、（）“”《》【】…—·「」"
# 句末标点（接断行时遇此即停）：解包阶段标点还是半角，故半角冒号与分号也算句末，否则“用法：”这类引导行会把后面的命令清单整段黏成一团。
TERMINAL = "。！？；：…”）》】」:;"
LIST_HEAD = re.compile(r"^(?:[#=\-*+~>]{2,}|[-*+•·]\s|\d+\s*[.)）、]|[（(]\d+[)）]|"
                       r"[①②③④⑤⑥⑦⑧⑨⑩]|\|)")
LABEL_HEAD = re.compile(r"^[^，。；\n]{0,12}[：:](?:\s|$)")   # 小标题式新句（如“注意：”）
RULE_END = re.compile(r"[\-=#*~]{3,}\s*$")
# 命令/代码示例开头：这类行属清单而非散文，不与前一行合并
CODE_START = re.compile(r"^(?:python3?\b|bash\b|sh\b|conda\b|sbatch\b|srun\b|scancel\b"
                        r"|echo\b|cd\b|export\b|kill\b|rm\b|mkdir\b|find\b|grep\b|tail\b"
                        r"|curl\b|pip3?\b|cargo\b|\./|~/|\$|/\w|--?\S|#\s|<|>"
                        r"|PEPOPT_|AF3_|TASKCTL_|HQ_)")
ALIGN = re.compile(r"\S {2,}\S")           # 行内有连续空格 → 对齐式清单/表格


def _cjkish(ch):
    return bool(re.match(r"[\u3400-\u4dbf\u4e00-\u9fff]", ch)) or ch in FULLWIDTH


def _sep(a, b):
    """拼接时是否需要空格：两侧都是中文/全角标点时不需，否则补一个。"""
    if not a or not b:
        return ""
    return "" if (_cjkish(a[-1]) and _cjkish(b[0])) else " "


def _can_join(a, b):
    """a、b 为去掉缩进与标记后的文本；判断能否把 b 接回 a 的句尾。"""
    if not a or not b:
        return False
    if a[-1] in TERMINAL or RULE_END.search(a):
        return False
    if LIST_HEAD.match(b) or LABEL_HEAD.match(b) or CODE_START.match(b):
        return False
    if ALIGN.search(a) or ALIGN.search(b):
        return False
    return True


def merge_block(items):
    """把同一注释块内的行接回断行：入参与返回都是（缩进、标记、文本）三元组列表，长度可变短。"""
    out = []
    for ws, marker, text in items:
        if out and _can_join(out[-1][2], text):
            pws, pmarker, ptext = out.pop()
            out.append((pws, pmarker, ptext + _sep(ptext, text) + text))
            continue
        out.append((ws, marker, text))
    return out


def _apply_edits(src, edits):
    """按编辑列表自底向上回写；每条编辑为行起、列起、行止、列止、新文本，行号从 1 起。"""
    lines = src.splitlines(keepends=True)
    for (r0, c0, r1, c1, new) in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        pre = lines[r0 - 1][:c0]
        post = lines[r1 - 1][c1:] if r1 - 1 < len(lines) else ""
        lines[r0 - 1:r1] = (pre + new + post).splitlines(keepends=True)
    return "".join(lines)


def unwrap_py(src):
    """Python：用 tokenize 拿到文档字符串与独立注释行的精确跨度，块内接回断行。"""
    try:
        toks = list(tokenize.tokenize(io.BytesIO(src.encode()).readline))
    except Exception:
        return src, 0
    raw = src.splitlines()
    edits, n = [], 0

    for t in toks:
        if t.type != tokenize.STRING:
            continue
        s = t.string
        if not (s.startswith('"""') and s.endswith('"""') and len(s) > 6):
            continue
        inner = s[3:-3]
        if not CJK.search(inner):
            continue
        rows = inner.split("\n")
        items = [(re.match(r"\s*", r).group(0), "", r.strip()) for r in rows]
        merged = merge_block(items)
        if len(merged) == len(items):
            continue
        n += len(items) - len(merged)
        new_inner = "\n".join(ws + txt for ws, _, txt in merged)
        edits.append((t.start[0], t.start[1], t.end[0], t.end[1], '"""' + new_inner + '"""'))

    groups, cur = [], []
    for t in toks:
        standalone = (t.type == tokenize.COMMENT
                      and raw[t.start[0] - 1][:t.start[1]].strip() == "")
        if standalone and (not cur or t.start[0] == cur[-1].end[0] + 1):
            cur.append(t)
        else:
            if cur:
                groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    for g in groups:
        if len(g) < 2:
            continue
        items = [(raw[t.start[0] - 1][:t.start[1]], "# ", t.string.lstrip("#").strip())
                 for t in g]
        merged = merge_block(items)
        if len(merged) == len(items):
            continue
        new_lines = [(ws + marker + txt) if txt else (ws + "#")
                     for ws, marker, txt in merged]
        r0, r1 = g[0].start[0], g[-1].end[0]
        edits.append((r0, 0, r1, len(raw[r1 - 1]), "\n".join(new_lines)))
        n += len(items) - len(merged)

    return (_apply_edits(src, edits), n) if edits else (src, 0)


SH_COMMENT = re.compile(r"^(\s*)(# |#$)(.*?)\s*$")


def unwrap_sh(src):
    """Shell：只处理“# ”注释行组成的块（shebang、#==== 分隔行、行尾注释不动）。"""
    lines = src.split("\n")
    out, i, n = [], 0, 0
    while i < len(lines):
        if i == 0 or not SH_COMMENT.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < len(lines):
            m = SH_COMMENT.match(lines[i])
            if not m:
                break
            block.append((m.group(1), m.group(2), m.group(3)))
            i += 1
        merged = merge_block(block) if len(block) >= 2 else block
        n += len(block) - len(merged)
        for ws, marker, txt in merged:
            out.append(ws + marker + txt if txt else ws + "#")
    return "\n".join(out), n


def unwrap_md(src):
    """Markdown：段落内接回断行；围栏代码、标题、列表行、表格行、硬换行一律不动。"""
    lines = src.split("\n")
    out, i, n, fence = [], 0, 0, False
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence
            out.append(ln)
            i += 1
            continue
        if (fence or not s or ln.startswith("#") or s.startswith(("|", ">"))
                or LIST_HEAD.match(s) or ln.rstrip().endswith("  ")):
            out.append(ln)
            i += 1
            continue
        block = [("", "", s)]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            ns = nxt.strip()
            if (not ns or ns.startswith("```") or nxt.startswith("#")
                    or ns.startswith(("|", ">")) or LIST_HEAD.match(ns)
                    or nxt.rstrip().endswith("  ")):
                break
            block.append(("", "", ns))
            i += 1
        merged = merge_block(block)
        n += len(block) - len(merged)
        out.extend(txt for _, _, txt in merged)
    return "\n".join(out), n


def unwrap(src, path):
    ext = os.path.splitext(path)[1]
    if ext == ".py":
        return unwrap_py(src)
    if ext == ".md":
        return unwrap_md(src)
    return unwrap_sh(src)


def fix_punct_py(src):
    """Python：用 tokenize 精确锚定，只改 COMMENT 与文档字符串；普通代码字符串原样保留（属代码语义）。"""
    try:
        toks = list(tokenize.tokenize(io.BytesIO(src.encode()).readline))
    except Exception:
        return src, 0
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, 0
    docspans = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                v = body[0].value
                docspans.add((v.lineno, v.col_offset))
    edits, n = [], 0
    for t in toks:
        keep = (t.type == tokenize.COMMENT
                or (t.type == tokenize.STRING and t.start in docspans))
        if not keep or not CJK.search(t.string):
            continue
        new = convert(t.string)
        if new != t.string:
            edits.append((t.start[0], t.start[1], t.end[0], t.end[1], new))
            n += 1
    return (_apply_edits(src, edits), n) if edits else (src, 0)


def fix_punct_sh(src):
    """Shell：只改 `#` 起的注释部分，命令行正文（含引号内字符串）一律不动。"""
    out, n = [], 0
    for line in src.splitlines(keepends=True):
        nl = "\n" if line.endswith("\n") else ""
        body = line[:-len(nl)] if nl else line
        code, com = split_comment(body)
        new = code + (convert(com) if com and CJK.search(com) else com)
        n += (new != body)
        out.append(new + nl)
    return "".join(out), n


def fix_punct_md(src):
    """Markdown：围栏代码块内不动，其余按行改标点（行内代码由 PROTECT 保护）。"""
    out, n, fence = [], 0, False
    for line in src.splitlines(keepends=True):
        nl = "\n" if line.endswith("\n") else ""
        body = line[:-len(nl)] if nl else line
        if body.lstrip().startswith("```"):
            fence = not fence
            out.append(line)
            continue
        if fence or not CJK.search(body):
            out.append(line)
            continue
        new = convert(body)
        n += (new != body)
        out.append(new + nl)
    return "".join(out), n


def fix_punct(src, path):
    ext = os.path.splitext(path)[1]
    if ext == ".py":
        return fix_punct_py(src)
    if ext == ".md":
        return fix_punct_md(src)
    return fix_punct_sh(src)


def main():
    total_p = total_w = 0
    for path in target_files():
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except (UnicodeDecodeError, OSError) as e:
            print(f"跳过 {path}: {e}")
            continue
        body, nw = (unwrap(src, path) if UNWRAP else (src, 0))
        new_src, np_ = fix_punct(body, path)
        if new_src != src:
            total_p += np_
            total_w += nw
            print(f"=== {path}\n    断行接回 {nw} 处, 标点 {np_} 行")
            if VERBOSE:
                import difflib
                for ln in difflib.unified_diff(src.splitlines(), new_src.splitlines(),
                                               lineterm="", n=0):
                    print("   " + ln)
            if APPLY:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_src)
    print(f"\n合计: 断行接回 {total_w} 处, 标点修正 {total_p} 行 "
          f"({'已写入' if APPLY else '干跑，未写入'})")


main()
