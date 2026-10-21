#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""注释规范修复：把中文语境里的英文标点换成中文全角标点，并把被硬换行打断的句子接回一行。

处理三类文本：
  1) 注释与文档字符串（Python 注释/docstring、Shell 注释、Markdown 正文）；
  2) **输出文本**：给人看的提示语 —— Python 的 print/日志/raise 消息与 argparse 的 help=/description=，
     Shell 的 echo/printf 引号内文本、die/错误消息、`cat <<TAG` 的 heredoc 正文；
  3) 文档正文（.md，围栏代码块内不动）。

**不碰代码**：数学公式、代码引用、CLI 参数值（如 --targets HTR1A,BIN1）、字典键、文件路径、
shell 展开 $(...) / ${VAR:-默认值} / $VAR、f-string 花括号内的表达式等一律原样保留；不含中文的行不处理。
用法：python3 注释规范.py [--apply] [--no-strings] [--no-unwrap] [--diff]；不加 --apply 为干跑。
"""
import re, sys, os, io, ast, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
# 两个项目一起治：AF3 优化（本目录）与通用的任务中间层
ROOTS = [HERE, os.path.join(os.path.dirname(HERE), "任务中间层")]
APPLY = "--apply" in sys.argv
UNWRAP = "--no-unwrap" not in sys.argv
VERBOSE = "--diff" in sys.argv
STRINGS = "--no-strings" not in sys.argv        # 输出文本（print/echo 的字面量）也一并转换

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
    # —— 以下两条必须排在 CLI 选项规则**之前**: 否则 --?[A-Za-z] 会把 `:-HTR1A,BIN1` 里的 `-HTR1A`
    #    当成短选项吃掉, 剩下的 `,BIN1` 就被转成中文逗号而改坏代码引用。
    r':-[A-Za-z0-9_.,/+*%:${}-]+',                       # shell 默认值后缀 ${VAR:-HTR1A,BIN1}: 整体属代码
    r'--?[A-Za-z][A-Za-z0-9_-]*(?:[= ])"[^"]*"',          # CLI 选项带引号实参 --run "命令": 是示例语法, 引号不得转中文引号
    r'(?<![A-Za-z0-9_])[A-Z0-9][A-Za-z0-9_]*(?:,[A-Z0-9][A-Za-z0-9_]*)+(?![A-Za-z0-9_])',  # 逗号紧凑的值列表 HTR1A,BIN1 / 0,1,2,3
    r'--?[A-Za-z][A-Za-z0-9_-]*(?:[= ][A-Za-z0-9_][A-Za-z0-9_./+*%-]*(?:,[A-Za-z0-9_.:/+*%-]+)*)?',  # CLI 选项及值 --targets HTR1A,BIN1（不吃尾随逗号与冒号）
    r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]*=[A-Za-z0-9_./+*%-]+(?:,[A-Za-z0-9_./+*%-]+)*',  # 键值 PEPOPT_TARGETS=HTR1A,BIN1（同上）
    r'\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',                       # shell 参数展开 ${j%% *} ${PEPOPT_DB_ROOT:-/path/x}
    r'\$[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?',                 # shell 变量 $REPO_DIR $arr[0]
    r'\$[0-9@*#?$!-]',                                          # shell 特殊变量 $@ $1 $# $?
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


def convert(text, tight=True, extra=(), eat_after=True):
    """把中文语境里的英文标点转成全角。
    tight=True（注释/文档）：顺手吃掉全角标点两侧多余空格；tight=False（**输出文本**）只动标点后的**一个**空格
    —— 标点前的空格是表格/标签的**对齐留白**, 绝不能吃（吃掉会把 `靶标      : …` 挤成 `靶标：…`, 实测踩过）。
    eat_after=False：连标点后那一个空格也保留（仅 Markdown 表格行需要 —— 单元格靠空格分隔, 见 fix_punct_md）。
    extra：追加的保护正则（如括号不配对时整体保护括号, 避免跨 token 的括号被改成一全一半）。"""
    spans = []
    s = text

    def rep(m):
        spans.append(m.group(0))
        return PH % (len(spans) - 1)

    # 迭代到不动点：min(a, f(b)) 这类嵌套要逐层套上占位符
    for _ in range(5):
        n0 = len(spans)
        for pat in list(PROTECT) + list(extra):
            s = re.sub(pat, rep, s)
        if len(spans) == n0:
            break

    # 省略号: 仅独立的 `...` 转成 `……`; 紧邻字母/数字/斜杠的（如 .../Optimize-peptide 路径示例、a...b 变量）保持原样
    s = re.sub(r"(?<![A-Za-z0-9_./\\])\.{3}(?![A-Za-z0-9_./\\])", "……", s)
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
    if tight:
        # 仅清水平空白：用 \s+ 会把全角标点两侧的行末换行一并吞掉，导致多行文档字符串整段黏连（已实测踩到）。
        s = re.sub(r"([，。；：？！、）”…])[^\S\n]+", r"\1", s)     # 全角标点后不留空格
        s = re.sub(r"[^\S\n]+([，。；：？！、）”…])", r"\1", s)     # 全角标点前不留空格
        s = re.sub(r"(?<=[\u4e00-\u9fff])[^\S\n]+（", "（", s)     # 中文后多余空格
    else:
        # 输出文本：标点**后**的那一个空格必须吃掉。理由不是美观而是列宽：
        #   「半角标点 + 一个空格」共 2 列, 恰好等于「全角标点」的 2 列 → 去掉后**显示宽度不变**,
        #   `靶标      : {值}` → `靶标      ：{值}` 的对齐列自动保住（且符合中文排版：全角标点后不空格）。
        #   早先只在标点后紧跟中文时才吃空格, 结果产出 191 处 `： ${VAR}` 这种中英混排错位（实测已改）。
        #   标点**前**的空格一律保留 —— 那是作者写的标签对齐留白, 它决定值列的位置。
        if eat_after:
            s = re.sub(r"([，。；：？！、）”…])[^\S\n]", r"\1", s)
        s = re.sub(r"(?<=[\u4e00-\u9fff])[^\S\n](?![^\S\n])（", "（", s)     # 中文与左全角括号间的单个空格
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


# ===================== 输出文本（给人看的提示语） =====================

SHTR = "\x01%d\x01"          # shell 展开占位符
FSTR = "\x03%d\x03"          # f-string 表达式占位符


def _restore(saved, tag, s):
    for k in range(len(saved) - 1, -1, -1):        # 倒序还原, 避免序号前缀互相干扰
        s = s.replace(tag % k, saved[k])
    return s


def _protect_fstring(s):
    """把 f-string 的 {...} 表达式段换成占位符（{{/}} 转义保留）: 括号内是**代码**（字典键/格式符/三元式）, 标点一律不得改。"""
    saved, out, i, n = [], [], 0, len(s)
    while i < n:
        if s.startswith("{{", i) or s.startswith("}}", i):
            out.append(s[i:i + 2])
            i += 2
            continue
        if s[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            if not depth:
                saved.append(s[i:j])
                out.append(FSTR % (len(saved) - 1))
                i = j
                continue
        out.append(s[i])
        i += 1
    return "".join(out), saved


def _protect_shell(s):
    """把 shell 展开段换成占位符: $(..)（含 $((..)) 与嵌套引号）、${..}（含 :-默认值）、`..`、$VAR、$1、$@。
    本行内未闭合的 $( 说明是跨行命令替换 → 其后内容整段当代码（宁可不转, 不可改坏）。"""
    saved, out, i, n = [], [], 0, len(s)

    def grab(a, b):
        saved.append(s[a:b])
        out.append(SHTR % (len(saved) - 1))

    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append(s[i:i + 2])
            i += 2
            continue
        if c == "`":
            j = s.find("`", i + 1)
            j = n if j < 0 else j + 1
            grab(i, j)
            i = j
            continue
        if c == "$" and i + 1 < n and s[i + 1] in "({":
            opener, closer = s[i + 1], ")" if s[i + 1] == "(" else "}"
            depth, j = 1, i + 2
            while j < n and depth:
                if s[j] == "\\":
                    j += 1
                elif s[j] == opener:
                    depth += 1
                elif s[j] == closer:
                    depth -= 1
                j += 1
            end = j if not depth else n
            grab(i, end)
            i = end
            continue
        if c == "$":
            m = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?|\$[0-9@*#?$!-]", s[i:])
            if m and m.end() > 1:
                grab(i, i + m.end())
                i += m.end()
                continue
        out.append(c)
        i += 1
    return "".join(out), saved


# 插入语括号转全角: 括号里含变量展开/f-string 表达式（即已换成占位符）且括号前**不是**标识符（排除 make_tag(var) 这类调用记法）
# → 说明它是中文句子里的插入语（如 (PID $pid)、(Python $PYVER)）, 应当用全角。
INSERT_PAREN = re.compile(r"(?<![A-Za-z0-9_])\(([^()]*(?:[\x01\x03]\d+[\x01\x03])[^()]*)\)")


def _full_width_insert_parens(s, tight, extra=(), eat_after=True):
    """先处理“带变量的中文插入语括号", 再跑通用转换（tight/eat_after 控制空格处理）。"""
    return convert(INSERT_PAREN.sub("（\\1）", s), tight=tight, extra=extra,
                   eat_after=eat_after)


def _paren_extra(inner):
    """括号不配对（多行隐式拼接的字符串常见）→ 整段保护括号, 避免一头转全角另一头还是半角。
    计数必须**半角与全角同等对待**：早先只数半角, 于是上一轮遗留的 `已装(legacy 驱动形态)` 这类
    文本转成半 `(` + 全 `）` 后就被永久判为“不配对”而整体保护, 混用状态再也修不回来（实测踩过）；
    按“开括号总数 == 闭括号总数”判定后, 混用的对会在下一轮自愈成完整全角。"""
    depth = 0
    for ch in inner:
        if ch in "(（":
            depth += 1
        elif ch in ")）":
            depth -= 1
            if depth < 0:
                return [r"\(", r"\)"]
    return [r"\(", r"\)"] if depth else []


def convert_shell_text(s, tight=False, eat_after=True):
    """转换 shell 文本（引号内提示语 / cat heredoc 正文 / 注释）: 先保护 shell 展开, 再按中文语境转全角。
    默认 tight=False —— 输出文本里的对齐留白在标点**前**, 不得吞；标点后那一个空格则必须吞（见 convert）。"""
    if not s or not CJK.search(s):
        return s
    prot, saved = _protect_shell(s)
    return _restore(saved, SHTR,
                    _full_width_insert_parens(prot, tight, _paren_extra(prot),
                                              eat_after=eat_after))


def _skip_expand(line, j):
    """line[j] 是 ` 或 $( 或 ${ 时, 返回跳过整个展开段后的下标（支持 $((..)) 嵌套）; 本行未闭合则返回行长。"""
    n = len(line)
    if line[j] == "`":
        k = line.find("`", j + 1)
        return n if k < 0 else k + 1
    opener = line[j + 1]                      # ( 或 {
    closer = ")" if opener == "(" else "}"
    depth, k = 0, j + 1
    while k < n:
        if line[k] == "\\":
            k += 2
            continue
        if line[k] == opener:
            depth += 1
        elif line[k] == closer:
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return n


def _scan_quoted(line, i):
    """从 line[i]（引号）起找到配对的结束引号: 途中跳过转义与 $(..) / ${..} / `..`（它们内部可再有引号,
    如 $(cat "$PID_FILE")，不跳过就会把内部的 " 当成字符串结尾 → 把代码当文本改坏）。
    返回结束下标, 本行内找不到则返回 -1（不冒险, 整段原样保留）。"""
    q, j, n = line[i], i + 1, len(line)
    while j < n:
        if line[j] == "\\":
            j += 2
            continue
        if q == '"' and (line[j] == "`" or (line[j] == "$" and j + 1 < n and line[j + 1] in "({")):
            j = _skip_expand(line, j)
            continue
        if line[j] == q:
            return j
        j += 1
    return -1


MSG_CMD = re.compile(r"\s*(?:echo|printf|die|log|err|warn|msg|say|cat|tee)\b")


def convert_shell_quoted(line, single_ok=False):
    """只转换 shell 代码行里**引号内**的中文文本（echo/printf 提示语、die/错误消息）; 引号外的 shell 语法一字不动。
    单引号段只在 single_ok 为真时转（代码正文里的 Python 字串、或 echo/die 的实参）—— 否则可能是
    `trap 'echo; exit 130'` 这类**代码**, 转了就直接改坏脚本。"""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "\\" and i + 1 < n:
            out.append(line[i:i + 2])
            i += 2
            continue
        if c in "\"'" and (c == '"' or single_ok):
            j = _scan_quoted(line, i)
            if j < 0:                                   # 未闭合（多行字符串/heredoc 起始行）→ 不冒险
                out.append(line[i:])
                i = n
                continue
            out.append(c + convert_shell_text(line[i + 1:j]) + c)
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _split_str_token(tok):
    """拆出 Python 字符串 token 的（前缀, 定界引号, 内容）; 不合法时返回 None。"""
    m = re.match(r'^([fFrRbBuU]{0,3})("""|\'\'\'|"|\')', tok)
    if not m:
        return None
    prefix, quote = m.group(1), m.group(2)
    if not tok.endswith(quote) or len(tok) <= len(prefix) + 2 * len(quote):
        return None
    return prefix, quote, tok[len(prefix) + len(quote):-len(quote)]


def convert_str_token(tok, parens=True, force=False):
    """Python 输出字符串 token → 只转换**引号内**文本, 保留 f/r/b 前缀与定界引号（否则 " 会被改成中文引号而坏掉语法）。
    parens=False 时括号整段保护（多行隐式拼接时防一头全角一头半角）; tight 固定为 False: 不吞对齐空格。
    force=True: 本段自身没中文, 但它是同一条拼接消息的一部分 → 也要转（否则另一头的 `）` 落单）。"""
    parts = _split_str_token(tok)
    if not parts:
        return tok
    prefix, quote, inner = parts
    if "r" in prefix.lower() or (not force and not CJK.search(inner)):
        return tok                      # 原始字符串（多为正则与路径）/ 与中文无关的独立字符串 → 不动
    saved = []
    if "f" in prefix.lower():
        inner, saved = _protect_fstring(inner)
    extra = [] if parens else [r"\(", r"\)"]
    new = convert(INSERT_PAREN.sub("（\\1）", inner), tight=False, extra=extra)
    return prefix + quote + _restore(saved, FSTR, new) + quote


# 输出型调用: 实参就是给人看的文本（字典键/路径不会被误伤, 因为只收这些函数的实参）
OUTPUT_FUNCS = {"print", "pprint", "input", "exit", "error", "warning", "warn",
                "info", "debug", "critical", "exception", "fatal", "log", "say"}
OUTPUT_KWARGS = {"help", "description", "epilog", "usage", "prompt", "metavar"}


def _collect_text(node, spans):
    """收集"给人看的文本"字符串的**跨度**（起, 止）: 字面量、f-string（不深入其内部表达式——那里可能是字典键）、
    + 拼接、str.format/join 的模板。**不收**下标（d["键"]）等代码里的字串。
    用跨度而非只记起点: 隐式拼接的多行字串（f"a" f"b"）在 AST 里合成一个节点, 却占多个 token。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        spans.add(((node.lineno, node.col_offset),
                   (node.end_lineno, node.end_col_offset)))
    elif isinstance(node, ast.JoinedStr):
        spans.add(((node.lineno, node.col_offset),
                   (node.end_lineno, node.end_col_offset)))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        _collect_text(node.left, spans)
        _collect_text(node.right, spans)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr in ("format", "join"):
        _collect_text(node.func.value, spans)


def output_str_ranges(tree):
    """定位输出文本字符串: print/日志/input/sys.exit 的实参、raise 消息、argparse 的 help=/description= 值。"""
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            if isinstance(exc, ast.Call):
                for a in exc.args:
                    _collect_text(a, spans)
            else:
                _collect_text(exc, spans)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name in OUTPUT_FUNCS:
                for a in node.args:
                    _collect_text(a, spans)
            for kw in node.keywords:
                if name in OUTPUT_FUNCS or kw.arg in OUTPUT_KWARGS:
                    _collect_text(kw.value, spans)
    return sorted(spans)


# 定界符可不带引号（<<TAG）也可带引号（<<'TAG'）; 带/不带要分开写, 因为 Python 里对**未参与匹配**的组做反向引用会失配
HEREDOC = re.compile(r"<<-?\s*(?:([\"'])([A-Za-z_][A-Za-z0-9_]*)\1|([A-Za-z_][A-Za-z0-9_]*))")
# 只有 cat/echo/printf/tee 的 heredoc 正文是纯文本; python heredoc 的正文是代码, 只转其引号内文本
TEXT_HEREDOC = re.compile(r"^\s*(?:cat|echo|printf|tee)\b")


def _quotes_balanced(s):
    i, n = 0, len(s)
    while i < n:
        if s[i] == "\\":
            i += 2
            continue
        if s[i] in "\"'":
            j = s.find(s[i], i + 1)
            if j < 0:
                return False
            i = j + 1
            continue
        i += 1
    return True


def _heredoc_tag(body):
    """本行 heredoc 的定界符（无则 None）: 要求 `<<` 不在引号内, 否则只是字符串里的普通字符。"""
    if "<<" not in body:
        return None
    for m in HEREDOC.finditer(body):
        if _quotes_balanced(body[:m.start()]):
            return m.group(2) or m.group(3)
    return None


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
    """Python：用 tokenize 精确锚定，改 COMMENT、文档字符串与**输出文本字符串**；其余代码字符串原样保留（字典键/路径/正则属代码语义）。"""
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
    parens_ok = _group_paren_ok(toks, output_str_ranges(tree) if STRINGS else [])
    edits, n = [], 0
    for idx, t in enumerate(toks):
        if t.type == tokenize.COMMENT:
            new = convert(t.string) if CJK.search(t.string) else t.string
        elif t.type == tokenize.STRING and t.start in docspans:
            new = convert(t.string) if CJK.search(t.string) else t.string
        elif t.type == tokenize.STRING and idx in parens_ok:
            ok, force = parens_ok[idx]
            new = convert_str_token(t.string, parens=ok, force=force)     # 输出文本: 保留前缀与定界引号
        else:
            continue
        if new != t.string:
            edits.append((t.start[0], t.start[1], t.end[0], t.end[1], new))
            n += 1
    return (_apply_edits(src, edits), n) if edits else (src, 0)


def _group_paren_ok(toks, ranges):
    """按字面量跨度分组（隐式拼接的多段属同一组）, 以**整组**的括号配对情况决定是否转括号。
    拼接串常把 `(iptm=…, ` 与 `…, clash=…)` 拆在不同 token 里, 逐 token 判会一头全角一头半角。
    返回 {token 下标: (可否转括号, 是否强制转)}；不在返回里的 token 就不是输出文本, 不转换。
    强制转的意义: 拼接消息里“中文头 + 纯 ASCII 尾”很常见（f"…（iptm=…, " f"clash=…）"）,
    尾巴那段自己不含中文就会被跳过 → 括号一头全角一头半角。整段含中文就要求全组一起转。"""
    out = {}
    for (r0, r1) in ranges:
        idxs = [i for i, t in enumerate(toks)
                if t.type == tokenize.STRING and r0 <= t.start < r1]
        if not idxs:
            continue
        inners = [(_split_str_token(toks[i].string) or ("", "", ""))[2] for i in idxs]
        ok = not _paren_extra("".join(inners))
        force = any(CJK.search(x) for x in inners)
        for i in idxs:
            out[i] = (ok, force)
    return out


def _sh_code_line(body, single_ok=None):
    """一行 shell 代码：注释转全角（tight）；代码部分只转**引号内**的中文文本（tight=False, 不吞对齐空格），
    引号外语法（含 ; && | >）绝不动。
    single_ok: None = 按命令词判定（echo/die/log 等消息命令才转单引号实参）; True = 转（heredoc 里的 Python 字串）。"""
    if not STRINGS:
        code, com = split_comment(body)
        return code + (convert_shell_text(com, tight=True) if com and CJK.search(com) else com)
    if single_ok is None:
        single_ok = bool(MSG_CMD.match(body))       # 普通 shell 行: 只有消息命令的实参才当文本
    code, com = split_comment(body)
    code = convert_shell_quoted(code, single_ok=single_ok)
    if com and CJK.search(com):
        com = convert_shell_text(com, tight=True)
    return code + com


def fix_punct_sh(src):
    """Shell：`#` 注释 + **输出文本**（echo/printf 引号内提示语、die/错误消息、`cat <<TAG` 的 heredoc 正文）转中文全角标点。
    非 cat 的 heredoc（正文是 Python 等代码）按代码行处理 → 只转其引号内文本；shell 语法与展开一概不动。"""
    out, n = [], 0
    delim, text_body = None, False
    for line in src.splitlines(keepends=True):
        nl = "\n" if line.endswith("\n") else ""
        body = line[:-len(nl)] if nl else line
        if delim is not None:
            if body.strip() == delim:               # 定界符行原样保留
                delim, text_body = None, False
                new = body
            else:
                new = (convert_shell_text(body) if text_body
                       else _sh_code_line(body, single_ok=True))   # 代码正文（python heredoc）里的单引号也是字串
        else:
            new = _sh_code_line(body)
            tag = _heredoc_tag(body)
            if tag:
                delim, text_body = tag, bool(TEXT_HEREDOC.match(body))
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
        # 表格行的单元格靠空格分隔 → 连全角标点后那一个空格也不能吃（否则 `） |` 挤成 `）|`）
        table_row = body.lstrip().startswith("|")
        new = convert(body, tight=not table_row, eat_after=not table_row)
        n += (new != body)
        out.append(new + nl)
    return "".join(out), n


MAX_PASSES = 4                 # 不动点上限：实测 2 轮即收敛, 预留余量防止意外振荡


def fix_punct(src, path):
    """标点全角化，**迭代到不动点**。
    必需原因：同一行上先改的片段会挠动后面的列偏移 —— python 侧按 AST 跨度定位输出串,
    首轮把 `调度      : ` 的冒号改完后, 同行后一个字面量的跨度就对不齐而被跳过
    （实测 小肽优化_节点内.py 的 `enumerate_space (不重复 + 有限步全覆盖)` 要第二遍才转）。
    循环至稳定 → 单次调用即幂等, 再跑一遍不会又改出一批。"""
    ext = os.path.splitext(path)[1]
    fn = fix_punct_py if ext == ".py" else (fix_punct_md if ext == ".md" else fix_punct_sh)
    total = 0
    for _ in range(MAX_PASSES):
        new, n = fn(src)
        total += n
        if new == src:
            break
        src = new
    return src, total


def normalize(src, path):
    """断行接回 + 标点全角化，整体迭代到不动点。返回 (新文本, 断行接回数, 标点修正行数)。
    两阶段必须循环而非单跑：首轮接回后会出现新的标点修正位; 而标点修正又会产生新的接回机会
    —— `，` 不在 TERMINAL 里, 行尾 `;`/`:` 转全角前后可否拼接的判定会翻转（实测需 2 轮）。"""
    nw = np_ = 0
    for _ in range(MAX_PASSES):
        body, w = (unwrap(src, path) if UNWRAP else (src, 0))
        nw += w
        new, p = fix_punct(body, path)
        np_ += p
        if new == src:                      # 本轮无变化（此时 w=p=0）→ 已收敛
            return src, nw, np_
        src = new
    return src, nw, np_


def main():
    total_p = total_w = 0
    for path in target_files():
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except (UnicodeDecodeError, OSError) as e:
            print(f"跳过 {path}：{e}")
            continue
        new_src, nw, np_ = normalize(src, path)
        if new_src != src:
            total_p += np_
            total_w += nw
            print(f"=== {path}\n    断行接回 {nw} 处，标点 {np_} 行")
            if VERBOSE:
                import difflib
                for ln in difflib.unified_diff(src.splitlines(), new_src.splitlines(),
                                               lineterm="", n=0):
                    print("   " + ln)
            if APPLY:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_src)
    print(f"\n合计：断行接回 {total_w} 处，标点修正 {total_p} 行 "
          f"（{'已写入' if APPLY else '干跑，未写入'}）")


main()
