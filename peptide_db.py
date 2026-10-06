"""
小肽序列优化 — 共享结果数据库（跨入口公共库，导入零副作用）。
=============================================================
为什么需要它：
  当前 bme_gpupub 分区全是 V100-32GB，官方仅支持 ≤1280 tokens 的复合物：
  肽×HTR1A ≈ 441、肽×BIN1 ≈ 612 均在限内，而肽×UNC13C ≈ 2233 远超上限，
  在 V100 上必然显存溢出。因此现阶段**去掉 UNC13C 约束**，只优化
  HTR1A（ipTM 尽量大）+ BIN1（ipTM < 0.8），并把结果**尽量多地**存成
  数据库；等申请到 A100/H100 能算 UNC13C 后，直接从本数据库补齐/筛选即可，
  已算的 HTR1A/BIN1 不必重算。

存放位置（共享目录，计算节点可见，家目录 ~500G 配额放不下）：
    <DB_ROOT>/
        peptide_database.csv    # 主表: 每序列一行, iptm_htr1a/iptm_bin1/iptm_unc13c
        sequences.csv           # tag → 可变区/全序列 注册表（无限扩池也能对上）
        af3_results/            # 全部 AF3 结果目录（input.json / 输出 / *_summary_confidences.json）
        BEST_TOP10/             # 当前最佳 10 个序列的 HTR1A 结构包副本 + BEST_TOP10.csv
    DB_ROOT 默认 /public_bme2/Share200T/管吉松/peptide_opt_db
    可用环境变量 PEPOPT_DB_ROOT 覆盖（便于测试）。

任务命名（结果目录名, AF3 的 sanitised_name 会转小写, 故必须小写）:
    <tag>_<target>      tag = s<5位十进制>（由可变区 crc32 稳定生成）
    例: s00129_htr1a、s00129_bin1。tag 只依赖序列本身, 与"第几个候选"无关,
    因此无限扩池时不同批次的候选也不会撞名, 且天然幂等（同序列同靶标同名,
    重跑不会重复计算）。

导入零副作用: 不会自动建目录/写文件, 只有调用 paths(ensure=True) 或写入类
函数时才落盘。
"""

import os
import re
import csv
import glob
import json
import zlib
import shutil
import zipfile
import tempfile
from datetime import datetime

# ---------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------

DEFAULT_DB_ROOT = "/public_bme2/Share200T/管吉松/peptide_opt_db"
SUMMARY_SUFFIX = "_summary_confidences.json"
TARGETS_ALL = ("HTR1A", "BIN1", "UNC13C")

MASTER_FIELDS = ["tag", "peptide", "var_region",
                 "iptm_htr1a", "score_htr1a", "ranking_htr1a",
                 "iptm_bin1", "ranking_bin1",
                 "iptm_unc13c", "ranking_unc13c",
                 "source", "created_at", "updated_at"]

SEQREG_FIELDS = ["tag", "var_region", "peptide", "first_seen"]

_SKIP_FILE_PREFIXES = ("slurm_",)
_SKIP_FILE_NAMES = {".submitted", ".oom_skipped", ".harvested"}


def db_root():
    """共享数据库根目录（PEPOPT_DB_ROOT 可覆盖）。"""
    return os.environ.get("PEPOPT_DB_ROOT", "").strip() or DEFAULT_DB_ROOT


def paths(ensure=False):
    """返回共享库各路径 dict；ensure=True 时创建所有子目录。"""
    root = db_root()
    p = {
        "root": root,
        "master_csv": os.path.join(root, "peptide_database.csv"),
        "seqreg_csv": os.path.join(root, "sequences.csv"),
        "results": os.path.join(root, "af3_results"),
        "best_dir": os.path.join(root, "BEST_TOP10"),
    }
    if ensure:
        os.makedirs(root, exist_ok=True)
        for k in ("results", "best_dir"):
            os.makedirs(p[k], exist_ok=True)
    return p


def results_dir():
    return paths()["results"]


# ---------------------------------------------------------------------
# 标签与任务名
# ---------------------------------------------------------------------

def make_tag(seq_var):
    """可变区序列 → 稳定任务标签（crc32 全 32bit 十六进制, 与扩池批次无关）。

    **旧规则 `s%05d`(crc32 mod 100000) 已被实测证伪**: 对"编辑距离≤2, 长6~10"
    的全空间 51091 个可变区枚举检验, 有 **11121 个(21.8%) 发生标签碰撞**
    （9326 个组, 最大组 5 个序列共用一标签）。碰撞的后果很隐蔽:
    register_sequence 抛错 → prepare_cycle 捕获后**静默跳过该序列** →
    这些序列永久无法评估, 而覆盖率看起来仍在增长 → "全覆盖"承诺落空。
    改用 crc32 全 32bit 后, 同一空间枚举实测**碰撞 0 个**。

    历史目录名（旧 5 位十进制, 如 s37548）由 resolve_tag() 按注册表原样沿用,
    因此本次扩位不会使已评估结果孤儿化、也不需要重算。
    碰撞的最终防线仍是 register_sequence() 的唯一性校验（现在几乎不可能触发）。
    """
    return f"s{zlib.crc32(seq_var.encode('utf-8')):08x}"


_TASK_RE = re.compile(r"^([a-z0-9]+)_([a-z0-9]+)$")
_TARGET_ALIASES = {"htr1a": "HTR1A", "bin1": "BIN1", "unc13c": "UNC13C"}

# AF3 的 run_alphafold.py 向**已存在且非空**的输出目录写结果时, 会另建
# <名>_YYYYMMDD_HHMMSS/ 兄弟目录以免覆盖。驱动为打包提交预先在
# <tag>_<target>/ 写了 input.json, 故实际输出常落在时间戳兄弟目录。
# 不把两者关联会导致: 规范目录看不到 summary → 被判为"未完成"而重贑,
# 且 harvest 也认不出。下列函数专门处理这一分歧。
_TS_RE = re.compile(r"_\d{8}_\d{6}$")

# resolve_tag 的反向索引缓存: 按 sequences.csv 的 mtime 失效。
# 必要性: 注册表最终会有 5 万行, 若每个序列都重读一次 CSV,
# 一个周期(24 序列)会多花数秒且随规模线性恶化。
_VAR2TAG_CACHE = {"mtime": None, "size": None, "index": {}}


def _var2tag_index():
    """可变区 → 已注册 tag 的反向索引（带 mtime 缓存）。"""
    p = paths()["seqreg_csv"]
    try:
        st = os.stat(p)
        key = (st.st_mtime, st.st_size)
    except OSError:
        return {}
    if (_VAR2TAG_CACHE["mtime"], _VAR2TAG_CACHE["size"]) == key:
        return _VAR2TAG_CACHE["index"]
    idx = {}
    for tag, info in load_sequence_registry().items():
        v = info.get("var")
        if v and v not in idx:              # 先到先占: 与注册表写入顺序一致
            idx[v] = tag
    _VAR2TAG_CACHE.update(mtime=key[0], size=key[1], index=idx)
    return idx


def resolve_tag(seq_var):
    """可变区 → 标签, **注册表优先**。

    写入路径一律用本函数而不是 make_tag(): 已登记过的序列必须沿用它当初的
    标签（否则扩位/改哈希规则后同一序列会得到新标签 → 已有结果被当成新任务
    重算, 旧目录变孤儿）。未登记的新序列才按当前 make_tag() 计算。"""
    return _var2tag_index().get(seq_var) or make_tag(seq_var)


def make_job_name(peptide, target):
    """(全序列, 靶标) → AF3 结果目录名（小写, 与 sanitised_name 一致）。"""
    from peptide_common import FIXED_A
    var = peptide[len(FIXED_A):] if peptide.startswith(FIXED_A) else peptide
    return f"{resolve_tag(var)}_{target.lower()}"


def _strip_ts(name):
    """去掉目录名可能追加的 _YYYYMMDD_HHMMSS 时间戳后缀。"""
    return _TS_RE.sub("", name)


def split_job_name(job_name):
    """结果目录名 → (tag, TARGET) 或 None（不符合 <tag>_<target> 或靶标未知）。
    兼容 AF3 追加的时间戳后缀: s05770_htr1a_20261002_111938 → (s05770, HTR1A)。"""
    m = _TASK_RE.match(_strip_ts(job_name))
    if not m:
        return None
    tag, tgt = m.group(1), m.group(2)
    if tgt.lower() not in _TARGET_ALIASES:
        return None
    return tag, _TARGET_ALIASES[tgt.lower()]


def canonical_dir(tag, target, results_root=None):
    """某任务的规范结果目录 <root>/<tag>_<target>（未必存在）。"""
    return os.path.join(results_root or results_dir(), f"{tag}_{target.lower()}")


def task_dirs(tag, target, results_root=None):
    """某(序列×靶标)任务在磁盘上的全部输出目录: 规范目录 + 时间戳兄弟目录
    （按名升序; 时间戳名即时间序, 末尾为最新）。不存在时返回 []。"""
    results_root = results_root or results_dir()
    if not os.path.isdir(results_root):
        return []
    canon = f"{tag}_{target.lower()}"
    out = []
    for name in sorted(os.listdir(results_root)):
        if _strip_ts(name) == canon and \
                os.path.isdir(os.path.join(results_root, name)):
            out.append(os.path.join(results_root, name))
    return out


def consolidate_task(tag, target, results_root=None):
    """把 AF3 时间戳兄弟目录的输出**合并进规范目录** <tag>_<target>/, 令下游
    （评分提取 / bundle has_result / 去重提交 / BEST 打包）只认单一规范位置:
    既不重复收割, 也不再对已完成任务重跑。多次重跑时取最新一份输出。

    仅在无并发写入时安全（调用方保证作业已终结）。返回规范目录内 summary
    路径或 None。"""
    results_root = results_root or results_dir()
    canon = canonical_dir(tag, target, results_root)
    sibs = [d for d in task_dirs(tag, target, results_root)
            if _TS_RE.search(os.path.basename(d))]
    if sibs and not find_summary(canon):
        done = [d for d in sibs if find_summary(d)]
        if done:
            src = sorted(done)[-1]                 # 最新一次输出
            os.makedirs(canon, exist_ok=True)
            for entry in sorted(os.listdir(src)):
                if entry in _SKIP_FILE_NAMES or entry.startswith(_SKIP_FILE_PREFIXES):
                    continue
                dst = os.path.join(canon, entry)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(src, entry), dst)
    for d in sibs:                                 # 折叠后清掉所有时间戳兄弟
        shutil.rmtree(d, ignore_errors=True)
    return find_summary(canon)


# ---------------------------------------------------------------------
# CSV 读写工具
# ---------------------------------------------------------------------

def _read_csv(path, fields):
    rows = []
    if not os.path.isfile(path):
        return rows
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({c: (r.get(c) or "") for c in fields})
    except (OSError, csv.Error):
        pass
    return rows


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rewrite_master(rows):
    """原子重写主表（先写临时文件再 os.replace, NFS 上比追加更稳）。"""
    p = paths(ensure=True)
    fd, tmp = tempfile.mkstemp(dir=p["root"], prefix=".pep_db_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in MASTER_FIELDS})
        os.replace(tmp, p["master_csv"])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------
# 序列注册表: tag ↔ (可变区, 全序列)
# ---------------------------------------------------------------------

def load_sequence_registry():
    """tag → {"var": 可变区, "peptide": 全序列}。"""
    reg = {}
    for r in _read_csv(paths()["seqreg_csv"], SEQREG_FIELDS):
        if r["tag"]:
            reg[r["tag"]] = {"var": r["var_region"], "peptide": r["peptide"]}
    return reg


def register_sequence(tag, var_region, peptide):
    """把 tag→序列 写入注册表。已存在且一致返回 False；不一致（撞标签）抛错。"""
    p = paths(ensure=True)
    reg = load_sequence_registry()
    if tag in reg:
        if reg[tag]["var"] == var_region:
            return False
        raise RuntimeError(
            f"标签冲突: {tag} 已对应 {reg[tag]['var']!r}, 新序列 {var_region!r}。"
            "（make_tag 已用 crc32 全 32bit, 实测全空间 0 碰撞; 若真出现说明"
            " 有旧目录被人工改名, 需核对 sequences.csv）")
    existed = os.path.isfile(p["seqreg_csv"]) and os.path.getsize(p["seqreg_csv"]) > 0
    with open(p["seqreg_csv"], "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SEQREG_FIELDS)
        if not existed:
            w.writeheader()
        w.writerow({"tag": tag, "var_region": var_region, "peptide": peptide,
                    "first_seen": _now()})
    return True


# ---------------------------------------------------------------------
# 主数据库: 每序列一行
# ---------------------------------------------------------------------

def load_database():
    """读主表 → dict[tag] = {"peptide","iptm_htr1a",...}（空/非法数值为 None）。"""
    rows = _read_csv(paths()["master_csv"], MASTER_FIELDS)
    by_tag = {}
    for r in rows:
        by_tag[r["tag"]] = {
            "peptide": r["peptide"],
            "var": r["var_region"],
            "iptm_htr1a": _to_float(r["iptm_htr1a"]),
            "score_htr1a": _to_float(r["score_htr1a"]),
            "ranking_htr1a": _to_float(r["ranking_htr1a"]),
            "iptm_bin1": _to_float(r["iptm_bin1"]),
            "ranking_bin1": _to_float(r["ranking_bin1"]),
            "iptm_unc13c": _to_float(r["iptm_unc13c"]),
            "ranking_unc13c": _to_float(r["ranking_unc13c"]),
        }
    return by_tag


def upsert_result(tag, target, iptm, ranking=None, peptide=None, var=None,
                  source="local"):
    """写入/更新某序列某靶标的 ipTM（行不存在则新建）。iptm/ranking 为 None
    时对应字段留空。返回该行 dict。"""
    target = target.upper()
    col = f"iptm_{target.lower()}"
    rk_col = f"ranking_{target.lower()}"
    if col not in MASTER_FIELDS:
        raise ValueError(f"主表无列 {col}, 请先扩展 MASTER_FIELDS")
    rows = _read_csv(paths()["master_csv"], MASTER_FIELDS)
    idx = next((i for i, r in enumerate(rows) if r["tag"] == tag), None)
    if idx is None:
        new_row = {c: "" for c in MASTER_FIELDS}
        new_row.update({"tag": tag, "peptide": peptide or "",
                        "var_region": var or "", "source": source,
                        "created_at": _now()})
        rows.append(new_row)
        idx = len(rows) - 1
    if iptm is not None:
        rows[idx][col] = f"{float(iptm):.4f}"
        if col == "iptm_htr1a":
            rows[idx]["score_htr1a"] = f"{round(1.0 - float(iptm), 4):.4f}"
    if rk_col in MASTER_FIELDS and ranking is not None:
        rows[idx][rk_col] = f"{float(ranking):.4f}"
    rows[idx]["updated_at"] = _now()
    if peptide and not rows[idx]["peptide"]:
        rows[idx]["peptide"] = peptide
    if var and not rows[idx]["var_region"]:
        rows[idx]["var_region"] = var
    _rewrite_master(rows)
    return rows[idx]


# ---------------------------------------------------------------------
# 结果提取与收割
# ---------------------------------------------------------------------

def find_summary(result_dir):
    """在结果目录中递归查找 *_summary_confidences.json, 找不到返回 None。"""
    hits = glob.glob(os.path.join(result_dir, "**", "*" + SUMMARY_SUFFIX),
                     recursive=True)
    return hits[0] if hits else None


def extract_scores(summary_path):
    """从磁盘上的 *_summary_confidences.json 提取 iptm/ptm/ranking。"""
    try:
        with open(summary_path) as f:
            c = json.load(f)
        return {"iptm": c.get("iptm", 0.0),
                "ptm": c.get("ptm", 0.0),
                "ranking_score": c.get("ranking_score", 0.0),
                "fraction_disordered": c.get("fraction_disordered", 0.0),
                "has_clash": c.get("has_clash", False)}
    except (OSError, ValueError):
        return None


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O", "UNK": "X",
}


def _seq_from_input_json(inp_path, chain_id="A"):
    """从 AF3 input.json 取指定链（默认 A = 小肽）的序列。"""
    try:
        with open(inp_path) as f:
            job = json.load(f)
    except (OSError, ValueError):
        return None
    for item in job.get("sequences", []):
        pv = item.get("protein", {}) if isinstance(item, dict) else {}
        ids = pv.get("id") if isinstance(pv, dict) else None
        if isinstance(ids, list) and ids and ids[0] == chain_id:
            return pv.get("sequence")
    return None


def _seq_from_cif(cif_path, entity_id="1"):
    """从 model.cif 的 _entity_poly_seq 循环还原某实体（默认 1 = A 链/小肽）序列。
    三字母码转单字母; 不认识的记为 X。旧时间戳结果常无 input.json, 以 cif 自证。"""
    try:
        lines = open(cif_path).read().splitlines()
    except OSError:
        return None
    n, i = len(lines), 0
    while i < n:
        if lines[i].strip() == "loop_":
            j, fields = i + 1, []
            while j < n and lines[j].strip().startswith("_entity_poly_seq."):
                fields.append(lines[j].strip()); j += 1
            ent = next((k for k, f in enumerate(fields)
                        if f == "_entity_poly_seq.entity_id"), None)
            mon = next((k for k, f in enumerate(fields)
                        if f == "_entity_poly_seq.mon_id"), None)
            if ent is not None and mon is not None:
                out, k = [], j
                while (k < n and lines[k].strip()
                       and not lines[k].strip().startswith("#")
                       and not lines[k].strip().startswith("loop_")):
                    parts = lines[k].split()
                    if len(parts) > max(ent, mon) and parts[ent] == entity_id:
                        out.append(_AA3TO1.get(parts[mon].upper(), "X"))
                    k += 1
                if out:
                    return "".join(out)
            i = j if j > i + 1 else i + 1
        else:
            i += 1
    return None


def pack_result_zip(result_dir, job_name):
    """把完成的输出目录打包成 <任务名>.zip（与服务器版下载包命名一致）。"""
    zip_path = os.path.join(results_dir(), f"{job_name}.zip")
    if os.path.exists(zip_path):
        return zip_path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(result_dir):
            for fn in files:
                if fn.startswith(_SKIP_FILE_PREFIXES) or fn in _SKIP_FILE_NAMES:
                    continue
                full = os.path.join(root, fn)
                zf.write(full, os.path.join(job_name, os.path.relpath(full, result_dir)))
    return zip_path


def harvest_results(results_root=None):
    """扫描结果目录, 把**新完成**的 AF3 结果提取评分并入库（幂等, 可反复调用）。

    以 (序列×靶标) 为一组处理: AF3 可能把输出写到规范目录 <tag>_<target>/
    或追加时间戳的兄弟目录 <tag>_<target>_<ts>/, 本函数先用 consolidate_task()
    折叠到规范目录再取分, 故两种布局都能收割; 折叠后该任务不再被误判为"未完成"
    而重跑。入库成功后在规范目录写 .harvested, 下次跳过。返回 (新增数, {名: 跳过原因})。"""
    results_root = results_root or results_dir()
    if not os.path.isdir(results_root):
        return 0, {}
    reg = load_sequence_registry()
    # 按 (tag, target) 归组（时间戳兄弟目录归入同一组）
    groups = {}
    for name in sorted(os.listdir(results_root)):
        full = os.path.join(results_root, name)
        if name.endswith(".zip") or not os.path.isdir(full) or name.startswith("_"):
            continue
        parsed = split_job_name(name)
        if parsed is not None:
            groups.setdefault(parsed, True)
    n_new, skipped = 0, {}
    for (tag, target) in sorted(groups):
        key = f"{tag}_{target.lower()}"
        canon = canonical_dir(tag, target, results_root)
        if os.path.isfile(os.path.join(canon, ".harvested")):
            continue
        summary = consolidate_task(tag, target, results_root)
        if summary is None:
            skipped[key] = "无 summary_confidences.json"
            continue
        scores = extract_scores(summary)
        if not scores:
            skipped[key] = "评分解析失败"
            continue
        info = reg.get(tag) or {}
        upsert_result(tag, target, scores["iptm"],
                      ranking=scores.get("ranking_score"),
                      peptide=info.get("peptide"), var=info.get("var"),
                      source="local")
        n_new += 1
        try:
            os.makedirs(canon, exist_ok=True)
            with open(os.path.join(canon, ".harvested"), "w") as f:
                f.write(_now() + "\n")
        except OSError:
            pass
    return n_new, skipped


# ---------------------------------------------------------------------
# 最佳结果维护: 复制 top-N HTR1A 结构包
# ---------------------------------------------------------------------

def refresh_best(off_threshold=0.8, top_n=10, targets=("HTR1A", "BIN1")):
    """按 (底线蛋白可行优先, iptm_htr1a 降序) 维护 BEST_TOP10/。

    只用启用靶标做可行性判定：UNC13C 未评估(None)时不参与, 待将来补齐
    后以 targets 含 UNC13C 重跑本函数即可用全约束重新排序、重选 top-N。"""
    by_tag = load_database()
    p = paths(ensure=True)
    rows = []
    for tag, r in by_tag.items():
        h = r.get("iptm_htr1a")
        if h is None:
            continue
        feas = True
        for t in targets:
            if t == "HTR1A":
                continue
            v = r.get(f"iptm_{t.lower()}")
            if v is not None and v >= off_threshold:
                feas = False
        rows.append((0 if feas else 1, -h, tag, r, feas))
    rows.sort(key=lambda x: (x[0], x[1]))
    best = rows[:top_n]

    best_dir = p["best_dir"]
    os.makedirs(best_dir, exist_ok=True)
    for old in glob.glob(os.path.join(best_dir, "*")):
        if os.path.isfile(old):
            os.remove(old)
        else:
            shutil.rmtree(old, ignore_errors=True)

    index = []
    for rank, (_, neg_h, tag, r, feas) in enumerate(best, 1):
        src = ensure_best_structure(tag)
        dest = os.path.join(best_dir, f"rank{rank:02d}_{tag}_HTR1A.zip")
        if src:
            shutil.copy(src, dest)
        index.append({"rank": rank, "tag": tag, "peptide": r.get("peptide", ""),
                      "iptm_htr1a": -neg_h, "iptm_bin1": r.get("iptm_bin1"),
                      "iptm_unc13c": r.get("iptm_unc13c"),
                      "feasible": feas, "has_structure": bool(src)})
    with open(os.path.join(best_dir, "BEST_TOP10.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "tag", "peptide", "iptm_htr1a",
                                          "iptm_bin1", "iptm_unc13c", "feasible",
                                          "has_structure"])
        w.writeheader()
        for it in index:
            w.writerow(it)
    return index


def ensure_best_structure(tag):
    """确保 tag 的 HTR1A 结果已在共享库打包成 zip, 返回 zip 路径或 None。"""
    rd = results_dir()
    zip_path = os.path.join(rd, f"{tag}_htr1a.zip")
    if os.path.isfile(zip_path):
        return zip_path
    d = os.path.join(rd, f"{tag}_htr1a")
    if os.path.isdir(d) and find_summary(d):
        return pack_result_zip(d, f"{tag}_htr1a")
    return None


# ---------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------

def stats():
    by_tag = load_database()
    return {
        "total": len(by_tag),
        "htr1a": sum(1 for r in by_tag.values() if r.get("iptm_htr1a") is not None),
        "bin1": sum(1 for r in by_tag.values() if r.get("iptm_bin1") is not None),
        "unc13c": sum(1 for r in by_tag.values() if r.get("iptm_unc13c") is not None),
        "root": db_root(),
    }


# ---------------------------------------------------------------------
# 旧结果迁移: 家目录 af3_local_results → 共享库（去时间戳 + 稳定标签）
# ---------------------------------------------------------------------

def migrate_legacy(legacy_results_dir=None, dry_run=False):
    """把家目录旧 af3_local_results/ 的**已完成**结果迁移进共享库。

    旧布局的问题:
      - 完成结果常被移到带时间戳的兄弟目录 seq_XXXX_htr1a_20260930_HHMMSS/,
        而当前代码按非时间戳名 seq_XXXX_htr1a 查找 → 永远找不到, 会被重算;
      - 旧标签用候选池下标(seq_0129), 无限扩池后会与新批次撞名。
    迁移做三件事:
      1) 按 input.json 还原 A 链(可变区)全序列, 生成稳定 tag=crc32(可变区);
      2) 去时间戳归并（同一 seq×靶标取最新时间戳目录的结果, 含 summary 优先）;
      3) 复制为 <tag>_<target>/ 并在 sequences.csv 注册 tag→序列, 直接入库,
         使已完成结果被复用（不重算）。UNC13C 超限目录跳过（留待高端 GPU）。
    返回 (迁移数, {名称: 跳过原因})。dry_run=True 只统计不写盘。"""
    legacy_results_dir = legacy_results_dir or "af3_local_results"
    if not os.path.isdir(legacy_results_dir):
        return 0, {"(missing)": legacy_results_dir}
    from peptide_common import FIXED_A
    p = paths(ensure=True)

    # 归组: (seqstem, target) → [(目录, 时间戳, 是否有 summary)];
    # 旧布局把同一结果的 input.json 与非时间戳目录、summary 与时间戳目录分开,
    # 故需收集同 stem 的全部目录: 序列从有 input.json 者取, 结果从有 summary 者取。
    groups = {}
    for name in sorted(os.listdir(legacy_results_dir)):
        src = os.path.join(legacy_results_dir, name)
        if not os.path.isdir(src) or name.startswith("_"):
            continue
        m = re.match(r"^(seq_\d+|step_init|step_\d+)_(htr1a|bin1|unc13c)"
                     r"(?:_(\d{8}_\d{6}))?$", name)
        if not m:
            continue
        stem, tgt, ts = m.group(1), m.group(2).lower(), (m.group(3) or "")
        groups.setdefault((stem, tgt), []).append(
            (src, ts, find_summary(src) is not None))

    migrated, skipped = 0, {}
    for (stem, tgt), dirs in sorted(groups.items()):
        label = f"{stem}_{tgt}"
        if tgt == "unc13c":
            skipped[label] = "UNC13C 在 V100 超限, 保留原状待高端 GPU"
            continue
        # 排序: 有 summary 优先 → 时间戳新优先; 首元素即结果目录（迁移复制对象）
        dirs.sort(key=lambda x: (x[2], x[1]), reverse=True)
        src = dirs[0][0]
        if not dirs[0][2]:
            skipped[label] = "未完成(无 summary), 跳过"
            continue
        # 还原 A 链序列: 优先组内任一 input.json, 回退到结果目录的 model.cif
        seq = None
        for s, _, _ in dirs:
            inp = os.path.join(s, "input.json")
            if os.path.isfile(inp):
                seq = _seq_from_input_json(inp)
                if seq:
                    break
        if not seq:
            for c in sorted(glob.glob(os.path.join(src, "**", "*_model.cif"),
                                      recursive=True)):
                seq = _seq_from_cif(c)
                if seq:
                    break
        if not seq:
            skipped[label] = "input.json 与 model.cif 均无法还原 A 链序列, 需人工核对"
            continue
        var = seq[len(FIXED_A):] if seq.startswith(FIXED_A) else seq[-8:]
        tag = resolve_tag(var)
        dest_name = f"{tag}_{tgt}"
        dest = os.path.join(p["results"], dest_name)
        if os.path.isdir(dest) and find_summary(dest):
            skipped[dest_name] = "共享库已有完成结果, 跳过"
            continue
        if not dry_run:
            register_sequence(tag, var, seq)
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest)
            for junk in (".submitted", ".harvested", ".oom_skipped"):
                jj = os.path.join(dest, junk)
                if os.path.isfile(jj):
                    os.remove(jj)
            sc = extract_scores(find_summary(dest))
            if sc:
                upsert_result(tag, tgt.upper(), sc["iptm"],
                              ranking=sc.get("ranking_score", 0.0),
                              peptide=seq, var=var, source="migrate")
                # 已内联入库, 打 .harvested 标记, 避免后续 harvest 重复扫描
                try:
                    with open(os.path.join(dest, ".harvested"), "w") as f:
                        f.write(_now() + " (migrate)\n")
                except OSError:
                    pass
        migrated += 1
    return migrated, skipped
