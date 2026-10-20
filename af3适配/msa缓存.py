#!/usr/bin/env python3
"""
msa缓存.py — AF3 MSA 磁盘缓存（跨进程/跨任务/跨作业复用 MSA 搜索结果）
=====================================================================
原理：
  AF3 输入 JSON 的每条蛋白链可内嵌 `unpairedMsa` / `pairedMsa` / `templates`三件套；三件齐全时 data pipeline **完全跳过搜索**（见 folding_input 与pipeline.process_protein_chain的分支："Has MSA and templates, don't search for anything"），特征与现场搜索逐字节等价 —— 因为缓存值全部取自 AF3 自己输出的 `<任务名>_data.json`（pipeline 的真实搜索结果），回放即复现，无任何近似。生效前提：数据库与 max_template_date 配置不变（本项目一直如此）。

为什么值得做：
  靶标链（HTR1A 422aa / BIN1 593aa / ASIC1A 526aa）序列恒定，其 MSA 搜索是每进程 ~78min 的固定开销；肽链 MSA ~708s/序列。内嵌后 HQ 单任务粒度（每任务独立进程）不再重复支付这些成本 —— 任务即可随时添加/取消而不破坏MSA摊销。

缓存布局：
  <缓存根>/<sha1（序列前20位）>.json = {"sequence":…, "unpairedMsa":…,"pairedMsa":…, "templates":[…]} 写入一律 tmp+rename 原子落盘，并发读写安全。

用法：
  python3 msa缓存.py extract <_data.json……>   # 从 AF3 输出提取链 MSA 入缓存
  python3 msa缓存.py embed   <任务.json……>     # 给输入 JSON 内嵌已缓存的 MSA
  python3 msa缓存.py harvest <_data.json……>   # 任务完成后的自愈入库（同 extract）
  python3 msa缓存.py stat                     # 缓存概况
"""

import hashlib
import json
import os
import sys
import glob

# 缓存根：共享盘，计算节点与登录节点都可见
CACHE_ROOT = os.environ.get(
    "PEPOPT_MSA_CACHE",
    "/public_bme2/Share200T/管吉松/peptide_opt_db/msa_cache",
)


def _key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()[:20]


def _path(sequence: str) -> str:
    return os.path.join(CACHE_ROOT, _key(sequence) + ".json")


def load(sequence: str):
    """查缓存；未命中返回 None。"""
    try:
        with open(_path(sequence)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save(sequence: str, unpaired: str, paired: str, templates: list):
    """原子写入缓存（tmp + rename）。已存在则跳过（首写优先，内容同源）。"""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    p = _path(sequence)
    if os.path.exists(p):
        return
    tmp = f"{p}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(
            {
                "sequence": sequence,
                "unpairedMsa": unpaired,
                "pairedMsa": paired,
                "templates": templates if templates is not None else [],
            },
            f,
        )
    os.replace(tmp, p)


def extract(paths) -> int:
    """从 AF3 输出的 _data.json 提取所有链的 MSA 三件套入缓存。返回新入库数。"""
    n_new = 0
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  ⚠️ 跳过 {p}: {e}")
            continue
        for entry in d.get("sequences", []):
            prot = entry.get("protein")
            if not prot:
                continue
            seq = prot.get("sequence", "")
            up = prot.get("unpairedMsa")
            if not seq or up is None:
                continue                      # 无 MSA 的链（如自定义）跳过
            if load(seq) is None:
                save(seq, up, prot.get("pairedMsa") or "",
                     prot.get("templates") or [])
                n_new += 1
    return n_new


def embed(paths) -> tuple:
    """把缓存中的 MSA 三件套内嵌进 AF3 输入 JSON；原地改写。

    返回（内嵌链数，缺缓存链数）。缺缓存的链保持原样 → AF3 会现场搜索，任务完成后 harvest 会把结果补入缓存（自愈）。"""
    n_hit = n_miss = 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        changed = False
        for entry in d.get("sequences", []):
            prot = entry.get("protein")
            if not prot or "unpairedMsa" in prot:
                continue                      # 已内嵌的链跳过
            seq = prot.get("sequence", "")
            c = load(seq) if seq else None
            if c is None:
                n_miss += 1
                continue
            prot["unpairedMsa"] = c["unpairedMsa"]
            prot["pairedMsa"] = c["pairedMsa"]
            prot["templates"] = c.get("templates") or []
            n_hit += 1
            changed = True
        if changed:
            tmp = f"{p}.tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, p)
    return n_hit, n_miss


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    op = sys.argv[1]
    args = sys.argv[2:]
    # 展开 shell 未展开的通配符（跨平台稳妥）
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])

    if op in ("extract", "harvest"):
        n = extract(paths)
        print(f"✅ {op}: 提取 {len(paths)} 个文件, 新入库 {n} 条链 MSA")
    elif op == "embed":
        hit, miss = embed(paths)
        print(f"✅ embed: 内嵌 {hit} 条链, 缺缓存 {miss} 条（将现场搜索并自愈）")
    elif op == "stat":
        files = glob.glob(os.path.join(CACHE_ROOT, "*.json"))
        tgt = pep = 0
        for p in files:
            try:
                with open(p) as f:
                    n = len(json.load(f).get("sequence", ""))
            except Exception:
                continue
            if n <= 35:
                pep += 1
            else:
                tgt += 1
        print(f"缓存根: {CACHE_ROOT}\n链条目: {len(files)}（靶标 {tgt} / 肽链 {pep}）")
    else:
        print(f"未知操作: {op}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
