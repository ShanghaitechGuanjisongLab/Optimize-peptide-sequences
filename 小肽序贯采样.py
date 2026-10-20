"""
序贯采样 × 目标乘积 × β 拟合 —— 一体化入库（集成层）
=====================================================
把用户的三件套连起来：
  1）逐样本提取（AF3 每 seed-K_sample-J 是一次独立扩散抽样）；
  2）目标乘积样本 = 各靶标逐样本配对相乘（固定因子用常数）；
  3）序贯规则消费样本 → 增量充分统计（n，Σx，Σx²，U）+ 缩放 β 拟合 → 入库。

关键口径决策（有实测依据）：
  - **必须逐样本取 max，不能用 AF3 汇总**——实测三聚体：逐样本界面 ipTM [0.20, 0.20, 0.22, 0.18, 0.18]，max=0.22；而汇总文件报 0.18（它按 ranking_score 0.8·ipTM_global+0.2·pTM 选样本，不是界面 ipTM 最大者）。
  - **U = Π（固定因子）**：归档行 HTR1A/BIN1 固定 → U = 二者乘积；全变记录 U = 1。此时若阈值 > U 则 P=0 结构性成立（该记录数学上不可能超越）。
  - **入库存 argmax 样本的各分量**：使 objective_product = Π iptm_<target> 严格成立。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import 小肽数据库 as db
from 小肽统计 import BetaFit, consolidation_pass, trigger_power_of_two


def build_product_samples(target_samples, fixed_factors):
    """构造目标乘积样本序列（β 用标准 [0,1]，乘积本身 ∈ [0,1] 无需缩放）。

    target_samples：{靶标：[逐样本 ipTM]} —— 本次新算的（变因子）。
    fixed_factors：{靶标：常数 ipTM}   —— 归档复用的（固定因子）。
    返回 products 列表。按样本下标配对相乘；长度取变因子的最短者。
    """
    if not target_samples:
        raise ValueError("target_samples 不能为空")
    lens = [len(v) for v in target_samples.values() if v]
    if not lens:
        raise ValueError("无有效逐样本数据")
    m = min(lens)
    fixed_prod = 1.0
    for v in fixed_factors.values():
        fixed_prod *= float(v)
    products = []
    for i in range(m):
        p = fixed_prod
        for vals in target_samples.values():
            p *= float(vals[i])
        products.append(p)
    return products


def ingest_record(tag, target_samples, fixed_factors, peptide=None, var=None):
    """一条记录的序贯采样入库：乘积样本 → 序贯规则 → β 拟合 → 写统计量。

    返回 {“used”，“max”，“alpha”，“beta”，“pool_size”} 或 None（无有效样本）。"""
    products = build_product_samples(target_samples, fixed_factors)
    fit = BetaFit()
    fit.consume_sequential(products)

    if fit.max_x is None or fit.n == 0:
        return None

    # 入库存 argmax 样本的各分量（使 objective_product = Π iptm_<t> 严格成立）
    # 找到 max 对应的样本下标（在已消费的前 fit.n 个中）
    consumed = products[:fit.n]
    argmax = max(range(len(consumed)), key=lambda i: consumed[i])
    for t, vals in target_samples.items():
        db.upsert_result(tag, t, float(vals[argmax]), peptide=peptide, var=var,
                         source="seq_sampled")
    for t, v in fixed_factors.items():
        db.upsert_result(tag, t, float(v), peptide=peptide, var=var,
                         source="archived")

    db.save_obj_stats(tag, fit)
    ab = fit.fit()
    return {"used": fit.n, "max": fit.max_x,
            "alpha": ab[0] if ab else None, "beta": ab[1] if ab else None,
            "pool_size": len(fit.pool)}


# ==================== 逐样本入库与整理补采（集成入口）====================

# 目标乘积因子：真源在 小肽数据库.PRODUCT_TARGETS（总开关；2026-10-20 停算 ASIC1A）。此处转发以免反向依赖入口模块。
PRODUCT_TARGETS = db.PRODUCT_TARGETS


def collect_product_factors(tag, row=None, product_targets=PRODUCT_TARGETS):
    """收集一条记录的乘积因子：结果目录里有逐样本数据的靶标 → 变因子（样本列表）；无逐样本但库中有值 → 固定因子（常数）。

    任一产品靶标两者皆无 → 返回 None（因子不齐，乘积无法构造）。"""
    row = row if row is not None else (db.load_database().get(tag) or {})
    target_samples, fixed_factors = {}, {}
    for t in product_targets:
        samples = db.extract_sample_iptms(db.canonical_dir(tag, t), t)
        if samples:
            target_samples[t] = samples
            continue
        v = row.get(f"iptm_{t.lower()}")
        if v is not None:
            fixed_factors[t] = v
            continue
        return None
    return target_samples, fixed_factors


def ingest_record_from_results(tag, row=None):
    """一条记录的序贯采样入库（幂等：已有 β 统计量则跳过）。返回 ingest_record 结果或 None。

    全固定因子（三个靶标都只有常数）时不开统计量：无可采样随机性，后续补采也无从谈起。"""
    row = row if row is not None else (db.load_database().get(tag) or {})
    if (row.get("obj_n") or 0) > 0:
        return None
    fac = collect_product_factors(tag, row)
    if fac is None:
        return None
    target_samples, fixed_factors = fac
    if not target_samples:
        return None
    return ingest_record(tag, target_samples, fixed_factors,
                         peptide=row.get("peptide"), var=row.get("var"))


def ingest_new_records(new_sample_fn=None):
    """序贯采样入库总入口：扫描库里“因子齐备且尚无 β 统计量”的记录逐条入库；每入一条做一次 2^k 整理触发检查。返回入库条数。

    new_sample_fn：整理阶段的批量补采生成器（见 run_consolidation）；留 None 时整理只消费池内免费样本。"""
    D = db.load_database()
    fitted = sum(1 for r in D.values() if (r.get("obj_n") or 0) > 0)
    n = 0
    for tag in sorted(D):
        row = D[tag]
        if (row.get("obj_n") or 0) > 0 or not row.get("peptide"):
            continue
        if ingest_record_from_results(tag, row) is None:
            continue
        n += 1
        fitted += 1
        maybe_consolidate(fitted, new_sample_fn)
    return n


def run_consolidation(new_sample_fn, max_rounds=24):
    """数据库整理主流程（用户规格，补采批量化）：
      1）按最优乘积 obj_max 降序；
      2）对第 K 名：阈值 = 第 K-1 名 obj_max（K=1 用（max+1）/2），P̃ = min（β尾概率，CP上界）；若 P̃×(N-K+2)>1 补采，更新拟合与排名，循环。

    补采以“批量真跑 AF3”实现，两阶段循环（AF3 一次可打包多任务，chain-B MSA 每靶标只建一次，逐条单跑太浪费）：
      阶段1  consolidation_pass 配“只登记需求”的 sampler 跑到收敛：免费消费各记录 pool 里的样本，并收集所有“池空但 P̃·N>M 仍成立”的记录；
      阶段2  对登记需求批量真跑 AF3 生成新乘积样本，全部挂进对应记录的 pool，交下一轮 pass 按设计逐个消费、重排、再判；直到无人达标、补采无果或补采批数用尽。
    new_sample_fn：fn(tags) -> {tag：[新乘积样本]}（批量接口，无法生成的 tag 不出现在结果里）；None 时只消费池内免费样本。
    返回 consolidation_pass 各轮统计之和 {"added", "rounds", "capped"}。"""
    max_batches = max(1, int(os.environ.get("PEPOPT_RESAMPLE_BATCHES", "2")))
    max_tags = max(1, int(os.environ.get("PEPOPT_RESAMPLE_TAGS", "2")))
    D = db.load_database()
    records, fits = [], {}
    for tag, r in D.items():
        fit = db.load_obj_fit(tag)
        if fit is not None and fit.max_x is not None:
            records.append({"tag": tag, "fit": fit})
            fits[tag] = fit
    if not records:
        return {"added": 0, "rounds": 0, "capped": 0}
    total = {"added": 0, "rounds": 0, "capped": 0}
    batches = 0
    for _ in range(max_rounds):
        demand = []

        def sampler(tag):                 # 只登记需求；样本由阶段2 批量补给 pool
            demand.append(tag)
            return None

        res = consolidation_pass(records, sampler)
        for k in total:
            total[k] += res.get(k, 0)
        if not demand or new_sample_fn is None or batches >= max_batches:
            break
        batches += 1
        batch = new_sample_fn(list(dict.fromkeys(demand))[:max_tags]) or {}
        if not batch:
            break
        for tag, xs in batch.items():
            fit = fits.get(tag)
            if fit is not None:
                fit.pool.extend(float(v) for v in xs)
        for rec in records:               # 每轮写回：中途断电最多丢一轮
            db.save_obj_stats(rec["tag"], rec["fit"])
    for rec in records:                   # 把更新后的统计量写回
        db.save_obj_stats(rec["tag"], rec["fit"])
    return total


def maybe_consolidate(total_records, new_sample_fn):
    """记录数达到 2 的整数次幂时触发整理。返回是否触发 + 结果。"""
    if trigger_power_of_two(total_records):
        print(f"  🔧 数据库整理触发（记录数 {total_records} = 2^k）", flush=True)
        return True, run_consolidation(new_sample_fn)
    return False, None
