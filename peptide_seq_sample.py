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

import peptide_db as db
from peptide_stats import BetaFit, consolidation_pass, trigger_power_of_two


def build_product_samples(target_samples, fixed_factors):
    """构造目标乘积样本序列（β 用标准 [0,1]，乘积本身 ∈ [0,1] 无需缩放）。

    target_samples: {靶标: [逐样本 ipTM]} —— 本次新算的（变因子）。
    fixed_factors:  {靶标: 常数 ipTM}   —— 归档复用的（固定因子）。
    返回 products 列表。按样本下标配对相乘; 长度取变因子的最短者。
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


def consolidation_sampler_factory(new_sample_fn):
    """为 consolidation_pass 构造 sampler：优先取池，池空调 new_sample_fn 生成。

    new_sample_fn(tag) → 新乘积样本值（如跑一次新 seed 的 AF3）；无法生成返回 None。"""
    def sampler(tag):
        fit = db.load_obj_fit(tag)
        if fit is None:
            return None
        x = fit.draw_from_pool()
        if x is not None:
            db.save_obj_stats(tag, fit)
            return x
        x = new_sample_fn(tag)
        if x is not None:
            fit.add(x)
            fit.extra_used += 1
            db.save_obj_stats(tag, fit)
        return x
    return sampler


def run_consolidation(new_sample_fn, active_targets=("ASIC1A", "HTR1A", "BIN1")):
    """数据库整理主流程（用户规格）：
      1）按最优乘积 obj_max 降序；
      2）对第 K 名：阈值 = 第 K-1 名 obj_max（K=1 用（max+1）/2），P̃ = min(β尾概率, CP上界)；若 P̃×(N-K+2)>1 补采，更新拟合与排名，循环。
    返回 consolidation_pass 的统计。"""
    D = db.load_database()
    records = []
    for tag, r in D.items():
        fit = db.load_obj_fit(tag)
        if fit is not None and fit.max_x is not None:
            records.append({"tag": tag, "fit": fit})
    if not records:
        return {"added": 0, "rounds": 0, "capped": 0}
    sampler = consolidation_sampler_factory(new_sample_fn)
    res = consolidation_pass(records, sampler)
    for rec in records:                    # 把更新后的统计量写回
        db.save_obj_stats(rec["tag"], rec["fit"])
    return res


def maybe_consolidate(total_records, new_sample_fn):
    """记录数达到 2 的整数次幂时触发整理。返回是否触发 + 结果。"""
    if trigger_power_of_two(total_records):
        print(f"  🔧 数据库整理触发（记录数 {total_records} = 2^k）", flush=True)
        return True, run_consolidation(new_sample_fn)
    return False, None
