"""
小肽序列优化 — 贝叶斯优化（有限预算）入口
=============================================================
⚠️ 目标口径声明（2026-10-22 校正）：本入口仍在旧口径上跑——采集函数 = 纯 EI，y = score_htr1a = 1 − ipTM(HTR1A)（单目标最大化 HTR1A），**未接入目标乘积**；而当前主推链路（`小肽优化_节点内.py` + `小肽优化_无限.py`，即共享数据库上的长驻循环作业）已改为 y = 1 − ipTM(HTR1A) × ipTM(BIN1) 的乘积最大化。两者不可混读，本入口的 BEST_STRUCTURE / result_bo.xlsx 排序不代表全库最佳。若要统一到乘积目标需改 5 处：y_obs（run_bo 主循环）、best_y、best_feasible_row 判定、history_y、收尾报表；改完必须在离线假分数上实测（本入口会真提交 AF3 任务，不能拿集群作业当测试台）。下面的“搜索策略”描述的是历史设计（含已取消的双底线约束），保留仅供考古对比。

为什么换掉蒙特卡洛：
  AF3 单次评估昂贵（每候选 ×3 次预测），本项目真正的瓶颈是“评估预算”（服务器版每日约 30 个任务 ≈ 每天 10 个候选）。贝叶斯优化用代理模型（高斯过程）从已评估样本中学习“序列特征 → 分数”的映射，每轮只把评估机会花在最有希望的候选上，样本效率远高于随机抽样式的蒙特卡洛/遗传算法。

搜索策略：
  1. 候选库：预生成约 1500 个编辑距离 ≤ 2 的可变区突变（与原版相同规则）
  2. 初始设计：评估初始序列 + 一组随机候选，作为冷启动
  3. 每轮：
     a）拟合 3 个独立高斯过程（GP）：
        - GP_target：score = 1 - ipTM(HTR1A)          （要最小化）
        - GP_unc：ipTM(UNC13C)                      （约束 < 0.8）
        - GP_bin：ipTM(BIN1)                        （约束 < 0.8）
     b）采集函数 = 约束期望改进（constrained Expected Improvement）：
        cEI(x) = EI(x) × P(ipTM_unc < 0.8 | x) × P(ipTM_bin < 0.8 | x) 即“改进潜力 × 对每个底线蛋白的可行概率”，比原版硬阈值（双 ≥0.8 即丢弃）更平滑，避免阈值附近的浪费。
        ❗ **现状（与上式不符）**：run_bo 主循环已改为传空约束表（`constrained_acquisition(gp_t, [], …)`），
        即约束因子已删除、GP_unc/GP_bin 不再拟合；下面的 fit_active_gp / feasible_prob / OFF_THRESHOLD
        因此处于**休眠状态**（无调用方），仅为“日后要恢复底线约束时不必重写”而保留。
     c）按 cEI 贪心选出 batch 个候选（带特征空间多样性约束），提交评估；完成后写入 CSV，进入下一轮。

复用的部分（从 小肽优化_云端 导入）：
  - 序列常量、generate_mutant() 突变规则、create_af3_json()、extract_scores_from_zip() 评分提取；
  - predict_and_score()（服务器版：JSON 生成/缓存评分）。

运行方式：
    python 小肽优化_BO.py             # 默认：本地集群推理（无每日配额限制）
    python 小肽优化_BO.py --server    # 改用 AlphaFold Server 云端预测（手动/自动模式同主脚本）
  与其他入口一样支持断点续跑：已评估候选记录在 output_bo/bo_evaluations.csv，已完成的 AF3 结果缓存直接复用。

输出：
    output_bo/bo_evaluations.csv    已评估候选：序列 + 三个靶标的分数/约束值
    output_bo/bo_curve.png          最优 score 随评估数的改进曲线
    output_bo/BEST_STRUCTURE/       最佳序列的 AF3 结果包（复制）
    output_bo/result_bo.xlsx        最终最优序列表

合规/评分说明：score = 1 - ipTM 仅是界面置信度的代理，并非真实亲和度，最优序列仍需湿实验验证。AlphaFold 服务器版受每日配额、本地版受 GPU 排队约束。
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_missing = []
for _m in ("sklearn", "scipy"):
    try:
        __import__(_m)
    except ImportError:
        _missing.append(_m)
if _missing:
    sys.exit(f"❌ 缺少依赖：{_missing}\n   请先执行：pip install scikit-learn scipy")

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from 小肽公共库 import (SEQ_202, FIXED_A, ORIG_TAIL,
                            HTR1A_SEQ, UNC13C_SEQ, BIN1_SEQ,
                            generate_mutant)

# ===================== 靶标配置 =====================

# 三个靶标各自独立预测。UNC13C 复合物约 2233 tokens，超过 bme_gpupub 分区 V100 的官方上限 1280 tokens （见 小肽优化_本地集群.GPU_MAX_TOKENS），在 V100 上必然显存溢出，因此支持 --targets HTR1A,BIN1 只跑 V100 放得下的两个靶标（结果独立存放）；待申请到 A100/H100 再以全靶标运行补齐 UNC13C 约束。
TARGET_SEQS    = {"HTR1A": HTR1A_SEQ, "UNC13C": UNC13C_SEQ, "BIN1": BIN1_SEQ}
TARGET_ORDER   = ["HTR1A", "UNC13C", "BIN1"]   # 固定评估顺序
OFF_TARGETS    = ["UNC13C", "BIN1"]            # 旧阶段“底线蛋白”语义（ipTM 越低越好）；本入口仅用作报表明细列名，不作判定
_OFF_COL       = {"UNC13C": "iptm_unc13c", "BIN1": "iptm_bin1"}   # 评估 CSV 列名

# ===================== 贝叶斯优化配置 =====================

BO_ROOT        = "output_bo"                 # 本方法输出根目录（靶标子集运行时指向独立目录）
EVAL_CSV       = os.path.join(BO_ROOT, "bo_evaluations.csv")
ACTIVE_TARGETS = list(TARGET_ORDER)            # 本次实际评估的靶标（--targets 可缩减）
ACTIVE_OFF     = [t for t in OFF_TARGETS if t in ACTIVE_TARGETS]
OFF_THRESHOLD  = 0.8                          # ❗休眠：仅被 feasible_prob 引用，而后者已无调用方（约束已从采集函数中删除）
N_INIT         = 8                            # 冷启动随机评估数（不含初始序列）
DEFAULT_POOL   = 1500                         # 候选库大小
DEFAULT_BATCH  = 8                            # 每轮评估数（≈ 每日配额 / 3）
DEFAULT_BUDGET = 120                          # 总评估预算（候选数，不含初始序列亦可含）
XI             = 1e-3                         # EI 探索项
TOP_K          = 50                           # 贪心多样性选择时的短名单长度

# ==========================================================


def set_active_targets(names):
    """只启用 names 中的靶标（必须含 HTR1A）。

    非全集时输出根目录改为 output_bo_<靶标小写连字符>，与全量评估的记录分开存放，避免同一 CSV 中混入 NaN 约束列。AF3 推理结果目录（af3_local_results/<seq_任务名>）按 序列×靶标 命名，与本设置无关，将来补齐 UNC13C 时已完成的 HTR1A/BIN1 结果仍直接复用。返回被搁置（未启用）的靶标列表。"""
    global ACTIVE_TARGETS, ACTIVE_OFF, BO_ROOT, EVAL_CSV
    ACTIVE_TARGETS = [t for t in TARGET_ORDER if t in names]
    ACTIVE_OFF = [t for t in OFF_TARGETS if t in ACTIVE_TARGETS]
    if len(ACTIVE_TARGETS) < len(TARGET_ORDER):
        BO_ROOT = "output_bo_" + "-".join(t.lower() for t in ACTIVE_TARGETS)
    else:
        BO_ROOT = "output_bo"
    EVAL_CSV = os.path.join(BO_ROOT, "bo_evaluations.csv")
    return [t for t in TARGET_ORDER if t not in ACTIVE_TARGETS]


# ---------------- 序列特征化 ----------------

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
_HYDROPHOBIC = set("AILMFWVP")
_AROMATIC = set("FWY")
_POLAR = set("STNQ")
_AA_INDEX = {a: i for i, a in enumerate(AA_LIST)}    # 位置 one-hot 反查用（AA_LIST 是 list，无 str.find）
# 位置 one-hot 的通道数：可变区枚举空间长 6~10（见 小肽优化_无限.PEPOPT_SPACE_LEN_MIN/MAX），按上限开通道、短序列尾部补零；更长的可变区被截断到前 POS_MAX_LEN 位（长度信息另由基础特征的 len 通道承载）。
POS_MAX_LEN = int(os.environ.get("PEPOPT_POS_MAX_LEN", "10") or 10)
# 特征构造器开关：positional = 逐位 one-hot + 全局组成（默认），composition = 26 维纯全局组成。所有 GP 调用点统一走 feat_of() 读这个开关，避免同一批数据被两种维度不同的特征混用（那会直接抛矩阵维不匹配的错）。
# 默认取 positional 的依据：共享库 745 条真实评估数据、计算节点上跑的 5-fold 交叉验证（划分各组合共用）把 Spearman 从 0.371 提到 0.542、TopK 命中率从 0.322 提到 0.411（K=18 = 每周期 24 序列中 GP 的配额），全空间 50322 条待办的预测乘积上限从 0.3884 提到 0.4886（库内真实榜首 0.5580）；定性上 composition 挑出的 18 条散落在 *VDVV*S 的各种替换位上，positional 则聚拢到 SIVD_V_CS 骨架的邻域（库里第 3、4 名 SIVDEVCS=0.5456、SIVDDVVCS=0.5368 都在这骨架上），即它才具备在已知高分骨架附近做局部精调的能力。代价是全空间打分端到端 4.4s→10.2s，占周期预算（n=24 时约 7.7h）的 0.04%。已知局限：时序留出（按 created_at 前 80% 训、后 20% 测）下 positional 并无优势（Spearman 0.505→0.487），因 226 维对 596 个样本更易过拟合位置模式，但随机 CV 样本量更大且提升显著。退回纯组成特征：PEPOPT_FEATURE_MODE=composition。
FEATURE_MODE = os.environ.get("PEPOPT_FEATURE_MODE", "positional").strip().lower()


def featurize(seq):
    """序列 → 26 维数值特征：20 个氨基酸组成 + 长度/净电荷/疏水/芳香/极性/破结构比例"""
    n = max(len(seq), 1)
    comp = [seq.count(a) / n for a in AA_LIST]
    charge = (sum(seq.count(a) for a in "RKH") - sum(seq.count(a) for a in "DE")) / n
    hydro = sum(seq.count(a) for a in _HYDROPHOBIC) / n
    aro = sum(seq.count(a) for a in _AROMATIC) / n
    pol = sum(seq.count(a) for a in _POLAR) / n
    disorder = sum(seq.count(a) for a in "PG") / n
    return np.array(comp + [len(seq), charge, hydro, aro, pol, disorder], dtype=float)


def featurize_positional(seq):
    """序列 → 位置分辨特征（POS_MAX_LEN×20 + 26 维）：可变区逐位 one-hot（左对齐、尾部补零）拼上 featurize 的全局组成/物性特征。

    为何需要：featurize 只有**全局**组成比例，对“编辑距离 ≤2、可变区仅 8 位”的近邻几乎无区分力——同一组氨基酸打乱位置得到的序列 26 维特征完全相同，而组成只差一两个残基的序列特征也只差几个百分点，于是 GP 无从知道“哪一位换成什么”才有用。升维前的病征（共享库 745 条真实评估数据、全空间 50322 条待办，计算节点实测）：预测乘积上限仅 0.3884 而库内真实榜首已达 0.5580，5-fold 的 Spearman 只有 0.371、TopK 命中率 0.322；升维后依次变为 0.4886 / 0.542 / 0.411。

    一个容易误判的点：EI 的改进项 imp = best_y − μ − ξ 在这份数据上恒为负（四个特征×标签组合下 imp>0 的候选都是 0 条），这**不是**特征或标签缺陷的证据而是 GP 回归的固有性质：μ 是训练标签的凸组合，除非靠核外推几乎不可能低于历史最优值，EI 的探索价值本来就主要由不确定性项 σ·φ(z) 承担。所以衡量特征好坏要看留出集上的预测精度（Spearman / TopK 命中率），不要看 imp 是否转正。

    左对齐的理由：固定区（TAT）在 N 端且长度恒定，所有候选完全相同，剔出去后逐位比对可变区才不会让前 11 个位置通道全是常数偏置。非标准氨基酸（如 X）所在位保持全零，等价于一个 gap 通道。
    """
    tail = seq[len(FIXED_A):] if seq.startswith(FIXED_A) else seq
    k = len(AA_LIST)
    onehot = [0.0] * (POS_MAX_LEN * k)
    for i, aa in enumerate(tail[:POS_MAX_LEN]):
        j = _AA_INDEX.get(aa)
        if j is not None:                            # 非标准氨基酸（如 X）→ 该位保持全零
            onehot[i * k + j] = 1.0
    return np.concatenate((np.asarray(onehot, dtype=float), featurize(seq)))


def feat_of(seq):
    """按 FEATURE_MODE 选特征构造器（唯一入口），本模块的 run_bo 与 小肽优化_无限 的两条挑选路径均经此。

    维度对照：composition = 26；positional = POS_MAX_LEN×20 + 26 = 226（默认上限 10）。GP 建核为 O(n²·d)、Cholesky 分解为 O(n³)，故升维只影响建核那一项（d 放大 8.7 倍，n 不变）；穷尽队列每周期要对**整个待办集**（枚举全集 51091 条减去已派发台账，当前约 5 万条）打分，这笔开销须在周期预算（n=24 时约 7.7h）内：升维前实测全空间打分仅 1.0s（26 维、n_train=34，见 PEPOPT_SCORE_POOL 注释），升维后（226 维、n_train=745）实测拟合 2.8s + 预测 5.9s、端到端 10.2s，仍是预算的 0.04%。POS_MAX_LEN 必须不小于枚举空间的长度上限（PEPOPT_SPACE_LEN_MAX，默认 10），否则可变区尾部会被静默截断而丢掉信息。"""
    return featurize_positional(seq) if FEATURE_MODE == "positional" else featurize(seq)


# ---------------- 候选库 ----------------

def build_candidate_pool(n_candidates, seed=42):
    """按原版突变规则预生成候选库（去重，含初始序列）"""
    random.seed(seed)
    pool = {SEQ_202}
    while len(pool) < n_candidates + 1:
        new_tail = generate_mutant(ORIG_TAIL)
        pool.add(FIXED_A + new_tail)
    pool = sorted(pool)
    print(f"📦 候选库：{len(pool)} 个唯一序列（含初始序列）")
    return pool


# ---------------- 高斯过程 ----------------

def make_gp():
    kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)) + \
             WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e1))
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0,
                                    normalize_y=True, random_state=0)


def fit_gp(X, y):
    gp = make_gp()
    gp.fit(X, y)
    return gp


def fit_active_gp(X_s, rows, target_name):
    """拟合底线蛋白 target_name 的约束 GP。该靶标未启用、或有限样本不足 3 个时返回 None（不参与采集函数）。

    ❗当前无调用方（约束已从采集函数中删除），休眠保留。"""
    if target_name not in ACTIVE_TARGETS:
        return None
    y = pd.to_numeric(rows[_OFF_COL[target_name]], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    return fit_gp(X_s[m], y[m])


# ---------------- 采集函数 ----------------

def expected_improvement(mu, sd, best_y, xi=XI):
    """最小化问题的期望改进"""
    sd = np.maximum(sd, 1e-9)
    imp = best_y - mu - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def feasible_prob(gp_off, X):
    """P（ipTM_off < 阈值 | x）。❗当前无调用方（run_bo 传空约束表）；保留备日后恢复底线语义时使用。"""
    mu, sd = gp_off.predict(X, return_std=True)
    sd = np.maximum(sd, 1e-9)
    return norm.cdf((OFF_THRESHOLD - mu) / sd)


def constrained_acquisition(gp_target, gp_con_list, X_cand, best_y):
    """cEI：期望改进 × 对每个已启用底线蛋白的可行概率（gp_con_list 只含本次运行激活的约束 GP，可为空 = 纯 EI）。"""
    mu, sd = gp_target.predict(X_cand, return_std=True)
    acq = expected_improvement(mu, sd, best_y)
    for gp in gp_con_list:
        acq = acq * feasible_prob(gp, X_cand)
    return acq


def select_batch(acq, X_cand, indices, batch_size):
    """按采集值取 TOP_K 短名单，再贪心挑选特征空间中彼此远离者（多样性）"""
    order = np.argsort(-acq)[:TOP_K]
    chosen = [order[0]]
    while len(chosen) < min(batch_size, len(order)):
        remaining = [i for i in order if i not in chosen]
        if not remaining:
            break
        def key(i):
            d = np.linalg.norm(X_cand[i] - X_cand[chosen], axis=1)
            return (acq[i], d.min())
        chosen.append(max(remaining, key=key))
    return [indices[i] for i in chosen]


# ---------------- 评估流程 ----------------

def evaluate_sequence(seq, seq_tag, out_root, predict_fn):
    """对单个序列评估全部激活靶标（默认 3 个，--targets 可缩减），全部就绪时返回 dict，否则 None。

    关键：先遍历所有激活靶标、逐个创建并提交预测任务，**不因某个靶标结果未就绪就提前 return**。否则该靶标后面的靶标任务永远不会被创建——例如 UNC13C 因token超限被拦截而始终未就绪时，排在它后面的 BIN1 会被连带锁死，表现为“BIN1 尚未生成任务”。所有激活靶标任务都创建/提交后，只要有任一未就绪就返回 None（已创建的任务下次重跑继续，不会重复提交）。"""
    res = {}
    for target_name in ACTIVE_TARGETS:
        target_seq = TARGET_SEQS[target_name]
        out_dir = os.path.join(out_root, f"{seq_tag}_{target_name}")
        score, iptm, path = predict_fn(seq, target_seq, target_name, out_dir,
                                       auto_submit=True)
        # 未就绪不 return，继续把后续靶标任务创建出来
        res[target_name] = None if iptm is None else {
            "score": score, "iptm": iptm, "path": path}
    if any(v is None for v in res.values()):
        return None
    return res


def load_evaluations():
    """续跑：读取已评估候选记录（返回 DataFrame；直接用 csv 模块避免pandas对 .csv 写入时的压缩依赖探测问题）"""
    cols = ["peptide", "score_htr1a", "iptm_unc13c", "iptm_bin1", "round"]
    if os.path.isfile(EVAL_CSV):
        rows = []
        with open(EVAL_CSV, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({c: r[c] for c in cols})
        df = pd.DataFrame(rows, columns=cols)
        for c in cols[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"⏩ 发现 {len(df)} 条已评估记录：{EVAL_CSV}（断点续跑）")
        return df
    return pd.DataFrame(columns=cols)


def append_eval_row(row):
    """追加写入一条评估记录（首次写入时带表头）"""
    header = not os.path.isfile(EVAL_CSV)
    with open(EVAL_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["peptide", "score_htr1a",
                                          "iptm_unc13c", "iptm_bin1", "round"])
        if header:
            w.writeheader()
        w.writerow(row)


def feasible(row):
    """约束已全部取消：所有行都可行（HTR1A 和 BIN1 不设上限）。"""
    return True


# ---------------- 主循环 ----------------

def run_bo(predict_fn, pool_size=DEFAULT_POOL, batch_size=DEFAULT_BATCH,
           budget=DEFAULT_BUDGET):
    os.makedirs(BO_ROOT, exist_ok=True)
    pool = build_candidate_pool(pool_size)
    X_pool = np.array([feat_of(s) for s in pool])
    scaler = StandardScaler()
    X_pool_s = scaler.fit_transform(X_pool)

    df_eval = load_evaluations()
    evaluated_seqs = set(df_eval["peptide"]) if len(df_eval) else set()

    # ---- 冷启动：初始序列 + 随机候选 ----
    init_needed = []
    if SEQ_202 not in evaluated_seqs:
        init_needed.append(pool.index(SEQ_202))
    rng = np.random.default_rng(42)
    others = [i for i in range(len(pool)) if pool[i] not in evaluated_seqs]
    n_pick = max(N_INIT - len(init_needed), 0)
    init_needed.extend(list(rng.choice(others, size=min(n_pick, len(others)),
                                       replace=False)))

    best_feasible_row = None
    best_path = None
    history_y = []          # 每个评估点的 best-so-far

    def track_best(row, path_htr1a):
        nonlocal best_feasible_row, best_path
        if feasible(row):
            if best_feasible_row is None or row["score_htr1a"] < best_feasible_row["score_htr1a"]:
                best_feasible_row = row
                best_path = path_htr1a

    def save_new_row(peptide, res, round_no):
        nonlocal df_eval
        row = {"peptide": peptide,
               "score_htr1a": res["HTR1A"]["score"],
               "iptm_unc13c": res["UNC13C"]["iptm"] if "UNC13C" in res else float("nan"),
               "iptm_bin1": res["BIN1"]["iptm"] if "BIN1" in res else float("nan"),
               "round": round_no}
        append_eval_row(row)
        df_eval = pd.concat([df_eval, pd.DataFrame([row])], ignore_index=True)
        track_best(row, res["HTR1A"]["path"])
        history_y.append(best_feasible_row["score_htr1a"]
                         if best_feasible_row is not None else np.nan)

    # 先把历史记录重放进 best 轨迹（续跑情形）
    for _, r in df_eval.iterrows():
        track_best(r, None)

    def evaluate_batch(idx_list, round_no, out_subdir):
        """评估一组候选；返回实际完成数"""
        done = 0
        for idx in idx_list:
            seq = pool[idx]
            if seq in set(df_eval["peptide"]):
                continue
            out_root = os.path.join(BO_ROOT, out_subdir, f"seq_{idx:04d}")
            res = evaluate_sequence(seq, f"seq_{idx:04d}", out_root, predict_fn)
            if res is None:
                print(f"  ⏳ [{round_no}] seq_{idx:04d} 结果未就绪，跳过（下次重跑继续）")
                continue
            save_new_row(seq, res, round_no)
            done += 1
        return done

    round_no = 0
    total_new = round_no

    # ---- 冷启动评估 ----
    if init_needed:
        round_no += 1
        print(f"\n== 轮次 {round_no}：冷启动评估 {len(init_needed)} 个候选 ==")
        evaluate_batch(init_needed, round_no, "init")

    # ---- 贝叶斯优化主循环 ----
    while len(df_eval) < budget and round_no < 500:
        n_eval = len(df_eval)
        round_no += 1
        print(f"\n== 轮次 {round_no}：拟合代理模型并挑选候选 "
              f"（已评估 {n_eval}/{budget}）==")

        rows = df_eval
        if len(rows) == 0:
            print("  ⚠️ 尚无已评估候选（预测任务均未就绪），等待结果完成后重跑本脚本。")
            break
        X_obs = np.array([feat_of(s) for s in rows["peptide"]])
        X_obs_s = scaler.transform(X_obs)
        y_obs = rows["score_htr1a"].to_numpy(dtype=float)

        if n_eval >= 3:
            gp_t = fit_gp(X_obs_s, y_obs)

            uneval = [i for i in range(len(pool))
                      if pool[i] not in set(rows["peptide"])]
            if not uneval:
                print("候选库已评完，停止。")
                break
            X_cand_s = X_pool_s[uneval]

            if best_feasible_row is not None:
                best_y = best_feasible_row["score_htr1a"]
                acq = constrained_acquisition(gp_t, [], X_cand_s, best_y)
                picked = select_batch(acq, X_cand_s, uneval, batch_size)
            else:
                # 无可行解时：用常数采集值，由多样性贪心决定短名单顺序
                acq = np.ones(len(X_cand_s))
                picked = select_batch(acq, X_cand_s, uneval, batch_size)

            print(f"  🎯 采集函数选出 {len(picked)} 个候选："
                  f"{[f'seq_{i:04d}' for i in picked]}")
            evaluate_batch(picked, round_no, f"round_{round_no:02d}")
        else:
            print("  样本不足，继续随机冷启动评估。")
            others = [i for i in range(len(pool))
                      if pool[i] not in set(df_eval["peptide"])]
            pick = list(rng.choice(others, size=min(batch_size, len(others)),
                                   replace=False))
            evaluate_batch(pick, round_no, "init")

        if len(df_eval) == n_eval:
            print("  ⚠️ 本轮无新结果就绪，等待计算完成后重跑本脚本。")
            break

    # ---------------- 收尾 ----------------
    print("\n" + "=" * 60)
    if best_feasible_row is not None:
        best_seq = best_feasible_row["peptide"]
        best_score = best_feasible_row["score_htr1a"]
        print(f"🏆 最佳序列：{best_seq}")
        print(f"📊 最佳 score：{best_score}  （ipTM = {1 - best_score:.3f}）")
        for t in OFF_TARGETS:
            v = best_feasible_row[_OFF_COL[t]]
            desc = f"{v:.3f}" if pd.notna(v) else "未评估（本次运行已搁置, 待高端 GPU 补跑）"
            print(f"   {t} ipTM = {desc}")
        if best_path and os.path.exists(best_path):
            dest_dir = os.path.join(BO_ROOT, "BEST_STRUCTURE")
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, "BEST_peptide_HTR1A.zip")
            import shutil
            shutil.copy(best_path, dest)
            print(f"📁 最佳结果已保存：{dest}")
        pd.DataFrame([[best_seq, best_score,
                       best_feasible_row["iptm_unc13c"],
                       best_feasible_row["iptm_bin1"]]],
                     columns=["最优序列", "最优分数(1-iptm)",
                              "UNC13C_ipTM", "BIN1_ipTM"]
                     ).to_excel(os.path.join(BO_ROOT, "result_bo.xlsx"),
                                index=False)
    else:
        print("❌ 尚无可行评估结果（可能作业仍在计算，重跑本脚本继续）。")

    # 改进曲线
    valid = [(i, y) for i, y in enumerate(history_y) if not np.isnan(y)]
    if valid:
        xs, ys = zip(*valid)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(1, len(ys) + 1), ys, "o-", markersize=3)
        plt.xlabel("评估数量 (可行候选累计)")
        plt.ylabel("best score = 1 - ipTM(HTR1A)")
        plt.title("贝叶斯优化: 最优界面置信度随评估数变化")
        plt.tight_layout()
        plt.savefig(os.path.join(BO_ROOT, "bo_curve.png"))
        plt.close()
        print(f"📈 曲线：{os.path.join(BO_ROOT, 'bo_curve.png')}")
    print("=" * 60)
    return best_feasible_row


# ---------------- 入口 ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="小肽序列优化的贝叶斯优化入口")
    parser.add_argument("--server", action="store_true",
                        help="改用 AlphaFold Server 云端预测（默认是本地集群推理，无每日配额限制）")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help=f"总评估预算（默认 {DEFAULT_BUDGET} 个候选）")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"每轮评估的候选数（默认 {DEFAULT_BATCH}）")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help=f"候选库大小（默认 {DEFAULT_POOL}）")
    parser.add_argument("--targets", default=",".join(TARGET_ORDER),
                        help="逗号分隔的待评估靶标（必须含 HTR1A）。V100 分区用--targets HTR1A,BIN1：UNC13C 复合物 ~2233 tokens 超过V100上限 1280，搁置待 A100/H100 补跑")
    args = parser.parse_args()

    want = {t.strip().upper() for t in args.targets.split(",") if t.strip()}
    unknown = want - set(TARGET_ORDER)
    if unknown:
        parser.error(f"--targets 含未知靶标 {sorted(unknown)}，可选：{TARGET_ORDER}")
    if "HTR1A" not in want:
        parser.error("--targets 必须包含 HTR1A（优化目标）")
    pending_targets = set_active_targets(want)

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    off_desc = "+".join(ACTIVE_OFF) if ACTIVE_OFF else "无"
    print(f"🧬 贝叶斯优化 — 小肽序列（**旧口径**：最大化 ipTM(HTR1A)，已无 {off_desc} 上限约束；"
          "未接入乘积目标，榜单不代表全库最佳）")
    print(f"   本次评估靶标：{ACTIVE_TARGETS} | 结果目录：{BO_ROOT}/")
    if pending_targets:
        print(f"   ⏸️  搁置靶标：{pending_targets}（V100 显存放不下，待 A100/H100 以全靶标重跑补齐；已有 HTR1A/BIN1 结果会被直接复用）")
    print("=" * 60)

    if args.server:
        import 小肽云端后端 as server
        server.AUTO_MODE = False
        predict_fn = server.predict_and_score
        print("运行模式：AlphaFold Server（--server，与主脚本相同的手动/自动流程，受每日配额限制）")
    else:
        import 小肽优化_本地集群 as loc
        # 预检查 + 解压作业实时进度跟随（运行中则等待，不继续后续步骤）
        loc.ensure_local_ready()
        predict_fn = loc.local_predict_and_score
        print("运行模式：本地集群推理（默认，自动提交 SLURM GPU 作业，无配额限制）")

    run_bo(predict_fn, pool_size=args.pool, batch_size=args.batch,
           budget=args.budget)
