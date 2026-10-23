"""
小肽序列优化 — 无限搜索入口（V100 现状：HTR1A + BIN1，结果无限追加进共享数据库）
=============================================================
背景（2026-10 决策）：
  - bme_gpupub 分区全是 V100-32GB（官方上限 ~1280 tokens）：肽×HTR1A ≈ 441、肽×BIN1 ≈ 612 可算；肽×UNC13C ≈ 2233 超限，暂时搁置。
  - **优化目标（2026-10-20 起，2026-10-22 用户定稿）= 目标乘积 ipTM(HTR1A) × ipTM(BIN1) 最大化**，两个因子**都越大越好**，不再有任何“底线上限/红线/可行域”概念（旧阶段的 BIN1 ipTM < 0.8 约束已彻底取消，UNC13C 同样不是约束而是待补算的因子）。真源开关 = `小肽数据库.PRODUCT_TARGETS`。策略仍是**多算多存**——结果持续追加进共享数据库，等 A100/H100 到位后补算 UNC13C、把三元组加进 PRODUCT_TARGETS 即可重排 BEST_TOP10（已完成结果不重算）。

与 小肽优化_BO.py（有限预算）的关系：
  本入口是 BO 的“无限 + 数据库化”变体：复用其序列特征化/高斯过程/约束期望改进采集函数，但
    1）单一真源改为共享数据库（小肽数据库.py，存于共享目录，计算节点可见）；
    2）无预算上限：候选来自 **精确穷尽枚举**（`enumerate_space()`，51091 条）；
    3）每轮：收割新完成结果入库 → 刷新 BEST_TOP10 → 按启发式挑选新候选提交→无进展则轮询等待；直到手动终止（SIGINT/SIGTERM 优雅退出）。

候选调度（2026-10-04 第二次定案：穷尽队列取代随机采样）：
  现在的主调度器是 `ExhaustiveQueue` —— 全集为 `小肽公共库.enumerate_space()`精确构造的 51091 条（编辑距离 ≤2、长 6~10、20 种标准氨基酸），每轮从「全集 − 已派发台账（sequences.csv）」中取 n 条。由此得到三条可证明的性质：
  **不重复**、待办集合每轮精确定量递减、至多 ⌈51091/24⌉=2129 轮 ≈680 天必然全覆盖（取空后 next_batch 返回 [] = 明确的完成信号）。GP只决定“先跑谁”（把预测**目标乘积**高者排前面），对上述性质零贡献；GP 失效时探索配额照样从待办集推进，覆盖与不重复依然成立。
  取代随机采样的理由（均实测）：旧 `CandidateStream` 每周期只撒 ~720 个随机点供 GP 排序（argmax 在子集上取，不在整个空间上），且 `generate_mutant` 有**665 条序列永远采不到**（模式B 的 delta=±1 只做单次插入或删除）。退回旧行为：`PEPOPT_SWEEP=0`。

仍保留的启发式先后顺序（只影响次序，不影响覆盖保证）：
  1）先评估初始序列 SEQ_202（基线锚点）；
  2）评估数 < MIN_GP_SAMPLES（默认 12）时，整批走探索抽样（冷启动）；
  3）之后每轮 batch 中 ~EXPLORE_FRAC（默认 25%，`PEPOPT_EXPLORE_FRAC` 可调）配额从待办集 **无放回随机抽样**（探索），其余按期望改进 EI(y = 1 − 目标乘积) 挑选（**无约束因子**：约束已于 2026-10-22 随目标改为乘积而取消，详见 `_gp_order` 注释）；探索批只影响“先跑谁”，不影响上述三条不变量（它们由枚举 + 台账保证，与取批次序无关）；
  4）GP 训练集超 TRAIN_CAP（默认 2000）时做确定性分层子采样 —— 因为拟合是O(n³)（实测 4000 条已 52s，外推 5 万条 ≈9.6h > 周期预算），而全空间打分实测仅 ≈1s，故瓶颈在训练侧不在候选侧。

用法（登录节点，建议经 运行.sh infinite 后台运行）：
    python 小肽优化_无限.py                  # 默认 --targets HTR1A,BIN1
    python 小肽优化_无限.py --migrate       # 先把家目录旧结果迁入共享库再开跑
    python 小肽优化_无限.py --batch 6 --poll 300
    python 小肽优化_无限.py --status         # 只打印数据库统计与 BEST_TOP10，不启动搜索
终止：kill <PID>（或 Ctrl+C）；已提交的 SLURM 作业不受影响，重跑即续作。

共享数据库（小肽数据库，默认 /public_bme2/Share200T/管吉松/peptide_opt_db）：
    peptide_database.csv  每序列一行：iptm_htr1a / iptm_bin1 / iptm_unc13c（现为空）
    sequences.csv         tag → 序列注册表
    af3_results/<tag>_<target>/   全部 AF3 结果（输入/输出/评分）
    BEST_TOP10/           当前 top10 的 HTR1A 结构包 + BEST_TOP10.csv

将来 UNC13C 解锁（申请到 A100/H100-80G）：
    PEPOPT_GPU_MAX_TOKENS=5120 PEPOPT_FLASH_ATTN=triton PEPOPT_XLA_7X=0 \
    PEPOPT_PARTITION=<新分区> python 小肽优化_无限.py --targets HTR1A,BIN1,UNC13C 已入库序列只补算 UNC13C（HTR1A/BIN1 直接复用），
    再把 db.PRODUCT_TARGETS 加回 "UNC13C" → refresh_best 自动按三因子乘积重排 BEST_TOP10。
"""

import os
import sys
import time
import signal
import random
import argparse
import numpy as np

# 注意：必须在 import 小肽优化_本地集群 **之前**设定结果目录环境变量，该模块在 import 时读取 PEPOPT_RESULTS_DIR（指向共享库的 af3_results/）。
import 小肽数据库 as db
os.environ.setdefault("PEPOPT_RESULTS_DIR", db.results_dir())

import 小肽优化_本地集群 as loc          # 本地集群提交后端（复用打包/节流/OOM拦截）
import 小肽优化_BO as bo              # 特征化 + GP + EI 采集函数（复用；其 gp_con_list 传空表）
import 小肽序贯采样 as seq_sample        # 序贯采样×目标乘积×β拟合入库与数据库整理

# 防御：无论 loc 以何种顺序被导入，都把其结果目录强制对齐到共享库，保证推理任务的 input.json 与结果一定落在共享盘（而非家目录 af3_local_results）。
loc.LOCAL_RESULTS_DIR = db.results_dir()

from 小肽公共库 import (SEQ_202, FIXED_A, ORIG_TAIL,HTR1A_SEQ, UNC13C_SEQ, BIN1_SEQ, ASIC1A_SEQ,generate_mutant, enumerate_space)

# ===================== 配置 =====================

TARGET_SEQS   = {"ASIC1A": ASIC1A_SEQ, "HTR1A": HTR1A_SEQ,
                 "UNC13C": UNC13C_SEQ, "BIN1": BIN1_SEQ}
TARGET_ORDER  = ["ASIC1A", "HTR1A", "BIN1", "UNC13C"]  # 固定评估顺序
# 旧阶段把 UNC13C/BIN1 当“底线蛋白（ipTM 越低越好）”；2026-10-20 起目标改为乘积最大化后 BIN1 已是正向因子（越高越好）、UNC13C 是待补算因子，二者都不再有上限语义，故下面两个常量如今只剩“明细列名映射”用途（cons 字典、启动与 --status 回显），不参与任何判定。
OFF_TARGETS   = ["UNC13C", "BIN1"]               # 历史名：目标乘积的非 HTR1A 因子
OFF_COL       = {"UNC13C": "iptm_unc13c", "BIN1": "iptm_bin1"}
# 同源多聚体靶标的拷贝数（AF3 输入挂几个链 ID）；也是目标乘积的候选因子之一
TARGET_COPIES = {"ASIC1A": 3}
# 目标乘积因子：真源在 小肽数据库.PRODUCT_TARGETS（**总开关**；2026-10-20 停算ASIC1A，目标 = ipTM(HTR1A)×ipTM(BIN1)），此处只转发不重复定义。
PRODUCT_TARGETS = list(db.PRODUCT_TARGETS)

DEFAULT_TARGETS      = "HTR1A,BIN1"              # 当前默认（V100 搁置 UNC13C）
# GP 训练标签口径（见 `_gp_objective_rows`）：max = 1 − objective（库里 iptm_* 分量对应的那次最佳样本乘积，与 best_y / BEST_TOP10 排名同源，也是交叉验证选出的口径）；mean = 1 − obj_mean（= obj_sum/obj_n，单次评估的期望乘积，已被实测证伪，仅供复现对照）。
GP_LABEL             = os.environ.get("PEPOPT_GP_LABEL", "max").strip().lower()
MIN_GP_SAMPLES       = 12                        # 冷启动评估数（不足时整批走探索）


def _env_float(name, default):
    """读浮点环境变量：未设/空串取默认，非法值告警后取默认（与 loc._env_int 同风格）。

    切勿写成 `float(os.environ.get(name) or default)`：那会让合法的显式 0（完全不探索）被当成假值而惄惄变成默认值。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        print(f"⚠️  {name}={raw!r} 不是合法浮点数，改用默认 {default}", flush=True)
        return default


# 每周期探索配额占比（其余配额给 GP 的 EI 次序）：探索批从待办集 **无放回随机抽样**，作用是在 GP 的模型盲区之外持续给数据库攒新数据（低分区锚点也喂 GP）。PEPOPT_EXPLORE_FRAC 可覆盖，取值先夹到 [0,1] 再算 n_exp = max(1, round(n × frac))——**下界恒为 1 条**（实测：n=24 时 frac=0/0.1/0.25/0.5/1 分别得 1/2/6/12/24 条），所以 frac=0 只是把探索降到最低而**无法完全关闭**；真想全听 GP 得同时置 MIN_GP_SAMPLES 让 GP 一直可用并把该值当 0 看待，而保留最低 1 条探索实际上是有意的安全底。冷启动（评估行 < MIN_GP_SAMPLES）则不看本变量，整批 n 条全走探索。
EXPLORE_FRAC         = max(0.0, min(1.0, _env_float("PEPOPT_EXPLORE_FRAC", 0.25)))
# 每轮 GP 挑选的序列数。**与 loc.BUNDLE_SIZE 解耦**：包容量大是为了摊薄55min固定 MSA 开销（见 小肽优化_本地集群），但每轮 GP 只应挑少量高信息量候选——样本还不多（几十条）时一次挑几十个会让采集函数退化、重复采样低价值区域。差额由 submit_pending_bundle 的“攒批”机制补齐：多轮各挑几个，攒满一包才提交，期间每轮先收割新结果 → GP 持续变强，既不空转 GPU 也不牺牲引导质量。（复用 loc._env_int：未设/空串取默认，非法值告警后取默认，切勿用 int(x or d) 写法）
DEFAULT_BATCH        = max(1, loc._env_int("PEPOPT_BATCH", 6))
DEFAULT_POLL         = 300                       # 无进展时的轮询间隔（秒）
BACKLOG_MULTIPLIER   = 2                         # 积压上限 = BUNDLE_SIZE × 该倍数（个任务）
TOP_N_BEST           = 10                        # BEST_TOP<N>

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n🛑 收到信号 {signum}，本轮结束后优雅退出（已提交的 SLURM 作业继续运行，重跑本脚本即续作）…", flush=True)


# ===================== 候选池（按需无限扩展）=====================

class CandidateStream:
    """编辑距离 ≤2 变异序列的惰性无限流（去重，固定种子可复现）。"""

    def __init__(self, seed=42):
        self._seen = {SEQ_202}
        # generate_mutant 用全局 random；固定种子保证扩池可复现
        random.seed(seed)

    def next_batch(self, n, exclude):
        """生成 n 个未见过且不在 exclude 中的全序列（去重上限保护）。"""
        out = []
        tries = 0
        while len(out) < n and tries < n * 200 + 1000:
            tries += 1
            seq = FIXED_A + generate_mutant(ORIG_TAIL)
            if seq in self._seen or seq in exclude:
                continue
            self._seen.add(seq)
            out.append(seq)
        return out

    def release(self, seqs):
        """把未最终提交的候选放回流（GP 短名单只是临时抽样，不释放会白白消耗候选空间）。"""
        for s in seqs:
            if s != SEQ_202:
                self._seen.discard(s)


# ===================== 数据库视图辅助 =====================

def db_evaluated_rows(active_targets):
    """已评估行（含目标乘积与各因子）：[dict]。

    row["objective"] = Π ipTM(t)，t ∈ db.PRODUCT_TARGETS（现为 HTR1A×BIN1，**越大越好**，因子齐备才非 None）；
    row["obj_mean"] = obj_sum/obj_n，即历次序贯补采的**单次评估期望乘积**（row["objective"] 则是同一批采样的极值，实测二者的最大差达 0.197）；GP_LABEL=mean 时用它做训练标签。
    row["score_htr1a"] = 1 − ipTM(HTR1A)，仅供乘积因子不齐时的冷启动回退与旧结果兼容显示；
    row["cons"] = {列名：ipTM}，旧阶段约束语义已废，现**仅用作明细列名映射**（不参与任何判定）。"""
    by_tag = db.load_database()
    rows = []
    for tag, r in by_tag.items():
        if r.get("iptm_htr1a") is None or not r.get("peptide"):
            continue
        cons = {OFF_COL[t]: r.get(OFF_COL[t]) for t in active_targets if t in OFF_COL}
        obj = db.objective_product(r)
        n_obj, s_obj = r.get("obj_n"), r.get("obj_sum")
        rows.append({"tag": tag, "peptide": r["peptide"],
                     "score_htr1a": r["score_htr1a"], "cons": cons,
                     "objective": obj,
                     "obj_n": n_obj,
                     "obj_mean": (s_obj / n_obj) if (n_obj and s_obj is not None) else None,
                     "iptm_asic1a": r.get("iptm_asic1a"),
                     "iptm_htr1a": r.get("iptm_htr1a"),
                     "iptm_bin1": r.get("iptm_bin1")})
    return rows


def in_flight_task_dirs(active_targets):
    """共享结果目录中“已创建但未完成”的任务目录（排队/运行中）。"""
    rd = db.results_dir()
    if not os.path.isdir(rd):
        return []
    hits = []
    for name in os.listdir(rd):
        d = os.path.join(rd, name)
        if not os.path.isdir(d):
            continue
        parsed = db.split_job_name(name)
        if parsed is None or parsed[1] not in active_targets:
            continue
        if db.find_summary(d):
            continue
        hits.append((name, d))
    return hits


def known_sequences():
    """数据库 + 结果目录里已出现过的全部序列（评估过/排队中/已提交）。"""
    seqs = {SEQ_202}
    for info in db.load_sequence_registry().values():
        if info.get("peptide"):
            seqs.add(info["peptide"])
    return seqs


# ===================== 启发式挑选 =====================

def heuristic_pick(stream, n, active_targets):
    """按启发式先后顺序挑 n 个新候选：评估样本 < MIN_GP_SAMPLES → 全部随机探索；否则 ~EXPLORE_FRAC 随机探索 + 其余按 EI（目标 = 目标乘积）。"""
    rows = db_evaluated_rows(active_targets)
    exclude = known_sequences()

    n_explore = n if len(rows) < MIN_GP_SAMPLES \
        else max(1, int(round(n * EXPLORE_FRAC)))
    n_gp = n - n_explore
    picked = []

    if n_gp > 0 and len(rows) >= 3:
        try:
            picked = _gp_pick(rows, n_gp, exclude, stream)
        except Exception as e:                        # GP 数值异常不应中断无限搜索
            print(f"  ⚠️ GP 挑选失败（{e}），本轮全部随机探索", flush=True)

    need = n - len(picked)
    if need > 0:                                      # 冷启动/GP 不足时随机补齐
        picked.extend(stream.next_batch(need, exclude | set(picked)))
    return picked[:n]


def _gp_pick(rows, n, exclude, stream):
    """EI 挑选（旧随机候选流路径）：拟合 GP → 在“新随机候选 + 已有样本邻域”组成的临时候选集上取 TOP。

    采集函数 = 纯 EI（无约束因子），与 `ExhaustiveQueue._gp_order` 保持一致。"""
    rows, y_of = _gp_objective_rows(rows)        # 目标 = 乘积，不足时回退旧目标
    X_obs = np.array([bo.feat_of(r["peptide"]) for r in rows])
    y_obs = np.array([y_of(r) for r in rows], dtype=float)

    # 临时候选集：随机撒 40×n 个新序列（无限流的天然探索）
    cand_seqs = stream.next_batch(max(40 * n, 200), exclude)
    if len(cand_seqs) < 2:
        return []
    X_cand = np.array([bo.feat_of(s) for s in cand_seqs])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.vstack([X_obs, X_cand]))
    X_obs_s, X_cand_s = scaler.transform(X_obs), scaler.transform(X_cand)

    gp_t = bo.fit_gp(X_obs_s, y_obs)
    best_y = float(np.min(y_obs))

    # 约束 GP 传空表 = 纯 EI（2026-10-22 定稿）：旧版在此拟合 P(ipTM_BIN1 < 0.8) 并乘进采集值，而目标改为乘积后 BIN1 已是正向因子，该因子会与 y 直接相互抵消——实测后果是榜单前 5 中 4 条 BIN1 ≥ 0.88、GP 却把它们所在的高分区当不可行而回避，那些序列只能靠 25% 探索配额偶然撞中。将来若要恢复任何底线语义，改 db.PRODUCT_TARGETS 或重新传入 gp_con_list 即可，bo.constrained_acquisition 本身保留了多约束能力。
    acq = bo.constrained_acquisition(gp_t, [], X_cand_s, best_y)
    if not np.all(np.isclose(acq, 0)) and bool(np.isfinite(acq).all()):
        idx = bo.select_batch(acq, X_cand_s, list(range(len(cand_seqs))), n)
        picked = [cand_seqs[i] for i in idx]
    else:
        # 采集值全零/异常 → 退化为按 GP 均值（预测 y 小者优先 = 预测乘积高者）排序
        mu, _ = gp_t.predict(X_cand_s, return_std=True)
        order = np.argsort(mu)[:n]
        picked = [cand_seqs[i] for i in order]
    stream.release([s for s in cand_seqs if s not in picked])   # 未入选的放回
    return picked


# ===================== 穷尽队列：不重复 + 有限步全覆盖 =====================

SPACE_EDIT     = loc._env_int("PEPOPT_SPACE_EDIT", 2)
SPACE_LEN_MIN  = loc._env_int("PEPOPT_SPACE_LEN_MIN", 6)
SPACE_LEN_MAX  = loc._env_int("PEPOPT_SPACE_LEN_MAX", 10)
# GP 打分池上限：0 = **不限制，对整个待办集打分**（默认）。实测（n_train=34）：候选 5000 条 0.1s / 20000 条 0.4s / **全部 51091 条仅 1.0s**，故候选池根本不是瓶颈；早期版本把它限在 20000 只是白费了 GP 的全局视野。保留旋钮仅为日后特征维度大幅上升时可重新收紧。
SCORE_POOL_CAP = loc._env_int("PEPOPT_SCORE_POOL", 0)
# GP **训练集**上限：真正的瓶颈在这里。实测拟合+预测耗时随 n_train 近似立方增长：
#   n_train=34 → 0.5s | 1000 → 7.8s | 2000 → 10.7s | 4000 → 52s | 外推 51091 → ≈9.6h
# 9.6h 已超过一个周期（n=24 时 7.7h）→ 若不封顶，穷尽搜索到后期会直接超时。
# 超限后按确定性分层子采样（见 _subsample_train），只影响“先跑谁”的次序，对不重复/全覆盖三条不变量无影响（best_y 仍取**全量**最优，不因子采样而失真）。
TRAIN_CAP = loc._env_int("PEPOPT_TRAIN_CAP", 2000)
# 总开关：置 0 退回旧的随机采样启发式（CandidateStream + heuristic_pick）。
SWEEP_ENABLED = os.environ.get("PEPOPT_SWEEP", "1").strip().lower() not in \
                ("0", "false", "no", "off")


def _gp_objective_rows(rows):
    """GP 训练目标选择：目标 = 乘积 objective（越大越好）→ 取 y = 1-objective（越小越好，与采集函数的最小化约定一致）；乘积齐备的行不足 MIN_GP_SAMPLES 时回退旧目标 score_htr1a（冷启动兼容，避免 GP 无米下锅）。返回（可用行，y 取值函数）。

    GP_LABEL=mean 时标签换成 obj_mean（= obj_sum/obj_n，单次采样的期望乘积）。**该口径已被实测证伪，故默认 max**：当初的设想是“未评估候选只能被预测单次评估的期望，而 objective 是采样后的极值统计量（实测 obj_n 有 5/10/15/20 四档，n 越大越偏乐观），用它当标签等于拟合一个随样本数漂移的量”；但拿共享库 745 条真实数据在计算节点上做的对照显示 mean 标签偏差——时序留出的 Spearman 在 composition 特征下从 0.505 跌到 0.268、positional 特征下从 0.487 跌到 0.429，全空间预测乘积上限从 0.4886 跌到 0.4562，5-fold 命中率只在 0.322→0.311 与 0.411→0.433 之间微摆；更关键的是即便把评估真值也换成 obj_mean 做同口径比较，max 标签仍然更准（composition 下 Spearman 0.542 vs 0.300）→ 说明 obj_max 本身就是更纯净的信号（它筛掉了未收敛到最优折叠的那些采样噪声），与排名取 max 的设计依据同源。开关保留仅为可复现该对照实验。含 obj_mean 的行同样不足 MIN_GP_SAMPLES 时退回 max 标签。"""
    full = [r for r in rows if r.get("objective") is not None]
    if len(full) < MIN_GP_SAMPLES:
        return rows, lambda r: r["score_htr1a"]
    if GP_LABEL == "mean":
        means = [r for r in full if r.get("obj_mean") is not None]
        if len(means) >= MIN_GP_SAMPLES:
            return means, lambda r: 1.0 - r["obj_mean"]
    return full, lambda r: 1.0 - r["objective"]


def _subsample_train(rows, y_of, cap=TRAIN_CAP):
    """训练集超限时做**确定性分层子采样**：一半取当前最优（保住高分区分辨率，EI 排序最关心这里），另一半按 y 排序等距抽样（保住全分布的形状信息）。两条规则都不含随机数 → 同样的库状态必得同样的子集，跨作业重启可复现。返回（子集，未截断的 best_y）；best_y 始终基于全量 rows，不因截断而偏移。"""
    srt = sorted(rows, key=lambda r: (y_of(r), r.get("tag", "")))
    best_y = float(min(y_of(r) for r in rows))
    if cap <= 0 or len(rows) <= cap:
        return rows, best_y
    k_best = max(1, cap // 2)
    head = srt[:k_best]
    got = {r.get("tag") for r in head}
    rest = [r for r in srt if r.get("tag") not in got]
    need = cap - len(head)
    if need > 0 and rest:
        stride = len(rest) / float(need)
        head.extend(rest[min(len(rest) - 1, int(i * stride))] for i in range(need))
    return head, best_y


def attempted_vars():
    """sequences.csv = **已派发台账**。prepare_cycle 在写任何输入之前先调用register_sequence，故凡被队列取走过的可变区必然登记在此 —— “不重复”这条性质由台账保证，不依赖任何内存状态，因而天然跨作业重启持久。"""
    out = set()
    for info in db.load_sequence_registry().values():
        v = info.get("var")
        if v:
            out.add(v)
    return out


def _sample_pool(P, n, exclude=None):
    """从待办集 P 无放回随机抽≤n 条（可排除 exclude 里的元素），**全程零复制**。

    P = 全集 减 台账，是一个随时可精确算出的已知集合，抽谁都对；探索配额只影响“先跑谁”，不影响不重复与全覆盖（那两条由枚举 + 台账保证）。为何不用旧的字典序旋转游标：① 相邻字典序 = 结构上最相似，实测同一 |P|=50298 上抽 6 条的两两编辑距离均值只有 1.50 位（而随机抽样 3.71 位，差 2.5×），且两个周期的探索批 12 条全属同一结构家族（N 端插 Cys 的 9/10 位肽）、长度只覆盖 9、10 两档；② 游标每周期只前进 n 位，在 5 万条的待办表上要爬几千个周期才能扫完；③ 构造 `rot = P[cur:] + P[:cur]` 要分配并复制**两个 5 万元素的列表**（每周期 ≈800KB、实测 0.618 ms），而下面的抽样实测只 0.005 ms（116× 更快）——所以换随机不仅多样性更高，**性能也是净赚**。

    零复制的两条路径：exclude 为空（即每周期的探索批，主路径）直接 `random.sample(P, n)`——CPython 在 n 小于其 setsize 阈值（n=6 时为 85）时走 set 分支，只做 O(n) 次 randrange + 下标取元素，**不拷贝 population**；带 exclude（GP 异常后的补齐路径，罕见）改用拒绝抽样，同样 O(n) 次下标、零分配，exclude 至多 n 条而 |P|≈5万 → 碰撞概率 <0.05%，期望尝试次数 ≈ n。

    兜底：池被 exclude 挤得更小（len(exclude)×4 ≥ |P|）或拒绝抽样撞太多次时，做一次全量过滤凑满剩余——探索配额**绝不得因凑不满而缩水**（凑不满意味着 |P| 未严格递减、伤到全覆盖的终止性），而这条路径上的 O(|P|) 开销比那点分配重要得多，只在罕见分支上发生。"""
    if n <= 0 or not P:
        return []
    if not exclude:
        return random.sample(P, min(n, len(P)))
    m = len(P)
    if len(exclude) * 4 >= m:
        pool = [v for v in P if v not in exclude]
        return random.sample(pool, min(n, len(pool)))
    out, dup = [], 0
    while len(out) < n and dup < 8 * n:
        v = P[random.randrange(m)]
        if v in exclude or v in out:     # `in out` 是 O(n) 线性查，总代价 O(n²)=百次比较，比开 set 分配更便宜
            dup += 1
            continue
        out.append(v)
    if len(out) < n:                     # 兜底：一次全量过滤凑满剩余
        got = set(out)
        pool = [v for v in P if v not in exclude and v not in got]
        out.extend(random.sample(pool, min(n - len(out), len(pool))))
    return out


class ExhaustiveQueue:
    """确定性穷尽调度器 —— 同时保证【不重复】与【有限步内全覆盖】。

    三条不变量（合起来构成覆盖证明）：
      A. 全集 S = 小肽公共库.enumerate_space()：**精确构造**而非随机采样，因此包含随机采样不可达的 665 条；|S| = 51091，枚举顺序固定（字典序，与随机种子和调用次数无关）。注：这里固定的是**全集的内容与顺序**，而不是“取批的次序”（探索批是从 P 里随机抽的）——不变量并不依赖取批次序。
      B. 待办 P_k = S \\ A_k，其中 A_k = sequences.csv 已登记台账。prepare_cycle先登记再写输入 → 凡被派发必入 A；于是每周期 |P| 至少减少本批大小（随机无放回抽样同理成立：抽中的 n 条必不同且全在 P 中），且**同一序列不可能被二次派发**。
      C. S 有限 → 至多 ceil(|S|/n) = ceil(51091/24) = 2129 个周期后 P = ∅，即全覆盖。由 B 的递减性得到，**是确定性上界而非概率保证**（随机抽样不会让它变成“几乎必然”）。届时 next_batch 返回空列表，调用方把它当**完成**而非失败（与旧的 EMPTY_CYCLE_LIMIT“采不到新样本”语义明确区分）。

    GP/贝叶斯在其中的角色：**只决定 P 里先跑谁**（把预测目标乘积高的排前面，让好结果尽早入库），对 A/B/C 三条不变量毫无贡献。即使 GP 完全失效（样本不足、数值异常、特征退化），探索配额照样从 P 里推进，不重复与全覆盖依然成立。这就是“调度算法保证不重复并最终全覆盖”的准确含义：**覆盖由枚举 + 台账保证，不由启发式保证**。
    """

    CENTER = ORIG_TAIL

    def __init__(self, active_targets=(), center=CENTER, max_edit=SPACE_EDIT,
                 min_len=SPACE_LEN_MIN, max_len=SPACE_LEN_MAX):
        self.active = tuple(active_targets) or ("HTR1A", "BIN1")
        self.center, self.max_edit = center, max_edit
        self.space = enumerate_space(center, max_edit, min_len, max_len)
        if not self.space:
            raise RuntimeError("穷尽枚举返回空集，无法保证覆盖")

    # ---- 状态视图 ----
    def pending(self, attempted=None):
        """待评估的可变区列表（= 全集 减 台账，作为探索抽样的池与 GP 打分的候选集；元素顺序沿用 space 的字典序，但取批已从其中随机抽，不再依赖顺序）。"""
        A = attempted_vars() if attempted is None else attempted
        return [v for v in self.space if v not in A]

    def coverage(self, attempted=None):
        P = self.pending(attempted)
        done = len(self.space) - len(P)
        return {"total": len(self.space), "dispatched": done, "pending": len(P),
                "frac": (done / float(len(self.space))) if self.space else 1.0}

    @staticmethod
    def _feat(seqs):
        """批量特征化：经 bo.feat_of 分发（维度由 PEPOPT_FEATURE_MODE 决定，全进程必须一致）。"""
        return np.array([bo.feat_of(s) for s in seqs], dtype=float)

    # ---- 取批 ----
    def next_batch(self, n, rows=None):
        """取 n 条未派发序列（返回**全序列**形式，与 prepare_cycle 约定一致），保证批内互不重复且不与台账重复；剩余不足 n 时返回剩余全部（不伪造）。返回空列表 = 空间已穷尽。"""
        P = self.pending()
        if n <= 0 or not P:
            return []
        if n >= len(P):
            return [FIXED_A + v for v in P]         # 末批：全取，顺序即字典序

        rows = db_evaluated_rows(self.active) if rows is None else rows

        # 冷启动（GP 不可拟合）时整批走探索；否则保留 EXPLORE_FRAC 配额。探索批 = 从待办集 P **无放回随机抽样**（见 _sample_pool：零复制，相比旧的字典序旋转游标多样性高 2.5×、单次取样快 116×，省掉每周期两次 5 万元素列表的分配与拷贝）。
        n_exp = n if len(rows) < MIN_GP_SAMPLES else max(1, int(round(n * EXPLORE_FRAC)))
        n_exp = max(0, min(n_exp, n))
        picked = _sample_pool(P, n_exp)             # 探索批：待办集上零复制随机抽样
        chosen = set(picked)

        n_gp = n - len(picked)
        if n_gp > 0 and len(rows) >= 3:
            try:
                for v in self._gp_order(P, chosen, rows, n_gp):
                    picked.append(v)
                    chosen.add(v)                   # chosen 与 _gp_order 的 exclude 同一对象
            except Exception as e:                  # GP 异常绝不阻断穷尽推进
                print(f"  ⚠️ GP 排序失败（{e}），本轮改为随机探索补齐", flush=True)

        if len(picked) < n:                         # 仍不足 → 随机补齐到 n（配额绝不缩水）
            picked.extend(_sample_pool(P, n - len(picked), exclude=chosen))
        return [FIXED_A + v for v in picked[:n]]

    def _gp_order(self, P, exclude, rows, n_gp):
        """按 EI 降序产出可变区（稳定排序 → 同分按字典序，可复现）。exclude 就地累积，便于调用方复用同一集合去重。"""
        if n_gp <= 0:
            return []
        if SCORE_POOL_CAP > 0 and len(P) > SCORE_POOL_CAP:
            stride = len(P) / float(SCORE_POOL_CAP)
            pool = [P[int(i * stride)] for i in range(SCORE_POOL_CAP)]
        else:
            pool = list(P)                  # 默认：对全部待办集打分（实测仅 ≈1s）
        pool = [v for v in pool if v not in exclude]
        if len(pool) < 2:
            return []

        rows, y_of = _gp_objective_rows(rows)   # 新阶段目标 = 乘积，不足时回退旧目标
        train, best_y = _subsample_train(rows, y_of)   # best_y 基于全量，不受截断影响
        X_obs = self._feat([r["peptide"] for r in train])
        y_obs = np.array([y_of(r) for r in train], dtype=float)
        X_cand = self._feat([FIXED_A + v for v in pool])
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(np.vstack([X_obs, X_cand]))
        X_obs_s, X_cand_s = sc.transform(X_obs), sc.transform(X_cand)

        gp_t = bo.fit_gp(X_obs_s, y_obs)
        # 约束 GP 传空表 = 纯 EI，与 y = 1 − 乘积 同口径；旧版叠乘的 P(ipTM_BIN1 < 0.8) 与乘积目标直接矛盾（把当前最优区当不可行而回避），2026-10-22 删除，前因后果详见 _gp_pick 里的同名注释。
        acq = bo.constrained_acquisition(gp_t, [], X_cand_s, best_y)
        if bool(np.isfinite(acq).all()) and not np.all(np.isclose(acq, 0)):
            order = np.argsort(-np.asarray(acq, dtype=float), kind="stable")
        else:                    # 采集退化 → 用 GP 均值（y 小者优先 = 预测乘积高者）
            mu, _ = gp_t.predict(X_cand_s, return_std=True)
            order = np.argsort(mu, kind="stable")

        out = []
        for i in order:
            v = pool[int(i)]
            if v in exclude:
                continue
            out.append(v)
            exclude.add(v)
            if len(out) >= n_gp:
                break
        return out


_SWEEP = None


def get_sweep(active_targets=None):
    """穷尽队列单例（枚举 51091 条只需约 0.1s，但避免每周期重建）。"""
    global _SWEEP
    if _SWEEP is None or (active_targets and tuple(active_targets) != _SWEEP.active):
        _SWEEP = ExhaustiveQueue(tuple(active_targets) if active_targets else None)
    return _SWEEP


def sweep_pick(active_targets, n):
    """穷尽队列取批（替代随机采样的 heuristic_pick）。返回 [] 表示**空间已穷尽**（应视为正常完成，而非采样失败）。"""
    return get_sweep(active_targets).next_batch(n)


def coverage_report(active_targets, cycle_secs=None, per_cycle=24):
    """覆盖进度 + ETA 文本。cycle_secs 给出时按每周期 per_cycle 条折算剩余天数。per_cycle由调用方显式传入（不 import 小肽优化_节点内，避免独立运行本模块时的循环导入）。"""
    q = get_sweep(active_targets)
    c = q.coverage()
    txt = (f"穷尽覆盖 {c['dispatched']}/{c['total']} ({c['frac'] * 100:.2f}%) "
           f"| 待评估 {c['pending']}")
    per_cycle = max(1, int(per_cycle or 1))
    if cycle_secs and cycle_secs > 0 and c["pending"]:
        eta_h = (c["pending"] / float(per_cycle)) * (cycle_secs / 3600.0)
        txt += f" | 全覆盖预计还需 {eta_h:.0f} h ({eta_h / 24.0:.1f} 天)"
    return txt


# ===================== 提交与收割 =====================

def submit_candidates(seqs, active_targets):
    """注册 + 写输入 + 触发打包提交；若某靶标结果**已就绪**则就地入库。返回提交的序列数。

    复用 小肽优化_本地集群.local_predict_and_score：它按<LOCAL_RESULTS_DIR>/<job_name>布局写 input.json，并由submit_pending_bundle()统一打包提交（含 MAX_INFLIGHT 节流与失败重试）。job_name 用共享库的稳定规则 <tag>_<target>，结果直接落共享盘；已就绪的（迁移过的旧结果/上轮完成的）返回 iptm，即刻 upsert 入库（并写 .harvested，与 harvest_results 幂等一致）。"""
    n = 0
    for seq in seqs:
        if _STOP:
            break
        var = seq[len(FIXED_A):]
        tag = db.resolve_tag(var)      # 注册表优先：沿用历史标签，不孤儿化已有结果
        try:
            db.register_sequence(tag, var, seq)
        except RuntimeError as e:
            print(f"  ⛔ {e}", flush=True)
            continue
        for t in active_targets:
            job_name = f"{tag}_{t.lower()}"
            out_dir = os.path.join(db.results_dir(), job_name)
            _, iptm, _ = loc.local_predict_and_score(
                seq, TARGET_SEQS[t], t, out_dir, auto_submit=True, verbose=False)
            if iptm is not None:                      # 已就绪 → 就地入库
                summary = db.find_summary(out_dir)
                sc = db.extract_scores(summary) if summary else None
                db.upsert_result(tag, t, iptm,
                                 ranking=(sc or {}).get("ranking_score"),
                                 peptide=seq, var=var, source="local")
                marker = os.path.join(out_dir, ".harvested")
                if not os.path.isfile(marker):
                    try:
                        open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    except OSError:
                        pass
        n += 1
    return n


def harvest_and_report(active_targets, new_sample_fn=None):
    """收割新完成结果入库，序贯采样入库（β 统计量），刷新 BEST_TOP10，打印统计。返回（收割新增条数，新增 tags 集合）。

    new_sample_fn 透传给数据库整理做补采（真跑 AF3 新 seed）；留 None 时整理只消费池内免费样本。"""
    n_new, skipped, new_tags = db.harvest_results()
    n_fit = seq_sample.ingest_new_records(new_sample_fn)
    if n_fit:
        print(f"  🧮 序贯采样入库 {n_fit} 条（β 统计量已写入 obj_* 列）", flush=True)
    if n_new:
        print(f"  📥 收割入库 {n_new} 条新结果"
              + (f"（跳过 {len(skipped)}）" if skipped else ""), flush=True)
    db.refresh_best(top_n=TOP_N_BEST)
    s = db.stats()
    rows = db_evaluated_rows(active_targets)
    # 可排名集 = 目标乘积齐备的行（无“可行/不可行”之分），再以原肽 202 自身乘积为基线报“多少条已击败起点”——比起旧的 BIN1<0.8 红线计数，这个口径才对当前目标有用。
    ranked = [r for r in rows if r.get("objective") is not None]
    baseline = next((r["objective"] for r in ranked
                     if r.get("peptide") == SEQ_202), None)
    n_beat = sum(1 for r in ranked
                 if baseline is not None and r["objective"] > baseline)
    # 排名：新阶段看目标乘积（降序）；乘积未齐（缺因子）的排后面按 score 兼容
    def _key(r):
        if r.get("objective") is not None:
            return (0, -r["objective"])
        return (1, r["score_htr1a"])
    best3 = sorted(rows, key=_key)[:3]
    print(f"  🗄️  数据库：总 {s['total']} 序列 | ASIC1A {s.get('asic1a', '-')} | "
          f"HTR1A {s['htr1a']} | BIN1 {s['bin1']} | UNC13C {s['unc13c']} | "
          f"完整评估 {len(rows)} | 乘积齐备可排名 {len(ranked)}"
          + (f" | 击败原肽202（{baseline:.4f}） {n_beat}" if baseline is not None else ""),
          flush=True)
    for r in best3:
        if r.get("objective") is not None:
            # 明细因子跟随 PRODUCT_TARGETS 开关：已停算的靶标值为 None，
            # 直接按旧三靶标硬格式化会 TypeError（2026-10-20 停算 ASIC1A 后实测崩过）。
            factors = " ".join(f"{t}={r[f'iptm_{t.lower()}']:.2f}"
                               for t in PRODUCT_TARGETS
                               if r.get(f"iptm_{t.lower()}") is not None)
            print(f"     🏅 {r['peptide']}  乘积={r['objective']:.4f}  （{factors}）",
                  flush=True)
        else:
            cons = " ".join(f"{t}={r['cons'].get(OFF_COL[t])}" for t in active_targets
                            if t in OFF_COL)
            print(f"     🏅 {r['peptide']}  ipTM_HTR1A={1 - r['score_htr1a']:.3f}  {cons}",
                  flush=True)
    return n_new, new_tags


# 首次爬山开关与轮数上限：每轮 = 激活靶标数个真跑 AF3 任务（实测 HTR1A+BIN1 ≈13~16min/序列）。
# 2026-10-21 用户定稿：首次采样一律反复爬山**直到不再破纪录，次数无上限** → CLIMB_ROUNDS 默认 0（无限），
# 由破纪录判定终止；需要临时限制算力时仍可手动设 PEPOPT_CLIMB_ROUNDS 为正整数。
CLIMB_ENABLED = os.environ.get("PEPOPT_CLIMB", "1").strip().lower() not in \
                ("0", "false", "no", "off")
CLIMB_ROUNDS = max(0, int(os.environ.get("PEPOPT_CLIMB_ROUNDS", "0")))   # 0 = 无上限


def climb_new_records(new_tags, active_targets, new_sample_fn, max_rounds=None):
    """对新入库的记录做首次爬山：反复补采新 seed，直到新样本没能打破自身最佳。2026-10-21 规格：**默认无轮数上限**（max_rounds=0 或 None 且 CLIMB_ROUNDS=0 时无限），仅由破纪录判定终止；补采无产物（时限护栏/AF3 失败）也会停，且不置 climb_done，留给后续数据库整理续爬。

    为什么必须直接用补采返回值、而不能像早期版本那样靠 ingest_new_records 刷新：
    后者内部对 `obj_n > 0` 的行直接 `continue`，而新记录入库时 obj_n 已为初次样本数，于是补采产物永远进不了库 → get_objective 读到旧值 → 首轮必判“未破纪录”。
    实测后果：32 个 bundle 真跑完 AF3（每个 12 份产物）而 obj_n 恒为 5，相当于把每周期一半的 GPU 时白烧。

    本实现：每轮把补采的**全部**乘积样本 add_many 进 BetaFit（含未破纪录的），n/sum_x/sum_x2 增量随 save_obj_stats 落库 → 采样次数与分布估计都计入 β 拟合；
    破纪录时用该样本的**分量** ipTM 回写 iptm_* 列（upsert 会自动重算 objective_product），因此库里的最高纪录与 β 拟合始终同源。每轮落盘：进程被 restart 打断最多丢一轮。

    返回 {"climbed"：参与爬山条数，"rounds"：补采总轮次，"improved"：被刷新条数，"samples"：新增计入拟合的样本数}。"""
    stat = {"climbed": 0, "rounds": 0, "improved": 0, "samples": 0}
    if not CLIMB_ENABLED:
        return stat
    if not new_tags or new_sample_fn is None:
        return stat
    rounds_cap = CLIMB_ROUNDS if max_rounds is None else max(0, int(max_rounds))
    cap_txt = f"至多 {rounds_cap} 轮" if rounds_cap > 0 else "轮数无上限"
    print(f"  🧗 首次爬山：{len(new_tags)} 条新记录，每条{cap_txt}"
          f"（新 seed 真跑 AF3，只要一直破纪录就一直采）", flush=True)
    for tag in sorted(new_tags):
        row = db.load_database().get(tag) or {}
        fit = db.load_obj_fit(tag)
        if fit is None or fit.max_x is None:
            continue                       # 尚无 β 统计量/无初次样本，无从爬山
        best = fit.max_x
        stat["climbed"] += 1
        print(f"    • {tag} {row.get('var_region') or ''}：初始 {best:.4f}",
              end="", flush=True)
        rnd = 0
        while True:                        # 2026-10-21 规格：无轮数上限，由破纪录判定终止
            rec = (new_sample_fn(tag, raw=True) or {}).get(tag) or {}
            prods = [float(x) for x in (rec.get("products") or [])]
            if not prods:                  # 时间护栏拦下/变因子不齐/AF3 失败
                print(" （无补采产物，停止）", flush=True)
                break
            rnd += 1
            t2s = rec.get("samples") or {}
            k = max(range(len(prods)), key=lambda i: prods[i])
            newmax = prods[k]
            fit.add_many(prods)            # 全部计入 β 拟合（含未破纪录的）
            stat["rounds"] += 1
            stat["samples"] += len(prods)
            broke = newmax > best + 1e-9
            if broke:
                best = newmax
                stat["improved"] += 1
                for t, vals in t2s.items():    # 最佳样本的分量 → iptm_* 列
                    if t in active_targets and k < len(vals):
                        db.upsert_result(tag, t, float(vals[k]),
                                         peptide=row.get("peptide"),
                                         var=row.get("var"), source="climb")
            else:
                # 爬山以“不再破纪录”收束：置 climb_done 后，若该记录仍是单点/退化（β 无解），
                # 数据库整理不会再对它重复爬山（见 小肽统计.consolidation_pass 的单点爬山分支）
                fit.climb_done = 1
            db.save_obj_stats(tag, fit)
            if not broke:
                print(f" → 第 {rnd} 轮最佳 {newmax:.4f} ≤ 最佳纪录 {best:.4f}，停止"
                      f"（已累计 {fit.n} 样本）", flush=True)
                break
            print(f" → {best:.4f}*", end="", flush=True)
            if rounds_cap > 0 and rnd >= rounds_cap:
                print(f" （达 {rounds_cap} 轮上限，最佳 {best:.4f}）", flush=True)
                break
    print(f"  🧗 爬山结束：{stat['climbed']} 条参与 / {stat['rounds']} 轮补采 / "
          f"{stat['improved']} 条刷新纪录 / {stat['samples']} 样本计入拟合",
          flush=True)
    return stat


# ===================== 主循环 =====================

def run_infinite(active_targets, batch, poll, backlog_limit):
    p = db.paths(ensure=True)
    os.makedirs(p["results"], exist_ok=True)
    stream = CandidateStream(seed=42)

    # 初始序列永远排第一（基线锚点）：任一激活靶标无结果即需提交（submit_candidates 幂等，已完成的靶标会被直接提取评分不重算）
    var0 = SEQ_202[len(FIXED_A):]
    tag0 = db.resolve_tag(var0)
    need_init = any(
        not db.find_summary(os.path.join(p["results"], f"{tag0}_{t.lower()}"))
        for t in active_targets)

    round_no = 0
    print(f"\n♾️  无限搜索启动：靶标={active_targets} | 目标=乘积最大化"
          f" {'×'.join(PRODUCT_TARGETS)}（无上限约束）"
          f" | 每轮≤{batch} 新序列 | "
          f"积压上限 {backlog_limit} 任务 | 轮询 {poll}s\n"
          f"    共享数据库：{p['root']}\n", flush=True)

    while not _STOP:
        round_no += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"== [{ts}] 轮次 {round_no} ==", flush=True)

        # 1）收割 + 最佳刷新 + 统计
        harvest_and_report(active_targets)
        if _STOP:
            break

        # 2）初始序列优先
        if need_init:
            print("  🧬 提交初始序列 SEQ_202（基线）", flush=True)
            submit_candidates([SEQ_202], active_targets)
            need_init = False

        # 3）计算积压，决定是否提交新候选
        inflight = in_flight_task_dirs(active_targets)
        print(f"  🚚 在途任务 {len(inflight)}/{backlog_limit} | "
              f"SLURM 队列深度 {loc.queue_depth()}", flush=True)
        if len(inflight) < backlog_limit:
            n_new = max(1, (backlog_limit - len(inflight)) // len(active_targets))
            n_new = min(n_new, batch)
            picks = heuristic_pick(stream, n_new, active_targets)
            if picks:
                print(f"  🎯 挑选 {len(picks)} 个新候选提交："
                      f"{[db.resolve_tag(s[len(FIXED_A):]) for s in picks]}", flush=True)
                submit_candidates(picks, active_targets)
        # 即便不提交新候选，也触发一次打包提交（消化攒批/失败重试）
        loc.submit_pending_bundle()

        # 4）等待一段再进入下一轮（让集群作业有时间产出结果；小步sleep保证 SIGTERM/SIGINT 信号响应及时）
        waited = 0
        while waited < poll and not _STOP:
            time.sleep(min(5, poll - waited))
            waited += 5

    print(f"\n👋 无限搜索已停止（共 {round_no} 轮）。数据库位于：{p['root']}",
          flush=True)
    print("   重跑本脚本即可续作；已提交的 GPU 作业不受影响。", flush=True)


# ===================== 入口 =====================

def main():
    parser = argparse.ArgumentParser(
        description="小肽序列无限搜索（结果持续追加进共享数据库，直到手动终止）")
    parser.add_argument("--targets", default=DEFAULT_TARGETS,
                        help=f"逗号分隔靶标，必须含 HTR1A（默认 {DEFAULT_TARGETS}；"
                             "V100 放不下 UNC13C，解锁后加 --targets HTR1A,BIN1,UNC13C）")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"每轮最多新提交的序列数（默认 {DEFAULT_BATCH}）")
    parser.add_argument("--poll", type=int, default=DEFAULT_POLL,
                        help=f"积压满时的轮询间隔秒数（默认 {DEFAULT_POLL}）")
    parser.add_argument("--migrate", action="store_true",
                        help="启动前把家目录旧 af3_local_results/ 的完成结果迁入共享库（复用不重算）")
    parser.add_argument("--status", action="store_true",
                        help="只打印数据库统计与 BEST_TOP10，不启动搜索")
    args = parser.parse_args()

    want = {t.strip().upper() for t in args.targets.split(",") if t.strip()}
    unknown = want - set(TARGET_ORDER)
    if unknown:
        parser.error(f"--targets 含未知靶标 {sorted(unknown)}，可选 {TARGET_ORDER}")
    if "HTR1A" not in want:
        parser.error("--targets 必须包含 HTR1A（优化目标）")
    active = [t for t in TARGET_ORDER if t in want]

    if args.migrate:
        print("📦 迁移家目录旧结果 → 共享数据库 …", flush=True)
        n, skipped = db.migrate_legacy()
        print(f"   迁移 {n} 个；跳过 {len(skipped)} 个", flush=True)
        for k, v in list(skipped.items())[:20]:
            print(f"     - {k}: {v}", flush=True)

    if args.status:
        s = db.stats()
        print(f"🗄️  {s['root']}：总 {s['total']} | HTR1A {s['htr1a']} | "
              f"BIN1 {s['bin1']} | UNC13C {s['unc13c']}")
        best_csv = os.path.join(s["root"], "BEST_TOP10", "BEST_TOP10.csv")
        if os.path.isfile(best_csv):
            with open(best_csv) as f:
                print(f.read())
        return

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    random.seed(42)
    np.random.seed(42)

    print("=" * 62)
    print("🧬 无限搜索 — 小肽序列（目标 = 乘积 "
          + " × ".join(f"ipTM({t})↑"
                      for t in PRODUCT_TARGETS if t in active)
          + "；无上限约束；结果持续追加共享数据库）")
    pend = [t for t in OFF_TARGETS if t not in active]
    if pend:
        print(f"   ⏸️  搁置因子：{pend}（超 V100 上限；A100/H100 到位后加进 PRODUCT_TARGETS 并补算，"
              " 已入库结果自动复用，仅补算搁置靶标并重排 BEST_TOP10）")
    print("=" * 62)

    # 就绪保障：预检查（数据库/权重/hmmer）+ 解压作业跟随（与有限入口一致）
    ok, errors = loc.ensure_local_ready()
    if not ok:
        print("\n⚠️  预检查未通过仍将尝试运行：旧结果可复用，新提交会被跳过\n",
              flush=True)

    backlog_limit = loc.BUNDLE_SIZE * BACKLOG_MULTIPLIER
    run_infinite(active, args.batch, args.poll, backlog_limit)


if __name__ == "__main__":
    main()
