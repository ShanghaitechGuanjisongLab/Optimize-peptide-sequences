"""
小肽序列优化 — 无限搜索入口（V100 现状: HTR1A + BIN1, 结果无限追加进共享数据库）
=============================================================
背景（2026-10 决策）：
  - bme_gpupub 分区全是 V100-32GB（官方上限 ~1280 tokens）：肽×HTR1A ≈ 441、肽×BIN1 ≈ 612 可算；肽×UNC13C ≈ 2233 超限, 暂时搁置。
  - 因此现阶段搜索目标调整为: **HTR1A ipTM 尽量大, BIN1 ipTM < 0.8**（UNC13C 约束去掉；由此得到的"最佳"未来可能部分因 UNC13C 越线被筛除, 所以策略是**多算多存**——结果持续追加进共享数据库, 等 A100/H100 到位后直接从数据库里补算 UNC13C 并筛选, 已完成结果不重算）。

与 optimize_peptide_BO.py（有限预算）的关系：
  本入口是 BO 的"无限 + 数据库化"变体: 复用其序列特征化/高斯过程/约束期望改进采集函数, 但
    1) 单一真源改为共享数据库（peptide_db.py, 存于共享目录, 计算节点可见）;
    2) 无预算上限: 候选来自 **精确穷尽枚举**（`enumerate_space()`, 51091 条）;
    3) 每轮: 收割新完成结果入库 → 刷新 BEST_TOP10 → 按启发式挑选新候选提交→无进展则轮询等待；直到手动终止（SIGINT/SIGTERM 优雅退出）。

候选调度（2026-10-04 第二次定案: 穷尽队列取代随机采样）：
  现在的主调度器是 `ExhaustiveQueue` —— 全集为 `peptide_common.enumerate_space()`精确构造的 51091 条（编辑距离 ≤2、长 6~10、20 种标准氨基酸）, 每轮从「全集 − 已派发台账(sequences.csv)」中取 n 条。由此得到三条可证明的性质：
  **不重复**、待办集合每轮精确定量递减、至多 ⌈51091/24⌉=2129 轮 ≈680 天必然全覆盖（取空后 next_batch 返回 [] = 明确的完成信号）。GP只决定"先跑谁"（cEI 把预测 HTR1A 分高、BIN1 可行的排前面）, 对上述性质零贡献；GP 失效则退化为字典序游标推进, 覆盖与不重复依然成立。
  取代随机采样的理由（均实测）: 旧 `CandidateStream` 每周期只撒 ~720 个随机点供 GP 排序（argmax 在子集上取, 不在整个空间上）, 且 `generate_mutant` 有**665 条序列永远采不到**（模式B 的 delta=±1 只做单次插入或删除）。退回旧行为: `PEPOPT_SWEEP=0`。

仍保留的启发式先后顺序（只影响次序, 不影响覆盖保证）：
  1) 先评估初始序列 SEQ_202（基线锚点）;
  2) 评估数 < MIN_GP_SAMPLES(默认 12) 时, 整批走字典序游标（冷启动）;
  3) 之后每轮 batch 中 ~EXPLORE_FRAC(默认 25%) 配额仍按字典序游标推进（游标 = 已派发数, 跨作业重启自动接续）, 其余按约束期望改进cEI = EI(score=1-ipTM_HTR1A) × P(ipTM_BIN1 < 0.8) 挑选;
  4) GP 训练集超 TRAIN_CAP(默认 2000) 时做确定性分层子采样 —— 因为拟合是O(n³)（实测 4000 条已 52s, 外推 5 万条 ≈9.6h > 周期预算）, 而全空间打分实测仅 ≈1s, 故瓶颈在训练侧不在候选侧。

用法（登录节点, 建议经 run.sh infinite 后台运行）：
    python optimize_peptide_infinite.py                  # 默认 --targets HTR1A,BIN1
    python optimize_peptide_infinite.py --migrate       # 先把家目录旧结果迁入共享库再开跑
    python optimize_peptide_infinite.py --batch 6 --poll 300
    python optimize_peptide_infinite.py --status         # 只打印数据库统计与最佳结果, 不开跑
终止：kill <PID>（或 Ctrl+C）；已提交的 SLURM 作业不受影响, 重跑即续作。

共享数据库（peptide_db, 默认 /public_bme2/Share200T/管吉松/peptide_opt_db）：
    peptide_database.csv  每序列一行: iptm_htr1a / iptm_bin1 / iptm_unc13c(现为空)
    sequences.csv         tag → 序列注册表
    af3_results/<tag>_<target>/   全部 AF3 结果（输入/输出/评分）
    BEST_TOP10/           当前 top10 的 HTR1A 结构包 + BEST_TOP10.csv

将来 UNC13C 解锁（申请到 A100/H100-80G）：
    PEPOPT_GPU_MAX_TOKENS=5120 PEPOPT_FLASH_ATTN=triton PEPOPT_XLA_7X=0 \
    PEPOPT_PARTITION=<新分区> python optimize_peptide_infinite.py --targets HTR1A,BIN1,UNC13C
  已入库序列只补算 UNC13C（HTR1A/BIN1 直接复用）, refresh_best 自动按全约束重排。
"""

import os
import sys
import time
import signal
import random
import argparse
import numpy as np

# 注意: 必须在 import optimize_peptide_local **之前**设定结果目录环境变量, 该模块在 import 时读取 PEPOPT_RESULTS_DIR（指向共享库的 af3_results/）。
import peptide_db as db
os.environ.setdefault("PEPOPT_RESULTS_DIR", db.results_dir())

import optimize_peptide_local as loc          # 本地集群提交后端（复用打包/节流/OOM拦截）
import optimize_peptide_BO as bo              # 特征化 + GP + cEI 采集函数（复用）

# 防御: 无论 loc 以何种顺序被导入, 都把其结果目录强制对齐到共享库, 保证推理任务的 input.json 与结果一定落在共享盘（而非家目录 af3_local_results）。
loc.LOCAL_RESULTS_DIR = db.results_dir()

from peptide_common import (SEQ_202, FIXED_A, ORIG_TAIL,
                            HTR1A_SEQ, UNC13C_SEQ, BIN1_SEQ, generate_mutant,
                            enumerate_space)

# ===================== 配置 =====================

TARGET_SEQS   = {"HTR1A": HTR1A_SEQ, "UNC13C": UNC13C_SEQ, "BIN1": BIN1_SEQ}
TARGET_ORDER  = ["HTR1A", "UNC13C", "BIN1"]      # 固定评估顺序
OFF_TARGETS   = ["UNC13C", "BIN1"]               # 底线蛋白（越低越好）
OFF_COL       = {"UNC13C": "iptm_unc13c", "BIN1": "iptm_bin1"}

DEFAULT_TARGETS      = "HTR1A,BIN1"              # V100 现状: 搁置 UNC13C
OFF_THRESHOLD        = 0.8                       # 底线蛋白可行阈值
MIN_GP_SAMPLES       = 12                        # 冷启动评估数（不足时纯随机探索）
EXPLORE_FRAC         = 0.25                      # 每轮随机探索比例（攒数据库多样性）
# 每轮 GP 挑选的序列数。**与 loc.BUNDLE_SIZE 解耦**: 包容量大是为了摊薄55min固定 MSA 开销(见 optimize_peptide_local), 但每轮 GP 只应挑少量高信息量候选——样本还不多(几十条)时一次挑几十个会让采集函数退化、重复采样低价值区域。差额由 submit_pending_bundle 的"攒批"机制补齐: 多轮各挑几个, 攒满一包才提交, 期间每轮先收割新结果 → GP 持续变强, 既不空转 GPU 也不牺牲引导质量。（复用 loc._env_int: 未设/空串取默认, 非法值告警后取默认, 切勿用 int(x or d) 写法）
DEFAULT_BATCH        = max(1, loc._env_int("PEPOPT_BATCH", 6))
DEFAULT_POLL         = 300                       # 无进展时的轮询间隔（秒）
BACKLOG_MULTIPLIER   = 2                         # 积压上限 = BUNDLE_SIZE × 该倍数（个任务）
TOP_N_BEST           = 10                        # BEST_TOP<N>

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n🛑 收到信号 {signum}, 本轮结束后优雅退出（已提交的 SLURM 作业继续运行, 重跑本脚本即续作）…", flush=True)


# ===================== 候选池（按需无限扩展） =====================

class CandidateStream:
    """编辑距离 ≤2 变异序列的惰性无限流（去重, 固定种子可复现）。"""

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
        """把未最终提交的候选放回流（GP 短名单只是临时抽样, 不释放会白白消耗候选空间）。"""
        for s in seqs:
            if s != SEQ_202:
                self._seen.discard(s)


# ===================== 数据库视图辅助 =====================

def db_evaluated_rows(active_targets):
    """已评估(所有激活靶标齐备)的行: [(peptide, score_htr1a, {约束值})]"""
    by_tag = db.load_database()
    rows = []
    for tag, r in by_tag.items():
        if r.get("iptm_htr1a") is None or not r.get("peptide"):
            continue
        cons = {OFF_COL[t]: r.get(OFF_COL[t]) for t in active_targets if t in OFF_COL}
        rows.append({"tag": tag, "peptide": r["peptide"],
                     "score_htr1a": r["score_htr1a"], "cons": cons})
    return rows


def in_flight_task_dirs(active_targets):
    """共享结果目录中"已创建但未完成"的任务目录（排队/运行中）。"""
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
    """按启发式先后顺序挑 n 个新候选：评估样本 < MIN_GP_SAMPLES → 全部随机探索；否则 ~EXPLORE_FRAC 随机探索 + 其余按 cEI(HTR1A 目标 × BIN1 可行概率)。"""
    rows = db_evaluated_rows(active_targets)
    exclude = known_sequences()
    active_off = [t for t in OFF_TARGETS if t in active_targets]

    n_explore = n if len(rows) < MIN_GP_SAMPLES \
        else max(1, int(round(n * EXPLORE_FRAC)))
    n_gp = n - n_explore
    picked = []

    if n_gp > 0 and len(rows) >= 3:
        try:
            picked = _gp_pick(rows, active_off, n_gp, exclude, stream)
        except Exception as e:                        # GP 数值异常不应中断无限搜索
            print(f"  ⚠️ GP 挑选失败({e}), 本轮全部随机探索", flush=True)

    need = n - len(picked)
    if need > 0:                                      # 冷启动/GP 不足时随机补齐
        picked.extend(stream.next_batch(need, exclude | set(picked)))
    return picked[:n]


def _gp_pick(rows, active_off, n, exclude, stream):
    """cEI 挑选: 拟合 GP → 在"新随机候选 + 已有样本邻域"组成的临时候选集上取 TOP。"""
    X_obs = np.array([bo.featurize(r["peptide"]) for r in rows])
    y_obs = np.array([r["score_htr1a"] for r in rows], dtype=float)

    # 临时候选集: 随机撒 40×n 个新序列（无限流的天然探索）
    cand_seqs = stream.next_batch(max(40 * n, 200), exclude)
    if len(cand_seqs) < 2:
        return []
    X_cand = np.array([bo.featurize(s) for s in cand_seqs])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(np.vstack([X_obs, X_cand]))
    X_obs_s, X_cand_s = scaler.transform(X_obs), scaler.transform(X_cand)

    gp_t = bo.fit_gp(X_obs_s, y_obs)
    best_y = float(np.min(y_obs))
    gp_cons = []
    for t in active_off:
        yv = np.array([r["cons"].get(OFF_COL[t], np.nan) for r in rows], dtype=float)
        m = np.isfinite(yv)
        if int(m.sum()) >= 3:
            gp_cons.append(bo.fit_gp(X_obs_s[m], yv[m]))

    acq = bo.constrained_acquisition(gp_t, gp_cons, X_cand_s, best_y)
    if not np.all(np.isclose(acq, 0)) and bool(np.isfinite(acq).all()):
        idx = bo.select_batch(acq, X_cand_s, list(range(len(cand_seqs))), n)
        picked = [cand_seqs[i] for i in idx]
    else:
        # 采集值全零/异常 → 退化为按 GP 均值(预测分低者优先)排序
        mu, _ = gp_t.predict(X_cand_s, return_std=True)
        order = np.argsort(mu)[:n]
        picked = [cand_seqs[i] for i in order]
    stream.release([s for s in cand_seqs if s not in picked])   # 未入选的放回
    return picked


# ===================== 穷尽队列: 不重复 + 有限步全覆盖 =====================

SPACE_EDIT     = loc._env_int("PEPOPT_SPACE_EDIT", 2)
SPACE_LEN_MIN  = loc._env_int("PEPOPT_SPACE_LEN_MIN", 6)
SPACE_LEN_MAX  = loc._env_int("PEPOPT_SPACE_LEN_MAX", 10)
# GP 打分池上限: 0 = **不限制, 对整个待办集打分**（默认）。实测（n_train=34）: 候选 5000 条 0.1s / 20000 条 0.4s / **全部 51091 条仅 1.0s**, 故候选池根本不是瓶颈；早期版本把它限在 20000 只是白费了 GP 的全局视野。保留旋钮仅为日后特征维度大幅上升时可重新收紧。
SCORE_POOL_CAP = loc._env_int("PEPOPT_SCORE_POOL", 0)
# GP **训练集**上限: 真正的瓶颈在这里。实测拟合+预测耗时随 n_train 近似立方增长:
#   n_train=34 → 0.5s | 1000 → 7.8s | 2000 → 10.7s | 4000 → 52s | 外推 51091 → ≈9.6h
# 9.6h 已超过一个周期（n=24 时 7.7h）→ 若不封顶, 穷尽搜索到后期会直接超时。
# 超限后按确定性分层子采样（见 _subsample_train）, 只影响"先跑谁"的次序, 对不重复/全覆盖三条不变量无影响（best_y 仍取**全量**最优, 不因子采样而失真）。
TRAIN_CAP = loc._env_int("PEPOPT_TRAIN_CAP", 2000)
# 总开关: 置 0 退回旧的随机采样启发式(CandidateStream + heuristic_pick)。
SWEEP_ENABLED = os.environ.get("PEPOPT_SWEEP", "1").strip().lower() not in \
                ("0", "false", "no", "off")


def _subsample_train(rows, cap=TRAIN_CAP):
    """训练集超限时做**确定性分层子采样**: 一半取当前最优（保住高分区分辨率, cEI 排序最关心这里）, 另一半按 score 排序等距抽样（保住全分布的形状信息）。两条规则都不含随机数 → 同样的库状态必得同样的子集, 跨作业重启可复现。返回 (子集, 未截断的 best_y)； best_y 始终基于全量 rows, 不因截断而偏移。"""
    srt = sorted(rows, key=lambda r: (r["score_htr1a"], r.get("tag", "")))
    best_y = float(min(r["score_htr1a"] for r in rows))
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
    """sequences.csv = **已派发台账**。prepare_cycle 在写任何输入之前先调用register_sequence, 故凡被队列取走过的可变区必然登记在此 —— "不重复"这条性质由台账保证, 不依赖任何内存状态, 因而天然跨作业重启持久。"""
    out = set()
    for info in db.load_sequence_registry().values():
        v = info.get("var")
        if v:
            out.add(v)
    return out


class ExhaustiveQueue:
    """确定性穷尽调度器 —— 同时保证【不重复】与【有限步内全覆盖】。

    三条不变量（合起来构成覆盖证明）:
      A. 全集 S = peptide_common.enumerate_space(): **精确构造**而非随机采样, 因此包含随机采样不可达的 665 条；|S| = 51091, 枚举顺序固定（字典序）, 与随机种子和调用次数无关。
      B. 待办 P_k = S \\ A_k, 其中 A_k = sequences.csv 已登记台账。prepare_cycle先登记再写输入 → 凡被派发必入 A；于是每周期 |P| 至少减少本批大小, 且**同一序列不可能被二次派发**。
      C. S 有限 → 至多 ceil(|S|/n) = ceil(51091/24) = 2129 个周期后 P = ∅, 即全覆盖。届时 next_batch 返回空列表, 调用方把它当**完成**而非失败（与旧的 EMPTY_CYCLE_LIMIT"采不到新样本"语义明确区分）。

    GP/贝叶斯在其中的角色: **只决定 P 里先跑谁**（把"预测 HTR1A 分高且BIN1可行"的排在前面, 让好结果尽早入库）, 对 A/B/C 三条不变量毫无贡献。即使 GP 完全失效（样本不足、数值异常、特征退化）, 次序也只是退化为字典序游标推进, 不重复与全覆盖依然成立。这就是"调整调度算法保证不重复并最终全覆盖"的准确含义: **覆盖由枚举 + 台账保证, 不由启发式保证**。
    """

    CENTER = ORIG_TAIL

    def __init__(self, active_targets=(), center=CENTER, max_edit=SPACE_EDIT,
                 min_len=SPACE_LEN_MIN, max_len=SPACE_LEN_MAX):
        self.active = tuple(active_targets) or ("HTR1A", "BIN1")
        self.center, self.max_edit = center, max_edit
        self.space = enumerate_space(center, max_edit, min_len, max_len)
        if not self.space:
            raise RuntimeError("穷尽枚举返回空集, 无法保证覆盖")

    # ---- 状态视图 ----
    def pending(self, attempted=None):
        """待评估的可变区（保持字典序 → 确定性游标）。"""
        A = attempted_vars() if attempted is None else attempted
        return [v for v in self.space if v not in A]

    def coverage(self, attempted=None):
        P = self.pending(attempted)
        done = len(self.space) - len(P)
        return {"total": len(self.space), "dispatched": done, "pending": len(P),
                "frac": (done / float(len(self.space))) if self.space else 1.0}

    @staticmethod
    def _feat(seqs):
        return np.array([bo.featurize(s) for s in seqs], dtype=float)

    # ---- 取批 ----
    def next_batch(self, n, rows=None, active_off=None):
        """取 n 条未派发序列（返回**全序列**形式, 与 prepare_cycle 约定一致）,保证批内互不重复且不与台账重复；剩余不足 n 时返回剩余全部（不伪造）。返回空列表 = 空间已穷尽。"""
        P = self.pending()
        if n <= 0 or not P:
            return []
        if n >= len(P):
            return [FIXED_A + v for v in P]         # 末批: 全取, 顺序即字典序

        rows = db_evaluated_rows(self.active) if rows is None else rows
        if active_off is None:
            active_off = [t for t in OFF_TARGETS if t in self.active]

        # 冷启动（GP 不可拟合）时整批走探索；否则保留 EXPLORE_FRAC 配额。探索用**字典序旋转游标**而非随机采样: 游标 = 已派发数（由台账派生,跨作业重启自动接续）, 因此既有系统性覆盖推进, 又完全可复现。
        n_exp = n if len(rows) < MIN_GP_SAMPLES else max(1, int(round(n * EXPLORE_FRAC)))
        n_exp = max(0, min(n_exp, n))
        cur = (len(self.space) - len(P)) % len(P)
        rot = P[cur:] + P[:cur]
        picked, chosen = rot[:n_exp], set(rot[:n_exp])

        n_gp = n - len(picked)
        if n_gp > 0 and len(rows) >= 3:
            try:
                for v in self._gp_order(P, chosen, rows, active_off, n_gp):
                    picked.append(v)
                    chosen.add(v)                   # chosen 与 _gp_order 的 exclude 同一对象
            except Exception as e:                  # GP 异常绝不阻断穷尽推进
                print(f"  ⚠️ GP 排序失败({e}), 本轮改为字典序游标推进", flush=True)

        for v in rot:                               # 仍不足 → 按游标补齐
            if len(picked) >= n:
                break
            if v not in chosen:
                picked.append(v)
                chosen.add(v)
        return [FIXED_A + v for v in picked[:n]]

    def _gp_order(self, P, exclude, rows, active_off, n_gp):
        """按 cEI 降序产出可变区（稳定排序 → 同分按字典序, 可复现）。exclude 就地累积, 便于调用方复用同一集合去重。"""
        if n_gp <= 0:
            return []
        if SCORE_POOL_CAP > 0 and len(P) > SCORE_POOL_CAP:
            stride = len(P) / float(SCORE_POOL_CAP)
            pool = [P[int(i * stride)] for i in range(SCORE_POOL_CAP)]
        else:
            pool = list(P)                  # 默认: 对全部待办集打分（实测仅 ≈1s）
        pool = [v for v in pool if v not in exclude]
        if len(pool) < 2:
            return []

        train, best_y = _subsample_train(rows)      # best_y 基于全量, 不受截断影响
        X_obs = self._feat([r["peptide"] for r in train])
        y_obs = np.array([r["score_htr1a"] for r in train], dtype=float)
        X_cand = self._feat([FIXED_A + v for v in pool])
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(np.vstack([X_obs, X_cand]))
        X_obs_s, X_cand_s = sc.transform(X_obs), sc.transform(X_cand)

        gp_t = bo.fit_gp(X_obs_s, y_obs)
        gp_cons = []
        for t in active_off:
            yv = np.array([r["cons"].get(OFF_COL[t], np.nan) for r in train], dtype=float)
            m = np.isfinite(yv)
            if int(m.sum()) >= 3:                   # 约束 GP 至少 3 个有限样本
                gp_cons.append(bo.fit_gp(X_obs_s[m], yv[m]))

        acq = bo.constrained_acquisition(gp_t, gp_cons, X_cand_s, best_y)
        if bool(np.isfinite(acq).all()) and not np.all(np.isclose(acq, 0)):
            order = np.argsort(-np.asarray(acq, dtype=float), kind="stable")
        else:                    # 采集退化 → 用 GP 均值(score=1-ipTM, 小者优先)
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
    """穷尽队列单例（枚举 51091 条只需约 0.1s, 但避免每周期重建）。"""
    global _SWEEP
    if _SWEEP is None or (active_targets and tuple(active_targets) != _SWEEP.active):
        _SWEEP = ExhaustiveQueue(tuple(active_targets) if active_targets else None)
    return _SWEEP


def sweep_pick(active_targets, n):
    """穷尽队列取批（替代随机采样的 heuristic_pick）。返回 [] 表示**空间已穷尽**（应视为正常完成, 而非采样失败）。"""
    return get_sweep(active_targets).next_batch(n)


def coverage_report(active_targets, cycle_secs=None, per_cycle=24):
    """覆盖进度 + ETA 文本。cycle_secs 给出时按每周期 per_cycle 条折算剩余天数。per_cycle由调用方显式传入（不 import optimize_peptide_node, 避免独立运行本模块时的循环导入）。"""
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

    复用 optimize_peptide_local.local_predict_and_score: 它按<LOCAL_RESULTS_DIR>/<job_name>布局写 input.json, 并由submit_pending_bundle()统一打包提交（含 MAX_INFLIGHT 节流与失败重试）。job_name 用共享库的稳定规则 <tag>_<target>, 结果直接落共享盘；已就绪的（迁移过的旧结果/上轮完成的）返回 iptm, 即刻 upsert 入库（并写 .harvested, 与 harvest_results 幂等一致）。"""
    n = 0
    for seq in seqs:
        if _STOP:
            break
        var = seq[len(FIXED_A):]
        tag = db.resolve_tag(var)      # 注册表优先: 沿用历史标签, 不孤儿化已有结果
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


def harvest_and_report(active_targets):
    """收割新完成结果入库, 刷新 BEST_TOP10, 打印统计。返回新增条数。"""
    n_new, skipped = db.harvest_results()
    if n_new:
        print(f"  📥 收割入库 {n_new} 条新结果"
              + (f"（跳过 {len(skipped)}）" if skipped else ""), flush=True)
    db.refresh_best(off_threshold=OFF_THRESHOLD, top_n=TOP_N_BEST,
                    targets=tuple(active_targets))
    s = db.stats()
    rows = db_evaluated_rows(active_targets)
    evaluable = []
    for r in rows:
        cons = [r["cons"].get(OFF_COL[t]) for t in active_targets if t in OFF_COL]
        if all(v is not None and v < OFF_THRESHOLD for v in cons):
            evaluable.append(r)
    best3 = sorted(rows, key=lambda r: r["score_htr1a"])[:3]
    print(f"  🗄️  数据库: 总 {s['total']} 序列 | HTR1A {s['htr1a']} | "
          f"BIN1 {s['bin1']} | UNC13C {s['unc13c']} | 完整评估 {len(rows)} | "
          f"当前可行 {len(evaluable)}", flush=True)
    for r in best3:
        cons = " ".join(f"{t}={r['cons'].get(OFF_COL[t])}" for t in active_targets
                        if t in OFF_COL)
        print(f"     🏅 {r['peptide']}  ipTM_HTR1A={1 - r['score_htr1a']:.3f}  {cons}",
              flush=True)
    return n_new


# ===================== 主循环 =====================

def run_infinite(active_targets, batch, poll, backlog_limit):
    p = db.paths(ensure=True)
    os.makedirs(p["results"], exist_ok=True)
    stream = CandidateStream(seed=42)

    # 初始序列永远排第一（基线锚点）: 任一激活靶标无结果即需提交（submit_candidates 幂等, 已完成的靶标会被直接提取评分不重算）
    var0 = SEQ_202[len(FIXED_A):]
    tag0 = db.resolve_tag(var0)
    need_init = any(
        not db.find_summary(os.path.join(p["results"], f"{tag0}_{t.lower()}"))
        for t in active_targets)

    round_no = 0
    print(f"\n♾️  无限搜索启动: 靶标={active_targets} | 底线约束="
          f"{[t for t in OFF_TARGETS if t in active_targets]} "
          f"(ipTM<{OFF_THRESHOLD}, UNC13C 搁置时其列留空) | 每轮≤{batch} 新序列 | "
          f"积压上限 {backlog_limit} 任务 | 轮询 {poll}s\n"
          f"    共享数据库: {p['root']}\n", flush=True)

    while not _STOP:
        round_no += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"== [{ts}] 轮次 {round_no} ==", flush=True)

        # 1) 收割 + 最佳刷新 + 统计
        harvest_and_report(active_targets)
        if _STOP:
            break

        # 2) 初始序列优先
        if need_init:
            print("  🧬 提交初始序列 SEQ_202（基线）", flush=True)
            submit_candidates([SEQ_202], active_targets)
            need_init = False

        # 3) 计算积压, 决定是否提交新候选
        inflight = in_flight_task_dirs(active_targets)
        print(f"  🚚 在途任务 {len(inflight)}/{backlog_limit} | "
              f"SLURM 队列深度 {loc.queue_depth()}", flush=True)
        if len(inflight) < backlog_limit:
            n_new = max(1, (backlog_limit - len(inflight)) // len(active_targets))
            n_new = min(n_new, batch)
            picks = heuristic_pick(stream, n_new, active_targets)
            if picks:
                print(f"  🎯 挑选 {len(picks)} 个新候选提交: "
                      f"{[db.resolve_tag(s[len(FIXED_A):]) for s in picks]}", flush=True)
                submit_candidates(picks, active_targets)
        # 即便不提交新候选, 也触发一次打包提交（消化攒批/失败重试）
        loc.submit_pending_bundle()

        # 4) 等待一段再进入下一轮（让集群作业有时间产出结果；小步sleep保证 SIGTERM/SIGINT 信号响应及时）
        waited = 0
        while waited < poll and not _STOP:
            time.sleep(min(5, poll - waited))
            waited += 5

    print(f"\n👋 无限搜索已停止（共 {round_no} 轮）。数据库位于: {p['root']}",
          flush=True)
    print("   重跑本脚本即可续作；已提交的 GPU 作业不受影响。", flush=True)


# ===================== 入口 =====================

def main():
    parser = argparse.ArgumentParser(
        description="小肽序列无限搜索（结果持续追加进共享数据库, 直到手动终止）")
    parser.add_argument("--targets", default=DEFAULT_TARGETS,
                        help=f"逗号分隔靶标, 必须含 HTR1A（默认 {DEFAULT_TARGETS}；"
                             "V100 放不下 UNC13C, 解锁后加 --targets HTR1A,BIN1,UNC13C）")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"每轮最多新提交的序列数（默认 {DEFAULT_BATCH}）")
    parser.add_argument("--poll", type=int, default=DEFAULT_POLL,
                        help=f"积压满时的轮询间隔秒数（默认 {DEFAULT_POLL}）")
    parser.add_argument("--migrate", action="store_true",
                        help="启动前把家目录旧 af3_local_results/ 的完成结果迁入共享库（复用不重算）")
    parser.add_argument("--status", action="store_true",
                        help="只打印数据库统计与 BEST_TOP10, 不启动搜索")
    args = parser.parse_args()

    want = {t.strip().upper() for t in args.targets.split(",") if t.strip()}
    unknown = want - set(TARGET_ORDER)
    if unknown:
        parser.error(f"--targets 含未知靶标 {sorted(unknown)}, 可选 {TARGET_ORDER}")
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
        print(f"🗄️  {s['root']}: 总 {s['total']} | HTR1A {s['htr1a']} | "
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
    print("🧬 无限搜索 — 小肽序列（HTR1A ipTM↑"
          + (" / " + "+".join(t for t in OFF_TARGETS if t in active) + " ipTM↓"
             if any(t in active for t in OFF_TARGETS) else "")
          + "；结果持续追加共享数据库）")
    if any(t not in active for t in OFF_TARGETS):
        pend = [t for t in OFF_TARGETS if t not in active]
        print(f"   ⏸️  搁置约束: {pend}（超 V100 上限；A100/H100 到位后全靶标重跑,"
              " 已入库结果自动复用, 仅补算搁置靶标并重排 BEST_TOP10）")
    print("=" * 62)

    # 就绪保障: 预检查(数据库/权重/hmmer) + 解压作业跟随（与有限入口一致）
    ok, errors = loc.ensure_local_ready()
    if not ok:
        print("\n⚠️  预检查未通过仍将尝试运行: 旧结果可复用, 新提交会被跳过\n",
              flush=True)

    backlog_limit = loc.BUNDLE_SIZE * BACKLOG_MULTIPLIER
    run_infinite(active, args.batch, args.poll, backlog_limit)


if __name__ == "__main__":
    main()
