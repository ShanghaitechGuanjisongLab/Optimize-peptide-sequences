"""
小肽优化 — 序贯采样 + 标准 β 分布拟合 + 数据库整理（增量充分统计）
=============================================================
用户定稿设计（2026-10-14 第二版；修正第一版“灵活上界”的缺陷）：

1. **首次采样 = 无上限爬山**（2026-10-21 用户定稿，替代旧“先算 2 个、首次下降即停”）：新记录一律反复补采新 seed（真跑 AF3），只要还在破纪录就继续，直到**整批补采样本都未破纪录**才停；次数无上限，由破纪录判定终止。取已算样本的最大值为“最优结果”。注：“算一次”= **真跑一次 AlphaFold**（新 diffusion 样本），不是从拟合的分布里抽随机数（拟合只用于估计 P）。
   实现注：AF3 一次运行批量产出多个 diffusion 样本（MSA 共享，边际成本低），初次批量样本全部计入拟合（consume_sequential 不再提前停）；爬山循环在 小肽优化_无限.climb_new_records 实现。

2. **β 分布采用标准 [0,1]**（不设灵活上界 U）：ipTM ∈ [0,1]、乘积 ∈ [0,1]，故 [0,1] 就是理论支撑。⚠️ 第一版曾用 MLE 估上界 U≈样本max，但早期采样运气差会把 U 压低 →P=0 → 该记录**永无翻身之日**（上界估计量必然早期低估）。故改固定 [0,1]：P 永不精确为 0，任何记录只要真实分布能破纪录就迟早被采到。

3. **P = 超过榜首的概率**（榜首自身除外）：榜首 K=1：阈值 = (max+1)/2（原设计不变）；其余 K>1：阈值 = **榜首的 max**（“能不能当第一”，而非“能不能爬升一名”）。副产品：远离榜首者 P 极小 → 自动不被采样，算力聚焦于竞争者。

4. **循环条件 P·N > M**（M = 该记录已采样次数）：
   终止性证明：每采一次 M←M+1；P≤1 ⇒ P·N≤N ⇒ M>N 时条件必失效。每记录补采 ≤N 次 ⇒ **必定有限步终止**，不依赖分布假设。（值-of-information 形式：P·N ≈ 期望收益（破纪录概率×一举超越的记录数），M = 已投入成本；收益递减+成本递增 ⇒ 必停。）
   ⚠️ 例外——**单点/退化记录**（n=1、或样本全同/全 0 全 1，β 拟合无解 → P 恒按 max 判定为 0 → 上述循环永远不给它补采）：2026-10-21 起整理时**改走与首次同规格的爬山**——只要还在破纪录就一直补采，一旦整批未破纪录即爬山结束（置 obj_climb 终止标记，之后不再重复烧算力）；终止性由“破纪录判定 + MAX_EXTRA_SAMPLES 工程上限”保证。

5. **增量充分统计** (n, Σx, Σx²) + 方法矩拟合：追加拟合 == 全数据批式拟合，不保存全数据。μ=Σx/n，σ²=Σx²/n-μ²，κ=μ(1-μ)/σ²-1，α=μκ，β=(1-μ)κ。
"""

import math
import json

try:
    from scipy.special import betainc
    _HAVE_SCIPY = True
except ImportError:                                  # 无 scipy 时退化为正态近似
    _HAVE_SCIPY = False

# 每记录补采硬上限。**纯工程保险**，正常不触发——终止性已由 P·N>M 保证。
MAX_EXTRA_SAMPLES = 50


# ---------------------------------------------------------------------
# 标准 [0,1] β 分布的增量拟合
# ---------------------------------------------------------------------

class BetaFit:
    """标准 [0,1] β 分布的增量拟合（方法矩）。

    只保存充分统计量（n、sum_x、sum_x2）与 max_x / pool，不保存已消费样本。
    add() 追加一个样本；fit() 由统计量方法矩拟合，与全数据批式拟合结果相同。
    """

    __slots__ = ("n", "sum_x", "sum_x2", "max_x", "pool", "extra_used",
                 "climb_done")

    def __init__(self):
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.max_x = None            # 最优结果（排名/阈值用它）
        self.pool = []               # 未消费样本（整理阶段补采用）
        self.extra_used = 0          # 整理阶段已补采次数（硬上限用）
        self.climb_done = 0          # 1 = 爬山已以“不再破纪录”收束（首次爬山或整理单点爬山写入；未置位则整理还会给退化记录重新爬山）

    # ---- 增量更新（充分统计量）----
    def add(self, x):
        x = float(x)
        if x < -1e-12 or x > 1.0 + 1e-12:
            raise ValueError(f"样本 {x} 超出 [0, 1]")
        x = min(max(x, 0.0), 1.0)
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x
        if self.max_x is None or x > self.max_x:
            self.max_x = x

    def add_many(self, xs):
        for x in xs:
            self.add(x)

    # ---- 方法矩拟合 ----
    def fit(self):
        """→ （alpha，beta）或 None（样本不足/方差退化）。
        由（μ，σ²）方法矩反解 Beta 参数，与全数据批式拟合一致。"""
        if self.n < 2:
            return None
        mu = self.sum_x / self.n
        var = self.sum_x2 / self.n - mu * mu
        if not (0.0 < mu < 1.0):
            return None                       # 全 0 或全 1，退化
        if var <= 0:
            return None                       # 零方差，退化（由 prob_gt 处理）
        kappa = mu * (1.0 - mu) / var - 1.0
        if kappa <= 0:
            # 过度离散：方法矩无解，夹到极小正值（宽分布，偏保守）
            kappa = 1e-3
        return mu * kappa, (1.0 - mu) * kappa

    # ---- 超阈概率（下一次 AlphaFold 采样 > t）----
    def prob_gt(self, t):
        """P（下一次 AlphaFold 采样 > t）= β 拟合的尾概率。

        标准 [0,1] 支撑保证**无死记录**：t<1 时 P 恒 >0，不会出现 “概率恒 0 → 永不补采”的闲死（第一版灵活上界的缺陷）。
        注：这里的 P 只用于估计概率；实际样本来自真跑 AlphaFold。"""
        t = float(t)
        if t <= 0:
            return 1.0
        if t >= 1.0:
            return 0.0
        if self.n == 0:
            return 1.0
        ab = self.fit()
        if ab is None:
            # 退化（零方差等）：按 max 判定
            return 1.0 if (self.max_x is not None and t < self.max_x) else 0.0
        a, b = ab
        if _HAVE_SCIPY:
            p = float(1.0 - betainc(a, b, t))
        else:
            p = _normal_tail(t, a / (a + b),
                             math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
        return max(0.0, min(1.0, p))

    # ---- 消费样本 ----
    def consume_sequential(self, xs):
        """消费 xs 的全部样本并返回个数。

        此处曾按序贯规则“首次下降即停、尾部留在 pool”，但 max_x 只在 add() 里更新→停在池中的更优样本永远进不了排名，iptm 回写用的 argmax 也一并偏移（实测206/698 条被低估，最重的一条真实 0.5313 记成 0.5159，第 3 名掉到第 8 名）。样本既已落盘，全用不额外耗算力，β 拟合拿到更多数据只会更准，故不再提前停止；pool 留为空字段以兼容 from_stats 反序列化旧行。"""
        if not xs:
            return 0
        for x in xs:
            self.add(x)
        return len(xs)

    def draw_from_pool(self):
        """整理阶段补采：优先取池中免费样本；池空返回 None（需外部生成）。"""
        while self.pool:
            x = self.pool.pop(0)
            if self.max_x is None or x > self.max_x - 1e-15:
                self.add(x)
                self.extra_used += 1
                return x
            # 池中样本不会超过当前 max 的情况仍可加入拟合（提供更多分布信息）
            self.add(x)
            self.extra_used += 1
            return x
        return None

    # ---- 序列化（供 CSV 存储）----
    def stats_dict(self):
        ab = self.fit()
        return {
            "obj_n": self.n,
            "obj_sum": f"{self.sum_x:.10g}",
            "obj_sum2": f"{self.sum_x2:.10g}",
            "obj_max": "" if self.max_x is None else f"{self.max_x:.6f}",
            "obj_alpha": "" if ab is None else f"{ab[0]:.6f}",
            "obj_beta": "" if ab is None else f"{ab[1]:.6f}",
            "obj_extra": self.extra_used,
            "obj_pool": json.dumps([round(x, 6) for x in self.pool]),
            "obj_climb": "" if not self.climb_done else "1",
        }

    @classmethod
    def from_stats(cls, d):
        """从 CSV 行恢复（含 pool），支持增量续算。"""
        f = cls()
        f.n = int(float(d.get("obj_n") or 0))
        f.sum_x = float(d.get("obj_sum") or 0.0)
        f.sum_x2 = float(d.get("obj_sum2") or 0.0)
        mx = d.get("obj_max") or ""
        f.max_x = float(mx) if mx != "" else None
        f.extra_used = int(float(d.get("obj_extra") or 0))
        f.climb_done = 1 if str(d.get("obj_climb") or "").strip() else 0
        try:
            f.pool = [float(x) for x in json.loads(d.get("obj_pool") or "[]")]
        except (ValueError, TypeError):
            f.pool = []
        return f


# ---------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------

def _normal_tail(y, mu, sd):
    """正态近似 P(X > y)（无 scipy 时的退化路径）。"""
    if sd <= 0:
        return 1.0 if y < mu else 0.0
    z = (y - mu) / sd
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# ---------------------------------------------------------------------
# 数据库整理（用户第二版设计的循环）
# ---------------------------------------------------------------------

def consolidation_pass(records, sampler, max_extra_per_record=MAX_EXTRA_SAMPLES):
    """一次数据库整理（用户第二版设计）。

    records：[{"tag", "fit": BetaFit}] —— 会被原地更新（补采样）。
    sampler：callable(tag) -> 新样本值（**真跑一次 AlphaFold**，非抽随机数）；
             返回 None 表示无法生成（跳过该记录）。

    规则：
      1）按最优结果 obj_max 降序排名，N = 记录数；
      2）阈值：榜首 K=1 用 (max+1)/2（原设计不变）；其余 K>1 用**榜首的 max**（“能不能当第一”，而非“能不能爬升一名”，避免在排名靠后的序列浪费算力）；
         P = β 拟合的 P（下一次 AF3 采样 > 阈值）；
      3）循环条件 **P·N > M**，M = 该记录已采样次数（含序贯初值）。
         终止性：每采一次 M<-M+1；P<=1 => P·N<=N => M>N 必失效 => **有限步终止**，不依赖分布假设。补采后更新拟合与 M，因榜首可能变而重排，直到无人满足。
      4）**单点/退化记录爬山**（2026-10-21 规格）：fit() 无解（n=1、样本全同、全 0/全 1）⇒ 尾概率恒 0 ⇒ 规则 3 永远不给它补采，等于判了“永无翻身”。此类记录改走与首次爬山同规格的循环：先整批消费 pool 里已补到的样本，只要还在破纪录就继续经 sampler 补采（批量需求模式下登记需求、等阶段2 填池后下一轮消费）；一旦整批未破纪录即置 climb_done 收束。终止性：破纪录判定 + extra_used ≤ max_extra_per_record 工程上限。

    返回 {"added"：补采总数，"rounds"：迭代轮数，"capped"：因硬上限停止的次数}
    """
    added = 0
    rounds = 0
    capped = 0

    while True:
        rounds += 1
        recs = sorted(records,
                      key=lambda r: (r["fit"].max_x is not None,
                                     r["fit"].max_x or 0.0),
                      reverse=True)
        N = len(recs)
        if N == 0:
            break
        top_max = recs[0]["fit"].max_x
        if top_max is None:
            break
        progressed = False

        for K, rec in enumerate(recs, start=1):
            fit = rec["fit"]
            if K == 1:
                threshold = (top_max + 1.0) / 2.0   # 榜首：原设计不变
            else:
                threshold = top_max                 # 其余：超过榜首

            if fit.max_x is not None and threshold <= fit.max_x:
                continue                        # 已达阈值，无需补采

            # ---- 单点/退化记录（β 无解 → P 恒 0 → 规则 3 永不补采）：改走爬山 ----
            if fit.n >= 1 and fit.fit() is None and not fit.climb_done:
                before = fit.max_x
                took = 0
                while fit.pool:                 # 整批消费池内已补到的样本（同一批产物全计入拟合）
                    fit.draw_from_pool()
                    took += 1
                if took:
                    added += took
                    progressed = True
                    if fit.max_x <= before + 1e-12:
                        fit.climb_done = 1      # 整批都没破纪录 → 爬山收束
                        continue
                best = fit.max_x
                while True:                     # 池空：经 sampler 真补采，破纪录就继续
                    if fit.extra_used >= max_extra_per_record:
                        capped += 1
                        fit.climb_done = 1      # 工程保险（正常不触发）
                        break
                    x = sampler(rec["tag"])
                    if x is None:
                        break                   # 批量需求模式：需求已登记，等阶段2 填池后下一轮消费
                    fit.add(x)
                    fit.extra_used += 1
                    added += 1
                    progressed = True
                    if x > best + 1e-12:
                        best = x
                        continue
                    fit.climb_done = 1          # 不再破纪录 → 爬山收束
                    break
                continue

            P = fit.prob_gt(threshold)
            M = fit.n                           # 已采样次数
            if P * N <= M:
                continue                        # P·N <= M：收益不及已投入，停

            # --- 补采一个样本 ---
            if fit.extra_used >= max_extra_per_record:
                capped += 1
                continue                        # 工程保险（正常不触发）
            x = fit.draw_from_pool()
            if x is None:
                x = sampler(rec["tag"])
                if x is None:
                    continue
                fit.add(x)
                fit.extra_used += 1
            added += 1
            progressed = True                   # 本轮有更新 -> 下轮重排重判

        if not progressed:
            break

    return {"added": added, "rounds": rounds, "capped": capped}


def trigger_power_of_two(total_records):
    """记录数达到 2 的整数次幂时触发整理。"""
    return total_records > 0 and (total_records & (total_records - 1)) == 0
