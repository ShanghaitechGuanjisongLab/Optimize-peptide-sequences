"""
小肽序列优化 — **计算节点内**的无限循环作业入口（单作业版, 无需登录节点常驻）
=============================================================
为什么是这个形态（2026-10-04 定案, 用户提问纠偏后）：

  此前的形态是"登录节点常驻 Python 驱动 + 反复 sbatch 小包作业"。它有两个
  结构性缺陷:
    1) **登录节点进程脆弱**: 本机把所有交互进程塞进同一个 cgroup
       `/system.slice/sshd.service`, 且 `who` / `loginctl` 查不到会话 →
       `setsid` 只能换会话/进程组而**逃不出该 cgroup**。VS Code 会话被清理时
       驱动**无声消失**（无异常栈、无退出日志）。已发生两次, 其中一次造成
       GPU 空转 8h22m、29 个已算完的结果滞留磁盘未入库。
    2) **QoS 使编排作业与计算作业互斥**: `partition_bme_gpupub` 的
       `MaxJobsPerUser=1` 是**运行并发**上限, 而分区 QoS 未给 MaxWall
       → 单作业最长可跑 `MaxTime=5 天`, 且 `PreemptMode=OFF`（不会被抢占）。
       实测 `--test-only` 下 48h / 2天 / 3天 / 5天 的预计启动时间**完全相同**
       → 申请长时限**不受精调度/回填惩罚**（瓶颈是 QoS 运行槽, 不是节点空间）。
       因此把整个搜索放进**一个长驻作业**里循环, 就不必在计算节点 sbatch
       下一包, 也不必在登录节点留任何进程。

  形态对比:
    旧: 登录节点常驻驱动 → sbatch 48 任务包 → 每 7.6h 重新排队/被调度
    新: sbatch **一个** 5 天作业 → 作业内部循环 {挑候选 → 跑 AF3 → 收割入库}
        登录节点只留一个 cron 看门狗, 职责退化为"队列里没作业了就再交一个"
        （每 5 天才触发一次; 它只跑 squeue/sbatch, 几秒即退, 极难出错）

循环内部为何仍要"分批"跑 AF3（而不是一口气喂几千条）：
  贝叶斯优化需要**已完成的评估结果**来拟合 GP 并挑选下一批候选。若把整个
  候选池一次性丢给 AF3, 就退化成纯随机搜索、失去引导。故每轮（cycle）大小
  = `CYCLE_SEQS`（默认 24 序列 = 48 任务 ≈ 7.7h）, 轮与轮之间做 GP 反馈。
  这个尺寸同时是最优摊销点: chain B(靶标) 的 MSA 是**每进程**重建一次的
  固定开销 ≈ 55min（HTR1A 1738s + BIN1 1555s, 实测）, 而 MSA 缓存
  (`data/pipeline.py` 的 `functools.cache`) 作用域仅限单个 python 进程 →
  轮太大则 GP 反馈太慢, 轮太小则 55min 摊不薄。单序列成本 ≈ 3293/N + 1010s,
  N=24 时 1147s（比 N=3 的 2108s 快 46%, 距渐近线 1010s 仅差 14%）。

时间预算自适应：
  作业临近 5 天上限时, 最后一轮按剩余时间**反向定尺寸**（能塞几个序列就塞
  几个）, 把 GPU 用到极限而不是提前空停; 完成后正常退出, 由看门狗续交下一个
  5 天作业（间隙 ≤ 5min）。被 SLURM 掐断也只损失"当前正在算的 1 个任务"
  —— 已完成任务的结果早就逐个落在共享盘上, 下一个作业的收割阶段即入库。

候选从哪来（2026-10-04 第二次定案: 穷尽队列取代随机采样）：
  旧形态用 `CandidateStream` 随机采样 + "重试并排除已评估"取候选。它有两个
  结构性缺陷, 都在实测中被量化过:
    1) **候选池是抽样的, 不是全集**: 每周期只撒 ~720 个随机点做 GP 排序,
       argmax 是在这个子集上取的, 不是在整个搜索空间上取的;
    2) **随机采样有 665 条序列永远采不到**（`generate_mutant` 的模式B 只做
       单次插入或删除, 故"净长度不变的插入+删除"两步编辑生成不出来）。
  现改为 `ExhaustiveQueue`: 候选全集 = `peptide_common.enumerate_space()`
  精确枚举的 51091 条（编辑距离 ≤2、长 6~10、20 种标准氨基酸）, 每周期从
  「全集 − 已派发台账」里按 GP 的 cEI 次序取 24 条。由此得到三条**可证明**的
  性质（详见该类文档）: 不重复、每周期待办精确定量递减、至多 2129 个周期
  ≈ 680 天必然全覆盖。GP 只决定"先跑谁", 对覆盖保证零贡献 —— 即便 GP 完全
  失效, 次序退化为字典序游标, 上述性质依然成立。
  代价护栏: 全空间打分实测仅 ≈1s（故 SCORE_POOL_CAP 默认 0 = 不限制）; 真正
  的瓶颈是 GP 拟合的 O(n_train^3)（4000 条已 52s, 外推 5 万条 ≈9.6h > 周期
  预算）, 故 TRAIN_CAP=2000 做确定性分层子采样, best_y 仍取全量最优。

优雅重启（换代码用）：
  正在运行的作业已把 .py 载入内存, **改源文件对它无效**。要把新代码投出去,
  在 `<db_root>/runs/` 下放一个名为 `restart.requested` 的空文件
  （`run.sh restart-loop`）即可: 作业会在**下一个周期起点**正常退出, 由 cron
  看门狗 ≤5min 内续交新作业。已完成周期的结果早已收割入库, 因此不丢数据,
  唯一代价是新作业首轮重建 chain-B MSA（实测 ≈55min）。标志会被自动消费,
  不会让续交的作业又立刻退出。

运行方式（都由 run.sh 封装, 手动亦可）：
    sbatch <resource flags> --export=ALL,PEPOPT_MAX_HOURS=118,... \\
        scripts/af3_loop_job.sh
    # 登录节点直接调试（会占用 QoS 运行槽, 仅在队列空时用）：
    PEPOPT_MAX_HOURS=0.05 python optimize_peptide_node.py --targets HTR1A,BIN1

依赖的环境分工（实测）：
    conda base   : sklearn / pandas / matplotlib（编排、GP、入库）
    conda af3_old: jax 0.4.34 / alphafold3 / jackhmmer（AF3 推理）—— **无**
                   sklearn、pandas, 故两套环境必须并存; 本脚本在 base 下运行,
                   并把 AF3 那一步整体委托给 scripts/af3_bundle_runner.sh
                   （它自己 conda activate af3_old）。这样两环境互不污染,
                   也复用了 runner 里已验证过的两阶段容错与输出折叠逻辑。
"""

import os
import re
import sys
import json
import time
import math
import signal
import shutil
import argparse
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ---- 必须先设结果目录再 import loc（该模块 import 时读取此环境变量）----
import peptide_db as db
os.environ.setdefault("PEPOPT_RESULTS_DIR", db.results_dir())

import optimize_peptide_local as loc
loc.LOCAL_RESULTS_DIR = db.results_dir()          # 防御: 强制对齐到共享库

import optimize_peptide_infinite as inf          # 复用候选流/启发式挑选/收割报告

# ===================== 配置 =====================

DEFAULT_TARGETS   = "HTR1A,BIN1"       # V100 现状: UNC13C(≈2233 tokens) 超限搁置
TARGET_ORDER      = ["HTR1A", "UNC13C", "BIN1"]
CYCLE_SEQS        = 24                 # 每轮序列数（×靶标数 = 每轮任务数）
CYCLE_FIXED_SEC   = 3293.0             # 每轮固定开销: chain B MSA（实测 1738+1555）
CYCLE_PER_SEQ_SEC = 1010.0             # 每序列边际成本: chain A MSA 670s + 推理 ~340s
SAFETY_MARGIN_SEC = 1200               # 距 SLURM 时限的安全余量（20min）
MAX_HOURS_DEFAULT = 118.0              # 5 天(120h)上限减去 2h 余量
EMPTY_CYCLE_LIMIT = 3                  # 连续 N 轮零产出即退出（避免空烧 5 天）
CYCLE_TAG_FMT     = "cycle_%Y%m%d_%H%M%S"
# 优雅重启标志: 在 <db_root>/runs/ 下放同名空文件, 作业会在**下一个周期起点**
# 正常退出(而非被 scancel 打断), 看门狗 ≤5min 内续交新作业 → 新代码生效, 且
# 不浪费已完成周期的计算。代价仅新作业首轮重建 chain-B MSA(实测 ~55min)。
# 为何不能靠改代码生效: 正在跑的作业已把 .py 载入内存, 编辑源文件对它无效。
RESTART_FLAG      = "restart.requested"

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n🛑 收到信号 {signum}: 完成当前轮后退出（已完成结果都在共享盘上, "
          "下个作业会收割）", flush=True)


def cycle_seconds(n_seq):
    """估计一轮（n_seq 个序列 × 激活靶标数）的墙钟秒数。"""
    # 任务数 = 序列数 × 靶标数, 但 chain B MSA 每靶标只建一次（进程内缓存）,
    # 故固定开销与靶标数成正比, 边际成本与任务数成正比。
    return CYCLE_FIXED_SEC + CYCLE_PER_SEQ_SEC * n_seq


def size_next_cycle(remaining_sec, max_seqs):
    """按剩余时间反推这一轮该放几个序列（尽量把 GPU 用满, 又不被时限掐断）。"""
    budget = remaining_sec - SAFETY_MARGIN_SEC
    if budget <= CYCLE_FIXED_SEC:
        return 0                              # 连一轮固定开销都放不下 → 收工
    n = int(math.floor((budget - CYCLE_FIXED_SEC) / CYCLE_PER_SEQ_SEC))
    return max(1, min(n, max_seqs))


# ===================== AF3 执行（委托给已验证的 bundle runner）=====================

def build_runner_env(bundle_dir):
    """组装 scripts/af3_bundle_runner.sh 需要的全部 AF3_* 环境变量。
    与 loc.build_bundle_sbatch_cmd() 的 --export 列表保持一致, 差别仅在于:
    这里是**进程内 subprocess 继承**而非 sbatch --export。"""
    env = os.environ.copy()
    env.update({
        "AF3_CODE_DIR": loc.AF3_CODE_DIR,
        "AF3_BUNDLE_DIR": os.path.abspath(bundle_dir),
        "AF3_OUT_DIR": os.path.abspath(db.results_dir()),
        "AF3_MODEL_DIR": loc.LOCAL_MODEL_DIR,
        "AF3_DB_DIR": loc.LOCAL_DB_DIR,
        "AF3_WORK_DIR": _ROOT,
        "AF3_CONDA_ENV": loc.CONDA_ENV,
        "AF3_FLASH_ATTN": loc.FLASH_ATTENTION_IMPL,
        "AF3_NEED_XLA_7X_FLAG": "1" if loc.NEED_XLA_7X_FLAG else "0",
        "AF3_MSA_CPUS": str(loc.MSA_CPUS_PER_GPU),
        "AF3_GPUS_PER_JOB": str(loc.LOCAL_GPUS_PER_JOB),
    })
    if loc.LOCAL_HMMER_DIR:
        env["AF3_HMMER_DIR"] = loc.LOCAL_HMMER_DIR
    return env


def run_af3_cycle(bundle_dir):
    """在**本作业内**直接跑 AF3（不再 sbatch）; 返回 runner 退出码。

    runner 先复制成**快照**再执行: bash 按字节偏移边读边执行脚本, 而编辑器
    写 scripts/*.sh 是**原地截断重写**（inode 不变）→ 若本长驻作业执行期间
    有人修改了仓库里的 runner, bash 回读到下一条命令时就会错位（语法崩
    或行为失控; 2026-10-04 实测踩到, 靠临时回滚才救回在跑的 5 天作业）。
    快照使长驻作业与仓库后续改动彻底解耦。

    runner 自带两阶段容错（整包失败时逐任务补跑）与时间戳输出折叠, 故这里
    只需容忍非零退出码：已完成的任务结果都在盘上, 交给收割阶段入库。"""
    src = os.path.join(_ROOT, "scripts", "af3_bundle_runner.sh")
    if not os.path.isfile(src):
        print(f"  ❌ 缺 runner: {src}", flush=True)
        return 2
    snap = os.path.join(os.path.abspath(bundle_dir), "_runner_snapshot.sh")
    try:
        shutil.copyfile(src, snap)
        os.chmod(snap, 0o755)
    except OSError as e:
        print(f"  ⚠️ 快照失败({e}), 回退直接执行仓库副本", flush=True)
        snap = src
    print("  🚀 启动 AF3 推理（作业内直接执行, 不再 sbatch）", flush=True)
    try:
        p = subprocess.run(["bash", snap], env=build_runner_env(bundle_dir))
        return p.returncode
    except Exception as e:                       # noqa: BLE001 - 不能让作业裸崩
        print(f"  ❌ runner 执行异常: {e}", flush=True)
        return 2


# ===================== 组装一轮 =====================

def prepare_cycle(seqs, active_targets, cycle_dir):
    """为本轮挑中的序列写 AF3 输入 JSON（同时落地两处）:
      1) 规范结果目录 <af3_results>/<tag>_<target>/input.json
         —— 收割/折叠/BEST 打包都以它为唯一位置（见 peptide_db.consolidate_task）;
      2) 本轮工作目录 <cycle_dir>/<job_name>.json —— runner 以 --input_dir 读它。
    返回 (已排入的任务数, 跳过的原因列表)。token 超限的任务在提交前拦截。"""
    os.makedirs(cycle_dir, exist_ok=True)
    n_task, skipped = 0, []
    for seq in seqs:
        var = seq[len(inf.FIXED_A):]
        # resolve_tag 而非 make_tag：已登记过的序列必须沿用当初的标签（旧的 5 位
        # 十进制），否则扩位后的新标签会把磁盘上已有的 34 条结果当成新任务重算。
        tag = db.resolve_tag(var)
        try:
            db.register_sequence(tag, var, seq)
        except RuntimeError as e:                # 撞标签: 跳过该序列, 不中断搜索
            skipped.append(f"{tag}: {e}")
            continue
        for t in active_targets:
            job_name = f"{tag}_{t.lower()}"
            canon = db.canonical_dir(tag, t)
            # consolidate_task 会先把 AF3 的时间戳兄弟目录折叠进规范目录再判定,
            # 比只查规范目录更可靠（漏判会导致已算完的任务被重算）。
            if db.consolidate_task(tag, t) is not None:
                continue                          # 已有结果 → 复用, 不重算
            os.makedirs(canon, exist_ok=True)
            inp = loc.write_input_json(seq, inf.TARGET_SEQS[t], job_name, canon)
            tk = loc.estimate_tokens(inp)
            if tk > loc.GPU_MAX_TOKENS:
                skipped.append(f"{job_name}: {tk} tokens > {loc.GPU_MAX_TOKENS}")
                continue
            shutil.copyfile(inp, os.path.join(cycle_dir, f"{job_name}.json"))
            n_task += 1
    return n_task, skipped


# ===================== 主循环 =====================

def drain_backlog(active_targets, cycle_dir, seq_budget):
    """把**历史遗留的未完成任务**排入本轮（防孤儿化）。返回 (序列数, 任务数)。

    为何必需: 旧形态（登录节点驱动 + 反复 sbatch 小包）被取消或驱动被杀时,
    会留下"已有 input.json、但永远不会有 summary"的任务目录。而新入口的候选
    来自 `inf.known_sequences()`（= sequences.csv 全量）的排除集 → 这些序列
    **再也不会被选中**, 于是永久缺半边结果: 既拖累"完整评估"计数, 也让 GP
    的 BIN1 约束少一批样本。本函数直接复用它们已生成的 input.json 把缺的补上。

    按 tag 分组计数, 每轮最多补 seq_budget 个序列, 以免一次塞爆时间预算。"""
    rd = db.results_dir()
    if not os.path.isdir(rd):
        return 0, 0

    pending = {}                       # tag -> [(任务名, input.json 路径)]

    # ---- 第1类: 已有 input.json 但永远等不到 summary 的遗留任务 ----
    for name in sorted(os.listdir(rd)):
        d = os.path.join(rd, name)
        if name.startswith("_") or name.endswith(".zip") or not os.path.isdir(d):
            continue
        parsed = db.split_job_name(name)
        if parsed is None or parsed[1] not in active_targets:
            continue
        if db.consolidate_task(*parsed) is not None:
            continue                              # 已完成（含时间戳兄弟折叠）
        inp = os.path.join(d, "input.json")
        if not os.path.isfile(inp):
            continue
        pending.setdefault(parsed[0], []).append((name, inp))
    n_leftover = sum(len(v) for v in pending.values())

    # ---- 第2类(gap-fill): 已注册序列缺某个激活靶标, 且连 input.json 都没有 ----
    # 来源: 旧形态被取消、或（如 2026-10-04）靶标参数被 SLURM 截断只跑了 HTR1A,
    # 于是库里出现"有 HTR1A 无 BIN1"的半行; 这类序列已在 sequences.csv 中,
    # 因而**永远不会被重新选中** → 不补就永久缺半边。此处按主表已有值快速预筛,
    # 避免对已完整的序列做无谓的目录探测。
    reg, rows = db.load_sequence_registry(), db.load_database()
    n_gap = 0
    for tag in sorted(reg):
        info = reg[tag]
        pep = info.get("peptide")
        row = rows.get(tag, {})
        if not pep:
            continue
        has = {nm for nm, _ in pending.get(tag, [])}
        for t in active_targets:
            if row.get(f"iptm_{t.lower()}") is not None:
                continue                          # 主表已有值 → 不缺
            name = f"{tag}_{t.lower()}"
            if name in has:
                continue
            if db.consolidate_task(tag, t) is not None:
                continue                          # 盘上已完成, 下次收割即入库
            canon = db.canonical_dir(tag, t)
            try:
                os.makedirs(canon, exist_ok=True)
                inp = loc.write_input_json(pep, inf.TARGET_SEQS[t], name, canon)
            except OSError as e:
                print(f"  ⚠️ gap-fill 写输入失败 {name}: {e}", flush=True)
                continue
            if loc.estimate_tokens(inp) > loc.GPU_MAX_TOKENS:
                continue                          # 注定 OOM（如 UNC13C 在 V100）
            pending.setdefault(tag, []).append((name, inp))
            n_gap += 1

    n_seq, n_task = 0, 0
    for tag in list(pending)[:max(0, seq_budget)]:
        n_seq += 1
        for name, inp in pending[tag]:
            try:
                shutil.copyfile(inp, os.path.join(cycle_dir, f"{name}.json"))
                n_task += 1
            except OSError:
                pass
    left_seq = len(pending) - n_seq
    print(f"  🧹 补跑遗留/缺失: {n_seq} 序列 / {n_task} 任务"
          f"（遗留任务 {n_leftover} + 缺失靶标补齐 {n_gap}）"
          + (f"；尚余 {left_seq} 序列待后续轮次" if left_seq > 0 else ""),
          flush=True)
    return n_seq, n_task


def _restart_flag_path(p):
    return os.path.join(p["root"], "runs", RESTART_FLAG)


def restart_requested(p):
    """是否有人在请求周期边界重启。"""
    try:
        return os.path.isfile(_restart_flag_path(p))
    except OSError:
        return False


def consume_restart_flag(p):
    """取走重启标志(幂等): 避免续交的新作业又立即退出。"""
    try:
        os.remove(_restart_flag_path(p))
        return True
    except OSError:
        return False


def run_loop(active_targets, cycle_seqs, max_hours):
    p = db.paths(ensure=True)
    start = time.time()
    deadline = start + max_hours * 3600.0
    stream = inf.CandidateStream(seed=42)      # 仅 SWEEP_ENABLED=0 时使用
    empty_cycles = 0
    cycle_no = 0

    print("=" * 68, flush=True)
    print(f"♾️  无限搜索**在计算节点作业内**运行（无需登录节点常驻进程）", flush=True)
    print(f"   靶标      : {active_targets}（搁置: "
          f"{[t for t in TARGET_ORDER if t not in active_targets]}）", flush=True)
    print(f"   调度      : " + ("穷尽队列 enumerate_space (不重复 + 有限步全覆盖)"
                               if inf.SWEEP_ENABLED else
                               "随机采样启发式 (CandidateStream)"), flush=True)
    if inf.SWEEP_ENABLED:
        print("   覆盖起点  : " + inf.coverage_report(active_targets), flush=True)
    print(f"   每轮      : ≤{cycle_seqs} 序列 × {len(active_targets)} 靶标 "
          f"≈ {cycle_seconds(cycle_seqs) / 3600:.1f}h", flush=True)
    print(f"   时间预算  : {max_hours:.1f}h（到期自动退出, 由看门狗续交下一作业）",
          flush=True)
    print(f"   共享数据库: {p['root']}", flush=True)
    print(f"   SLURM 作业: {os.environ.get('SLURM_JOB_ID', '(非作业环境)')} @ "
          f"{os.environ.get('SLURMD_NODENAME', '(登录节点)')}", flush=True)
    print("=" * 68, flush=True)

    while not _STOP:
        remaining = deadline - time.time()
        if remaining <= SAFETY_MARGIN_SEC:
            print(f"\n⏰ 时间预算用尽（剩 {remaining / 60:.0f}min < 安全余量 "
                  f"{SAFETY_MARGIN_SEC / 60:.0f}min）, 正常退出。", flush=True)
            break

        # 周期边界优雅重启: 已完成周期的结果已在上一轮 step6 收割入库,
        # 此处退出不会丢数据; 标志必须先消费再 break, 否则新作业会立即又退出。
        if restart_requested(p):
            consume_restart_flag(p)
            print(f"\n🔁 检测到重启请求 → 在周期边界优雅退出（已完成 {cycle_no} 个周期）; "
                  f"看门狗 ≤5min 内续交新作业以加载新代码。", flush=True)
            break

        cycle_no += 1
        n_seq = size_next_cycle(remaining, cycle_seqs)
        if n_seq < 1:
            print("\n⏰ 剩余时间不足一轮, 退出。", flush=True)
            break

        elapsed_h = (time.time() - start) / 3600.0
        print(f"\n== [周期 {cycle_no}] {time.strftime('%F %T')} | "
              f"已用 {elapsed_h:.1f}h / {max_hours:.0f}h | "
              f"剩 {remaining / 3600:.1f}h → 本轮 {n_seq} 序列 "
              f"(≈{(cycle_seconds(n_seq)) / 3600:.1f}h)", flush=True)

        # 1) 先收割上一个周期的成果（首轮则收割历史遗留/其它入口的完成结果）
        n_new = inf.harvest_and_report(active_targets)

        # 2) 本轮工作目录（补跑遗留与新候选共用）
        cycle_dir = os.path.join(p["root"], "cycles",
                                 time.strftime(CYCLE_TAG_FMT))
        os.makedirs(cycle_dir, exist_ok=True)

        # 3) 先补上次遗留的未完成任务（防孤儿化）, 剩余预算再挑新候选
        d_seq, d_task = drain_backlog(active_targets, cycle_dir, n_seq)
        picks, space_done = [], False
        want = max(0, n_seq - d_seq)
        if want > 0:
            if inf.SWEEP_ENABLED:
                # 穷尽队列: 从「精确全集(51091) - 已派发台账(sequences.csv)」中
                # 按 GP 次序取 want 条 → 保证不重复, 且有限周期内必然全覆盖。
                # 返回空列表 = 空间已穷尽（**正常完成**, 不是采样失败）。
                picks = inf.sweep_pick(active_targets, want)
                if not picks:
                    space_done = True
                    print("  🏁 待评估集合为空 — 搜索空间已全覆盖, 不再有新序列。",
                          flush=True)
            else:
                picks = inf.heuristic_pick(stream, want, active_targets)
        if inf.SWEEP_ENABLED:
            print("  " + inf.coverage_report(active_targets,
                                            cycle_secs=cycle_seconds(n_seq),
                                            per_cycle=max(want, 1)), flush=True)
        if picks:
            print(f"  🎯 选中 {len(picks)} 个新序列: "
                  f"{[db.resolve_tag(s[len(inf.FIXED_A):]) for s in picks[:8]]}"
                  + (" ..." if len(picks) > 8 else ""),
                  flush=True)

        # 4) 组装本轮 AF3 输入（新候选; 遗留任务已在 drain 阶段复制进 cycle_dir）
        n_task, skipped = (0, [])
        if picks:
            n_task, skipped = prepare_cycle(picks, active_targets, cycle_dir)
        n_task += d_task
        if skipped[:6]:
            for s in skipped[:6]:
                print(f"     ⏭️  跳过 {s}", flush=True)
        if n_task == 0:
            if space_done:
                # 与"连续空轮"区分开: 这里是穷尽搜索的**正常终点**, 不该再
                # 耗掉 EMPTY_CYCLE_LIMIT 轮, 也不该被看门狗误判为异常退出。
                print("  🏁 空间已全覆盖且无遗留任务 → 搜索完成, 优雅退出。",
                      flush=True)
                break
            print("  ⚠️ 本轮无待算任务（新候选与历史遗留都已有结果）, 计入空轮",
                  flush=True)
            empty_cycles += 1
            if empty_cycles >= EMPTY_CYCLE_LIMIT:
                print(f"  🛑 连续 {empty_cycles} 轮零产出, 退出。", flush=True)
                break
            continue
        print(f"  📝 已排入 {n_task} 个任务 → {os.path.basename(cycle_dir)}",
              flush=True)
        empty_cycles = 0

        # 5) 作业内直接跑 AF3（阻塞至本周期结束）
        rc = run_af3_cycle(cycle_dir)
        status = "✅" if rc == 0 else f"⚠️ 退出码 {rc}"
        print(f"  {status} 周期 {cycle_no} 的 AF3 结束 "
              f"({(time.time() - start) / 3600:.1f}h)", flush=True)

        # 6) 立刻收割本周期的结果, 使下一轮的 GP 能用上最新数据
        got = inf.harvest_and_report(active_targets)
        if got == 0 and n_new == 0:
            empty_cycles += 1
        else:
            empty_cycles = 0

        # 轮间歇: 无实质作用, 仅让共享文件系统的元数据落盘/日志可读
        time.sleep(5)

    print(f"\n👋 作业内无限循环结束（共 {cycle_no} 个周期）", flush=True)
    inf.harvest_and_report(active_targets)
    print(f"   数据库: {p['root']}", flush=True)
    print("   本作业退出后, 登录节点的 cron 看门狗会在 ≤5min 内续交下一个作业。",
          flush=True)


# ===================== 入口 =====================

def parse_targets(raw):
    """靶标字符串 → 集合。**同时**用 ',' 与 '+' 作为分隔符。

    为何要支持 '+': SLURM 的 `--export=a,b,c` 列表本身以**逗号**分隔,
    而 `PEPOPT_TARGETS=HTR1A,BIN1` 的值内部含逗号 → 会被 SLURM 截断成
    `PEPOPT_TARGETS=HTR1A` 加一个孤立的 `BIN1`, 靶标静默退化为单靶标
    （2026-10-04 实测踩到: 已跑起的 5 天作业只算 HTR1A, BIN1 约束全丢）。
    故下发时用 '+' 作分隔符; 人手输入（CLI/README）两种都接受。"""
    return {t.strip().upper() for t in re.split(r"[,+]", str(raw)) if t.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="计算节点内的无限搜索循环作业（结果持续追加进共享数据库）")
    ap.add_argument("--targets", default=os.environ.get("PEPOPT_TARGETS")
                    or DEFAULT_TARGETS,
                    help="靶标列表, 用 ',' 或 '+' 分隔, 必须含 HTR1A（默认 HTR1A,BIN1; "
                         "UNC13C 需 A100/H100 解锁后加进来）。注意经 SLURM --export "
                         "下发时只能用 '+', 逗号会被截断（见 parse_targets）")
    ap.add_argument("--cycle-seqs", type=int,
                    default=int(os.environ.get("PEPOPT_CYCLE_SEQS") or CYCLE_SEQS),
                    help=f"每轮最多序列数（默认 {CYCLE_SEQS}）")
    ap.add_argument("--max-hours", type=float,
                    default=float(os.environ.get("PEPOPT_MAX_HOURS")
                                  or MAX_HOURS_DEFAULT),
                    help=f"时间预算小时数（默认 {MAX_HOURS_DEFAULT} = 5 天上限减余量）")
    args = ap.parse_args()

    want = parse_targets(args.targets)
    if want - set(TARGET_ORDER):
        ap.error(f"--targets 含未知靶标 {sorted(want - set(TARGET_ORDER))}, "
                 f"可选 {TARGET_ORDER}")
    if "HTR1A" not in want:
        ap.error("--targets 必须包含 HTR1A（优化目标）")
    active = [t for t in TARGET_ORDER if t in want]

    # 硬守卫: 只跑 HTR1A 而没有 BIN1, 等于**丢掉底线约束**（本项目目标是
    # HTR1A↑ 且 BIN1<0.8）, 产出的序列无法筛选 → 白烧几天机时。
    # 历史上这个退化是由 PEPOPT_TARGETS 逗号截断造成的, 故在此拦住。
    # 确实只想跑单靶标（如调试）时, 显式 PEPOPT_ALLOW_SINGLE_TARGET=1。
    if len(active) == 1 and os.environ.get("PEPOPT_ALLOW_SINGLE_TARGET") != "1":
        ap.error(
            f"靶标只有 {active}: BIN1 约束丢失, 产出无法用于筛选, 已拒绝启动以免"
            "白烧机时。\n"
            f"  原始参数: {args.targets!r}\n"
            "  常见原因: 经 SLURM `--export` 下发时用了逗号 → 被 SLURM 按逗号截断"
            "（--export 列表自身以逗号分隔）。\n"
            "  修正: 下发时用 '+' 分隔, 如 PEPOPT_TARGETS=HTR1A+BIN1（"
            "submit_loop_job() 已如此生成）。\n"
            "  如确实只想跑单靶标, 显式设 PEPOPT_ALLOW_SINGLE_TARGET=1。")

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    # 就绪检查: 数据库/权重/hmmer（与登录节点入口一致; 计算节点上通常都可用）
    ok, _ = loc.ensure_local_ready()
    if not ok:
        print("\n⚠️  预检查未通过, 仍尝试运行（已完成结果可复用）\n", flush=True)

    run_loop(active, args.cycle_seqs, args.max_hours)


if __name__ == "__main__":
    main()
