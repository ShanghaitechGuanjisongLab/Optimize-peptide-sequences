"""
小肽序列优化 — **计算节点内**的无限循环作业入口（单作业版，无需登录节点常驻）
=============================================================
为什么是这个形态（2026-10-04 定案，用户提问纠偏后）：

  此前的形态是“登录节点常驻 Python 驱动 + 反复 sbatch 小包作业”。它有两个结构性缺陷：
    1）**登录节点进程脆弱**：本机把所有交互进程塞进同一个 cgroup `/system.slice/sshd.service`，且 `who` / `loginctl` 查不到会话 →`setsid`只能换会话/进程组而**逃不出该 cgroup**。VS Code 会话被清理时驱动**无声消失**（无异常栈、无退出日志）。已发生两次，其中一次造成GPU空转 8h22m、29 个已算完的结果滞留磁盘未入库。
    2）**QoS 使编排作业与计算作业互斥**：`partition_bme_gpupub` 的`MaxJobsPerUser=1`是**运行并发**上限，而分区 QoS 未给 MaxWall→单作业最长可跑 `MaxTime=5 天`，且 `PreemptMode=OFF`（不会被抢占）。实测 `--test-only` 下 48h / 2天 / 3天 / 5天 的预计启动时间**完全相同**→申请长时限**不受精调度/回填惩罚**（瓶颈是 QoS 运行槽，不是节点空间）。因此把整个搜索放进**一个长驻作业**里循环，就不必在计算节点 sbatch下一包，也不必在登录节点留任何进程。

  形态对比：
    旧：登录节点常驻驱动 → sbatch 48 任务包 → 每 7.6h 重新排队/被调度新：sbatch **一个** 5 天作业 → 作业内部循环 {挑候选 → 跑 AF3 → 收割入库} 登录节点只留一个 cron 看门狗，职责退化为“队列里没作业了就再交一个”（每 5 天才触发一次；它只跑 squeue/sbatch，几秒即退，极难出错）

循环内部为何仍要“分批”跑 AF3（而不是一口气喂几千条）：
  贝叶斯优化需要**已完成的评估结果**来拟合 GP 并挑选下一批候选。若把整个候选池一次性丢给 AF3，就退化成纯随机搜索、失去引导。故每轮（cycle）大小=`CYCLE_SEQS`（默认 24 序列 = 48 任务 ≈ 7.7h），轮与轮之间做 GP 反馈。
  这个尺寸同时是最优摊销点：chain B（靶标）的 MSA 是**每进程**重建一次的固定开销 ≈ 55min（HTR1A 1738s + BIN1 1555s，实测），而 MSA 缓存（`data/pipeline.py` 的 `functools.cache`）作用域仅限单个 python 进程→轮太大则 GP 反馈太慢，轮太小则 55min 摊不薄。单序列成本 ≈ 3293/N + 1010s，N=24时 1147s（比 N=3 的 2108s 快 46%，距渐近线 1010s 仅差 14%）。

时间预算自适应：
  作业临近 5 天上限时，最后一轮按剩余时间**反向定尺寸**（能塞几个序列就塞几个），把 GPU 用到极限而不是提前空停；完成后正常退出，由看门狗续交下一个5天作业（间隙 ≤ 5min）。被 SLURM 掐断也只损失“当前正在算的 1 个任务”——已完成任务的结果早就逐个落在共享盘上，下一个作业的收割阶段即入库。

候选从哪来（2026-10-04 第二次定案：穷尽队列取代随机采样）：
  旧形态用 `CandidateStream` 随机采样 + “重试并排除已评估”取候选。它有两个结构性缺陷，都在实测中被量化过：
    1）**候选池是抽样的，不是全集**：每周期只撒 ~720 个随机点做 GP 排序，argmax是在这个子集上取的，不是在整个搜索空间上取的；
    2）**随机采样有 665 条序列永远采不到**（`generate_mutant` 的模式B 只做单次插入或删除，故“净长度不变的插入+删除”两步编辑生成不出来）。
  现改为 `ExhaustiveQueue`：候选全集 = `小肽公共库.enumerate_space()`精确枚举的 51091 条（编辑距离 ≤2、长 6~10、20 种标准氨基酸），每周期从「全集 − 已派发台账」里按 GP 的 EI(y = 1 − 目标乘积) 次序取 24 条（其中 ~25% 是从待办集无放回随机抽的探索配额，见 小肽优化_无限._sample_pool）。由此得到三条**可证明**的性质（详见该类文档）：不重复、每周期待办精确定量递减、至多 2129 个周期≈680 天必然全覆盖（三条都只依赖“枚举 + 台账”，与取批是随机还是有序无关）。GP 只决定“先跑谁”，对覆盖保证零贡献 —— 即便 GP 完全失效，探索配额照样推进，上述性质依然成立。
  代价护栏：全空间打分实测仅 ≈1s（故 SCORE_POOL_CAP 默认 0 = 不限制）；真正的瓶颈是 GP 拟合的 O(n_train^3)（4000 条已 52s，外推 5 万条 ≈9.6h > 周期预算），故 TRAIN_CAP=2000 做确定性分层子采样，best_y 仍取全量最优。

优雅重启（换代码用，两条路径都不释放计算节点）：
  正在运行的作业已把 .py 载入内存，**改源文件对它无效**。要把新代码投出去：
    · 立即换：`任务中间层/任务控制.sh restart-service` → 守护杀掉本进程并重新拉起（计算跑在 HQ worker 里，杀编排者不丢在跑的预测；未完成部分由 drain_backlog 接管）；
    · 周期边界换：在 `<db_root>/runs/` 下放 `restart.requested` 空文件（`运行.sh restart-loop`），本进程会在下一个周期起点以**重载码 42**（EXIT_RELOAD）退出，守护据此重启它。
  两者都不会让 SLURM 作业退出，因此不必重新排队（分区满载时重排可能要等数小时）。已完成周期的结果早已收割入库，故不丢数据；标志会被自动消费，不会让重启后的进程又立刻退出。

运行方式（都由 运行.sh 封装，手动亦可）：
    sbatch <resource flags> --export=ALL,PEPOPT_MAX_HOURS=118……\\ 作业脚本/AF3循环作业.sh
    # 登录节点直接调试（会占用 QoS 运行槽，仅在队列空时用）：
    PEPOPT_MAX_HOURS=0.05 python 小肽优化_节点内.py --targets HTR1A,BIN1

依赖的环境分工（实测）：
    conda base：sklearn / pandas / matplotlib（编排、GP、入库）
    conda af3_old：jax 0.4.34 / alphafold3 / jackhmmer（AF3 推理）—— **无**sklearn、pandas，故两套环境必须并存；本脚本在 base 下运行，并把 AF3 那一步整体委托给作业脚本/AF3打包推理.sh（它自己 conda activate af3_old）。这样两环境互不污染，也复用了 runner 里已验证过的两阶段容错与输出折叠逻辑。
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
import 小肽数据库 as db
os.environ.setdefault("PEPOPT_RESULTS_DIR", db.results_dir())

import 小肽优化_本地集群 as loc
loc.LOCAL_RESULTS_DIR = db.results_dir()          # 防御：强制对齐到共享库

import 小肽优化_无限 as inf          # 复用候选流/启发式挑选/收割报告
import 小肽序贯采样 as seq_sample        # 序贯采样×目标乘积×β拟合入库与数据库整理
from 小肽公共库 import create_af3_json     # 补采任务的 AF3 输入构造

# ===================== 配置 =====================

# 四靶标使用同一套模板参数；默认数据库与旧二因子战役隔离。
TARGET_ORDER      = inf.TARGET_ORDER
DEFAULT_TARGETS   = ",".join(db.PRODUCT_TARGETS)
CYCLE_SEQS        = 8                  # 每轮序列数（×4 = 每轮最多 32 个任务）A100四靶标保守预算；前三靶标有旧实测，UNC13C 先取保守值，后续由日志标定。
TARGET_MSA_SEC    = {"ASIC1A": 1410.0, "HTR1A": 1738.0,
                     "BIN1": 1555.0, "UNC13C": 3000.0}
TARGET_INFER_SEC  = {"ASIC1A": 1800.0, "HTR1A": 93.0,
                     "BIN1": 230.0, "UNC13C": 1200.0}
PEPTIDE_MSA_SEC   = 708.0
SAFETY_MARGIN_SEC = 1200               # 距 SLURM 时限的安全余量（20min）
MAX_HOURS_DEFAULT = 118.0              # 5 天(120h)上限减去 2h 余量
EMPTY_CYCLE_LIMIT = 3                  # 连续 N 轮零产出即退出（避免空烧 5 天）
CYCLE_TAG_FMT     = "cycle_%Y%m%d_%H%M%S"
# 优雅重启标志：在 <db_root>/runs/ 下放同名空文件，作业会在**下一个周期起点**
# 正常退出（而非被 scancel 打断），看门狗 ≤5min 内续交新作业 → 新代码生效，且
# 不浪费已完成周期的计算。代价仅新作业首轮重建 chain-B MSA（实测 ~55min）。
# 为何不能靠改代码生效：正在跑的作业已把 .py 载入内存，编辑源文件对它无效。
RESTART_FLAG      = "restart.requested"
# 与 任务守护.sh 的 --reload-code 默认值一致：本进程以此码退出时，守护会重启它而不是结束整个 SLURM 作业 → 换代码不必放弃已握到手的计算节点。
EXIT_RELOAD       = 42

_STOP = False


def _handle_stop(signum, frame):
    global _STOP
    _STOP = True
    print(f"\n🛑 收到信号 {signum}：完成当前轮后退出（已完成结果都在共享盘上，"
          "下个作业会收割）", flush=True)


def cycle_seconds(n_seq, active_targets=None):
    """按实际靶标估计墙钟；同靶标 MSA 固定开销每轮只付一次。"""
    targets = active_targets or inf.PRODUCT_TARGETS
    fixed = sum(TARGET_MSA_SEC[t] for t in targets)
    per_seq = PEPTIDE_MSA_SEC + sum(TARGET_INFER_SEC[t] for t in targets)
    return fixed + per_seq * n_seq


def size_next_cycle(remaining_sec, max_seqs, active_targets=None):
    """按剩余时间反推本轮序列数。"""
    targets = active_targets or inf.PRODUCT_TARGETS
    budget = remaining_sec - SAFETY_MARGIN_SEC
    fixed = sum(TARGET_MSA_SEC[t] for t in targets)
    per_seq = PEPTIDE_MSA_SEC + sum(TARGET_INFER_SEC[t] for t in targets)
    if budget <= fixed:
        return 0
    n = int(math.floor((budget - fixed) / per_seq))
    return max(1, min(n, max_seqs))


# ===================== AF3 执行（委托给已验证的 bundle runner）=====================

def build_runner_env(bundle_dir, out_dir=None):
    """组装 作业脚本/AF3打包推理.sh 需要的全部 AF3_* 环境变量。
    与 loc.build_bundle_sbatch_cmd() 的 --export 列表保持一致，差别仅在于：这里是**进程内 subprocess 继承**而非 sbatch --export。out_dir 覆盖输出根（补采任务写到 _resamples/ 下，与主结果隔离，收割路径自动忽略）。

    两个必须排除的变量：
      BASH_FUNC_* —— bash 函数导出（如 module()），写进生成脚本的 export 语句会变成 `export BASH_FUNC_module()='() {...}'` → 语法错误 → 任务全 FAILED。
      CUDA_VISIBLE_DEVICES —— 卡号应由中间层的 任务包装.sh 按 HQ 分配结果映射。
        启动器若把它快照成固定值，会在包装器**之后**覆盖掉正确映射，使中间层的 GPU 资源语义形同虚设（多卡时全部任务挤到编排进程那一块卡上）。"""

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("BASH_FUNC_", "TASKCTL_GPU_")) and k != "CUDA_VISIBLE_DEVICES"}
    env.update({
        "AF3_CODE_DIR": loc.AF3_CODE_DIR,
        "AF3_BUNDLE_DIR": os.path.abspath(bundle_dir),
        "AF3_OUT_DIR": os.path.abspath(out_dir or db.results_dir()),
        "AF3_MODEL_DIR": loc.LOCAL_MODEL_DIR,
        "AF3_DB_DIR": loc.LOCAL_DB_DIR,
        "AF3_WORK_DIR": _ROOT,
        "AF3_CONDA_ENV": loc.CONDA_ENV,
        "AF3_ENV_PREFIX": loc.AF3_ENV_PREFIX,
        "AF3_PYTHON": loc.AF3_PYTHON,
        "AF3_FLASH_ATTN": loc.FLASH_ATTENTION_IMPL,
        "AF3_NEED_XLA_7X_FLAG": "1" if loc.NEED_XLA_7X_FLAG else "0",
        "AF3_MSA_CPUS": str(loc.MSA_CPUS_PER_GPU),
        "AF3_GPUS_PER_JOB": str(loc.LOCAL_GPUS_PER_JOB),
    })
    if loc.LOCAL_HMMER_DIR:
        env["AF3_HMMER_DIR"] = loc.LOCAL_HMMER_DIR
    return env


# ---- 任务中间层（通用 HyperQueue 封装）接入；AF3 适配层在本项目内 ----
TASKCTL_DIR = os.path.join(os.path.dirname(_ROOT), "任务中间层")
AF3_TASKS_PY = os.path.join(_ROOT, "af3适配", "af3任务.py")
_HQ_REFILL = None


def load_hq_adapter():
    import importlib.util
    spec = importlib.util.spec_from_file_location("af3任务", AF3_TASKS_PY)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def active_hq_names():
    if _HQ_REFILL is None:
        return set()
    return {job.get("name") for job in load_hq_adapter().queue_jobs()
            if (job.get("task_stats") or {}).get("running")
            or (job.get("task_stats") or {}).get("waiting")}


class HqQueueRefiller:
    """有任务可派发时，为每种靶标保留最低 waiting 数量，缺项优先。"""

    def __init__(self, targets, deadline, adapter, per_target=None, stream=None):
        self.targets = tuple(targets)
        self.deadline = deadline
        self.adapter = adapter
        self.per_target = max(1, int(per_target if per_target is not None
                                    else os.environ.get("PEPOPT_QUEUE_MIN_PER_TARGET", "1")))
        self.stream = stream
        self.directory = os.path.join(db.db_root(), "queue_inputs")

    def refill(self, jobs):
        if _STOP or time.time() >= self.deadline - SAFETY_MARGIN_SEC:
            return
        counts = self.adapter.waiting_target_counts(jobs, self.targets)
        missing = {target: max(0, self.per_target - counts[target]) for target in self.targets}
        if not any(missing.values()):
            return
        active = {job.get("name") for job in jobs
                  if (job.get("task_stats") or {}).get("running")
                  or (job.get("task_stats") or {}).get("waiting")}
        registry, rows = db.load_sequence_registry(), db.load_database()
        靶标缺口 = 全库靶标缺口(self.targets, registry, rows)
        submitted = []

        def priority(tag):
            row = rows.get(tag, {})
            values = [row.get(f"iptm_{target.lower()}") for target in self.targets]
            known = [value for value in values if value is not None and math.isfinite(value)]
            return (0, -math.prod(known), tag) if known else (1, 0.0, tag)

        def enqueue(tag, peptide, target, task_priority=0):
            name = f"{tag}_{target.lower()}"
            canonical = db.canonical_dir(tag, target)
            if name in active or db.find_summary(canonical):
                return
            os.makedirs(canonical, exist_ok=True)
            source = loc.write_input_json(peptide, inf.TARGET_SEQS[target], name,
                                          canonical, **inf.target_af3_options(target))
            if loc.estimate_tokens(source) > loc.GPU_MAX_TOKENS:
                return
            os.makedirs(self.directory, exist_ok=True)
            path = os.path.join(self.directory, f"{name}.json")
            shutil.copyfile(source, path)
            self.adapter.submit(path, build_runner_env(self.directory),
                                os.path.join(self.directory, "_hq_tasks"),
                                priority=task_priority)
            active.add(name)
            missing[target] -= 1
            submitted.append(name)

        for tag in sorted(registry, key=priority):
            if not any(missing.values()):
                break
            peptide = registry[tag].get("peptide")
            if not peptide:
                continue
            row = rows.get(tag, {})
            for target in self.targets:
                if missing[target] and row.get(f"iptm_{target.lower()}") is None:
                    enqueue(tag, peptide, target, resample_priority(靶标缺口[target]))
        if any(missing.values()):
            count = max(missing.values())
            if inf.SWEEP_ENABLED:
                candidates = inf.sweep_pick(self.targets, count)
            else:
                if self.stream is None:
                    self.stream = inf.CandidateStream(seed=42)
                candidates = inf.heuristic_pick(self.stream, count, self.targets)
            for peptide in candidates:
                variable = peptide[len(inf.FIXED_A):]
                tag = db.resolve_tag(variable)
                db.register_sequence(tag, variable, peptide)
                for target in self.targets:
                    if missing[target]:
                        enqueue(tag, peptide, target)
        if submitted:
            print(f"  📥 按靶标补充等待队列：{submitted}；剩余缺口={missing}", flush=True)


def run_af3_cycle_hq(bundle_dir, out_dir=None, priority=0):
    """把 bundle 里每个 JSON 提交为独立 HQ 任务，等待全部终态。

    priority 越大越先运行；提交、查询或补队列失败立即抛错，已提交任务仍由 HQ 管理。
    不自动直跑、不重复提交；任务内 MSA 经磁盘缓存复用。"""
    names = sorted(f for f in os.listdir(bundle_dir)
                   if f.endswith(".json") and not f.startswith(("_", ".")))
    def job_priority(filename):
        return (priority.get(os.path.splitext(filename)[0], 0)
                if isinstance(priority, dict) else priority)

    names.sort(key=lambda filename: -job_priority(filename))
    if not names:
        print("  ⚠️ bundle 内无 JSON，跳过", flush=True)
        return 0
    mod = load_hq_adapter()

    env = build_runner_env(bundle_dir, out_dir)
    env["AF3_GPUS_PER_JOB"] = "1"             # 每任务独占 1 个 GPU 槽
    task_root = os.path.join(os.path.abspath(bundle_dir), "_hq_tasks")
    os.makedirs(task_root, exist_ok=True)

    ids, adopted = [], []
    # 接管而非重提：编排进程被 restart-service 重启后，它上次提交的任务往往**仍在
    # HQ 队列里跑**（worker 不随编排进程生死）。只查磁盘 summary 会把这些在途任务
    # 误判为遗留并重新提交 → 同一个预测算两遍（单卡下等于翻倍工时）。
    active = {job["name"]: job["id"] for job in mod.queue_jobs()
              if (job.get("task_stats") or {}).get("running")
              or (job.get("task_stats") or {}).get("waiting")}
    for n in names:
        stem = os.path.splitext(n)[0]
        if active.get(stem) is not None:
            ids.append(active[stem])         # 直接等它跑完，不重复提交
            adopted.append(stem)
            continue
        try:
            ids.append(mod.submit(os.path.join(bundle_dir, n), env, task_root,
                                  priority=job_priority(n)))
        except Exception as error:
            raise RuntimeError(f"AF3 提交失败 {n}：{error}；已提交任务 {ids} 仍由 HQ 管理") from error
    if not ids:
        raise RuntimeError("没有可等待的 HQ 任务")
    print(f"  🚀 已提交 {len(ids) - len(adopted)}/{len(names)} 个 AF3 任务到任务中间层"
          f"（可随时 add/cancel）：{ids}"
          + (f"\n     ♻️ 其中 {len(adopted)} 个为接管重启前仍在队列的任务，未重复提交"
             if adopted else ""), flush=True)
    mod.wait(ids, on_poll=_HQ_REFILL)
    print("  ✅ 本周期全部 AF3 任务已到终态", flush=True)
    return 0


def run_af3_cycle(bundle_dir, out_dir=None, priority=0):
    """只允许 HQ 执行；查询、补队列或提交失败立即报错，绝不直跑回退。"""
    return run_af3_cycle_hq(bundle_dir, out_dir, priority=priority)


# ===================== 组装一轮 =====================

# 补采输出隔离根：以下划线开头，harvest_results/in_flight/drain_backlog 都会跳过
RESAMPLE_ROOT = "_resamples"
# 补采产物消费完即可删：样本已折算进数据库统计量与 iptm_* 列，留着只是白占盘
# （实测：爬山空转期间累积 35 个 bundle = 4.2G，而 obj_n 始终为 5，产物一个没用上）。
# 为何不能靠「非空目录直接覆盖」省盘：AF3 见到非空输出目录会自动改用带时间戳的兄弟目录
# （日志 `exists and non-empty, using instead …_20261021_163012`），只会越盖越多。
# 故采「**固定目录名 + 每次调用先 rmtree**」：磁盘占用恒定 ≤1 个 bundle，且被
# SIGKILL 中断（如 restart-service 撞上在跑的补采）的残留会在下次调用开头自愈，
# 不必依赖只在正常返回路径上执行的那句清理。
KEEP_RESAMPLES = os.environ.get("PEPOPT_KEEP_RESAMPLES", "0").strip().lower() in \
                 ("1", "true", "yes", "on")
RESAMPLE_BUNDLE = "current"          # 固定名；置 KEEP_RESAMPLES=1 时不清理，便于事后追查
# 补采任务必须**插队**：首次爬山与 2^k 整理都阻塞在 `mod.wait(ids)` 上，
# 而它们的任务若按默认 priority=0 入队，会被几十个主批次任务顶到队尾
# （实测：38 个 WAITING 时排第 37/38 位，单任务 ~15min → 要等 ~9 小时才执行，
#   整条流水线就卡在第一条记录的爬山里，主批次连提交都轮不到）。
# 每轮补采只有 1~2 个任务，插队对主批次的扰动可忽略。
RESAMPLE_PRIORITY = int(os.environ.get("PEPOPT_RESAMPLE_PRIORITY", "20"))


def resample_priority(deficit):
    deficit = int(deficit)
    if deficit < 1 or RESAMPLE_PRIORITY < 1:
        raise ValueError("补采缺口和基础优先级必须为正数")
    return RESAMPLE_PRIORITY + deficit


def 全库靶标缺口(靶标列表, 注册表=None, 评分表=None):
    注册表 = db.load_sequence_registry() if 注册表 is None else 注册表
    评分表 = db.load_database() if 评分表 is None else 评分表
    标签列表 = (set(注册表) | set(评分表)) - {""}
    return {靶标: sum(评分表.get(标签, {}).get(f"iptm_{靶标.lower()}") is None or not math.isfinite(评分表[标签][f"iptm_{靶标.lower()}"]) for 标签 in 标签列表) for 靶标 in 靶标列表}


def make_resample_fn(deadline):
    """构造数据库整理的批量补采生成器（真跑 AF3 新 seed，非抽随机数）。

    契约 fn(tag, n=1) 或 fn(tags, n={tag: 样本缺口}) -> {tag：[新乘积样本…]}。
    n 仅决定补采优先级；每轮仍跑一个新 seed，由统计器消费样本后重新判断缺口。
    raw=True 时改返回 {tag：{"products"：[乘积…]，"samples"：{靶标：[逐样本 ipTM]}}}，供首次爬山把最佳样本的**分量**写回 iptm_* 列（只有乘积无从定位分量）。
    实现：对每个 tag 的**变因子**靶标（有逐样本数据者，固定因子是常数无需重跑）各跑一次新 seed 的 AF3 → 逐样本配对相乘得新乘积样本；一次调用打包成一个 bundle 跑（chain-B MSA 每靶标只建一次，逐条单跑太浪费）。
    时间护栏：距 deadline 不足 SAFETY_MARGIN×3 时不再开新补采（宁可少补，不能把时限烧在补采上）。"""
    def resample(tags, n=1, raw=False):
        # 兼容单条调用（爬山）与批量调用（整理）
        if isinstance(tags, str):
            tags = [tags]
        if time.time() > deadline - 3 * SAFETY_MARGIN_SEC:
            print("  ⏭️  接近作业时限，跳过本轮补采", flush=True)
            return {}
        D = db.load_database()
        plan = {}                      # tag -> [（target，samples旧列表）]（只为取靶标）
        fixed = {}
        for tag in tags:
            row = D.get(tag) or {}
            fac = seq_sample.collect_product_factors(tag, row)
            if fac is None:
                continue
            target_samples, fixed_factors = fac
            var_t = [t for t in seq_sample.PRODUCT_TARGETS if t in target_samples]
            if not var_t:
                continue               # 全固定因子：无可采样随机性
            plan[tag] = var_t
            fixed[tag] = fixed_factors
        if not plan:
            return {}
        seed = int(time.time()) % 100000 + 2   # 新 seed（初跑 seed=1）；时间戳保证不与历史重复
        res_root = os.path.join(db.results_dir(), RESAMPLE_ROOT)
        bundle = os.path.join(res_root, RESAMPLE_BUNDLE)
        out_root = os.path.join(bundle, "out")
        if not KEEP_RESAMPLES:
            # 先清掉上一轮的残留（含被 SIGKILL 中断而未走到末尾清理的），再开新目录
            shutil.rmtree(bundle, ignore_errors=True)
        os.makedirs(out_root, exist_ok=True)
        jobs = []                       # (tag, target, job_name)
        for tag in list(plan):
            pep = (D.get(tag) or {}).get("peptide")
            if not pep:
                plan.pop(tag, None)
                continue
            for t in plan[tag]:
                job = f"{tag}_{t.lower()}_s{seed}"
                job_json = create_af3_json(
                    pep, inf.TARGET_SEQS[t], job, model_seeds=[seed],
                    **inf.target_af3_options(t))
                with open(os.path.join(bundle, f"{job}.json"), "w") as f:
                    json.dump(job_json, f, indent=2)
                jobs.append((tag, t, job))
        if not jobs:
            return {}
        print(f"  🔁 整理补采：{len(plan)} 条记录 × 新 seed={seed} → {len(jobs)} 个 AF3 任务",
              flush=True)
        priorities = {job: resample_priority(n[tag] if isinstance(n, dict) else n)
                  for tag, target, job in jobs}
        run_af3_cycle(bundle, out_dir=out_root, priority=priorities)
        # 逐任务提取新样本，按靶标配对相乘（固定因子为常数）；
        # 变因子缺任何一个都放弃该记录——乘积缺因子会得到错误的样本值
        per_tag = {}
        for tag, t, job in jobs:
            samples = db.extract_sample_iptms(os.path.join(out_root, job), t)
            if samples:
                per_tag.setdefault(tag, {})[t] = samples
        out, rawout = {}, {}
        for tag, t2s in per_tag.items():
            if len(t2s) != len(plan[tag]):
                continue
            try:
                prods = seq_sample.build_product_samples(t2s, fixed.get(tag) or {})
            except ValueError:
                continue
            if not prods:
                continue
            out[tag] = prods
            rawout[tag] = {"products": prods, "samples": t2s}
        if not KEEP_RESAMPLES:
            shutil.rmtree(bundle, ignore_errors=True)
        return rawout if raw else out
    return resample

def prepare_cycle(seqs, active_targets, cycle_dir):
    """为本轮挑中的序列写 AF3 输入 JSON（同时落地两处）：
      1）规范结果目录 <af3_results>/<tag>_<target>/input.json —— 收割/折叠/BEST 打包都以它为唯一位置（见 小肽数据库.consolidate_task）；
      2）本轮工作目录 <cycle_dir>/<job_name>.json —— runner 以 --input_dir 读它。
    返回（已排入的任务数，跳过的原因列表）。token 超限的任务在提交前拦截。"""
    os.makedirs(cycle_dir, exist_ok=True)
    n_task, skipped = 0, []
    active_jobs = active_hq_names()
    for seq in seqs:
        var = seq[len(inf.FIXED_A):]
        # resolve_tag 而非 make_tag：已登记过的序列必须沿用当初的标签（旧的 5 位
        # 十进制），否则扩位后的新标签会把磁盘上已有的 34 条结果当成新任务重算。
        tag = db.resolve_tag(var)
        try:
            db.register_sequence(tag, var, seq)
        except RuntimeError as e:                # 撞标签：跳过该序列，不中断搜索
            skipped.append(f"{tag}: {e}")
            continue
        for t in active_targets:
            job_name = f"{tag}_{t.lower()}"
            if job_name in active_jobs:
                continue
            canon = db.canonical_dir(tag, t)
            # consolidate_task 会先把 AF3 的时间戳兄弟目录折叠进规范目录再判定，
            # 比只查规范目录更可靠（漏判会导致已算完的任务被重算）。
            if db.consolidate_task(tag, t) is not None:
                continue                          # 已有结果 → 复用，不重算
            os.makedirs(canon, exist_ok=True)
            inp = loc.write_input_json(
                seq, inf.TARGET_SEQS[t], job_name, canon,
                **inf.target_af3_options(t))
            tk = loc.estimate_tokens(inp)
            if tk > loc.GPU_MAX_TOKENS:
                skipped.append(f"{job_name}: {tk} tokens > {loc.GPU_MAX_TOKENS}")
                continue
            shutil.copyfile(inp, os.path.join(cycle_dir, f"{job_name}.json"))
            n_task += 1
    return n_task, skipped


# ===================== 主循环 =====================

def drain_backlog(active_targets, cycle_dir, seq_budget, task_priorities=None):
    """把**历史遗留的未完成任务**排入本轮（防孤儿化）。返回（序列数，任务数）。

    为何必需：旧形态（登录节点驱动 + 反复 sbatch 小包）被取消或驱动被杀时，会留下“已有 input.json、但永远不会有 summary”的任务目录。而新入口的候选来自 `inf.known_sequences()`（= sequences.csv 全量）的排除集 → 这些序列
    **再也不会被选中**，于是永久缺半边结果：既拖累“完整评估”计数，也让目标乘积缺因子（objective 为 None → 不进排名、GP 训练集也少一条）。本函数直接复用它们已生成的 input.json 把缺的补上。

    每轮最多补 seq_budget 个序列；按已有激活靶标 ipTM 乘积降序选批，
    缺失因子不参与乘积，无已知评分者排最后，同分按 tag 排序。"""
    rd = db.results_dir()
    if not os.path.isdir(rd):
        return 0, 0
    os.makedirs(cycle_dir, exist_ok=True)

    pending = {}                       # tag -> [（任务名，input.json 路径）]
    active_jobs = active_hq_names()

    # ---- 第1类：已有 input.json 但永远等不到 summary 的遗留任务 ----
    for name in sorted(os.listdir(rd)):
        d = os.path.join(rd, name)
        if name.startswith("_") or name.endswith(".zip") or not os.path.isdir(d):
            continue
        parsed = db.split_job_name(name)
        if parsed is None or parsed[1] not in active_targets:
            continue
        if f"{parsed[0]}_{parsed[1].lower()}" in active_jobs:
            continue
        if db.consolidate_task(*parsed) is not None:
            continue                              # 已完成（含时间戳兄弟折叠）
        inp = os.path.join(d, "input.json")
        if not os.path.isfile(inp):
            continue
        pending.setdefault(parsed[0], []).append((name, inp))
    n_leftover = sum(len(v) for v in pending.values())

    # ---- 第2类(gap-fill)：已注册序列缺某个激活靶标，且连 input.json 都没有 ----
    # 来源：旧形态被取消、或（如 2026-10-04）靶标参数被 SLURM 截断只跑了 HTR1A，
    # 于是库里出现“有 HTR1A 无 BIN1”的半行；这类序列已在 sequences.csv 中，
    # 因而**永远不会被重新选中** → 不补就永久缺半边。此处按主表已有值快速预筛，
    # 避免对已完整的序列做无谓的目录探测。
    reg, rows = db.load_sequence_registry(), db.load_database()
    靶标缺口 = 全库靶标缺口(active_targets, reg, rows)
    leftovers, pending = pending, {}

    def priority_key(tag):
        row = rows.get(tag, {})
        values = [row.get(f"iptm_{target.lower()}") for target in active_targets]
        known_values = [value for value in values
                        if value is not None and math.isfinite(value)]
        if not known_values:
            return (1, 0.0, tag)
        return (0, -math.prod(known_values), tag)

    n_gap = 0
    more_backlog = False
    for tag in sorted(set(reg) | set(leftovers), key=priority_key):
        if len(pending) >= seq_budget:
            more_backlog = True             # 内嵌 9E4E 较大，只生成本批次的输入
            break
        if tag in leftovers:
            pending[tag] = leftovers[tag]
        info = reg.get(tag, {})
        pep = info.get("peptide")
        row = rows.get(tag, {})
        if not pep:
            continue
        has = {nm for nm, _ in pending.get(tag, [])}
        for t in active_targets:
            if row.get(f"iptm_{t.lower()}") is not None:
                continue                          # 主表已有值 → 不缺
            name = f"{tag}_{t.lower()}"
            if name in has or name in active_jobs:
                continue
            if db.consolidate_task(tag, t) is not None:
                continue                          # 盘上已完成，下次收割即入库
            canon = db.canonical_dir(tag, t)
            try:
                os.makedirs(canon, exist_ok=True)
                inp = loc.write_input_json(
                    pep, inf.TARGET_SEQS[t], name, canon,
                    **inf.target_af3_options(t))
            except OSError as e:
                print(f"  ⚠️ gap-fill 写输入失败 {name}：{e}", flush=True)
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
                if task_priorities is not None:
                    靶标 = db.split_job_name(name)[1]
                    task_priorities[name] = resample_priority(max(1, 靶标缺口[靶标]))
                n_task += 1
            except OSError as e:
                print(f"  ⚠️ 补算输入复制失败 {name}：{e}", flush=True)
    left_seq = len(pending) - n_seq
    left_text = (f"；尚余 {left_seq} 序列待后续轮次" if left_seq > 0 else
                 "；后续仍有历史序列待补" if more_backlog else "")
    print(f"  🧹 补跑遗留/缺失：{n_seq} 序列 / {n_task} 任务"
          f"（遗留任务 {n_leftover} + 缺失靶标补齐 {n_gap}）" + left_text,
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
    """取走重启标志（幂等）：避免续交的新作业又立即退出。"""
    try:
        os.remove(_restart_flag_path(p))
        return True
    except OSError:
        return False


def run_loop(active_targets, cycle_seqs, max_hours):
    global _HQ_REFILL
    if db.db_root() == db.DEFAULT_DB_ROOT:
        print("   战役初始化：", db.initialize_four_target_campaign(), flush=True)
    口径重建 = db.确保乘积口径()
    if 口径重建 is not None:
        print("   乘积口径已切换：", 口径重建, flush=True)
    p = db.paths(ensure=True)
    start = time.time()
    deadline = start + max_hours * 3600.0
    stream = inf.CandidateStream(seed=42)      # 仅 SWEEP_ENABLED=0 时使用
    _HQ_REFILL = HqQueueRefiller(active_targets, deadline, load_hq_adapter(), stream=stream).refill
    empty_cycles = 0
    cycle_no = 0
    reload_exit = False                        # True → 以 EXIT_RELOAD 退出，让守护重启本进程

    print("=" * 68, flush=True)
    print(f"♾️  无限搜索**在计算节点作业内**运行（无需登录节点常驻进程）", flush=True)
    print(f"   靶标      ：{active_targets}（搁置："
          f"{[t for t in TARGET_ORDER if t not in active_targets]}）", flush=True)
    print(f"   调度      ：" + ("穷尽队列 enumerate_space （不重复 + 有限步全覆盖）"
                               if inf.SWEEP_ENABLED else
                               "随机采样启发式 (CandidateStream)"), flush=True)
    if inf.SWEEP_ENABLED:
        print("   覆盖起点  ：" + inf.coverage_report(active_targets), flush=True)
    print(f"   每轮      ：≤{cycle_seqs} 序列 × {len(active_targets)} 靶标 "
          f"≈ {cycle_seconds(cycle_seqs, active_targets) / 3600:.1f}h", flush=True)
    print(f"   时间预算  ：{max_hours:.1f}h（到期自动退出，由看门狗续交下一作业）",
          flush=True)
    print(f"   共享数据库：{p['root']}", flush=True)
    print(f"   SLURM 作业：{os.environ.get('SLURM_JOB_ID', '(非作业环境)')} @ "
          f"{os.environ.get('SLURMD_NODENAME', '(登录节点)')}", flush=True)
    print("=" * 68, flush=True)

    while not _STOP:
        remaining = deadline - time.time()
        if remaining <= SAFETY_MARGIN_SEC:
            print(f"\n⏰ 时间预算用尽（剩 {remaining / 60:.0f}min < 安全余量 "
                  f"{SAFETY_MARGIN_SEC / 60:.0f}min），正常退出。", flush=True)
            break

        # 周期边界优雅重启：已完成周期的结果已在上一轮 step6 收割入库，
        # 此处退出不会丢数据；标志必须先消费再 break，否则重启后的进程会立即又退出。
        if restart_requested(p):
            consume_restart_flag(p)
            reload_exit = True
            print(f"\n🔁 检测到重启请求 → 在周期边界以重载码 {EXIT_RELOAD} 退出（已完成 {cycle_no} 个周期）；任务守护将立即拉起新进程以加载新代码，作业与节点保持不变。", flush=True)
            break

        cycle_no += 1
        n_seq = size_next_cycle(remaining, cycle_seqs, active_targets)
        if n_seq < 1:
            print("\n⏰ 剩余时间不足一轮，退出。", flush=True)
            break

        elapsed_h = (time.time() - start) / 3600.0
        print(f"\n== [周期 {cycle_no}] {time.strftime('%F %T')} | "
              f"已用 {elapsed_h:.1f}h / {max_hours:.0f}h | "
              f"剩 {remaining / 3600:.1f}h → 本轮 {n_seq} 序列 "
              f"（≈{cycle_seconds(n_seq, active_targets) / 3600:.1f}h）", flush=True)

        # 1）先收割上一个周期的成果（首轮则收割历史遗留/其它入口的完成结果）
        #    收割内部做序贯采样入库（β 统计量），并在入库数过 2^k 时触发数据库整理（P̃·N>M 决定每条记录重复搜索多少次，补采真跑 AF3 新 seed）
        n_new, new_tags = inf.harvest_and_report(active_targets,
                                                 new_sample_fn=make_resample_fn(deadline),
                                                 exclude_jobs=active_hq_names())
        if new_tags:
            inf.climb_new_records(new_tags, active_targets,
                                  make_resample_fn(deadline))

        # 2）本轮工作目录（补跑遗留与新候选共用）
        cycle_dir = os.path.join(p["root"], "cycles",
                                 time.strftime(CYCLE_TAG_FMT))
        os.makedirs(cycle_dir, exist_ok=True)

        # 3）先补上次遗留的未完成任务（防孤儿化），剩余预算再挑新候选
        task_priorities = {}
        d_seq, d_task = drain_backlog(active_targets, cycle_dir, n_seq,
                        task_priorities=task_priorities)
        picks, space_done = [], False
        want = max(0, n_seq - d_seq)
        if want > 0:
            if inf.SWEEP_ENABLED:
                # 穷尽队列：从「精确全集(51091) - 已派发台账(sequences.csv)」中
                # 按 GP 次序取 want 条 → 保证不重复，且有限周期内必然全覆盖。
                # 返回空列表 = 空间已穷尽（**正常完成**，不是采样失败）。
                picks = inf.sweep_pick(active_targets, want)
                if not picks:
                    space_done = True
                    print("  🏁 待评估集合为空 — 搜索空间已全覆盖，不再有新序列。",
                          flush=True)
            else:
                picks = inf.heuristic_pick(stream, want, active_targets)
        if inf.SWEEP_ENABLED:
            print("  " + inf.coverage_report(active_targets,
                                            cycle_secs=cycle_seconds(n_seq, active_targets),
                                            per_cycle=max(want, 1)), flush=True)
        if picks:
            print(f"  🎯 选中 {len(picks)} 个新序列："
                  f"{[db.resolve_tag(s[len(inf.FIXED_A):]) for s in picks[:8]]}"
                  + (" ..." if len(picks) > 8 else ""),
                  flush=True)

        # 4）组装本轮 AF3 输入（新候选；遗留任务已在 drain 阶段复制进 cycle_dir）
        n_task, skipped = (0, [])
        if picks:
            n_task, skipped = prepare_cycle(picks, active_targets, cycle_dir)
        n_task += d_task
        if skipped[:6]:
            for s in skipped[:6]:
                print(f"     ⏭️  跳过 {s}", flush=True)
        if n_task == 0:
            if space_done:
                # 与“连续空轮”区分开：这里是穷尽搜索的**正常终点**，不该再
                # 耗掉 EMPTY_CYCLE_LIMIT 轮，也不该被看门狗误判为异常退出。
                print("  🏁 空间已全覆盖且无遗留任务 → 搜索完成，优雅退出。",
                      flush=True)
                break
            print("  ⚠️ 本轮无待算任务（新候选与历史遗留都已有结果），计入空轮",
                  flush=True)
            empty_cycles += 1
            if empty_cycles >= EMPTY_CYCLE_LIMIT:
                print(f"  🛑 连续 {empty_cycles} 轮零产出，退出。", flush=True)
                break
            continue
        print(f"  📝 已排入 {n_task} 个任务 → {os.path.basename(cycle_dir)}",
              flush=True)
        empty_cycles = 0

        rc = run_af3_cycle(cycle_dir, priority=task_priorities)
        status = "✅" if rc == 0 else f"⚠️ 退出码 {rc}"
        print(f"  {status} 周期 {cycle_no} 的 AF3 结束 "
              f"（{(time.time() - start) / 3600:.1f}h）", flush=True)

        # 6）立刻收割本周期的结果，使下一轮的 GP 能用上最新数据
        got, got_tags = inf.harvest_and_report(active_targets,
                                               new_sample_fn=make_resample_fn(deadline),
                                               exclude_jobs=active_hq_names())
        if got_tags:
            inf.climb_new_records(got_tags, active_targets,
                                  make_resample_fn(deadline))
        if got == 0 and n_new == 0:
            empty_cycles += 1
        else:
            empty_cycles = 0

        # 轮间歇：无实质作用，仅让共享文件系统的元数据落盘/日志可读
        time.sleep(5)

    print(f"\n👋 作业内无限循环结束（共 {cycle_no} 个周期）", flush=True)
    final_n, final_tags = inf.harvest_and_report(active_targets,
                                                  new_sample_fn=make_resample_fn(deadline),
                                                  exclude_jobs=active_hq_names())
    if final_tags:
        inf.climb_new_records(final_tags, active_targets,
                              make_resample_fn(deadline))
    print(f"   数据库：{p['root']}", flush=True)
    return EXIT_RELOAD if reload_exit else 0


# ===================== 入口 =====================

def parse_targets(raw):
    """靶标字符串 → 集合。**同时**用 ',' 与 '+' 作为分隔符。

    为何要支持 '+'：SLURM 的 `--export=a,b,c` 列表本身以**逗号**分隔，而 `PEPOPT_TARGETS=HTR1A,BIN1` 的值内部含逗号 → 会被 SLURM 截断成 `PEPOPT_TARGETS=HTR1A` 加一个孤立的 `BIN1`，靶标静默退化为单靶标（2026-10-04 实测踩到：已跑起的 5 天作业只算 HTR1A，BIN1 因子全丢→乘积恒为 None）。
    故下发时用 '+' 作分隔符；人手输入（CLI/README）两种都接受。"""
    return {t.strip().upper() for t in re.split(r"[,+]", str(raw)) if t.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="计算节点内的无限搜索循环作业（结果持续追加进共享数据库）")
    ap.add_argument("--targets", default=DEFAULT_TARGETS, help="默认读取靶标配置；显式调试名单可用 ',' 或 '+' 分隔")
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
        ap.error(f"--targets 含未知靶标 {sorted(want - set(TARGET_ORDER))}，"
                 f"可选 {TARGET_ORDER}")
    if "HTR1A" not in want:
        ap.error("--targets 必须包含 HTR1A（优化目标）")
    active = [t for t in TARGET_ORDER if t in want]

    # 默认必须是完整四因子；调试子集须显式放行，防止静默退回旧目标。
    if set(active) != set(db.PRODUCT_TARGETS) \
            and os.environ.get("PEPOPT_ALLOW_TARGET_SUBSET") != "1":
        ap.error(f"生产扫描必须覆盖 {db.PRODUCT_TARGETS}，实际为 {active}。"
                 "调试子集请显式设 PEPOPT_ALLOW_TARGET_SUBSET=1。")
    max_tokens = max(len(inf.SEQ_202) + len(inf.TARGET_SEQS[t])
                     * inf.TARGET_COPIES.get(t, 1) for t in active)
    if max_tokens > loc.GPU_MAX_TOKENS:
        ap.error(f"最大复合物 {max_tokens} tokens 超过当前上限 "
                 f"{loc.GPU_MAX_TOKENS}；UNC13C 需 A100/H100 80GB。")
    from 小肽模板 import load as load_template
    load_template()                         # 缺模板/哈希不符时在提交前立即失败

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    # 就绪检查：数据库/权重/hmmer（与登录节点入口一致；计算节点上通常都可用）
    ok, _ = loc.ensure_local_ready()
    if not ok:
        print("\n⚠️  预检查未通过，仍尝试运行（已完成结果可复用）\n", flush=True)

    rc = run_loop(active, args.cycle_seqs, args.max_hours)
    sys.exit(rc)


if __name__ == "__main__":
    main()
