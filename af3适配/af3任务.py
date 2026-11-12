#!/usr/bin/env python3
"""
af3任务.py — 把 AF3 预测任务接入任务中间层（HyperQueue）
=========================================================
职责：把「一个 AF3 输入 JSON」变成「一个可随时添加/取消的 HQ 任务」：
  1. 任务目录：JSON 副本 + **冻结的** AF3打包推理.sh 快照 + `_启动器.sh`（快照使长驻作业与仓库后续改动彻底解耦，与 小肽优化_节点内 既有设计一致：bash 边读边执行，编辑器原地截断重写会让执行错位）。
  2. 启动器在任务**开始时**内嵌 MSA（查 msa缓存，命中即免搜索，特征逐字节等价于现场搜索），**结束时**把本任务产出的 _data.json 收割回缓存——于是同批后续任务自动复用前序任务算出的肽链 MSA（自愈）。
  3. 经 任务控制.sh 提交（gpus=1，优先级可调），返回 HQ job id。

任务粒度：一任务 = 一条（序列×靶标）预测 = 一个 AF3 JSON，与 HQ 的一任务一job语义对齐 → 取消即精确取消单个预测。

用法（库）：
    from af3任务 import submit_all，wait ids = submit_all(jsons, env, task_root, priority=0) wait(ids)
用法（命令行）：
    python3 af3任务.py submit --env-json <env.json> [--priority 0] <任务.json……>
    python3 af3任务.py wait <job_id……>
"""

import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# 任务控制.sh 在通用任务中间层（与本项目平级的 任务中间层/）；TASKCTL_HOME 可覆盖。本目录是 AF3 专属适配层，中间层本身不含任何任务适配。
_MW = os.environ.get("TASKCTL_HOME") or os.path.join(HERE, "..", "..", "任务中间层")
TASKCTL = os.path.join(_MW, "任务控制.sh")
MSA_CACHE = os.path.join(HERE, "msa缓存.py")


def pause_at_batch_boundary():
    """维护切换时在加载下一批适配层之前暂停，不打断本批结果入库。"""
    state = os.environ.get("TASKCTL_STATE", os.path.join(_MW, "运行状态"))
    requested = os.path.join(state, "af3-drain.requested")
    if not os.path.isfile(requested):
        return
    ready = requested + ".ready"
    temporary = ready + f".{os.getpid()}.tmp"
    with open(temporary, "w") as stream:
        json.dump({"pid": os.getpid(), "host": os.uname().nodename,
                   "time": time.time()}, stream)
    os.replace(temporary, ready)
    print("AF3 已在下一批适配层加载边界暂停，等待维护切换", flush=True)
    while os.path.isfile(requested):
        time.sleep(1)


pause_at_batch_boundary()


def _sh_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\\''") + "'"


def gen_task(json_path: str, env: dict, task_root: str) -> str:
    """为单个 AF3 JSON 生成自包含任务目录；返回任务目录路径。

    env：AF3_* 环境变量字典（即 小肽优化_节点内.build_runner_env() 的产物）；其中 AF3_BUNDLE_DIR 会被任务目录覆盖，其余原样烧进启动器。
    """
    name = os.path.splitext(os.path.basename(json_path))[0]
    tdir = os.path.join(task_root, name)
    os.makedirs(tdir, exist_ok=True)

    dst_json = os.path.join(tdir, name + ".json")
    shutil.copyfile(json_path, dst_json)

    # 冻结 runner 快照（见模块文档：防止仓库脚本被原地重写时执行错位）。
    # 生成物一律以 `_` 开头：收割与打包逻辑靠该前缀跳过它们，不会当作任务。
    runner_src = os.path.join(env.get("AF3_WORK_DIR", ""), "作业脚本",
                              "AF3打包推理.sh")
    runner_snap = os.path.join(tdir, "_AF3打包推理快照.sh")
    shutil.copyfile(runner_src, runner_snap)
    os.chmod(runner_snap, 0o755)
    cache_snap = os.path.join(tdir, "_编译缓存快照.py")
    shutil.copyfile(os.path.join(HERE, "编译缓存.py"), cache_snap)
    msa_snap = os.path.join(tdir, "_msa缓存快照.py")
    shutil.copyfile(MSA_CACHE, msa_snap)
    env = dict(env)
    env["AF3_COMPILE_CACHE_CONFIG"] = cache_snap
    env["AF3_MSA_CACHE"] = msa_snap

    out_dir = env["AF3_OUT_DIR"]
    lines = [
        "#!/bin/bash",
        "# 自动生成: AF3 单任务启动器（内嵌 MSA → runner → 收割 MSA 入缓存）",
        "set -uo pipefail",
    ]
    for k in sorted(env):
        if k == "AF3_BUNDLE_DIR":
            continue
        lines.append(f"export {k}={_sh_quote(env[k])}")
    lines.append(f"export AF3_BUNDLE_DIR={_sh_quote(tdir)}")
    lines += [
        # 显式模板缺 MSA 时 embed 非零：立即失败，不能带着不完整输入启动 AF3。
        f"{env.get('AF3_PYTHON', 'python3')} {_sh_quote(msa_snap)} embed {_sh_quote(dst_json)}",
        "rc=$?",
        "if [ $rc -ne 0 ]; then exit $rc; fi",
        f"bash {_sh_quote(runner_snap)}",
        "rc=$?",
        # 任务结束时自愈：把本任务算出的 MSA 收割入缓存，供后续任务复用
        f"find {_sh_quote(out_dir)} -maxdepth 2 -name '*_data.json' "
        f"-path {_sh_quote('*' + name + '*')} "
        f"-exec {env.get('AF3_PYTHON', 'python3')} {_sh_quote(msa_snap)} harvest {{}} + 2>/dev/null || true",
        "exit $rc",
    ]
    launcher = os.path.join(tdir, "_启动器.sh")
    with open(launcher, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(launcher, 0o755)
    return tdir


def queue_jobs():
    """严格读取 HQ 队列；查询失败时不能把队列当成空队列重提任务。"""
    result = subprocess.run(["bash", TASKCTL, "list", "--json"],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"无法读取 HQ 队列：{result.stderr.strip()}")
    jobs = json.loads(result.stdout or "[]")
    if not isinstance(jobs, list):
        raise ValueError("HQ 队列返回格式异常")
    return jobs


def waiting_target_counts(jobs, targets):
    """只统计 waiting；运行中任务不占每种靶标的待运行保有量。"""
    counts = dict.fromkeys(targets, 0)
    for job in jobs:
        waiting = (job.get("task_stats") or {}).get("waiting", 0)
        if not waiting:
            continue
        for target in targets:
            if re.search(rf"_{re.escape(target.lower())}(?:_s\d+)?$", job.get("name", "").lower()):
                counts[target] += waiting
                break
    return counts


def pending_jobs():
    """HQ 中尚未终态的任务：{name: job_id}。

    为什么必须有它：提交前只看磁盘有无 summary（drain_backlog / prepare_cycle）是不够的。
    编排进程被 restart-service 杀掉时，它已提交的任务**仍在 HQ 队列里继续跑**，
    新进程扫磁盘会发现它们“未完成”而重新提交 → 同一个预测算两遍（单卡下翻倍工时）。
    故提交前先拿这份名单，命中的直接接管其 job id 去等，而非重提。

    终态判据：task_stats 的 running+waiting 为零（HQ 0.27 的 json 输出**无 state 字段**，
    实测字段为 id/name/is_open/task_count/task_stats；is_open 恒为 False，不得用作判据）。
    查询失败直接报错，不能把未知状态当作空队列重提。"""
    jobs = queue_jobs()
    out = {}
    for j in jobs:
        ts = j.get("task_stats") or {}
        if (ts.get("running") or 0) + (ts.get("waiting") or 0) > 0:
            name = j.get("name")
            if name:
                out[name] = j.get("id")
    return out


COMPILATION_BUCKETS = (256, 512, 768, 1024, 1280, 1536, 2048,
                       2560, 3072, 3584, 4096, 4608, 5120)


def minimum_gpu_memory(json_path, env=None):
    """按蛋白复合物大小估算充足显存 GiB，作为范围上限，不映射显卡容量。

    当前默认 bfloat16/5 个扩散样本的保守初始容量模型，尚无峰值实测标定：
    ceil((固定开销 + pair 系数 * (bucket / 1024)^2) * 余量)。
    默认固定开销 6 GiB、pair 系数 8 GiB、余量 1.15；三个 AF3_MEMORY_*
    参数可据实测调整。超过默认最大 bucket 时，AF3 按实际 token 数编译。
    """
    settings = os.environ if env is None else env
    fixed = float(settings.get("AF3_MEMORY_FIXED_GB", "6"))
    pair = float(settings.get("AF3_MEMORY_PAIR_GB", "8"))
    margin = float(settings.get("AF3_MEMORY_MARGIN", "1.15"))
    if any(not math.isfinite(value) or value <= 0 for value in (fixed, pair, margin)) \
            or margin < 1:
        raise ValueError("AF3 显存估算系数必须为有限正数，余量必须至少为 1")
    with open(json_path, encoding="utf-8") as stream:
        job = json.load(stream)
    tokens = 0
    for entry in job.get("sequences", []):
        protein = entry.get("protein")
        if not protein:
            raise ValueError("当前显存估算仅支持蛋白体系，不能忽略非蛋白实体")
        chain_ids = protein.get("id")
        copies = len(chain_ids) if isinstance(chain_ids, list) else 1
        tokens += len(protein["sequence"]) * copies
    if tokens <= 0:
        raise ValueError(f"AF3 输入无有效蛋白序列，无法声明所需显存：{json_path}")
    bucket = next((size for size in COMPILATION_BUCKETS if size >= tokens), tokens)
    return math.ceil((fixed + pair * (bucket / 1024) ** 2) * margin)


def 显存需求范围(json_path, env=None):
    上限 = minimum_gpu_memory(json_path, env)
    return 上限 / 2, 上限


def submit(json_path: str, env: dict, task_root: str, priority: int = 0,
           gpus: int = 1, cpus: int = 6, name: str = None) -> int:
    """生成并提交单个 AF3 任务；返回 HQ job id。"""
    tdir = gen_task(json_path, env, task_root)
    name = name or os.path.splitext(os.path.basename(json_path))[0]
    下限, 上限 = 显存需求范围(json_path, env) if gpus else (0, 0)
    r = subprocess.run(
        ["bash", TASKCTL, "add",
         "--name", name, "--gpus", str(gpus), "--cpus", str(cpus),
         "--gpu-memory", f"{下限:g}:{上限:g}" if gpus else "0",
         "--priority", str(priority),
         "--", "bash", os.path.join(tdir, "_启动器.sh")],
        capture_output=True, text=True)
    m = re.search(r"ID=(\d+)", r.stdout)
    if r.returncode or not m:
        raise RuntimeError(f"提交失败：{r.stdout}{r.stderr}")
    return int(m.group(1))


def submit_all(jsons, env: dict, task_root: str, priority: int = 0,
               gpus: int = 1, cpus: int = 6):
    """批量提交；返回 job id 列表。

    按文件名排序，同序列的靶标相邻入队。首个错误立即抛出；已提交任务继续运行，禁止直跑回退。"""
    ids = []
    for jp in sorted(jsons):
        ids.append(submit(jp, env, task_root, priority=priority,
                          gpus=gpus, cpus=cpus))
    return ids


def wait(ids, on_poll=None, poll_seconds=15):
    """等待全部成功，失败、取消或队列异常立即抛错；等待时可补充队列。"""
    owned = set(ids)
    while owned:
        jobs = queue_jobs()
        tracked = {job["id"]: job for job in jobs if job["id"] in owned}
        if owned - tracked.keys():
            raise RuntimeError(f"HQ 队列缺少待等待任务：{sorted(owned - tracked.keys())}")
        failed = {job_id: job.get("task_stats") for job_id, job in tracked.items()
                  if any((job.get("task_stats") or {}).get(state)
                         for state in ("failed", "canceled", "aborted"))}
        if failed:
            raise RuntimeError(f"AF3 HQ 任务未成功完成：{failed}")
        if on_poll is not None:
            on_poll(jobs)
        active = {job_id for job_id, job in tracked.items()
                  if (job.get("task_stats") or {}).get("running")
                  or (job.get("task_stats") or {}).get("waiting")}
        if not active:
            if any(not (job.get("task_stats") or {}).get("finished")
                   for job in tracked.values()):
                raise RuntimeError("HQ 任务既未运行也未成功完成，状态异常")
            return
        time.sleep(poll_seconds)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    op = sys.argv[1]
    if op == "submit":
        args = sys.argv[2:]
        priority = 0
        env_json = None
        task_root = None
        rest = []
        while args:
            a = args.pop(0)
            if a == "--priority":
                priority = int(args.pop(0))
            elif a == "--env-json":
                env_json = args.pop(0)
            elif a == "--task-root":
                task_root = args.pop(0)
            else:
                rest.append(a)
        if not env_json or not rest:
            print("用法：af3任务.py submit --env-json <env.json> [--priority N] "
                  "[--task-root DIR] <任务.json...>")
            return 2
        with open(env_json) as f:
            env = json.load(f)
        task_root = task_root or os.path.join(
            os.path.dirname(os.path.abspath(rest[0])), "_hq_tasks")
        ids = submit_all(rest, env, task_root, priority=priority)
        print("✅ 已提交任务：" + " ".join(str(i) for i in ids))
    elif op == "wait":
        wait(sys.argv[2:])
        print("✅ 全部任务已到终态")
    else:
        print(f"未知操作：{op}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
