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
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 任务控制.sh 在通用任务中间层（与本项目平级的 任务中间层/）；TASKCTL_HOME 可覆盖。本目录是 AF3 专属适配层，中间层本身不含任何任务适配。
_MW = os.environ.get("TASKCTL_HOME") or os.path.join(HERE, "..", "..", "任务中间层")
TASKCTL = os.path.join(_MW, "任务控制.sh")
MSA_CACHE = os.path.join(HERE, "msa缓存.py")


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
        # 任务开始时内嵌缓存 MSA：命中 → AF3 全跳过搜索；未命中 → 现场搜索
        f"python3 {_sh_quote(MSA_CACHE)} embed {_sh_quote(dst_json)}",
        f"bash {_sh_quote(runner_snap)}",
        "rc=$?",
        # 任务结束时自愈：把本任务算出的 MSA 收割入缓存，供后续任务复用
        f"find {_sh_quote(out_dir)} -maxdepth 2 -name '*_data.json' "
        f"-path {_sh_quote('*' + name + '*')} "
        f"-exec python3 {_sh_quote(MSA_CACHE)} harvest {{}} + 2>/dev/null || true",
        "exit $rc",
    ]
    launcher = os.path.join(tdir, "_启动器.sh")
    with open(launcher, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(launcher, 0o755)
    return tdir


def pending_jobs():
    """HQ 中尚未终态的任务：{name: job_id}。

    为什么必须有它：提交前只看磁盘有无 summary（drain_backlog / prepare_cycle）是不够的。
    编排进程被 restart-service 杀掉时，它已提交的任务**仍在 HQ 队列里继续跑**，
    新进程扫磁盘会发现它们“未完成”而重新提交 → 同一个预测算两遍（单卡下翻倍工时）。
    故提交前先拿这份名单，命中的直接接管其 job id 去等，而非重提。

    终态判据：task_stats 的 running+waiting 为零（HQ 0.27 的 json 输出**无 state 字段**，
    实测字段为 id/name/is_open/task_count/task_stats；is_open 恒为 False，不得用作判据）。
    任何异常都返回空集：宁可退化为重提（不丢计算），也不能让作业裸崩。"""
    try:
        r = subprocess.run(["bash", TASKCTL, "list", "--json"],
                           capture_output=True, text=True, timeout=60)
        jobs = json.loads(r.stdout or "[]")
    except Exception as e:                   # noqa: BLE001
        print(f"  ⚠️  无法读取任务中间层队列({e}), 本次不做接管查重", flush=True)
        return {}
    out = {}
    for j in jobs:
        ts = j.get("task_stats") or {}
        if (ts.get("running") or 0) + (ts.get("waiting") or 0) > 0:
            name = j.get("name")
            if name:
                out[name] = j.get("id")
    return out


def submit(json_path: str, env: dict, task_root: str, priority: int = 0,
           gpus: int = 1, cpus: int = 6, name: str = None) -> int:
    """生成并提交单个 AF3 任务；返回 HQ job id。"""
    tdir = gen_task(json_path, env, task_root)
    name = name or os.path.splitext(os.path.basename(json_path))[0]
    r = subprocess.run(
        ["bash", TASKCTL, "add",
         "--name", name, "--gpus", str(gpus), "--cpus", str(cpus),
         "--priority", str(priority),
         "--", "bash", os.path.join(tdir, "_启动器.sh")],
        capture_output=True, text=True)
    m = re.search(r"ID=(\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"提交失败：{r.stdout}{r.stderr}")
    return int(m.group(1))


def submit_all(jsons, env: dict, task_root: str, priority: int = 0,
               gpus: int = 1, cpus: int = 6):
    """批量提交；返回 job id 列表。

    按文件名排序 → 同序列的各靶标任务相邻入队 → 1 GPU 串行时 MSA 自愈链（任务1收割 → 任务2内嵌）自然成立。首个提交失败即抛异常（此时队列为空，调用方可安全回退直接执行）；中途失败则抛异常但已提交的任务继续运行。"""
    ids = []
    last_err = None
    for jp in sorted(jsons):
        try:
            ids.append(submit(jp, env, task_root, priority=priority,
                              gpus=gpus, cpus=cpus))
        except Exception as e:                      # noqa: BLE001
            last_err = e
            break
    if last_err is not None:
        if ids:
            # 已有任务入队：等它们跑完再抛，让收割阶段收走已完成结果
            wait(ids)
        raise last_err
    return ids


def wait(ids):
    """阻塞至所有任务终态（hq job wait 逐个等，总时长 = 最慢者）。"""
    for jid in ids:
        subprocess.run(["bash", TASKCTL, "wait", str(jid)],
                       capture_output=True, text=True)


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
