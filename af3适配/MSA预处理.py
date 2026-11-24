"""用小肽 CPU MSA 任务为 GPU 推理准备共享缓存；不加载模型、不申请 GPU。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msa缓存 as cache


def peptide(json_path):
    data = json.loads(Path(json_path).read_text())
    for entry in data.get("sequences", []):
        protein = entry.get("protein")
        if protein and "A" in (protein.get("id") or []) and protein.get("sequence"):
            return protein["sequence"]
    raise ValueError(f"{json_path} 没有小肽 A 链")


def needed(sequence):
    return cache.load(sequence) is None


def prepare(sequence, work_dir, cpus):
    """用官方 AF3 数据管线产出真实 MSA，随后原子写入共享缓存。"""
    if not needed(sequence):
        return {"cached": True}
    if not 1 <= int(cpus) <= 256:
        raise ValueError("MSA CPU 数必须为正整数")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "peptide.json"
    temporary = source.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps({"name": "peptide_msa", "modelSeeds": [1], "sequences": [{"protein": {"id": "A", "sequence": sequence}}], "dialect": "alphafold3", "version": 1}), encoding="utf-8")
    temporary.replace(source)
    shutil.rmtree(work / "out", ignore_errors=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", AF3_NEED_XLA_7X_FLAG="0")
    python = os.environ.get("AF3_PYTHON") or os.environ.get("PEPOPT_AF3_PYTHON", "/public_bme2/Share200T/管吉松/envs/af3/bin/python")
    env["PATH"] = str(Path(python).parent) + ":" + env.get("PATH", "")
    code_dir = os.environ.get("AF3_CODE_DIR", "/public_bme2/Share200T/管吉松/AlphaFold3")
    db_dir = os.environ.get("AF3_DB_DIR", "/public_bme2/Share200T/管吉松/databases/alphafold3")
    # 官方数据管线并行跑 4 个蛋白数据库；分摊线程，避免 N 核预约变成 4N 线程。
    search_cpus = max(1, int(cpus) // 4)
    command = [python, code_dir + "/run_alphafold.py",
               "--json_path=" + str(source), "--output_dir=" + str(work / "out"),
               "--db_dir=" + db_dir, "--norun_inference",
               f"--jackhmmer_n_cpu={search_cpus}", f"--nhmmer_n_cpu={search_cpus}"]
    print(f"MSA 开始：{int(cpus)} CPU；不使用 GPU", flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise RuntimeError(f"AF3 小肽数据管线失败，退出码 {result.returncode}")
    data = work / "out" / "peptide_msa" / "peptide_msa_data.json"
    if not data.is_file():
        raise RuntimeError("MSA 数据管线未产生小肽 _data.json")
    cache.extract([str(data)])
    if needed(sequence):
        raise RuntimeError("MSA 数据管线未产生可入库的小肽缓存")
    print("MSA 已写入共享缓存", flush=True)
    return {"cached": False, "cpus": int(cpus)}


def _state_root():
    root = Path(os.environ.get("TASKCTL_STATE", os.environ.get("PEPOPT_DB_ROOT", HERE.parent.parent / "任务中间层" / "运行状态"))) / "msa_pipeline"
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_path(sequence):
    return _state_root() / (hashlib.sha1(sequence.encode()).hexdigest()[:20] + ".json")


def read_state(sequence):
    path = state_path(sequence)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_state(sequence, state):
    path = state_path(sequence)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def hq(*arguments, output="json"):
    root = Path(os.environ.get("TASKCTL_STATE", HERE.parent.parent / "任务中间层" / "运行状态"))
    middleware = Path(os.environ.get("TASKCTL_HOME", HERE.parent.parent / "任务中间层"))
    env = dict(os.environ, HQ_SERVER_DIR=str(root / "server"))
    env["LD_LIBRARY_PATH"] = "/public_bme2/Share200T/管吉松/envs/af3/lib:" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run([str(middleware / "bin" / "hq"), f"--output-mode={output}", *arguments], env=env, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError("HQ 查询/提交失败：" + result.stderr.strip())
    return result.stdout if output != "json" else json.loads(result.stdout or "null")


def free_cpu_capacity():
    """返回在线 GPU worker 的剩余整核数；仅 GPU 空闲 worker 留 1 核给推理。"""
    workers = hq("worker", "list")
    jobs = hq("job", "list")
    active = [job["id"] for job in jobs if (job.get("task_stats") or {}).get("running")]
    details = hq("job", "info", ",".join(map(str, active))) if active else []
    used = {}
    gpu_busy = set()
    for job in details:
        resources = {item["resource"]: next(iter(item["request"].values()), 0) for variant in job["submits"][0]["array"]["resources"] for item in variant.get("resources", [])}
        for task in job.get("tasks", []):
            if task.get("state") == "running" and task.get("worker") is not None:
                used[task["worker"]] = used.get(task["worker"], 0) + resources.get("cpus", 0) / 10000
                if resources.get("gpus", 0) > 0:
                    gpu_busy.add(task["worker"])
    capacity = {}
    for worker in workers:
        if worker.get("ended") is not None:
            continue
        resources = {item["name"]: item for item in worker["configuration"]["resources"].get("resources", [])}
        cpus = resources.get("cpus", {})
        total = cpus.get("end", -1) - cpus.get("start", 0) + 1 if cpus.get("kind") == "range" else 0
        if resources.get("gpus", {}).get("values") and worker["id"] not in gpu_busy:
            total -= 1
        capacity[worker["id"]] = max(0, int(total - used.get(worker["id"], 0)))
    return capacity


def msa_cpu_count():
    capacity = free_cpu_capacity()
    return max(1, max(capacity.values(), default=1))


def submit_msa(sequence, task_root, priority=0):
    state = read_state(sequence)
    if not needed(sequence):
        return None
    job_id = state.get("msa_job")
    if job_id is not None:
        return int(job_id)
    if state.get("gpu_ids"):
        raise RuntimeError("小肽 MSA 已放行 GPU，不应重复提交预处理")
    cpus = msa_cpu_count()
    directory = _state_root() / (hashlib.sha1(sequence.encode()).hexdigest()[:20] + "_work")
    command = [sys.executable, str(HERE / "MSA预处理.py"), "run", "--sequence", sequence, "--work-dir", str(directory), "--cpus", str(cpus)]
    name = "msa_" + hashlib.sha1(sequence.encode()).hexdigest()[:12]
    logs = _state_root() / "logs"
    logs.mkdir(exist_ok=True)
    result = hq("submit", "--name", name, "--cpus", str(cpus), "--priority", str(max(1000, int(priority) + 1000)), "--stdout", str(logs / "%{JOB_ID}.stdout"), "--stderr", str(logs / "%{JOB_ID}.stderr"), "--", *command, output="quiet")
    job_id = int(result.strip())
    state.update({"sequence": sequence, "msa_job": job_id, "cpus": cpus})
    write_state(sequence, state)
    print(f"MSA 任务 {job_id}：{cpus} CPU；GPU 等待 MSA 完成", flush=True)
    return job_id


def bind_prefetch_gpu(sequence):
    """给预取小肽绑定同序列各靶标的真实 GPU 请求，避免 MSA 完成后成为孤儿。"""
    state = read_state(sequence)
    if state.get("gpu_pending") or state.get("gpu_ids"):
        return bool(state.get("gpu_pending") or state.get("gpu_ids"))
    sys.path.insert(0, str(HERE.parent))
    import 小肽数据库 as database
    import 小肽优化_节点内 as node
    import 小肽优化_无限 as search
    variable = sequence[len(search.FIXED_A):]
    tag = database.resolve_tag(variable)
    database.register_sequence(tag, variable, sequence)
    directory = _state_root() / "inputs"
    node.prepare_cycle([sequence], database.PRODUCT_TARGETS, str(directory))
    env = node.build_runner_env(str(directory))
    for target in database.PRODUCT_TARGETS:
        name = f"{tag}_{target.lower()}"
        path = directory / f"{name}.json"
        if path.is_file():
            register_gpu(str(path), env, str(_state_root() / "gpu_tasks"), 0, 1, name)
    return bool(read_state(sequence).get("gpu_pending"))


def ensure_prefetch(jobs=None, picker=None):
    """MSA 就绪、GPU 未启动的新小肽数量等于当前运行中的不同节点数。"""
    jobs = hq("job", "list", "--all") if jobs is None else jobs
    在线 = [worker for worker in hq("worker", "list") if worker.get("ended") is None]
    目标水位 = len({worker.get("configuration", {}).get("hostname") for worker in 在线})
    states = [json.loads(path.read_text()) for path in _state_root().glob("*.json")]
    索引 = {job["id"]: job for job in jobs}
    可供应 = []
    for state in states:
        sequence = state.get("sequence")
        if not sequence or state.get("gpu_ids"):
            continue
        job_id = state.get("msa_job")
        if job_id is not None:
            job = 索引.get(job_id)
            stats = (job or {}).get("task_stats") or {}
            if any(stats.get(key) for key in ("failed", "canceled", "aborted")):
                continue  # 已取消/失败的旧任务不永久占用预取水位
            if stats.get("running") or stats.get("waiting"):
                if bind_prefetch_gpu(sequence):
                    可供应.append(state)
                continue
        if not needed(sequence):
            if bind_prefetch_gpu(sequence):
                可供应.append(state)
        elif job_id is not None:
            if bind_prefetch_gpu(sequence):
                可供应.append(state)
    sys.path.insert(0, str(HERE.parent))
    import 小肽数据库 as database
    import 小肽优化_无限 as search
    已登记 = search.attempted_vars()
    rows = database.load_database()
    for tag, entry in database.load_sequence_registry().items():
        peptide_sequence = entry.get("peptide")
        if not peptide_sequence or needed(peptide_sequence):
            continue
        row = rows.get(tag) or {}
        if any(row.get(f"iptm_{target.lower()}") is not None for target in database.PRODUCT_TARGETS):
            continue
        if bind_prefetch_gpu(peptide_sequence):
            可供应.append({"sequence": peptide_sequence, "ready": True})
    if len(可供应) >= 目标水位:
        return 可供应[0] if 可供应 else None
    for peptide_sequence in (picker(目标水位 - len(可供应)) if picker is not None else search.sweep_pick(search.PRODUCT_TARGETS, 目标水位 - len(可供应))):
        if peptide_sequence[len(search.FIXED_A):] in 已登记:
            continue
        job_id = submit_msa(peptide_sequence, str(_state_root()), 0)
        bind_prefetch_gpu(peptide_sequence)
        print(f"预取下一条新小肽 MSA：{job_id}；GPU 请求已绑定，等待 MSA 完成", flush=True)
        可供应.append(read_state(peptide_sequence))
        if len(可供应) >= 目标水位:
            break
    return 可供应[0] if 可供应 else None


def advance_pending(only_ids=None):
    """CPU MSA 完成后提交真实 GPU 任务；只把当前等待者的失败向上抛。"""
    import af3任务 as adapter
    replacements = {}
    states = [json.loads(path.read_text()) for path in _state_root().glob("*.json")]
    待查 = [s["msa_job"] for s in states if s.get("msa_job") is not None and not s.get("gpu_ids")]
    无需查 = [s for s in states if s.get("msa_job") is None and not s.get("gpu_ids")]
    if not 待查 and not 无需查:
        return replacements
    索引 = {}
    if 待查:
        try:
            详情 = hq("job", "info", ",".join(map(str, sorted(set(待查)))))
            for item in 详情:
                信息 = item.get("info", item)
                编号 = 信息.get("id", item.get("id"))
                if 编号 is not None:
                    索引[编号] = 信息
        except (RuntimeError, OSError, ValueError):
            print("MSA 状态查询暂时失败，保持当前等待状态", flush=True)
            return replacements
    for state in states:
        job_id = state.get("msa_job")
        if state.get("gpu_ids"):
            continue
        if job_id is not None:
            job = 索引.get(job_id)
            if job is None and needed(state["sequence"]):
                continue
            stats = (job or {}).get("task_stats") or {}
            if any(stats.get(key) for key in ("failed", "canceled", "aborted")):
                if only_ids is None or job_id in only_ids:
                    raise RuntimeError(f"MSA 任务 {job_id} 未成功，拒绝放行 GPU 推理")
                continue
            if stats.get("running") or stats.get("waiting"):
                continue
            if not stats.get("finished") and needed(state["sequence"]):
                continue
        if needed(state["sequence"]):
            continue
        if needed(state["sequence"]):
            continue
        if not state.get("gpu_pending"):
            continue  # 独立预取保持就绪，不得用空 gpu_ids 冒充已放行
        gpu_ids = []
        for request in state.get("gpu_pending", []):
            name = request.get("name") or os.path.splitext(os.path.basename(request["json_path"]))[0]
            gpu_ids.append(adapter.submit(request["json_path"], request["env"], request["task_root"], priority=request["priority"], gpus=request.get("gpus", 1), cpus=1, name=request.get("name"), allow_msa=False))
        state["gpu_ids"] = gpu_ids
        write_state(state["sequence"], state)
        replacements[job_id] = gpu_ids
        print(f"MSA {job_id} 完成，已提交 GPU 任务 {gpu_ids}", flush=True)
    return replacements


def resolve_ids(ids):
    current = set(ids)
    replacements = {state["msa_job"]: state["gpu_ids"] for state in (json.loads(path.read_text()) for path in _state_root().glob("*.json")) if state.get("msa_job") in current and state.get("gpu_ids")}
    replacements.update(advance_pending(current))
    return [new for old in ids for new in replacements.get(old, [old])]


def register_gpu(json_path, env, task_root, priority, gpus, name):
    sequence = peptide(json_path)
    state = read_state(sequence)
    pending = state.setdefault("gpu_pending", [])
    request = {"json_path": str(json_path), "env": dict(env), "task_root": str(task_root), "priority": int(priority), "gpus": int(gpus), "name": name}
    if not any(item["json_path"] == request["json_path"] and item["name"] == name for item in pending):
        pending.append(request)
    state["sequence"] = sequence
    write_state(sequence, state)
    job_id = submit_msa(sequence, task_root, priority)
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "advance", "capacity"))
    parser.add_argument("--sequence")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    if args.command == "capacity":
        print(json.dumps(free_cpu_capacity()))
    elif args.command == "advance":
        print(json.dumps(advance_pending()))
    else:
        prepare(args.sequence, args.work_dir, args.cpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
