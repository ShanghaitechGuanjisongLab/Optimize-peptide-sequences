"""用小肽 CPU MSA 任务为 GPU 推理准备共享缓存；不加载模型、不申请 GPU。"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


def _read_states():
    """读全部序列状态；容忍共享盘跨节点瞬时不一致（文件刚被另一节点原子替换），本轮跳过下轮再读。"""
    states = []
    for path in _state_root().glob("*.json"):
        try:
            states.append(json.loads(path.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return states


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
        # 陈旧号盲信是 10 连崩根因：server 重置后旧号消失/撞新任务，
        # 必须核实号在队且名字匹配才复用，否则清掉重提。
        try:
            详情 = hq("job", "info", str(job_id))
            信息 = (详情[0].get("info", 详情[0]) if 详情 else {})
            stats = 信息.get("task_stats") or {}
            名字 = 信息.get("name")
            if 信息.get("id") == int(job_id) and (名字 is None or 名字 == _msa_name(sequence)) and not any(stats.get(k) for k in ("failed", "canceled", "aborted")):
                return int(job_id)
        except (RuntimeError, OSError, ValueError, IndexError):
            pass  # 号不存在/查询异常：视为丢失重提
        state["msa_job"] = None
        write_state(sequence, state)
    if state.get("gpu_ids"):
        raise RuntimeError("小肽 MSA 已放行 GPU，不应重复提交预处理")
    # 数据管线固定并行 4 个数据库搜索，至少预约 4 核；容量不足时宁可排队也不超订
    cpus = max(4, msa_cpu_count())
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


def _msa_name(sequence):
    return "msa_" + hashlib.sha1(sequence.encode()).hexdigest()[:12]


def 各靶标优先级(sequence, fresh=False):
    """与补采队列同口径：全新探索候选=0；已登记缺分靶标=20+全库缺口；已有分=0。"""
    sys.path.insert(0, str(HERE.parent))
    import 小肽数据库 as database
    import 小肽优化_节点内 as node
    import 小肽优化_无限 as search
    tag = database.resolve_tag(sequence[len(search.FIXED_A):])
    rows = database.load_database()
    row = rows.get(tag) or {}
    缺口 = node.全库靶标缺口(database.PRODUCT_TARGETS, database.load_sequence_registry(), rows)
    priorities = {}
    for target in database.PRODUCT_TARGETS:
        value = row.get(f"iptm_{target.lower()}")
        缺分 = value is None or not math.isfinite(value)
        priorities[target] = 0 if fresh else (node.resample_priority(max(1, 缺口[target])) if 缺分 else 0)
    return tag, priorities


def bind_prefetch_gpu(sequence, fresh=False):
    """给预取小肽绑定同序列各靶标的真实 GPU 请求，避免 MSA 完成后成为孤儿。

    fresh=True 表示刚从穷举队列抽出的全新探索候选，与 refill 同口径用 0 优先级；
    否则为已登记缺分补全，用 20+全库缺口，且已有分靶标不重复预测。"""
    state = read_state(sequence)
    if state.get("gpu_pending") or state.get("gpu_ids"):
        return bool(state.get("gpu_pending") or state.get("gpu_ids"))
    sys.path.insert(0, str(HERE.parent))
    import 小肽数据库 as database
    import 小肽优化_节点内 as node
    import 小肽优化_无限 as search
    variable = sequence[len(search.FIXED_A):]
    tag, priorities = 各靶标优先级(sequence, fresh=fresh)
    database.register_sequence(tag, variable, sequence)
    directory = _state_root() / "inputs"
    node.prepare_cycle([sequence], database.PRODUCT_TARGETS, str(directory))
    env = node.build_runner_env(str(directory))
    submit_msa(sequence, str(_state_root() / "gpu_tasks"), max(priorities.values(), default=0))
    for target in database.PRODUCT_TARGETS:
        if not fresh and priorities[target] <= 0:
            continue  # 已有分的靶标不重复预测（fresh 全 0 但仍需首评）
        name = f"{tag}_{target.lower()}"
        path = directory / f"{name}.json"
        if path.is_file():
            register_gpu(str(path), env, str(_state_root() / "gpu_tasks"), priorities[target], 1, name)
    return bool(read_state(sequence).get("gpu_pending"))


def ensure_prefetch(jobs=None, picker=None):
    """MSA 就绪、GPU 未启动的新小肽数量等于当前运行中的不同节点数。"""
    jobs = hq("job", "list", "--all") if jobs is None else jobs
    在线 = [worker for worker in hq("worker", "list") if worker.get("ended") is None]
    目标水位 = len({worker.get("configuration", {}).get("hostname") for worker in 在线})
    states = _read_states()
    索引 = {job["id"]: job for job in jobs}
    可供应 = []
    for state in states:
        sequence = state.get("sequence")
        if not sequence or state.get("gpu_ids"):
            continue
        job_id = state.get("msa_job")
        if job_id is not None:
            job = 索引.get(job_id)
            if job is not None and job.get("name") and job["name"] != _msa_name(sequence):
                job = None  # 日志重置后旧号可能撞到新任务，不得误认
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
    # 休眠恢复：已抽取但 MSA 丢失/失败的小肽，按水位逐条重提，不得一次性洪水
    for state in states:
        if len(可供应) >= 目标水位:
            break
        seq = state.get("sequence")
        if not seq or state.get("gpu_ids") or state.get("msa_job") is not None or not needed(seq):
            continue
        _, priorities = 各靶标优先级(seq)
        submit_msa(seq, str(_state_root()), max(priorities.values(), default=0))
        if not state.get("gpu_pending"):
            bind_prefetch_gpu(seq)
        可供应.append(read_state(seq))
        print(f"按水位恢复待算小肽 MSA：{seq[:16]}…", flush=True)
    if len(可供应) >= 目标水位:
        return 可供应[0] if 可供应 else None
    for peptide_sequence in (picker(目标水位 - len(可供应)) if picker is not None else search.sweep_pick(search.PRODUCT_TARGETS, 目标水位 - len(可供应))):
        if peptide_sequence[len(search.FIXED_A):] in 已登记:
            continue
        _, priorities = 各靶标优先级(peptide_sequence, fresh=True)
        job_id = submit_msa(peptide_sequence, str(_state_root()), max(priorities.values(), default=0))
        bind_prefetch_gpu(peptide_sequence, fresh=True)
        print(f"预取下一条新小肽 MSA：{job_id}；GPU 请求已绑定，等待 MSA 完成", flush=True)
        可供应.append(read_state(peptide_sequence))
        if len(可供应) >= 目标水位:
            break
    return 可供应[0] if 可供应 else None


def advance_pending(only_ids=None):
    """CPU MSA 完成后提交真实 GPU 任务；缓存就绪即视为 MSA 成功，不依赖旧 HQ 日志。"""
    import af3任务 as adapter
    replacements = {}
    states = _read_states()
    _活动名缓存 = {}
    _评分缓存 = {}

    def 活动任务名():
        if "map" not in _活动名缓存:
            try:
                _活动名缓存["map"] = {j.get("name"): j["id"] for j in hq("job", "list")
                                     if (j.get("task_stats") or {}).get("running") or (j.get("task_stats") or {}).get("waiting")}
            except (RuntimeError, OSError, ValueError):
                _活动名缓存["map"] = {}  # 查询失败退化为仅按名去重，不阻断放行
        return _活动名缓存["map"]

    def 靶标已有分(sequence, name):
        if re.search(r"_s\d+$", name or ""):
            return False  # 补采种子是有意重跑
        sys.path.insert(0, str(HERE.parent))
        import 小肽数据库 as database
        import 小肽优化_无限 as search
        if "rows" not in _评分缓存:
            _评分缓存["rows"] = database.load_database()
        tag = database.resolve_tag(sequence[len(search.FIXED_A):])
        target = next((t for t in database.PRODUCT_TARGETS if (name or "").endswith("_" + t.lower())), None)
        if target is None:
            return False
        value = (_评分缓存["rows"].get(tag) or {}).get(f"iptm_{target.lower()}")
        return value is not None and math.isfinite(value)

    待查 = sorted({s["msa_job"] for s in states if s.get("msa_job") is not None and not s.get("gpu_ids") and s.get("sequence") and needed(s["sequence"])})
    索引 = {}
    查询失败 = False
    if 待查:
        try:
            详情 = hq("job", "info", ",".join(map(str, 待查)))
            for item in 详情:
                信息 = item.get("info", item)
                编号 = 信息.get("id", item.get("id"))
                if 编号 is not None:
                    索引[编号] = 信息
        except (RuntimeError, OSError, ValueError):
            查询失败 = True
            print("MSA 状态查询暂时失败，未就绪序列保持等待", flush=True)
    for state in states:
        job_id = state.get("msa_job")
        sequence = state.get("sequence")
        if state.get("gpu_ids") or not sequence:
            continue
        if needed(sequence):
            if 查询失败 or job_id is None:
                continue
            job = 索引.get(job_id)
            if job is not None and job.get("name") and job["name"] != _msa_name(sequence):
                job = None  # 日志重置后旧号撞新任务：视为丢失
            stats = (job or {}).get("task_stats") or {}
            if job is not None and not any(stats.get(key) for key in ("failed", "canceled", "aborted")):
                continue  # 运行/等待/已完成但缓存未就绪：继续等
            if job is not None and (only_ids is None or job_id in only_ids):
                raise RuntimeError(f"MSA 任务 {job_id} 未成功，拒绝放行 GPU 推理")
            # MSA 丢失/失败：置休眠，由 ensure_prefetch 按水位逐条恢复，禁止一次性全部重提
            state["msa_job"] = None
            write_state(sequence, state)
            continue
        有效请求 = [r for r in state.get("gpu_pending", []) if os.path.isfile(r["json_path"])]
        if not 有效请求:
            continue  # 独立预取保持就绪，不得用空 gpu_ids 冒充已放行
        # 同名去重（历史跨目录重复请求），保留优先级较高者
        按名 = {}
        for r in 有效请求:
            old = 按名.get(r.get("name"))
            if old is None or int(r.get("priority", 0)) > int(old.get("priority", 0)):
                按名[r.get("name")] = r
        gpu_ids = []
        for request in 按名.values():
            name = request.get("name")
            同名活动 = 活动任务名().get(name)
            if 同名活动 is not None:
                gpu_ids.append(同名活动)  # refill 已直提同名任务，接管不重提
                continue
            if 靶标已有分(sequence, name):
                continue  # 已收割出分，不重复预测
            gpu_ids.append(adapter.submit(request["json_path"], request["env"], request["task_root"], priority=request["priority"], gpus=request.get("gpus", 1), cpus=1, name=name, allow_msa=False))
        state["gpu_pending"] = list(按名.values())
        state["gpu_ids"] = gpu_ids
        write_state(sequence, state)
        if job_id is not None:
            replacements[job_id] = gpu_ids
        print(f"MSA 就绪，已提交 GPU 任务 {gpu_ids}", flush=True)
    return replacements


def resolve_ids(ids):
    current = set(ids)
    states = _read_states()
    replacements = {state["msa_job"]: state["gpu_ids"] for state in states if state.get("msa_job") in current and state.get("gpu_ids")}
    改号 = {}
    for state in states:
        改号.update({int(old): int(new) for old, new in (state.get("gpu_replaced") or {}).items()})
    for old in current:
        new, seen = old, set()
        while new in 改号 and new not in seen:
            seen.add(new)
            new = 改号[new]
        if new != old:
            replacements[old] = [new]
    replacements.update(advance_pending(current))
    return [new for old in ids for new in replacements.get(old, [old])]


def reprioritize_waiting():
    """把已放行但仍在纯等待的 GPU 任务按全库缺口重算优先级；过低则取消重提。

    先写状态（gpu_ids 换新号 + gpu_replaced 旧→新）再取消旧任务，
    编排侧 resolve_ids 会据此把等待集里的旧号无缝换成新号。"""
    import af3任务 as adapter
    states = _read_states()
    replacements = {}
    for state in states:
        sequence = state.get("sequence")
        pending = state.get("gpu_pending") or []
        if not sequence or not pending:
            continue
        _, priorities = 各靶标优先级(sequence)
        if not state.get("gpu_ids"):
            changed = False
            for request in pending:
                target = next((t for t in priorities if (request.get("name") or "").endswith("_" + t.lower())), None)
                if target is not None and priorities[target] > int(request.get("priority", 0)):
                    request["priority"] = priorities[target]
                    changed = True
            if changed:
                write_state(sequence, state)
            continue
        if len(pending) != len(state["gpu_ids"]) or needed(sequence):
            continue
        ids = state["gpu_ids"]
        索引 = {}
        查询号 = sorted(set(ids))
        try:
            for start in range(0, len(查询号), 100):
                for item in hq("job", "info", ",".join(map(str, 查询号[start:start + 100]))):
                    信息 = item.get("info", item)
                    编号 = 信息.get("id", item.get("id"))
                    if 编号 is not None:
                        索引[编号] = 信息
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"序列 {sequence[:16]}… 状态查询暂时失败，本轮跳过（重跑可续）：{exc}", flush=True)
            continue
        已处理 = set()
        for index, request in enumerate(pending):
            job_id = ids[index]
            if job_id in 已处理:
                continue
            stats = (索引.get(job_id) or {}).get("task_stats") or {}
            if job_id not in 索引 or any(stats.get(key) for key in ("running", "finished", "failed", "canceled", "aborted")):
                continue  # 只纠正纯等待任务，绝不碰运行中/已终态
            target = next((t for t in priorities if (request.get("name") or "").endswith("_" + t.lower())), None)
            want = priorities.get(target, 0) if target is not None else 0
            if want <= int(request.get("priority", 0)):
                continue
            old_priority = int(request.get("priority", 0))
            new_id = adapter.submit(request["json_path"], request["env"], request["task_root"], priority=want, gpus=request.get("gpus", 1), cpus=1, name=request.get("name"), allow_msa=False)
            request["priority"] = want
            state["gpu_ids"] = [new_id if 旧 == job_id else 旧 for 旧 in state["gpu_ids"]]
            state.setdefault("gpu_replaced", {})[str(job_id)] = new_id
            已处理.add(job_id)
            write_state(sequence, state)
            for attempt in range(5):
                try:
                    hq("job", "cancel", str(job_id), output="quiet")
                    break
                except (RuntimeError, OSError):
                    if attempt == 4:
                        raise RuntimeError(f"新任务 {new_id} 已提交但旧任务 {job_id} 取消失败，请手动取消以免重复计算")
                    time.sleep(1)
            replacements[job_id] = new_id
            print(f"GPU 任务 {job_id} 优先级 {old_priority} 过低，已重提为 {new_id}（优先级 {want}，{request.get('name')}）", flush=True)
    return replacements


def requeue_lost():
    """HQ 日志重置后：gpu_ids 已不在队列（或撞号改名）的状态重新放行；已有分或输入已清理的请求丢弃。"""
    sys.path.insert(0, str(HERE.parent))
    import 小肽数据库 as database
    try:
        现存 = {job["id"]: job.get("name") for job in hq("job", "list", "--all")}
    except (RuntimeError, OSError, ValueError):
        try:
            hq("server", "info")
        except (RuntimeError, OSError, ValueError):
            现存 = {}  # HQ 服务已停止：旧任务号全部失效
        else:
            raise  # 服务在线但查询失败：不能把未知当丢失
    rows = database.load_database()
    released = 0
    for state in _read_states():
        ids = state.get("gpu_ids")
        sequence = state.get("sequence")
        if not sequence or not state.get("gpu_pending"):
            continue
        请求名 = {r.get("name") for r in state["gpu_pending"]}
        # 日志重置后旧号可能撞到新任务：仅当号在队列且名字仍匹配本序列请求才算存活
        if ids and any(编号 in 现存 and 现存[编号] in 请求名 for 编号 in ids):
            continue
        tag, priorities = 各靶标优先级(sequence)
        pending = []
        for request in state["gpu_pending"]:
            if not os.path.isfile(request["json_path"]):
                continue  # 临时输入已被清理，无法重提
            name = request.get("name") or ""
            if re.search(r"_s\d+$", name):
                pending.append(request)  # 补采种子任务是有意重跑，不受已有分判重约束
                continue
            target = next((t for t in priorities if name.endswith("_" + t.lower())), None)
            if target is not None:
                value = (rows.get(tag) or {}).get(f"iptm_{target.lower()}")
                if (value is not None and math.isfinite(value)) or database.find_summary(database.canonical_dir(tag, target)):
                    continue  # 已有分或已有结果文件，不重算
                request["priority"] = priorities[target]
            pending.append(request)
        fresh = read_state(sequence)
        if fresh.get("gpu_ids") != ids:
            continue  # 编排已处理，不覆盖
        if pending == state["gpu_pending"] and not (ids and pending):
            continue
        state["gpu_pending"] = pending
        if pending:
            state.pop("gpu_ids", None)
            state.pop("gpu_replaced", None)
        write_state(sequence, state)
        released += 1
        print(f"序列 {tag} 的 HQ 任务已丢失，重新放行 {len(pending)} 个 GPU 请求", flush=True)
    return released


def register_gpu(json_path, env, task_root, priority, gpus, name):
    sequence = peptide(json_path)
    state = read_state(sequence)
    pending = state.setdefault("gpu_pending", [])
    request = {"json_path": str(json_path), "env": dict(env), "task_root": str(task_root), "priority": int(priority), "gpus": int(gpus), "name": name}
    # 同一序列同一靶标只允许一个请求，与输入目录无关（refill 与预取写不同目录，
    # 若按路径判重会各放一条 → advance_pending 重复提交同一预测）。优先级取较高者。
    existing = next((item for item in pending if item["name"] == name), None)
    if existing is None:
        pending.append(request)
    elif int(priority) > int(existing.get("priority", 0)):
        existing["priority"] = int(priority)
    state["sequence"] = sequence
    write_state(sequence, state)
    job_id = submit_msa(sequence, task_root, priority)
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "advance", "capacity", "reprioritize", "requeue"))
    parser.add_argument("--sequence")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cpus", type=int)
    args = parser.parse_args()
    if args.command == "capacity":
        print(json.dumps(free_cpu_capacity()))
    elif args.command == "advance":
        print(json.dumps(advance_pending()))
    elif args.command == "reprioritize":
        print(json.dumps(reprioritize_waiting()))
    elif args.command == "requeue":
        print(json.dumps(requeue_lost()))
    else:
        prepare(args.sequence, args.work_dir, args.cpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
