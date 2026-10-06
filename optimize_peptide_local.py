"""
蒙特卡洛优化小肽序列 — AlphaFold3 本地推理入口
=============================================================
与 optimize_peptide_AF3.py（官方服务器版）的区别：
  - 结构预测在本集群 GPU 计算节点上运行（自动用 sbatch 提交作业）
  - 复用本机已下载的 AlphaFold3 代码、模型权重与数据库（见下方路径）
  - 无每日配额限制，无需 Google 账号，任务全部并行排队在集群上

复用的部分（来自公共库 peptide_common）：
  - 序列常量（SEQ_202 / HTR1A_SEQ / UNC13C_SEQ / BIN1_SEQ）
  - generate_mutant() 突变规则、run_monte_carlo() 蒙特卡洛主循环（后端无关）
  本文件提供本地集群推理的预测实现：生成 .sbatch → sbatch 提交 → 计算节点推理。

工作目录约定：
  af3_local_results/<任务名>/
      input.json          ← AlphaFold3 输入（与服务器版相同格式）
      af3_local.sbatch    ← 生成的作业脚本
      slurm_<jobid>.out/err
      <任务名>/           ← run_alphafold.py 的输出（*_summary_confidences.json 等）
  完成的任务会打包为 af3_local_results/<任务名>.zip，供优化主循环读取评分与保存最佳结构。

使用步骤：
  1. 首次准备（一次性，大部分已自动处理）：
     a) 推理环境: conda 环境 af3_old 已含 alphafold3 + jax
        若缺 hmm 工具: conda install -n af3_old -c bioconda hmmer
     b) 驱动依赖: 运行本脚本的环境需要 pandas / matplotlib / openpyxl
        pip install pandas matplotlib openpyxl
     c) 数据库: 共享目录中是压缩的 .zst 文件; 首次运行会自动提交一个解压作业到计算节点异步执行，无需保持 SSH 连接，解压完成后重跑本脚本即可继续。
  2. 运行:
        conda activate base      # 或任意装好驱动依赖的环境
        python optimize_peptide_local.py
     首次运行会为初始序列×3 个靶标提交推理作业；作业完成后重跑本脚本即可自动评分并继续推进（支持断点续跑）。

检查策略：入口的预检查只做提示、不会退出——即使资源尚未就绪也会照常运行流程：已完成的旧结果仍可被评分复用，新的提交失败会打印原因，补齐条件后重跑即可。

数据库自动准备（异步在计算节点解压，无需保持 SSH 连接）:
  预检查发现数据库缺失时:
    1. 先读取各 .zst 帧头汇总预计解压体积，用 statvfs 检查目标盘剩余空间，不足（低于预计体积×1.05）则中止并提示；
    2. 通过后把解压作为 SLURM 作业提交到计算节点异步执行（--prepare-db 模式），提交后即可断开 SSH；用 tail -f db_prep_<jobid>.out 查看进度，完成后重跑继续。
  解压目标: /public_bme2/Share200T/管吉松/databases/alphafold3（家目录有 ~500G NFS 配额放不下, 用共享大容量目录, 计算节点可见）。
  环境变量: SKIP_DB_PREPARE=1 跳过自动解压。

合规提示：AlphaFold3 模型权重与输出受官方条款约束（见 /public_bme2/Share200T/管吉松/AlphaFold3/WEIGHTS_TERMS_OF_USE.md），仅限非商业学术用途。
"""

import os
import sys
import json
import glob
import re
import shutil
import zipfile
import subprocess

import numpy as np
import random
import time

# ---- 驱动依赖检查（缺失时给出明确提示）----
_missing = []
for _m in ("pandas", "matplotlib", "openpyxl"):
    try:
        __import__(_m)
    except ImportError:
        _missing.append(_m)
if _missing:
    sys.exit(f"驱动环境缺少依赖: {_missing}，请先执行: pip install {' '.join(_missing)}")

# 导入公共库（导入零副作用，不含任何预测后端逻辑）
from peptide_common import create_af3_json, run_monte_carlo, ROOT_DIR

# ===================== 本地模式配置区域 =====================

AF3_CODE_DIR   = "/public_bme2/Share200T/管吉松/AlphaFold3"         # 本地 AF3 代码目录
LOCAL_MODEL_DIR = "/public_bme2/Share200T/管吉松/weights"        # 含 af3.bin 权重文件的目录

# 数据库: 共享目录中是压缩的 .zst 文件（只读）; 预检查发现缺失时会先检查磁盘空间,再把解压作为作业提交到计算节点异步执行。
# 注意: 家目录有 ~500G NFS 配额(不够放全量数据库), 必须放共享大容量目录。
SHARED_DB_SRC = "/public_bme2/Share200T/AlphaFold3/DB_DIR/alphafold3"
LOCAL_DB_DIR  = "/public_bme2/Share200T/管吉松/databases/alphafold3"

# AF3 的数据管线构造时会解析全部数据库路径, 即使纯蛋白体系也要求 RNA 库文件存在（仅在含 RNA 链时真正参与搜索）, 因此下列文件均为必需。
DB_FASTA_FILES = [
    "uniref90_2022_05.fa",                       # 主 MSA 库 (UniRef90)
    "bfd-first_non_consensus_sequences.fasta",   # 深度补充 MSA 库 (BFD)
    "mgy_clusters_2022_05.fa",                   # 宏基因组补充库 (MGnify)
    "uniprot_all_2021_04.fa",                    # 链间 MSA 配对 (UniProt全量)
    "pdb_seqres_2022_09_28.fasta",               # 模板搜索序列索引 (PDB)
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",     # RNA 库
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",             # RNA 库
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta",          # RNA 库
]
DB_MMCIF_ARCHIVE = "pdb_2022_09_28_mmcif_files.tar.zst"   # 解压为目录: <DB>/mmcif_files/

# 本地推理结果目录。环境变量 PEPOPT_RESULTS_DIR 可覆盖（无限搜索/共享库模式把它指向共享目录 <PEPOPT_DB_ROOT>/af3_results, 结果直接落共享盘）。
# 必须是绝对路径: 计算节点在 AF3_WORK_DIR(仓库目录)下解析相对路径，家目录相对路径在共享盘场景会指错位置。
LOCAL_RESULTS_DIR = os.environ.get("PEPOPT_RESULTS_DIR", "").strip() \
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "af3_local_results")

# 外置作业脚本目录: SLURM 实际执行的 .sh 脚本存放处, 不再把 shell 内嵌进 Python 字符串
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

# SLURM 资源配置（每个"序列×靶标"为一个推理任务, 多任务打包进单个作业）
# PEPOPT_PARTITION 可覆盖（将来申请到 A100/H100 分区时切换）
LOCAL_PARTITION = os.environ.get("PEPOPT_PARTITION", "").strip() or "bme_gpupub"
LOCAL_ACCOUNT   = "v-jiamh"                   # SLURM 账户（无此分区/账户组合会被拒）
LOCAL_MEMORY    = "64G"                       # **每 GPU** 的主机内存配额（总量按卡数放大）
LOCAL_CPUS      = 8                           # **每 GPU** 的 CPU 配额（受 QoS cpu 上限钳制）
LOCAL_TIME      = "48:00:00"                  # 单个作业最长时限（打包模式下含多个任务）


# ---------------------------------------------------------------------
# 分区 QoS 自动探测（关键: 可用 GPU 数不由分区总卡数决定, 而由 QoS 决定）
# ---------------------------------------------------------------------
# 分区总容量 16×V100 是**公共**的, 但分区自带 QoS 对每个用户硬限资源:
#   partition_bme_gpupub: MaxTRESPU=cpu=8,gres/gpu=1  MaxJobsPerUser=1
#                         MaxSubmitJobsPerUser=2
# 即**每用户峰值只有 1 张 GPU / 8 CPU**, 同时 1 个作业在跑 + 1 个排队。
# 管理员策略, 代码无法绕过; 超限作业不会报错而是**永远 PENDING**(QOSMaxTRESPULimit / QOSMaxJobsPerUserLimit), 因此必须主动钳制。


def _probe_qos_limits(partition="bme_gpupub"):
    """读分区的 QoS 限制 → {gpu, cpu, running, submit}; 探测失败返回保守默认。

    running = MaxJobsPerUser（可同时运行）, submit = MaxSubmitJobsPerUser
    （运行+排队总数）。gpu/cpu 取自 MaxTRESPU 的 gres/gpu 与 cpu（每用户合计）。"""
    limits = {"gpu": 1, "cpu": 8, "running": 1, "submit": 2}
    try:
        part = subprocess.run(["scontrol", "show", "partition", partition],
                              capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"QoS=(\S+)", part)
        if not m:
            return limits
        out = subprocess.run(
            ["sacctmgr", "-n", "-p", "show", "qos", f"name={m.group(1)}",
             "format=MaxJobsPerUser,MaxSubmitJobsPerUser,MaxTRESPU"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        if not out:
            return limits
        f = out.split("|")
        tres = f[2] if len(f) > 2 else ""
        g = re.search(r"gres/gpu=(\d+)", tres)
        c = re.search(r"(?<![\w/])cpu=(\d+)", tres)
        if g:
            limits["gpu"] = int(g.group(1))
        if c:
            limits["cpu"] = int(c.group(1))
        if f[0].isdigit():
            limits["running"] = int(f[0])
        if len(f) > 1 and f[1].isdigit():
            limits["submit"] = int(f[1])
    except Exception:
        pass
    return limits


QOS_LIMITS = _probe_qos_limits(LOCAL_PARTITION)


def _env_int(name, default):
    """读整数环境变量: 未设/空串用默认, 非法值告警后用默认。

    注意不能用 `os.environ.get(name, "0") or default` 的写法 —— 字符串 "0"
    是真值, 会把默认值吃掉（实际踩过: BUNDLE_SIZE 被错置为 1）。"""
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        print(f"⚠️ 环境变量 {name}={v!r} 不是整数, 改用默认值 {default}")
        return default


# 单作业 GPU 数: QoS 的 gpu 上限是**每用户所有运行中作业的合计**, 故平分给可同时
# 运行的作业数。当前 1÷1=1（包内任务串行, runner 的 stage1_parallel 不启用）;
# 管理员放开 MaxTRESPU 后本值自动变大 → 每作业多卡并行, 无需改代码。
# PEPOPT_GPUS_PER_JOB 可手动指定（仍被 QoS 上限钳制, 避免永远 PENDING）。
_GPU_SHARE = max(1, QOS_LIMITS["gpu"] // max(1, QOS_LIMITS["running"]))
_want_gpu = max(0, _env_int("PEPOPT_GPUS_PER_JOB", 0))
if _want_gpu > QOS_LIMITS["gpu"]:
    print(f"⚠️  PEPOPT_GPUS_PER_JOB={_want_gpu} 超过分区 QoS 上限 "
          f"gres/gpu={QOS_LIMITS['gpu']}（超限作业会永远 PENDING）, 已钳到 "
          f"{QOS_LIMITS['gpu']}。")
LOCAL_GPUS_PER_JOB = max(1, min(_want_gpu, QOS_LIMITS["gpu"])) if _want_gpu else _GPU_SHARE

# 每 GPU 分得的 CPU（合计不超 QoS 的 cpu 上限）
LOCAL_CPUS_PER_GPU = max(1, min(LOCAL_CPUS, QOS_LIMITS["cpu"] // LOCAL_GPUS_PER_JOB))

# 每条 GPU 流喂给 jackhmmer/nhmmer 的线程数。
# 实测 MSA 占总时长 ~85%, 是绝对的临界路径; 而 QoS 给了 8 CPU 却只用 6 是浪费。
# AF3 用 pyhmmer(OpenMP 多线程), 数据管线阶段主线程阻塞等待, 并不额外抢 CPU,
# 故线程数 = 全部可用 CPU 最优（不再像早期版本那样留 2 核余量）。
# PEPOPT_MSA_CPUS 可覆盖。
MSA_CPUS_PER_GPU = max(1, _env_int("PEPOPT_MSA_CPUS", LOCAL_CPUS_PER_GPU))

# 单个作业打包的任务数。**这是 1 卡限额下最大的吞吐杠杆**, 依据实测(2026-10-03,
# bme_gpu09, V100-SXM2, 6 任务包 2h)的 MSA/推理拆解:
#   chain B(靶标) MSA: HTR1A 1738s + BIN1 1555s ≈ 55min **每进程只建一次**(包内复用);
#   chain A(肽)    MSA: ~670s **每个不同肽都要重算**(不可摊销, 占大头);
#   推理: HTR1A 93s / BIN1 230s, 仅占总时长 ~14%。
# 关键: chain B 的两个靶标序列恒定不变, 但 MSA 缓存(data/pipeline.py 的
# functools.cache)作用域仅单个 python 进程 → 每个作业都要白重建 55min。
# 故包越大, 这 55min 摊得越薄。按实测值建模的**单序列**成本 3293/N + 1010s:
#   N=3(包6)  → 2108s   N=6(包12)  → 1559s   N=12(包24) → 1284s
#   N=24(包48)→ 1147s   N=48(包96) → 1079s   N=∞        → 1010s
# 取 48（24 序列×2 靶标, 约 7.7h/包, 分区 MaxTime=5 天绰绰有余）: 比包 24 提升
# 12% 吞吐; 再大一档只多 6%, 却要 14.4h/包且降低 GP 反馈频率, 不划算。
# 注意: 提交的 --time 恒为 LOCAL_TIME(48h), 与包大小无关, 故加大包不会增加排队等待。
# PEPOPT_BUNDLE_SIZE 可覆盖。
BUNDLE_SIZE     = max(1, _env_int("PEPOPT_BUNDLE_SIZE", 48))
# 提交窗口 = QoS 允许的运行+排队总数: 用满它可让下个作业在上个作业结束
# 的瞬间接续（零空隙），又不会因超限而被拒
MAX_INFLIGHT    = max(1, QOS_LIMITS["submit"])

# 单个任务的最大 token 数。超出者注定显存溢出, 提交前直接拦截, 避免白烧机时。
# 依据 AlphaFold3 官方 docs/performance.md 的硬件上限（均指已启用 unified memory）:
#   V100 (CUDA 算力 7.x) -> 1,280 tokens   <- 当前 bme_gpupub 分区全是 V100 32GB
#   A100 40GB            -> 4,352 tokens（还需改 model_config.py 的 pair_transition_shard_spec）
#   A100/H100 80GB       -> 官方支持全尺寸（默认最大 bucket 5,120 tokens）
# 若申请到 A100/H100-80G: 设环境变量 PEPOPT_GPU_MAX_TOKENS=5120 与
# PEPOPT_FLASH_ATTN=triton（见下）, 并把 runner 的 AF3_NEED_XLA_7X_FLAG 置 0,
# 即可补齐 UNC13C; 代码无需改动。
GPU_MAX_TOKENS  = int(os.environ.get("PEPOPT_GPU_MAX_TOKENS", "") or 1280)

# flash attention 实现（对应 run_alphafold.py 的 --flash_attention_implementation）:
#   "xla"    : 不用 flash attention, 跨设备可用, 但注意力显存 O(N^2)。
#              CUDA 算力 7.x（V100/T4）**必须**用此值（AF3 会强制校验）。
#   "triton" : 真正的 flash attention, 显存 O(N) 且更快, 但需 Ampere（算力 8.0+,
#              即 A100/H100）。A100-80G 上换成 triton 后, 大体系不再受显存平方律限制。
#   "cudnn"  : cuDNN 实现, 同样需 Ampere, 测试不如 triton 充分。
FLASH_ATTENTION_IMPL = os.environ.get("PEPOPT_FLASH_ATTN", "").strip() or "xla"
# CUDA 算力 7.x (V100/T4) 需要给 JAX 注入禁用 HLO 融合通道的 XLA_FLAGS;
# Ampere+ (A100/H100) 不需要, 置 "0" 跳过。PEPOPT_XLA_7X=0 可覆盖默认值。
NEED_XLA_7X_FLAG = os.environ.get("PEPOPT_XLA_7X", "1") == "1"

# token 数超限的任务不提交, 写入此标记, 以免每次重跑都重复刷警告
OOM_SKIP_MARKER = ".oom_skipped"

# hmm 工具（jackhmmer/nhmmer/hmmalign/hmmsearch/hmmbuild）所在目录；
# 留空则依赖作业节点 PATH 中可直接找到
LOCAL_HMMER_DIR = ""

CONDA_ENV       = "af3_old"                 # 推理用 conda 环境（含 alphafold3+jax）

FORCE_RESUBMIT  = os.environ.get("FORCE_RESUBMIT", "0") == "1"   # 强制重提交失败作业

# =====================================================================


# ---------------- 数据库自动解压 ----------------

def zstd_content_size(path):
    """从 zstd 帧头解析解压后体积（只读文件前 ≤18 字节, 瞬时完成）。
    解析失败返回 None（帧头未记录内容大小时会发生）。"""
    try:
        with open(path, "rb") as f:
            header = f.read(18)
    except OSError:
        return None
    if len(header) < 5 or int.from_bytes(header[:4], "little") != 0xFD2FB528:
        return None
    fhd = header[4]
    fcs_flag = fhd >> 6                    # Frame_Content_Size_Flag
    single_segment = bool(fhd & 0x20)      # Single_Segment_Flag
    dict_flag = fhd & 0x03                 # Dictionary_ID_Flag
    pos = 5
    if not single_segment:
        pos += 1                           # Window_Descriptor (1 字节)
    pos += {0: 0, 1: 1, 2: 2, 3: 4}[dict_flag]
    if fcs_flag == 0:
        if not single_segment:
            return None
        if len(header) < pos + 1:
            return None
        return header[pos]
    widths = {1: 2, 2: 4, 3: 8}
    w = widths[fcs_flag]
    if len(header) < pos + w:
        return None
    value = int.from_bytes(header[pos:pos + w], "little")
    return value + 256 if fcs_flag == 1 else value


# PDB 2022-09 全量约 20 万个 .cif; 顶层铺开的 .cif 少于此数视为解压不完整
MMCIF_MIN_CIFS = 100000


def mmcif_ready():
    """mmCIF 模板库就绪判定: 有完成标记, 或顶层已铺开足够 .cif
    （仅看目录存在会把残缺解压误判为就绪）。"""
    mmcif_dir = os.path.join(LOCAL_DB_DIR, "mmcif_files")
    if not os.path.isdir(mmcif_dir):
        return False
    if os.path.isfile(os.path.join(LOCAL_DB_DIR, ".mmcif_done")):
        return True
    n = 0
    try:
        with os.scandir(mmcif_dir) as it:
            for e in it:
                if e.name.endswith(".cif"):
                    n += 1
                    if n >= MMCIF_MIN_CIFS:
                        return True
    except OSError:
        return False
    return False


def database_ready():
    """检查 LOCAL_DB_DIR 中所需数据库是否齐备。返回 (ok, missing)。"""
    missing = [n for n in DB_FASTA_FILES
               if not os.path.isfile(os.path.join(LOCAL_DB_DIR, n))]
    if not mmcif_ready():
        missing.append("mmcif_files/ (目录)")
    return (not missing), missing


def prepare_database():
    """解压共享目录中的压缩数据库: 先检查磁盘空间, 不足则中止；
    够用则逐项解压（中断后重跑自动续作, 不留半个文件）。返回是否全部就绪。
    本函数在计算节点上由 '--prepare-db' 作业调用（见 submit_db_prep_job）。"""
    os.makedirs(LOCAL_DB_DIR, exist_ok=True)

    if not shutil.which("zstd"):
        print("❌ 未找到 zstd 命令行工具, 无法解压数据库")
        return database_ready()[0]

    # ---- 收集缺失项及解压后体积 ----
    todo = []   # (kind, src, dst, expected_bytes)
    for name in DB_FASTA_FILES:
        dst = os.path.join(LOCAL_DB_DIR, name)
        if os.path.isfile(dst):
            continue
        src = os.path.join(SHARED_DB_SRC, name + ".zst")
        if not os.path.isfile(src):
            print(f"⚠️  共享目录缺少源文件: {src}")
            continue
        todo.append(("fasta", src, dst, zstd_content_size(src)))

    mmcif_src = os.path.join(SHARED_DB_SRC, DB_MMCIF_ARCHIVE)
    mmcif_dir = os.path.join(LOCAL_DB_DIR, "mmcif_files")
    mmcif_marker = os.path.join(LOCAL_DB_DIR, ".mmcif_done")
    if not mmcif_ready():
        if os.path.isfile(mmcif_src):
            size = zstd_content_size(mmcif_src)
            if size is not None:
                size = int(size * 1.10)   # 文件系统额外开销余量（小文件多）
            todo.append(("mmcif", mmcif_src, mmcif_dir, size))
        else:
            print(f"⚠️  共享目录缺少 mmCIF 归档: {mmcif_src}")

    if not todo:
        return database_ready()[0]

    # ---- 磁盘空间检查（解压前必须通过）----
    unknown = [os.path.basename(t[1]) for t in todo if t[3] is None]
    if unknown:
        print(f"⚠️  无法从帧头读取体积, 未计入空间核算: {unknown}")
    required = sum(t[3] for t in todo if t[3] is not None)
    st = os.statvfs(LOCAL_DB_DIR)
    avail = st.f_bavail * st.f_frsize
    print(f"📦 数据库准备: 需解压 {len(todo)} 项, 预计占用 {required / 2**30:.0f} GB；"
          f"目标盘当前可用 {avail / 2**30:.0f} GB ({LOCAL_DB_DIR})")
    if avail < int(required * 1.05):
        print(f"❌ 磁盘空间不足: 需要 ≥ {int(required * 1.05) / 2**30:.0f} GB, 中止解压。"
              f"请把 LOCAL_DB_DIR 改到容量更大的目录后重试。")
        return False

    # ---- 逐项解压 ----
    print("⏳ 开始解压（总体积较大, 可能耗时数小时）", flush=True)
    for kind, src, dst, size in todo:
        size_gb = f" ({size / 2**30:.1f} GB)" if size else ""
        if kind == "fasta":
            tmp = dst + ".partial"
            print(f"   解压 {os.path.basename(src)}{size_gb} → {dst}", flush=True)
            rc = subprocess.run(["zstd", "-d", "-f", "-T4", src, "-o", tmp]).returncode
            if rc != 0 or not os.path.isfile(tmp):
                print(f"   ❌ 解压失败: {os.path.basename(src)}")
                if os.path.exists(tmp):
                    os.remove(tmp)
                continue
            os.replace(tmp, dst)   # 原子改名, 避免残留半个文件被误认为就绪
        else:
            os.makedirs(dst, exist_ok=True)
            log_path = os.path.join(LOCAL_DB_DIR, ".mmcif_uncompress.log")
            print(f"   解压 mmCIF 归档{size_gb} → {dst}/", flush=True)
            try:
                with open(log_path, "w") as logf:
                    p1 = subprocess.Popen(["zstd", "-dc", src], stdout=subprocess.PIPE)
                    p2 = subprocess.Popen(["tar", "-xf", "-", "-C", dst],
                                          stdin=p1.stdout, stderr=logf)
                    p1.stdout.close()
                    rc = p2.wait()
                    p1.wait()
            except Exception as e:
                print(f"   ❌ mmCIF 解压异常: {e}")
                continue
            if rc != 0 or p1.returncode != 0:
                print(f"   ❌ mmCIF 解压失败（见 {log_path}）")
                continue
            with open(mmcif_marker, "w") as f:
                f.write("done\n")
        print(f"   ✅ 完成: {os.path.basename(src)}", flush=True)
    return database_ready()[0]


def db_prep_job_state():
    """查询数据库解压作业状态。返回 (state, jobid)，
    state ∈ none/pending/running/done/failed/unknown。"""
    marker = os.path.join(LOCAL_DB_DIR, ".db_prep_submitted")
    if not os.path.isfile(marker):
        return "none", None
    try:
        with open(marker) as f:
            jobid = f.read().strip().splitlines()[0].strip()
    except OSError:
        return "none", None
    if not jobid:
        return "none", None
    # 先查 squeue（活动作业的权威来源, 无记账延迟）; 不在队列再用 sacct 查终态
    try:
        out = subprocess.run(["squeue", "-j", jobid, "-h", "-o", "%T"],
                             capture_output=True, text=True, timeout=15)
        q = out.stdout.strip().splitlines()
        if q:
            s = q[0].strip()
            return ("running" if s == "RUNNING" else "pending"), jobid
    except Exception:
        pass
    try:
        out = subprocess.run(["sacct", "-j", jobid, "-n", "-P", "-o", "State"],
                             capture_output=True, text=True, timeout=15)
        states = [s.strip() for s in out.stdout.splitlines() if s.strip()]
    except Exception:
        return "unknown", jobid
    if not states:
        return "unknown", jobid
    s = states[0]
    if s == "RUNNING":
        return "running", jobid
    if s in ("PENDING", "CONFIGURING", "COMPLETING"):
        return "pending", jobid
    if s == "COMPLETED":
        return "done", jobid
    return "failed", jobid


def follow_db_prep_job(jobid):
    """实时跟踪解压作业进度直到结束（Ctrl+C 退出跟踪不影响作业运行）。
    检测到解压作业在运行时应转入本函数的进度回报, 而不是继续后续步骤。"""
    out_file = os.path.join(LOCAL_DB_DIR, f"db_prep_{jobid}.out")
    print(f"\n⏳ 检测到解压作业正在运行 (jobid={jobid})，转入实时进度跟踪…")
    print(f"   日志: {out_file}")
    print("   （Ctrl+C 仅退出跟踪, 解压作业继续运行；完成后重跑本脚本继续）")
    tail_proc = None
    if os.path.isfile(out_file):
        tail_proc = subprocess.Popen(["tail", "-f", "-n", "+1", out_file])
    try:
        while True:
            state, _ = db_prep_job_state()
            if state not in ("pending", "running"):
                break
            time.sleep(15)
    except KeyboardInterrupt:
        if tail_proc:
            tail_proc.terminate()
        print("\n👀 已停止进度跟踪, 解压作业仍在运行；完成后重跑本脚本继续。")
        return
    if tail_proc:
        time.sleep(1)
        tail_proc.terminate()
    state, _ = db_prep_job_state()
    ready, missing = database_ready()
    print(f"\n{'✅ 解压作业完成' if state == 'done' and ready else '⚠️ 解压作业结束: ' + state}")
    if not ready:
        print(f"   仍缺: {missing}；排查日志: {out_file}")


def submit_db_prep_job():
    """把数据库解压作为 SLURM 作业提交到计算节点异步执行；
    提交后即可断开 SSH，无需维持连接。返回正在执行/新提交的 jobid, 否则 None。"""
    os.makedirs(LOCAL_DB_DIR, exist_ok=True)
    marker = os.path.join(LOCAL_DB_DIR, ".db_prep_submitted")

    state, jobid = db_prep_job_state()
    if state in ("pending", "running"):
        return jobid                      # 已在跑, 由调用方转入进度跟踪
    if state in ("done", "failed", "unknown") and os.path.isfile(marker):
        print(f"📦 上次解压作业已结束 ({state}, jobid={jobid})，清理陈旧标记后重新提交。")
        os.remove(marker)

    # 作业脚本外置于 scripts/db_prep_runner.sh；SLURM 资源走参数, 配置走 --export,
    # Python 侧不再内嵌 shell, 也不生成临时 .sbatch 文件。
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(SCRIPTS_DIR, "db_prep_runner.sh")
    env_pairs = [
        f"PEPOPT_WORK_DIR={repo_dir}",
        "PEPOPT_CONDA_ENV=base",
        "PEPOPT_ENTRY=optimize_peptide_local.py",
    ]
    cmd = [
        "sbatch",
        "--job-name=af3_db_prep",
        f"--output={LOCAL_DB_DIR}/db_prep_%j.out",
        f"--error={LOCAL_DB_DIR}/db_prep_%j.err",
        f"--partition={LOCAL_PARTITION}",
        f"--account={LOCAL_ACCOUNT}",
        "--nodes=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=24:00:00",
        "--export=ALL," + ",".join(env_pairs),
        runner,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jobid = out.stdout.strip().split()[-1]
        with open(marker, "w") as f:
            f.write(jobid + "\n")
        print(f"🚀 数据库解压作业已提交到计算节点 (jobid={jobid})，提交后无需保持 SSH 连接。")
        print(f"   进度查看: tail -f {LOCAL_DB_DIR}/db_prep_{jobid}.out")
        print(f"   体积约 330+ GB, 预计数小时, 磁盘不足会自动中止。")
        return jobid
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ 提交数据库解压作业失败: {e}")
        return None


# =====================================================================

# 预检查状态: 未通过时 submit_local_job 会跳过提交，
# 避免向队列投递注定失败的 GPU 作业
PREFLIGHT_OK = False


def preflight_check():
    """启动前检查本地推理所需资源是否齐备。
    只打印提示、绝不中断流程；返回 (ok, errors)，并把结果记入 PREFLIGHT_OK。"""
    global PREFLIGHT_OK
    errors = []

    script = os.path.join(AF3_CODE_DIR, "run_alphafold.py")
    if not os.path.isfile(script):
        errors.append(f"AF3 代码不存在: {script}")

    weights = os.path.join(LOCAL_MODEL_DIR, "af3.bin")
    if not os.path.isfile(weights):
        errors.append(f"模型权重不存在: {weights}")

    # 数据库：要求解压后的文件（AF3 不接受 .zst）；缺失时把解压提交为
    # 计算节点异步作业，无需在登录节点维持 SSH 连接，也不占用登录节点资源。
    ready, missing_db = database_ready()
    if not ready and os.environ.get("SKIP_DB_PREPARE") != "1":
        submit_db_prep_job()
        ready, missing_db = database_ready()
    if not ready:
        errors.append(
            f"数据库缺少: {missing_db}\n"
            f"   目录: {LOCAL_DB_DIR}\n"
            f"   解压作业在计算节点运行中, 新的推理作业会跳过提交"
        )

    # hmm 工具检查（数据管线必需；推理用环境为 CONDA_ENV）
    env_bin = os.path.expanduser(f"~/miniconda3/envs/{CONDA_ENV}/bin")
    if not shutil.which("jackhmmer") and not os.path.isfile(os.path.join(env_bin, "jackhmmer")) \
            and not LOCAL_HMMER_DIR:
        errors.append(
            f"未找到 jackhmmer（数据管线搜索 MSA 必需）\n"
            f"   解决其一: conda install -n {CONDA_ENV} -c bioconda hmmer\n"
            f"   或将现有目录写入 LOCAL_HMMER_DIR / 确保计算节点 PATH 可用"
        )

    if errors:
        PREFLIGHT_OK = False
        print("⚠️  本地推理预检查发现以下问题（不阻断运行，相关步骤可能失败）：")
        for e in errors:
            print("   -", e)
        return False, errors

    PREFLIGHT_OK = True
    print("✅ 预检查通过（代码/权重/数据库）")
    return True, []


def ensure_local_ready():
    """本地推理就绪保障: 预检查 → 若解压作业运行中则转入其实时进度回报
    （不继续后续步骤）→ 完成后复查。返回 (ok, errors)。
    本地入口与 BO 入口共用此函数。"""
    ok, errors = preflight_check()
    state, jobid = db_prep_job_state()
    if state in ("pending", "running") and jobid:
        follow_db_prep_job(jobid)
        ok, errors = preflight_check()      # 解压结束后重新检查
    return ok, errors


def extract_scores_from_summary(summary_path):
    """从 _summary_confidences.json 提取评分（字段与服务器版结果一致）。"""
    try:
        with open(summary_path) as f:
            conf = json.load(f)

        iptm = conf.get("iptm", 0.0)
        chain_pair_iptm = conf.get("chain_pair_iptm", [])
        chain_pair_iptm_AB = 0.0
        if len(chain_pair_iptm) > 1 and len(chain_pair_iptm[0]) > 1:
            chain_pair_iptm_AB = chain_pair_iptm[0][1]

        return {
            "iptm": iptm,
            "ptm": conf.get("ptm", 0.0),
            "ranking_score": conf.get("ranking_score", 0.0),
            "chain_pair_iptm_AB": chain_pair_iptm_AB,
            "fraction_disordered": conf.get("fraction_disordered", 0.0),
            "has_clash": conf.get("has_clash", False),
        }
    except Exception as e:
        print(f"  ❌ 解析 {summary_path} 失败: {e}")
        return None


_TS_DIR_RE = re.compile(r"_\d{8}_\d{6}$")


def _summary_in(d):
    hits = glob.glob(os.path.join(d, "**", "*_summary_confidences.json"),
                     recursive=True)
    return hits[0] if hits else None


def find_summary_json(result_dir):
    """在 AF3 输出目录中查找 *_summary_confidences.json。

    AF3 向已存在且非空的输出目录写结果时, 会另建 <名>_YYYYMMDD_HHMMSS/ 兄弟
    目录以免覆盖（本流程为打包提交预先在 result_dir 写了 input.json, 故输出常落在
    兄弟目录）。因此同时搜索 result_dir 及其时间戳兄弟, 避免把已完成任务误判为
    "未完成"而重复提交重跑。"""
    s = _summary_in(result_dir)
    if s:
        return s
    for sib in sorted(glob.glob(result_dir + "_*_*")):
        if os.path.isdir(sib) and _TS_DIR_RE.search(os.path.basename(sib)):
            s = _summary_in(sib)
            if s:
                return s
    return None


def pack_result_zip(result_dir, job_name):
    """把完成的输出目录打包成 <任务名>.zip（与服务器版下载包命名一致）。"""
    zip_path = os.path.join(LOCAL_RESULTS_DIR, f"{job_name}.zip")
    if os.path.exists(zip_path):
        return zip_path
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(result_dir):
            for fn in files:
                if fn.startswith("slurm_") or fn == ".submitted":
                    continue
                full = os.path.join(root, fn)
                zf.write(full, os.path.join(job_name, os.path.relpath(full, result_dir)))
    return zip_path


def job_state(jobid):
    """查询任意 SLURM 作业状态 ∈ pending/running/done/failed/unknown。
    squeue 为活动作业权威来源（sacct 对新作业有记账延迟）。"""
    if not jobid:
        return "none"
    try:
        out = subprocess.run(["squeue", "-j", jobid, "-h", "-o", "%T"],
                             capture_output=True, text=True, timeout=15)
        q = out.stdout.strip().splitlines()
        if q:
            return "running" if q[0].strip() == "RUNNING" else "pending"
    except Exception:
        pass
    try:
        out = subprocess.run(["sacct", "-j", jobid, "-n", "-P", "-o", "State"],
                             capture_output=True, text=True, timeout=15)
        states = [s.strip() for s in out.stdout.splitlines() if s.strip()]
    except Exception:
        return "unknown"
    if not states:
        return "unknown"
    s = states[0]
    if s == "COMPLETED":
        return "done"
    if s in ("PENDING", "CONFIGURING", "COMPLETING"):
        return "pending"
    return "failed"


def queue_depth():
    """当前用户名下在途作业数。"""
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h"],
                             capture_output=True, text=True, timeout=15)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


def build_bundle_sbatch_cmd(bundle_id, bundle_dir):
    """构建提交打包推理作业的 sbatch 命令（返回 argv 列表）。

    作业脚本外置于 scripts/af3_bundle_runner.sh（两阶段容错逻辑见该文件头注释）;
    SLURM 资源指令走命令行参数, 作业配置走 --export 环境变量,
    Python 侧不内嵌任何 shell 脚本字符串。

    **提交的是 runner 的快照副本**（复制进 bundle 目录, 命名下划线开头以免被
    收割/打包逻辑当作任务）。原因: bash 边读边执行、按字节偏移回读脚本, 而
    编辑器写文件是**原地截断重写**（inode 不变）——若在作业运行期间修改
    scripts/ 下的 runner, 正在执行的作业会在下一条命令处读到错位内容而
    语法崩溃或行为失控。快照副本让作业与本仓库后续改动完全解耦
    （2026-10-04 实测踩到, 当时靠临时回滚救回运行中的作业）。"""
    work_dir = os.path.dirname(os.path.abspath(__file__))
    src_runner = os.path.join(SCRIPTS_DIR, "af3_bundle_runner.sh")
    runner = os.path.join(os.path.abspath(bundle_dir), "_runner_snapshot.sh")
    shutil.copyfile(src_runner, runner)          # 快照: 与仓库后续修改解耦
    os.chmod(runner, 0o755)
    n_gpu = LOCAL_GPUS_PER_JOB
    cpu_total = LOCAL_CPUS_PER_GPU * n_gpu            # 合计不超 QoS 的 cpu 上限
    mem_total = f"{int(LOCAL_MEMORY.rstrip('Gg')) * n_gpu}G"
    env_pairs = [
        f"AF3_CODE_DIR={AF3_CODE_DIR}",
        f"AF3_BUNDLE_DIR={os.path.abspath(bundle_dir)}",
        f"AF3_OUT_DIR={os.path.abspath(LOCAL_RESULTS_DIR)}",
        f"AF3_MODEL_DIR={LOCAL_MODEL_DIR}",
        f"AF3_DB_DIR={LOCAL_DB_DIR}",
        f"AF3_WORK_DIR={work_dir}",
        f"AF3_CONDA_ENV={CONDA_ENV}",
        f"AF3_FLASH_ATTN={FLASH_ATTENTION_IMPL}",
        f"AF3_NEED_XLA_7X_FLAG={'1' if NEED_XLA_7X_FLAG else '0'}",
        f"AF3_MSA_CPUS={MSA_CPUS_PER_GPU}",
        f"AF3_GPUS_PER_JOB={n_gpu}",
    ]
    if LOCAL_HMMER_DIR:
        env_pairs.append(f"AF3_HMMER_DIR={LOCAL_HMMER_DIR}")
    _check_export_pairs(env_pairs)
    return [
        "sbatch",
        f"--job-name=af3l_{bundle_id}",
        f"--output={os.path.abspath(bundle_dir)}/slurm_%j.out",
        f"--error={os.path.abspath(bundle_dir)}/slurm_%j.err",
        f"--partition={LOCAL_PARTITION}",
        f"--account={LOCAL_ACCOUNT}",
        "--nodes=1",
        f"--cpus-per-task={cpu_total}",
        f"--mem={mem_total}",
        f"--gres=gpu:{n_gpu}",
        f"--time={LOCAL_TIME}",
        "--export=ALL," + ",".join(env_pairs),
        runner,
    ]


# =====================================================================
# 单作业无限循环形态（推荐; 见 scripts/af3_loop_job.sh 文件头的取舍理由）
# ---------------------------------------------------------------------
# 旧形态是"登录节点常驻驱动 + 反复 sbatch 小包"。它的结构性缺陷：
#   ① 登录节点进程脆弱 —— 本机把所有交互进程塞进同一 cgroup
#      (/system.slice/sshd.service), setsid 逃不出去, 会话被清理即无声消失
#      （已发生两次, 其中一次 GPU 空转 8h22m、29 个结果滞留未入库）;
#   ② QoS MaxJobsPerUser=1 是**运行并发**上限 → 编排作业与计算作业互斥,
#      故也不能把编排做成一个独立的计算节点作业。
# 解法 = 把整个搜索放进**一个长驻作业内部循环**（optimize_peptide_node.py）:
#   分区 MaxTime=5 天、PreemptMode=OFF（不被抢占）, 且 --test-only 实测
#   48h/2天/3天/5天 的预计启动时间**完全相同** → 长时限不受精调度惩罚。
#   登录节点从此不留常驻进程, 只有一个 cron 看门狗负责"5 天到期/意外死亡
#   后再交一个作业"。
# =====================================================================

LOOP_JOB_NAME     = "pepopt-loop"
LOOP_SAFETY_HOURS = 2.0        # 作业时限 减去 该余量 = Python 侧时间预算
DEFAULT_MAX_HOURS_FALLBACK = 120.0   # 探测分区 MaxTime 失败时的回退值


def probe_partition_max_hours(fallback=DEFAULT_MAX_HOURS_FALLBACK):
    """读分区 MaxTime → 小时数（向下取整到整点）; 探测失败返回 fallback。"""
    try:
        out = subprocess.run(["scontrol", "show", "partition", LOCAL_PARTITION],
                             capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"MaxTime=(\d+)-(\d+):(\d+):(\d+)", out)
        if m:
            d, h, mi, s = (int(x) for x in m.groups())
            return d * 24 + h + mi // 60
        m = re.search(r"MaxTime=(\d+):(\d+):(\d+)", out)
        if m:
            h, mi, s = (int(x) for x in m.groups())
            return h + mi // 60
    except Exception:
        pass
    return fallback


def loop_job_active():
    """队列里是否已有无限循环作业（运行中或排队中）。"""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-n", LOOP_JOB_NAME, "-o", "%i"],
            capture_output=True, text=True, timeout=20).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def _check_export_pairs(env_pairs):
    """校验 `--export=a,b,c` 的各值不得含逗号。

    SLURM 的 --export 列表自身以**逗号**分隔, 若某个值内部含逗号（如
    `PEPOPT_TARGETS=HTR1A,BIN1`）, SLURM 会把它截断成 `PEPOPT_TARGETS=HTR1A`
    加一个孤立的 `BIN1` → 参数**静默丢失**且不报错。
    2026-10-04 实测踩到: 已跑起的 5 天作业因此只算 HTR1A, BIN1 约束全丢。
    故在提交前就报错, 不让它变成一个跑几天才被发现的数据事故。"""
    for kv in env_pairs:
        if "," in kv:
            raise ValueError(
                f"--export 的值不得含逗号（SLURM 以逗号分隔该列表, 会静默截断）: "
                f"{kv!r}\n    其余: {env_pairs}")
    return env_pairs


def build_loop_job_cmd(targets, cycle_seqs, max_hours=None):
    """构建"单个长驻循环作业"的 sbatch 命令（返回 argv 列表）。

    资源与 bundle 作业一致（受 QoS 的 gpu=1 / cpu=8 钳制）, 但时限取分区
    MaxTime（默认 5 天）, 并把 MaxTime-余量 作为 Python 侧时间预算下发,
    使作业能自适应地把最后一轮排满、到期前优雅退出（而非被 SLURM 掐断）。"""
    import peptide_db as db                       # 延迟导入: 保持本模块可独立使用
    p = db.paths(ensure=True)
    runs = os.path.join(p["root"], "runs")
    os.makedirs(runs, exist_ok=True)

    limit_h = max(1, int(max_hours or probe_partition_max_hours()))
    time_str = f"{limit_h}:00:00"
    budget_h = round(max(0.5, limit_h - LOOP_SAFETY_HOURS), 1)

    n_gpu = LOCAL_GPUS_PER_JOB
    cpu_total = LOCAL_CPUS_PER_GPU * n_gpu
    mem_total = f"{int(LOCAL_MEMORY.rstrip('Gg')) * n_gpu}G"
    # ⭐ 靶标用 '+' 而非 ',' 分隔（见 _check_export_pairs）; node 入口两种都认
    targets_s = "+".join(targets)
    env_pairs = [
        f"PEPOPT_DB_ROOT={p['root']}",
        f"PEPOPT_RESULTS_DIR={p['results']}",
        f"PEPOPT_WORK_DIR={os.path.dirname(os.path.abspath(__file__))}",
        f"PEPOPT_TARGETS={targets_s}",
        f"PEPOPT_CYCLE_SEQS={cycle_seqs}",
        f"PEPOPT_MAX_HOURS={budget_h}",
        f"PEPOPT_GPUS_PER_JOB={n_gpu}",           # banner 展示用（权威值）
        f"PEPOPT_TIME_REQUESTED={limit_h}h",
    ]
    _check_export_pairs(env_pairs)
    return [
        "sbatch",
        f"--job-name={LOOP_JOB_NAME}",
        f"--output={runs}/loop_%j.out",
        f"--error={runs}/loop_%j.err",
        f"--partition={LOCAL_PARTITION}",
        f"--account={LOCAL_ACCOUNT}",
        "--nodes=1",
        f"--cpus-per-task={cpu_total}",
        f"--mem={mem_total}",
        f"--gres=gpu:{n_gpu}",
        f"--time={time_str}",
        "--export=ALL," + ",".join(env_pairs),
        os.path.join(SCRIPTS_DIR, "af3_loop_job.sh"),
    ], budget_h


def submit_loop_job(targets, cycle_seqs, max_hours=None):
    """提交长驻循环作业; 已在队列中则不重复提交。返回 jobid 或 None。"""
    if not PREFLIGHT_OK:
        print("  ⏸️  预检查未通过, 跳过提交（请先补齐数据库/权重/工具条件）")
        return None
    active = loop_job_active()
    if active:
        print(f"  ⏭️  无限循环作业已在队列: {active}（不重复提交）")
        return None
    cmd, budget_h = build_loop_job_cmd(targets, cycle_seqs, max_hours)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jobid = out.stdout.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"  ❌ sbatch 提交失败: {(e.stderr or '').strip() or e}")
        return None
    except FileNotFoundError as e:
        print(f"  ❌ sbatch 提交失败: {e}")
        return None
    print(f"  🚀 已提交长驻循环作业 jobid={jobid}: "
          f"靶标={targets} | 每轮 {cycle_seqs} 序列 | 时间预算 {budget_h}h")
    return jobid


def estimate_tokens(input_json_path):
    """从 AF3 输入 JSON 估算 token 数（蛋白/核酸链按序列长度计）。
    用于提交前判断是否会超出 GPU 的 token 上限而 OOM。无法解析时返回 0（不拦截）。"""
    try:
        with open(input_json_path, encoding="utf-8") as f:
            job = json.load(f)
    except (OSError, ValueError):
        return 0
    total = 0
    for item in job.get("sequences", []) or []:
        for val in item.values():
            if not isinstance(val, dict):
                continue
            seq = val.get("sequence", "")
            if isinstance(seq, str):
                total += len(seq)
            elif isinstance(seq, list):        # 配体/修饰等非字符串条目
                total += len(seq)
    return total


def submit_pending_bundle():
    """把积压任务打包成一个作业提交（集群每用户在途作业有限, 打包提高吞吐）。
    自动跳过在途任务; 作业已终结但无结果的任务会清理标记后重新排队（失败自动重试）。
    token 数超过 GPU_MAX_TOKENS 的任务注定 OOM, 直接拦截不提交, 避免白烧机时。
    返回新提交的作业号, 无提交则 None。"""
    if not PREFLIGHT_OK:
        print("  ⏸️  预检查未通过, 跳过提交, 请先补齐数据库/工具条件")
        return None

    pending = []
    if os.path.isdir(LOCAL_RESULTS_DIR):
        for name in sorted(os.listdir(LOCAL_RESULTS_DIR)):
            d = os.path.join(LOCAL_RESULTS_DIR, name)
            if not os.path.isdir(d) or name.startswith("_"):
                continue
            if find_summary_json(d) or not os.path.isfile(os.path.join(d, "input.json")):
                continue
            # token 预检: 超出当前 GPU 官方上限者必然 OOM, 拦截并留痕（仅首次警告）
            oom_marker = os.path.join(d, OOM_SKIP_MARKER)
            tk = estimate_tokens(os.path.join(d, "input.json"))
            if tk > GPU_MAX_TOKENS:
                if not os.path.isfile(oom_marker):
                    with open(oom_marker, "w") as f:
                        f.write(f"{tk}\n")
                    print(f"  ⛔ {name}: {tk} tokens > 当前 GPU 上限 {GPU_MAX_TOKENS}"
                          "（V100 官方仅支持 ≤1280 tokens）, 必然显存溢出, 已跳过未提交。")
                    print("     对策: 换 A100/H100-80G 并调高 GPU_MAX_TOKENS;"
                          " 或把该靶标截断到与小肽相关的结构域（会改变约束语义, 需组内确认）。")
                continue
            marker = os.path.join(d, ".submitted")
            if os.path.isfile(marker):
                try:
                    jid = open(marker).read().strip().splitlines()[0]
                except OSError:
                    jid = ""
                st = job_state(jid)
                if st in ("pending", "running") and not FORCE_RESUBMIT:
                    continue                  # 在途, 不重复打包
                os.remove(marker)             # 终结但无结果 → 失败, 重新排队
            pending.append((name, d))

    if not pending:
        return None
    if queue_depth() >= MAX_INFLIGHT:
        print(f"  ⏸️  在途作业已达上限 {MAX_INFLIGHT}, 暂不提交（已有作业完成后重跑即续作）")
        return None

    bundle = pending[:BUNDLE_SIZE]
    # 攒批: 未满一包且队列仍有作业在跑时, 继续积压以提高单作业吞吐
    if len(bundle) < BUNDLE_SIZE and queue_depth() > 0:
        print(f"  📦 攒批中: 已积压 {len(pending)}/{BUNDLE_SIZE} 个任务"
              "（满包或队列空闲时提交）")
        return None
    bundle_id = f"bundle_{int(time.time())}"
    bundle_dir = os.path.join(LOCAL_RESULTS_DIR, "_bundles", bundle_id)
    os.makedirs(bundle_dir, exist_ok=True)
    for name, d in bundle:
        shutil.copy(os.path.join(d, "input.json"),
                    os.path.join(bundle_dir, f"{name}.json"))
    cmd = build_bundle_sbatch_cmd(bundle_id, bundle_dir)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        jobid = out.stdout.strip().split()[-1]
    except subprocess.CalledProcessError as e:
        print(f"  ❌ sbatch 提交失败: {(e.stderr or '').strip() or e}")
        return None
    except FileNotFoundError as e:
        print(f"  ❌ sbatch 提交失败: {e}")
        return None

    for name, d in bundle:
        with open(os.path.join(d, ".submitted"), "w") as f:
            f.write(jobid)
    print(f"  🚀 打包提交 {len(bundle)} 个任务 → jobid={jobid}: "
          f"{[n for n, _ in bundle]}")
    return jobid


def write_input_json(peptide_seq, target_seq, job_name, result_dir):
    """写入单任务的 AF3 输入 JSON（提交由打包机制统一处理）。"""
    input_json = os.path.join(result_dir, "input.json")
    job = create_af3_json(peptide_seq, target_seq, job_name)
    with open(input_json, "w") as f:
        json.dump(job, f, indent=2)
    return input_json


def local_predict_and_score(peptide_seq, target_seq, target_name, out_dir,
                            auto_submit=True, verbose=True):
    """
    服务器版 predict_and_score 的本地替身：
      - 已有结果 → 提取评分并返回 (score, iptm, zip路径)
      - 已提交未完成 → 返回 (None, None, None)，等下次重跑
      - 未提交 → 生成并提交本地作业，返回 (None, None, None)
    verbose=False 时不打印逐靶标评分行（无限搜索入口自己有收割日志）。
    """
    job_name = os.path.basename(out_dir).lower()   # AF3 的 sanitised_name() 转小写, 必须一致
    result_dir = os.path.join(LOCAL_RESULTS_DIR, job_name)
    os.makedirs(result_dir, exist_ok=True)

    # ---- 已完成：提取评分 ----
    summary = find_summary_json(result_dir)
    if summary:
        scores = extract_scores_from_summary(summary)
        if scores:
            iptm = scores["iptm"]
            score = round(1.0 - iptm, 4)
            if verbose:
                print(f"  📊 [{target_name}] 该靶标界面 ipTM={iptm:.3f} | "
                      f"score={score} (=1-ipTM, 优化目标, 越小越好) | "
                      f"ranking={scores['ranking_score']:.3f} (AF3综合分, 仅供参考)")
            # 结果可能在 AF3 的时间戳兄弟目录: 以含 summary 的顶层目录为打包源,
            # 保证 zip 非空（规范目录仅 input.json 时尤其重要）。
            src_root = result_dir
            try:
                top = os.path.relpath(summary, LOCAL_RESULTS_DIR).split(os.sep)[0]
                cand = os.path.join(LOCAL_RESULTS_DIR, top)
                if os.path.isdir(cand):
                    src_root = cand
            except ValueError:
                pass
            zip_path = pack_result_zip(src_root, job_name)
            return score, iptm, zip_path

    # ---- 未完成：写入输入并触发打包提交（含失败自动重试）----
    write_input_json(peptide_seq, target_seq, job_name, result_dir)
    marker = os.path.join(result_dir, ".submitted")
    if os.path.isfile(marker) and not FORCE_RESUBMIT:
        try:
            jid = open(marker).read().strip().splitlines()[0]
        except OSError:
            jid = ""
        if job_state(jid) in ("pending", "running"):
            print(f"  ⏳ [{target_name}] {job_name} 在集群上计算中, 等待结果…")
            return None, None, None
    submit_pending_bundle()      # 本任务连同其他积压任务一起打包提交
    return None, None, None


def job_queue_info(jobid):
    """查询作业的排队/运行信息: 状态 + SLURM 预计启动时间。"""
    state = job_state(jobid)
    start = ""
    try:
        out = subprocess.run(["squeue", "--start", "-j", jobid, "-h",
                              "-o", "%S %r"],
                             capture_output=True, text=True, timeout=15)
        parts = out.stdout.strip().split(None, 1)
        if parts:
            start = parts[0]
    except Exception:
        pass
    desc = {"running": "运行中", "pending": f"排队中(预计启动 {start})" if start else "排队中",
            "done": "已结束", "failed": "已失败"}.get(state, state)
    return state, desc


def check_local_status():
    """汇报本地任务的完成/进行中状态（含排队位置与预计启动时间）。"""
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    completed, running = [], []
    for name in sorted(os.listdir(LOCAL_RESULTS_DIR)):
        d = os.path.join(LOCAL_RESULTS_DIR, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        if find_summary_json(d):
            completed.append(name)
        elif os.path.exists(os.path.join(d, ".submitted")):
            running.append(name)

    print("\n📊 本地任务状态报告")
    print(f"   已完成: {len(completed)} | 运行中/排队中: {len(running)}")
    shown = set()
    for n in running[:20]:
        try:
            jid = open(os.path.join(LOCAL_RESULTS_DIR, n, ".submitted")).read().strip().splitlines()[0]
        except OSError:
            jid = ""
        _, desc = job_queue_info(jid)
        key = (jid, desc)
        suffix = f" [{jid} {desc}]" if jid and key not in shown else ""
        shown.add(key)
        print(f"     ⏳ {n}{suffix}")
    return len(completed), len(running)


if __name__ == "__main__":
    # ---- 计算节点上的数据库准备作业入口 ----
    if "--prepare-db" in sys.argv:
        ok = prepare_database()
        sys.exit(0 if ok else 1)

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("🧬 AlphaFold3 本地推理 — 蒙特卡洛肽优化")
    print("=" * 60)

    # 就绪保障: 预检查 + 解压作业实时进度跟随（运行中则等待，不继续后续步骤）
    ok, errors = ensure_local_ready()

    if not ok:
        print("\n⚠️  存在未满足的条件，仍将尝试运行流程：")
        print("   - 资源就绪前，已完成的旧结果仍可被评分复用；")
        print("   - 新的提交会跳过（见上方提示），请补齐条件后重跑。\n")

    # 蒙特卡洛主循环（后端无关的公共循环 + 本地推理预测函数）
    best_seq, best_score = run_monte_carlo(local_predict_and_score, ROOT_DIR)

    check_local_status()
    if best_seq is None:
        print("\n⏳ 尚无可用结果。等集群作业完成后重新运行本脚本即可继续。")
        print("   查看作业: squeue -u $USER")
    else:
        print("\n🏆 优化完成，详见 output_af3/202_htr1a/")
