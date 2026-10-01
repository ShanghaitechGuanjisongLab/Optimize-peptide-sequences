#!/usr/bin/env bash
# =============================================================================
# AlphaFold3 打包推理作业脚本（独立文件）
#
# 由 optimize_peptide_local.py 的 submit_pending_bundle() 提交：
#     sbatch <SLURM 选项> --export=ALL,AF3_CODE_DIR=...,... \
#         scripts/af3_bundle_runner.sh
# SLURM 资源指令走命令行参数，作业配置走环境变量，本脚本只负责作业逻辑，
# 因此没有任何 shell 代码内嵌在 Python 字符串里。
#
# 【两阶段容错设计】
#   阶段1  用 --input_dir 整包一次跑。AF3 的 MSA 缓存（data/pipeline.py 里的
#          functools.cache）作用域是**单个 python 进程**：同一靶标的 chain B MSA
#          第二次起仅需 ~0.06 s，而首次可达数千秒，整包跑能省下大量重复 MSA。
#   阶段2  阶段1 中断时，逐任务用 --json_path 补跑，每个任务独立容错。
#          必要性：AF3 主循环（run_alphafold.py 的 for fold_input 循环）**没有
#          try/except**，单任务异常（如显存 OOM）会终止整个进程，导致同包排在
#          它之后的任务全部得不到执行（真实踩过的坑）。
#   两阶段开始前都把已出结果的任务移入 _done/ 子目录，故 SLURM 重试或阶段2
#   接管时都不会重复计算。_done/ 是子目录，而 AF3 用非递归 glob('*.json') +
#   is_file() 加载输入，不会被误读。
#
# 必需环境变量（未设置即报错退出）：
#   AF3_CODE_DIR    AF3 代码目录（含 run_alphafold.py）
#   AF3_BUNDLE_DIR  本包任务目录（内含多个 <任务名>.json）
#   AF3_OUT_DIR     run_alphafold.py 的 --output_dir
#   AF3_MODEL_DIR   模型权重目录（含 af3.bin）
#   AF3_DB_DIR      序列数据库目录
#   AF3_WORK_DIR    仓库工作目录（AF3_OUT_DIR 为相对路径时在此 cd 后使用）
#
# 可选环境变量（含默认值）：
#   AF3_CONDA_ENV=af3_old    推理用 conda 环境
#   AF3_CONDA_SH             conda.sh 路径（留空则按常见安装位置探测）
#   AF3_HMMER_DIR            hmm 工具目录（留空则沿用节点 PATH）
#   AF3_FLASH_ATTN=xla       flash attention 实现；CUDA 算力 7.x 必须为 xla
#   AF3_MSA_CPUS=6           jackhmmer / nhmmer 各用的 CPU 数
#   AF3_NEED_XLA_7X_FLAG=1   1=注入 7.x 算力 GPU 必需的 XLA_FLAGS
# =============================================================================

set -uo pipefail
shopt -s nullglob

: "${AF3_CODE_DIR:?缺少环境变量 AF3_CODE_DIR}"
: "${AF3_BUNDLE_DIR:?缺少环境变量 AF3_BUNDLE_DIR}"
: "${AF3_OUT_DIR:?缺少环境变量 AF3_OUT_DIR}"
: "${AF3_MODEL_DIR:?缺少环境变量 AF3_MODEL_DIR}"
: "${AF3_DB_DIR:?缺少环境变量 AF3_DB_DIR}"
: "${AF3_WORK_DIR:?缺少环境变量 AF3_WORK_DIR}"

AF3_CONDA_ENV="${AF3_CONDA_ENV:-af3_old}"
AF3_CONDA_SH="${AF3_CONDA_SH:-}"
AF3_HMMER_DIR="${AF3_HMMER_DIR:-}"
AF3_FLASH_ATTN="${AF3_FLASH_ATTN:-xla}"
AF3_MSA_CPUS="${AF3_MSA_CPUS:-6}"
AF3_NEED_XLA_7X_FLAG="${AF3_NEED_XLA_7X_FLAG:-1}"

RUN_AF3="${AF3_CODE_DIR%/}/run_alphafold.py"

echo "AF3 打包推理开始: $(date) | 节点: $(hostname)"

if [ ! -f "$RUN_AF3" ]; then
    echo "❌ 找不到 AF3 入口: $RUN_AF3" >&2
    exit 2
fi

# ---- 激活推理环境 ----
if [ -z "$AF3_CONDA_SH" ]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh"; do
        if [ -f "$c" ]; then AF3_CONDA_SH="$c"; break; fi
    done
fi
if [ -n "$AF3_CONDA_SH" ] && [ -f "$AF3_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$AF3_CONDA_SH"
    if ! conda activate "$AF3_CONDA_ENV"; then
        echo "❌ 激活 conda 环境 '$AF3_CONDA_ENV' 失败" >&2
        exit 2
    fi
else
    echo "⚠️ 未定位到 conda.sh，沿用当前环境的 python" >&2
fi
[ -n "$AF3_HMMER_DIR" ] && export PATH="$AF3_HMMER_DIR:$PATH"

# ---- CUDA 算力 7.x GPU (V100/T4 等) 必须禁用该 HLO 融合通道, 否则 JAX 启动即报错 ----
if [ "$AF3_NEED_XLA_7X_FLAG" = "1" ]; then
    export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

# ---- 官方推荐的 unified memory: 显存不足时溢出到主机内存以防 OOM（代价是变慢）----
# 见 alphafold3/docs/performance.md 的 "Unified Memory" 一节
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=3.2

cd "$AF3_WORK_DIR" || { echo "❌ 无法进入 $AF3_WORK_DIR" >&2; exit 2; }

# has_result <任务名> —— 判断该任务是否已出结果
has_result() {
    local f
    for f in "$AF3_OUT_DIR/$1"/*_summary_confidences.json; do
        [ -e "$f" ] && return 0
    done
    return 1
}

# run_once <"--input_dir=DIR" | "--json_path=FILE">
run_once() {
    python "$RUN_AF3" "$1" \
        --output_dir="$AF3_OUT_DIR" \
        --model_dir="$AF3_MODEL_DIR" \
        --db_dir="$AF3_DB_DIR" \
        --flash_attention_implementation="$AF3_FLASH_ATTN" \
        --jackhmmer_n_cpu="$AF3_MSA_CPUS" \
        --nhmmer_n_cpu="$AF3_MSA_CPUS"
}

# ---- 已出结果的任务移出待跑集 ----
done_dir="$AF3_BUNDLE_DIR/_done"
mkdir -p "$done_dir"
for jf in "$AF3_BUNDLE_DIR"/*.json; do
    base="$(basename "$jf" .json)"
    if has_result "$base"; then
        echo "已有结果, 移出待跑集: $base"
        mv "$jf" "$done_dir/"
    fi
done

remaining=("$AF3_BUNDLE_DIR"/*.json)
if [ ${#remaining[@]} -eq 0 ]; then
    echo "AF3 打包推理结束: $(date) | 本包任务均已有结果, 无需运行"
    exit 0
fi
echo "本包待跑任务 ${#remaining[@]} 个: $(printf '%s ' "${remaining[@]##*/}" | tr ' ' '\n' | sed 's/\.json$//' | tr '\n' ' ')"

echo "=== 阶段1: 整包运行（共享同靶标 MSA 缓存）==="
if run_once "--input_dir=$AF3_BUNDLE_DIR"; then rc1=0; else rc1=1; fi

rc="$rc1"
if [ "$rc1" -ne 0 ]; then
    echo "=== 阶段2: 阶段1 中断（退出码 $rc1）, 逐任务容错补跑 ==="
    nfail=0
    for jf in "$AF3_BUNDLE_DIR"/*.json; do
        base="$(basename "$jf" .json)"
        if has_result "$base"; then
            echo "  ✓ 已有结果, 跳过: $base"
            continue
        fi
        echo "  ▶ 补跑: $base"
        if ! run_once "--json_path=$jf"; then
            echo "  ✗ 失败（已隔离, 不影响同包其它任务）: $base"
            nfail=$((nfail + 1))
        fi
    done
    rc="$nfail"
fi

echo "AF3 打包推理结束: $(date) | 阶段1退出码: $rc1 | 最终失败任务数: $rc"
exit "$rc"
