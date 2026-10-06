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
#   AF3_MSA_CPUS=6           jackhmmer / nhmmer 各用的 CPU 数（每条 GPU 流各一份）
#   AF3_NEED_XLA_7X_FLAG=1   1=注入 7.x 算力 GPU 必需的 XLA_FLAGS
#   AF3_GPUS_PER_JOB=1       单作业 GPU 数；>1 时阶段1把任务按靶标分组
#                            拆到各卡并行（QoS 限作业数不限每作业 GPU 数）
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
# 单作业可用 GPU 数: >1 时阶段1把任务按靶标分组拆到各卡并行（见 stage1_parallel）
AF3_GPUS_PER_JOB="${AF3_GPUS_PER_JOB:-1}"

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
    local f sib
    for f in "$AF3_OUT_DIR/$1"/*_summary_confidences.json; do
        [ -e "$f" ] && return 0
    done
    # AF3 向已存在的输出目录写结果时会另建 <名>_YYYYMMDD_HHMMSS/ 兄弟目录,
    # 一并识别, 否则已完成任务会被当"无结果"重跑
    for sib in "$AF3_OUT_DIR/$1"_[0-9]*_[0-9]*; do
        [ -d "$sib" ] || continue
        for f in "$sib"/*_summary_confidences.json; do
            [ -e "$f" ] && return 0
        done
    done
    return 1
}

# consolidate_outputs —— 把 AF3 生成的 <名>_<时间戳>/ 兄弟目录折叠进规范
# 目录 <名>/, 使规范目录成为唯一结果位置（登录节点据此收割/去重/打包,
# 也不再对已完成任务重跑）。多次重跑时任意一份含 summary 的兄弟保留到规范目录。
consolidate_outputs() {
    local sib base canon entry b
    for sib in "$AF3_OUT_DIR"/*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]; do
        [ -d "$sib" ] || continue
        base="$(basename "$sib")"
        canon="$AF3_OUT_DIR/${base%_*_*}"      # 去掉末尾 _<8位>_<6位>
        mkdir -p "$canon"
        if ! compgen -G "$canon/*_summary_confidences.json" > /dev/null; then
            for entry in "$sib"/*; do
                [ -e "$entry" ] || continue
                b="$(basename "$entry")"
                case "$b" in .submitted|.harvested|.oom_skipped|slurm_*) continue ;; esac
                [ -e "$canon/$b" ] || mv -- "$entry" "$canon/"
            done
        fi
        rm -rf -- "$sib"
    done
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

# ---- 多卡阶段1: 把包内任务按靶标后缀分组拆到 AF3_GPUS_PER_JOB 张卡并行 ----
# 分组策略: 出现 ≥2 种靶标时同靶标同流（MSA 每流只建一次、流内全复用,
# 与串行整包的缓存收益一致）; 仅单一靶标时轮转均分到各卡（每流各建一次
# MSA, 但并行把壁钟时间减半以上, 净赚）。
# 卡号优先用 SLURM 已注入的 CUDA_VISIBLE_DEVICES（即本作业分到的真实卡）,
# 未注入时才退回 0..N-1, 避免在共享节点上踩到其他作业的卡。
stage1_parallel() {
    local ngpu="$1"
    local split_root="$AF3_BUNDLE_DIR/_split"
    rm -rf "$split_root"
    local i
    for ((i = 0; i < ngpu; i++)); do mkdir -p "$split_root/g$i"; done

    local -a files=("$AF3_BUNDLE_DIR"/*.json)
    local -a suffixes=()
    local jf base suffix k found
    for jf in "${files[@]}"; do
        base="$(basename "$jf" .json)"; suffix="${base##*_}"
        found=""
        for ((k = 0; k < ${#suffixes[@]}; k++)); do
            [ "${suffixes[$k]}" = "$suffix" ] && { found=1; break; }
        done
        [ -z "$found" ] && suffixes+=("$suffix")
    done

    if [ "${#suffixes[@]}" -ge 2 ]; then
        local -A smap=()
        for ((k = 0; k < ${#suffixes[@]}; k++)); do smap["${suffixes[$k]}"]=$((k % ngpu)); done
        for jf in "${files[@]}"; do
            base="$(basename "$jf" .json)"; suffix="${base##*_}"
            cp "$jf" "$split_root/g${smap[$suffix]}/"
        done
        echo "  分组: ${#suffixes[@]} 种靶标同靶同流 → $ngpu 条 GPU 流"
    else
        i=0
        for jf in "${files[@]}"; do
            cp "$jf" "$split_root/g$((i % ngpu))/"; i=$((i + 1))
        done
        echo "  分组: 单一靶标 ${#files[@]} 任务轮转均分 → $ngpu 条 GPU 流"
    fi

    local -a gpu_ids=()
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    else
        for ((i = 0; i < ngpu; i++)); do gpu_ids+=("$i"); done
    fi
    [ "${#gpu_ids[@]}" -lt "$ngpu" ] && ngpu="${#gpu_ids[@]}"

    local -a pids=() used=()
    for ((i = 0; i < ngpu; i++)); do
        [ -z "$(ls -A "$split_root/g$i" 2>/dev/null)" ] && continue
        echo "  ▶ 流[i=${gpu_ids[$i]}]: $(cd "$split_root/g$i" && printf '%s ' *.json)"
        CUDA_VISIBLE_DEVICES="${gpu_ids[$i]}" \
            python "$RUN_AF3" "--input_dir=$split_root/g$i" \
            --output_dir="$AF3_OUT_DIR" \
            --model_dir="$AF3_MODEL_DIR" \
            --db_dir="$AF3_DB_DIR" \
            --flash_attention_implementation="$AF3_FLASH_ATTN" \
            --jackhmmer_n_cpu="$AF3_MSA_CPUS" \
            --nhmmer_n_cpu="$AF3_MSA_CPUS" \
            >"$split_root/g$i.log" 2>&1 &
        pids+=("$!")
        used+=("$i")
    done

    local rc_all=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || rc_all=1
    done
    for i in "${used[@]}"; do
        echo "---- 流 g$i 日志尾部 ----"
        tail -n 6 "$split_root/g$i.log" 2>/dev/null
    done
    return $rc_all
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
    consolidate_outputs        # 折叠可能残留的时间戳兄弟目录, 令规范目录成为唯一位置
    exit 0
fi
echo "本包待跑任务 ${#remaining[@]} 个: $(printf '%s ' "${remaining[@]##*/}" | tr ' ' '\n' | sed 's/\.json$//' | tr '\n' ' ')"

if [ "$AF3_GPUS_PER_JOB" -gt 1 ] && [ ${#remaining[@]} -gt 1 ]; then
    echo "=== 阶段1: 多卡并行（$AF3_GPUS_PER_JOB 张卡, 任务按靶标分组）==="
    if stage1_parallel "$AF3_GPUS_PER_JOB"; then rc1=0; else rc1=1; fi
else
    echo "=== 阶段1: 整包运行（共享同靶标 MSA 缓存）==="
    if run_once "--input_dir=$AF3_BUNDLE_DIR"; then rc1=0; else rc1=1; fi
fi

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

# 把本次产生的 <名>_<时间戳>/ 输出折叠进规范目录, 供登录节点统一收割与去重
consolidate_outputs
exit "$rc"
