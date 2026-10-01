#!/usr/bin/env bash
# =============================================================================
# AlphaFold3 序列数据库解压作业脚本（独立文件）
#
# 由 optimize_peptide_local.py 的 submit_db_prep_job() 提交：
#     sbatch <SLURM 选项> --export=ALL,PEPOPT_WORK_DIR=...,... \
#         scripts/db_prep_runner.sh
#
# 实际的磁盘空间检查与逐项解压逻辑在 Python 侧
# （optimize_peptide_local.py --prepare-db，见 prepare_database()），
# 本脚本只负责在计算节点上准备好环境并调用它，因此提交后即可断开 SSH。
# 用 python -u 关闭输出缓冲，使进度可被 `tail -f db_prep_<jobid>.out` 实时看到。
#
# 必需环境变量：
#   PEPOPT_WORK_DIR   仓库目录（含 optimize_peptide_local.py）
# 可选环境变量：
#   PEPOPT_CONDA_ENV=base   解压用 conda 环境（需含 zstd 工具或能调用到）
#   PEPOPT_CONDA_SH         conda.sh 路径（留空则按常见安装位置探测）
#   PEPOPT_ENTRY=optimize_peptide_local.py
# =============================================================================

set -uo pipefail

: "${PEPOPT_WORK_DIR:?缺少环境变量 PEPOPT_WORK_DIR}"
PEPOPT_CONDA_ENV="${PEPOPT_CONDA_ENV:-base}"
PEPOPT_CONDA_SH="${PEPOPT_CONDA_SH:-}"
PEPOPT_ENTRY="${PEPOPT_ENTRY:-optimize_peptide_local.py}"

echo "数据库解压作业开始: $(date) | 节点: $(hostname)"

# ---- 激活环境 ----
if [ -z "$PEPOPT_CONDA_SH" ]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh"; do
        if [ -f "$c" ]; then PEPOPT_CONDA_SH="$c"; break; fi
    done
fi
if [ -n "$PEPOPT_CONDA_SH" ] && [ -f "$PEPOPT_CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$PEPOPT_CONDA_SH"
    if ! conda activate "$PEPOPT_CONDA_ENV"; then
        echo "❌ 激活 conda 环境 '$PEPOPT_CONDA_ENV' 失败" >&2
        exit 2
    fi
else
    echo "⚠️ 未定位到 conda.sh，沿用当前环境的 python" >&2
fi

cd "$PEPOPT_WORK_DIR" || { echo "❌ 无法进入 $PEPOPT_WORK_DIR" >&2; exit 2; }

if [ ! -f "$PEPOPT_ENTRY" ]; then
    echo "❌ 找不到入口脚本: $PEPOPT_ENTRY" >&2
    exit 2
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "✅ 提示: 当前 PATH 无 zstd, prepare_database() 会自行处理并报错中止" >&2
fi

# -u: 关闭缓冲, 让解压进度能被 tail -f 实时看到
python -u "$PEPOPT_ENTRY" --prepare-db
rc=$?

echo "数据库解压作业结束: $(date) | 退出码: $rc"
exit "$rc"
