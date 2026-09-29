#!/bin/bash
#SBATCH --job-name=pepopt202
#SBATCH --output=pepopt_%j.out
#SBATCH --error=pepopt_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=bme_gpupub
#SBATCH --account=v-jiamh

# =====================================================
# 小肽序列蒙特卡洛优化 — SLURM 委托脚本
# -----------------------------------------------------
# 说明:
#   本脚本的"重计算"(结构预测)在 AlphaFold Server 云端
#   完成, 本地仅做 JSON 生成 / 评分提取 / 蒙特卡洛优化,
#   计算量极小, 用 1 核 4G 内存即可。
#
# 提交:  sbatch run_peptide.sbatch
# 查看:  squeue -u $USER
# =====================================================

echo "作业开始: $(date) | 节点: $(hostname)"

# 初始化 conda 并激活 base 环境（依赖已齐: numpy/pandas/matplotlib/openpyxl）
# 注意: 计算节点无外网, 不能依赖 pip 联网安装
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd "$SLURM_SUBMIT_DIR" || exit 1

# -u 关闭缓冲, 使日志可实时跟踪 (tail -f pepopt_%j.out)
python -u optimize_peptide_AF3.py

echo "作业结束: $(date)"
