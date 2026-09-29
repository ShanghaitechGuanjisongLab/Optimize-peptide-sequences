#!/bin/bash
#SBATCH --job-name=pepopt202-bo
#SBATCH --output=pepopt_bo_%j.out
#SBATCH --error=pepopt_bo_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=bme_gpupub
#SBATCH --account=v-jiamh

# =====================================================
# 小肽序列贝叶斯优化（服务器版后端）— SLURM 委托脚本
# -----------------------------------------------------
# 注意: optimize_peptide_BO.py 的默认后端是本地集群推理，
# 请直接在登录节点运行:
#   python optimize_peptide_BO.py
# 它会自动为每个候选提交独立的 GPU 作业（无每日配额限制），
# 支持断点续跑；无需本委托脚本。
#
# 仅当确需改用 AlphaFold Server 云端预测（手动上传/下载，
# 受每日配额限制）时，才用本脚本把轻量编排提交到计算节点:
#   sbatch run_peptide_bo.sbatch
# =====================================================

echo "作业开始: $(date) | 节点: $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base    # 依赖已齐, 计算节点无外网, 不能联网 pip 安装

cd "$SLURM_SUBMIT_DIR" || exit 1

# 贝叶斯优化入口(服务器版后端); 预算/批量/候选库可调, 例如:
#   python optimize_peptide_BO.py --server --budget 120 --batch 8 --pool 1500
# -u 关闭缓冲, 使日志可实时跟踪 (tail -f pepopt_bo_%j.out)
python -u optimize_peptide_BO.py --server

echo "作业结束: $(date)"
