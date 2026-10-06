#!/usr/bin/env bash
# =============================================================================
# 「单个长驻循环作业」的 cron 自愈看门狗
#
# 与旧 scripts/driver_watchdog.sh 的区别（架构换代）:
#   旧: 监督**登录节点上的常驻 Python 驱动**。它脆弱, 因为本机把所有交互进程
#       塞进同一个 cgroup /system.slice/sshd.service 且 loginctl 查不到会话,
#       setsid 逃不出去 → 会话被清理时驱动无声消失（已发生两次, 一次 GPU
#       空转 8h22m、29 个结果滞留未入库）。
#   新: 监督**计算节点上的长驻 SLURM 作业**（optimize_peptide_node.py,
#       由 scripts/af3_loop_job.sh 包装, 时限 = 分区 MaxTime 5 天）。
#       搜索循环整个跑在作业内部, 计算节点不受登录会话清理影响, 也不会被
#       抢占（PreemptMode=OFF）。看门狗职责退化为"队列里没有它了就再交一个"
#       —— 正常情况下每 5 天才动作一次。
#
# 为何仍需看门狗: 作业跑满 5 天时限会正常退出; 也可能因节点故障/被管理员
# 取消而中断。此时需要有人续交下一个作业, 而登录节点上不该常驻任何进程。
# crond 运行在自己的 service cgroup, 天然免疫 sshd 会话清理, 是唯一可靠宿主。
#
# 安装: ./run.sh watchdog   （每 5min; cron 的 PATH 极简, run.sh 会写入 SLURM 路径）
# 卸载: ./run.sh no-watchdog
# 停用: ./run.sh stop  → 写 runs/loop.disabled, 本脚本见到即不动作
# =============================================================================
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_ROOT="${PEPOPT_DB_ROOT:-/public_bme2/Share200T/管吉松/peptide_opt_db}"
RUNS_DIR="$DB_ROOT/runs"
DISABLED="$RUNS_DIR/loop.disabled"
LOCK="$RUNS_DIR/loop_watchdog.lock"
WLOG="$RUNS_DIR/loop_watchdog.log"
JOB_NAME="pepopt-loop"
TARGETS="${PEPOPT_TARGETS:-HTR1A,BIN1}"
CYCLE_SEQS="${PEPOPT_CYCLE_SEQS:-24}"

cd "$REPO" || exit 1
mkdir -p "$RUNS_DIR"

log() { echo "$(date '+%F %T') [loop-watchdog] $*" >> "$WLOG"; }

# cron 环境不含 SLURM 与 conda; 兜底补全（run.sh 安装时也会显式写 PATH=）
[[ ":$PATH:" == *"/slurm/bin:"* ]] || export PATH="/opt/gridview/slurm/bin:$PATH"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
[[ -f "$CONDA_SH" ]] && source "$CONDA_SH" && conda activate base 2>/dev/null

# ---- 并发保护 ----
exec 9>"$LOCK" 2>/dev/null || exit 0
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || exit 0
fi

# ---- 用户显式停用 → 尊重, 不提交 ----
[[ -f "$DISABLED" ]] && exit 0

# ---- 队列里已有循环作业? ----
alive="$(squeue -u "${USER:-$LOGNAME}" -h -n "$JOB_NAME" -o '%i' 2>/dev/null | tr -d ' ')"
if [[ -n "$alive" ]]; then
  # 正常路径: 什么都不做, 也不刷日志（避免 5 天里堆出上万行空记录）
  exit 0
fi

log "队列中无 $JOB_NAME 作业, 准备续交（靶标=$TARGETS, 每轮 $CYCLE_SEQS 序列）"

# 预检查由 submit_loop_job() 内部把关（PREFLIGHT_OK 为假时它会拒绝提交）。
# 这里把 stdout/stderr 一并留痕到日志, 便于事后排查"为什么没交上去"。
out="$(python -u - <<PY 2>&1
import re
import sys
sys.path.insert(0, "$REPO")
import optimize_peptide_local as loc
loc.ensure_local_ready()                                  # 预检查 + 数据库解压跟进
# 分隔符兼容 ',' 与 '+'：经 SLURM --export 下发时只能用 '+'（逗号会被截断）
targets = [t for t in re.split(r"[,+]", "$TARGETS") if t]
print("submit:", loc.submit_loop_job(targets, $CYCLE_SEQS))
PY
)"
rc=$?
echo "$out" | tail -6 | sed 's/^/    | /' >> "$WLOG"
if echo "$out" | grep -qE "submit: [0-9]+"; then
  log "已续交, jobid=$(echo "$out" | grep -oE 'submit: [0-9]+' | awk '{print $2}')"
else
  log "提交未成功(rc=$rc), 下个周期重试"
fi
exit 0
