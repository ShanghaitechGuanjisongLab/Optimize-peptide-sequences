#!/usr/bin/env bash
# =============================================================================
# 无限搜索驱动 —— cron 自愈看门狗
#
# 为什么需要它（实测结论, 2026-10-04）:
#   登录节点上所有用户交互进程（含 VS Code 的 sshd 会话）都在同一个
#   cgroup `/system.slice/sshd.service` 里。本机的 loginctl 里查不到任何
#   会话（who / loginctl list-sessions 均空）, 因此 `setsid` 只能换掉
#   会话与进程组, **逃不出这个 cgroup**：一旦该 sshd 会话被清理（关闭
#   VS Code 窗口、连接重建等）, cgroup 内进程会被整体回收。
#   后果是驱动进程**无声消失**（无异常栈、无优雅退出日志）,
#   而 GPU 作业跑完后没人提交下一包、也没人收割结果 → GPU 长时间空转、
#   已完成结果滞留磁盘不入库。已实际发生两次（10-02 12:19、10-03 20:08,
#   第二次导致 GPU 空转 8.3h、29 个结果未及时入库）。
#
#   crond 运行在自己的 service cgroup, 不受 sshd 会话清理影响,
#   故由它定期拉起驱动是唯一可靠的自愈手段（无需 root）。
#
# 行为:
#   - 每 5 分钟被 cron 调用一次（见 ./run.sh watchdog 安装）;
#   - 用户显式 ./run.sh stop 后**不会**擅自复活（靠 infinite.disabled 标记）;
#   - flock 防并发: 上一次拉起还没完成就直接退出;
#   - 存活判定看 cmdline（防 PID 复用误判）;
#   - 所有动作追加进 runs/watchdog.log 便于事后审计。
#
# cron 环境要点（安装时已处理, 改动此脚本需注意）:
#   cron 的 PATH 极简, 不含 SLURM 与 conda → 驱动会调不到 sbatch/squeue。
#   故 ./run.sh watchdog 会把登录时的完整 PATH 写进 crontab 的 PATH= 行。
# =============================================================================
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_ROOT="${PEPOPT_DB_ROOT:-/public_bme2/Share200T/管吉松/peptide_opt_db}"
RUNS_DIR="$DB_ROOT/runs"
PID_FILE="$RUNS_DIR/infinite.pid"
DISABLED="$RUNS_DIR/infinite.disabled"
LOCK="$RUNS_DIR/watchdog.lock"
WLOG="$RUNS_DIR/watchdog.log"

cd "$REPO" || exit 1
mkdir -p "$RUNS_DIR"

log() { echo "$(date '+%F %T') [watchdog] $*" >> "$WLOG"; }

# 存活判定: PID 文件存在 + 进程在 + cmdline 确为本驱动（防 PID 复用）
driver_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local p
  p="$(cat "$PID_FILE" 2>/dev/null)"
  [[ -n "$p" ]] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  ps -p "$p" -o args= 2>/dev/null | grep -q "optimize_peptide_infinite.py"
}

# ---- 并发保护: 已有看门狗在跑就立刻退出 ----
exec 9>"$LOCK" 2>/dev/null || exit 0
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || exit 0
fi

# ---- 用户显式停止 → 尊重其意愿, 不复活 ----
if [[ -f "$DISABLED" ]]; then
  exit 0
fi

if driver_alive; then
  exit 0
fi

# ---- 驱动已死: 清理陈旧 PID 文件并重新拉起 ----
if [[ -f "$PID_FILE" ]]; then
  log "驱动已死（陈旧 PID $(cat "$PID_FILE" 2>/dev/null)）, 准备重启"
  rm -f "$PID_FILE"
else
  log "无 PID 文件, 准备启动驱动"
fi

# cron 的最小环境不含 conda; 显式加载。PATH 由 crontab 的 PATH= 行提供,
# 若缺失（手工调用）则补上 SLURM 与 conda 目录, 保证 sbatch/squeue 可用。
[[ ":$PATH:" == *"/slurm/bin:"* ]] || export PATH="/opt/gridview/slurm/bin:$PATH"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
[[ -f "$CONDA_SH" ]] && source "$CONDA_SH" && conda activate base 2>/dev/null

ts="$(date +%Y%m%d_%H%M%S)"
DLOG="$RUNS_DIR/infinite_${ts}.log"

setsid nohup python -u optimize_peptide_infinite.py --migrate \
      </dev/null >"$DLOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
log "已重新拉起驱动 PID=$PID, 日志 $DLOG"

# 稍等片刻确认它没立刻崩（例如 conda/依赖问题）, 否则记一条便于排查
sleep 8
if kill -0 "$PID" 2>/dev/null; then
  log "确认存活 (8s 后仍在运行)"
else
  log "警告: 拉起后 8s 内退出, 请查看 $DLOG"
fi
exit 0
