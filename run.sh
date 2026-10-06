#!/bin/bash
# =====================================================
# 一键运行优化任务并实时输出进度（隐藏作业号等中间细节）
#
# 用法:
#   ./run.sh            # 默认: 提交**单个长驻循环作业**（当前形态, 推荐）
#   ./run.sh loop       # 同上（显式写法）; 计算节点作业内部无限循环搜索
#   ./run.sh stop       # 停止（scancel 循环作业 + 阻止看门狗续交）
#   ./run.sh status     # 作业/看门狗状态 + 穷尽覆盖进度 + 数据库统计与 BEST_TOP10
#   ./run.sh restart-loop # 改完代码后**在周期边界**优雅重启（新代码生效, 不废已完成周期）
#   ./run.sh watchdog   # 单独安装 cron 看门狗（loop 会自动装）
#   ./run.sh no-watchdog# 卸载看门狗
#   ./run.sh infinite   # 【legacy】登录节点常驻驱动形态（脆弱, 仅保留兼容）
#   ./run.sh bo         # 有限预算贝叶斯优化（前台, output_bo*/）
#   ./run.sh v100       # 同上, 但只跑 V100 放得下的 HTR1A+BIN1
#   ./run.sh mc         # 蒙特卡洛 + 本地集群推理后端
#   ./run.sh bo-server  # 贝叶斯优化 + AlphaFold Server 云端后端
#   ./run.sh mc-server  # 蒙特卡洛 + AlphaFold Server 云端后端
#
# 两种形态的取舍（2026-10-04 定案, 详见 optimize_peptide_node.py 文件头）:
#   - **loop（当前形态）**: sbatch 一个时限=分区 MaxTime（5 天）的作业, 搜索循环
#     整个跑在**计算节点内部**（optimize_peptide_node.py）: 挑候选 → 跑 AF3 →
#     收割入库 → 下一轮。登录节点**不留任何常驻进程**, 只有一个 cron 看门狗
#     负责"队列里没这个作业了就再交一个"（正常每 5 天才动作一次）。
#     可行性实测依据: PreemptMode=OFF 不被抢占; 分区无 MaxWall; --test-only 下
#     48h/2天/3天/5天 的预计启动时间**完全相同** → 长时限不受精调度惩罚。
#   - infinite（legacy）: 登录节点常驻 Python 驱动 + 反复 sbatch 小包。它脆弱:
#     本机把所有交互进程塞进同一 cgroup `/system.slice/sshd.service` 且 loginctl
#     查不到会话 → setsid 只换会话/进程组而**逃不出 cgroup** → 会话被清理时
#     驱动无声消失（已发生两次, 一次 GPU 空转 8h22m、29 个结果滞盘未入库）。
#   - 另注: 编排**不能**做成独立的计算节点作业——QoS MaxJobsPerUser=1 是运行
#     并发上限, 编排作业会与 GPU 作业互斥（实测 --test-only 预测 24h 后才启动）。
#     把编排塞进 GPU 作业内部循环, 正是为了绕开这个约束。
#   - 云端后端: 提交轻量编排作业到计算节点, 实时跟踪其日志
#   - Ctrl+C 只是停止查看进度, 已提交的计算任务仍会继续运行
# =====================================================
set -u
cd "$(dirname "$0")"

mode="${1:-loop}"

REPO_DIR="$(pwd)"
DB_ROOT="${PEPOPT_DB_ROOT:-/public_bme2/Share200T/管吉松/peptide_opt_db}"
RUNS_DIR="$DB_ROOT/runs"
WATCHDOG_INTERVAL=5                            # cron 自愈周期（分钟）
BLOCK_BEGIN="# >>> pepopt-watchdog >>>"
BLOCK_END="# <<< pepopt-watchdog <<<"

# ---- 【当前形态】单个长驻循环作业 ----
LOOP_JOB_NAME="pepopt-loop"
LOOP_TARGETS="${PEPOPT_TARGETS:-HTR1A,BIN1}"
LOOP_CYCLE_SEQS="${PEPOPT_CYCLE_SEQS:-24}"
LOOP_DISABLED="$RUNS_DIR/loop.disabled"       # stop 写入 → 看门狗不再续交
LOOP_WLOG="$RUNS_DIR/loop_watchdog.log"
LOOP_WATCHDOG="$REPO_DIR/scripts/loop_watchdog.sh"

# ---- 【legacy】登录节点常驻驱动 ----
PID_FILE="$RUNS_DIR/infinite.pid"
DISABLED_FILE="$RUNS_DIR/infinite.disabled"
WLOG="$RUNS_DIR/watchdog.log"
WATCHDOG="$LOOP_WATCHDOG"                     # 看门狗已换代为监督循环作业

# 存活判定: PID 在 + cmdline 确为本驱动（单看 PID 会被重用误导,
# 导致“以为活着”而不拉起 → GPU 空转）
driver_alive() {
  [[ -f "$PID_FILE" ]] || return 1
  local p; p="$(cat "$PID_FILE" 2>/dev/null)"
  [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null || return 1
  ps -p "$p" -o args= 2>/dev/null | grep -q "optimize_peptide_infinite.py"
}

install_watchdog() {
  local wd="${1:-$LOOP_WATCHDOG}"          # 默认监督循环作业; legacy 传驱动看门狗
  command -v crontab >/dev/null 2>&1 || { echo "⚠️  无 crontab, 无法装看门狗"; return 1; }
  [[ -f "$wd" ]] || { echo "⚠️  缺看门狗脚本 $wd"; return 1; }
  chmod +x "$wd" 2>/dev/null
  local tmp; tmp="$(mktemp)"
  # 保留用户既有 crontab, 仅剔除我们上次装的标记块（幂等）
  crontab -l 2>/dev/null | awk -v b="$BLOCK_BEGIN" -v e="$BLOCK_END" \
      'index($0,b){f=1;next} index($0,e){f=0;next} !f' > "$tmp"
  {
    echo "$BLOCK_BEGIN"
    echo "SHELL=/bin/bash"
    # cron 环境 PATH 极简, 不含 SLURM(/opt/gridview/slurm/bin) 与 conda,
    # 而驱动要调 sbatch/squeue/scontrol → 必须把登录时的完整 PATH 写进来。
    echo "PATH=$PATH"
    echo "*/$WATCHDOG_INTERVAL * * * * $wd >/dev/null 2>&1"
    echo "$BLOCK_END"
  } >> "$tmp"
  if crontab "$tmp"; then
    [[ "$wd" == "$LOOP_WATCHDOG" ]] \
      && echo "🩺 cron 看门狗已装: 每 ${WATCHDOG_INTERVAL} 分钟检查（循环作业不在队列就续交）" \
      || echo "🩺 cron 看门狗已装(legacy 驱动形态): 每 ${WATCHDOG_INTERVAL} 分钟自愈一次"
    echo "   卸载: ./run.sh no-watchdog"
  else
    echo "⚠️  crontab 安装失败"
  fi
  rm -f "$tmp"
}

uninstall_watchdog() {
  local tmp; tmp="$(mktemp)"
  crontab -l 2>/dev/null | awk -v b="$BLOCK_BEGIN" -v e="$BLOCK_END" \
      'index($0,b){f=1;next} index($0,e){f=0;next} !f' > "$tmp"
  if [[ -s "$tmp" ]]; then crontab "$tmp"; else crontab -r 2>/dev/null || crontab "$tmp"; fi
  rm -f "$tmp"
  echo "🧹 cron 看门狗已卸载（驱动不再自动复活）"
}

# ================= 【当前形态】单个长驻循环作业 =================

loop_jobs() { squeue -u "$USER" -h -n "$LOOP_JOB_NAME" -o '%i' 2>/dev/null | tr '\n' ' '; }
loop_log()  { ls -t "$RUNS_DIR"/loop_*.out 2>/dev/null | head -1; }
# 按 jobid 定位循环作业日志（sbatch 为 --output=runs/loop_%j.out）。
# loop_log() 取"最近修改"的日志, 在新作业还 PENDING（日志未创建）时会
# 错误地指向上一个已结束作业 → 把人引到死文件上。本函数优先返回活跃
# 作业的日志, 不存在时退回 loop_log() 并说明原因。
active_loop_log() {
  local jid="$1" lg="$RUNS_DIR/loop_${1}.out"
  [[ -n "$jid" && -f "$lg" ]] && { echo "$lg"; return 0; }
  loop_log
}

kill_legacy_driver() {
  driver_alive || return 1
  local pid; pid="$(cat "$PID_FILE" 2>/dev/null)"
  echo "🧹 停掉旧形态的登录节点常驻驱动 (PID $pid), 避免两套编排抢 QoS 队列…"
  mkdir -p "$RUNS_DIR"; date '+%F %T' > "$DISABLED_FILE"
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  rm -f "$PID_FILE"
  return 0
}

submit_loop() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base
  # submit_loop_job 内部会做预检查(PREFLIGHT_OK)并自行避免重复提交;
  # 若 QoS 提交窗口被旧 bundle 作业占满而失败, 看门狗会在窗口空出后自动补交。
  python -u - <<PY
import re
import sys
sys.path.insert(0, "$REPO_DIR")
import optimize_peptide_local as loc
loc.ensure_local_ready()
# 分隔符兼容 ',' 与 '+'：submit_loop_job 内部改用 '+' 下发给 SLURM（逗号会被截断）
targets = [t for t in re.split(r"[,+]", "$LOOP_TARGETS") if t]
jid = loc.submit_loop_job(targets, $LOOP_CYCLE_SEQS)
print("SUBMITTED:", jid)
PY
}

start_loop() {
  rm -f "$LOOP_DISABLED" 2>/dev/null
  mkdir -p "$RUNS_DIR"
  kill_legacy_driver
  local j; j="$(loop_jobs)"
  if [[ -n "${j// /}" ]]; then
    echo "♾️  循环作业已在队列: $j（不重复提交）"
  else
    echo "🚀 提交长驻循环作业: 靶标=$LOOP_TARGETS | 每轮 $LOOP_CYCLE_SEQS 序列 …"
    submit_loop
    j="$(loop_jobs)"
  fi
  install_watchdog
  echo
  echo "♾️  无限搜索由**单个计算节点作业**承担, 登录节点不再需要常驻进程:"
  echo "   🗄️  数据库: $DB_ROOT"
  # 日志路径按 jobid 推算（sbatch 用 --output=runs/loop_%j.out），而不是取
  # "最近修改的 loop_*.out"：新作业还在 PENDING 时它的日志尚未创建，后者会
  # 错误地指向上一个已结束作业的日志，把人引到死文件上。
  local lg jid
  jid="${j%% *}"; jid="${jid// /}"
  if [[ -n "$jid" ]]; then
    lg="$RUNS_DIR/loop_${jid}.out"
    echo "   📄 作业日志: tail -f $lg"
    [[ ! -f "$lg" ]] && echo "      （作业 ${jid} 尚在排队, 启动后该文件才出现）"
  else
    lg="$(loop_log)"
    [[ -n "$lg" ]] && echo "   📄 最近日志: tail -f $lg"
  fi
  echo "   🛑 停止: ./run.sh stop   |   📊 状态: ./run.sh status"
  if [[ -z "${j// /}" ]]; then
    echo "   ⏳ 本次未能提交（多为 QoS 提交窗口被旧作业占满）; 看门狗会在"
    echo "      窗口空出后自动补交（≤${WATCHDOG_INTERVAL}min）, 无需人工重跑。"
  fi
}

stop_loop() {
  mkdir -p "$RUNS_DIR"; date '+%F %T' > "$LOOP_DISABLED"
  local j; j="$(loop_jobs)"
  if [[ -n "${j// /}" ]]; then
    echo "🛑 取消循环作业: $j"
    # shellcheck disable=SC2086
    scancel $j 2>/dev/null
    for _ in $(seq 1 10); do [[ -z "$(loop_jobs)" ]] && break; sleep 1; done
    echo "✅ 已取消。已算完的结果都在共享盘上（下个作业的收割阶段即入库）。"
  else
    echo "循环作业不在队列。"
  fi
  kill_legacy_driver || true
  echo "   已写停用标记 → 看门狗不再续交; 彻底移除: ./run.sh no-watchdog"
  echo "   重新开跑: ./run.sh loop"
}

# 穷尽覆盖进度（候选全集 51091 的已派发/待评估 + 全覆盖 ETA）
coverage_line() {
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base
  python -u - <<PY 2>/dev/null
import re, sys
sys.path.insert(0, "$REPO_DIR")
import optimize_peptide_infinite as inf
import optimize_peptide_node as N        # cycle_seconds 在节点入口里定义
targets = tuple(t for t in re.split(r"[,+]", "$LOOP_TARGETS") if t)
n = max(1, $LOOP_CYCLE_SEQS)
if not inf.SWEEP_ENABLED:
    print("   📈 随机采样模式 (PEPOPT_SWEEP=0): 无穷尽覆盖保证")
else:
    print("   📈 " + inf.coverage_report(
        targets, cycle_secs=N.cycle_seconds(n), per_cycle=n))
PY
}

# 在**周期边界**优雅重启循环作业 → 让新代码生效, 不浪费已完成周期的计算。
# 必要性: 正在跑的作业已把 .py 载入内存, 改源文件对它无效; 直接 scancel 则会
# 丢掉当前周期尚未算完的任务。标志由 optimize_peptide_node.run_loop 在每轮
# 开头消费(取走后 break), cron 看门狗 ≤5min 内续交加载了新代码的作业。
restart_loop() {
  local j; j="$(loop_jobs)"
  if [[ -z "${j// /}" ]]; then
    echo "循环作业不在队列 → 无需重启; 直接 ./run.sh loop 即以新代码提交。"
    return 0
  fi
  if [[ -f "$LOOP_DISABLED" ]]; then
    echo "⚠️  存在停用标记 $LOOP_DISABLED → 重启后看门狗不会续交。"
    echo "   先执行: ./run.sh loop"
    return 1
  fi
  mkdir -p "$RUNS_DIR"; date '+%F %T' > "$RUNS_DIR/restart.requested"
  echo "🔁 已放置重启标志: $RUNS_DIR/restart.requested  (作业 $j)"
  echo "   作业会在**当前周期算完后**正常退出, 看门狗 ≤${WATCHDOG_INTERVAL}min 内续交新作业。"
  echo "   代价仅新作业首轮重建 chain-B MSA（实测 ≈55min）; 已完成周期的结果早已入库。"
  echo "   撤销（不想重启了）: rm -f $RUNS_DIR/restart.requested"
}

status_loop() {
  local j; j="$(loop_jobs)"
  if [[ -n "${j// /}" ]]; then
    echo "♾️  循环作业在队列: $j"
    squeue -u "$USER" -n "$LOOP_JOB_NAME" -o "   %.9i %.2t %10M %6D %R" 2>/dev/null
    local lim
    lim="$(scontrol show job ${j%% *} 2>/dev/null | grep -oE 'TimeLimit=[^ ]+' | head -1)"
    echo "   ${lim:-}  |  名称 $LOOP_JOB_NAME"
  elif [[ -f "$LOOP_DISABLED" ]]; then
    echo "♾️  已停用（$(cat "$LOOP_DISABLED") 手动停止; 看门狗不续交）"
  else
    echo "♾️  循环作业不在队列（无停用标记 → 看门狗将在 ≤${WATCHDOG_INTERVAL}min 内补交）"
  fi
  coverage_line   # 穷尽覆盖进度（只在 PEPOPT_SWEEP 开启时有意义）
  driver_alive && {
    echo "⚠️  legacy 登录节点驱动仍在运行 (PID $(cat "$PID_FILE")) → ./run.sh loop 会自动接管"; }
  if crontab -l 2>/dev/null | grep -q "$BLOCK_BEGIN"; then
    echo "🩺 看门狗: 已装 (每 ${WATCHDOG_INTERVAL}min)"
    [[ -s "$LOOP_WLOG" ]] && { echo "   最近动作:"; tail -4 "$LOOP_WLOG" | sed 's/^/     /'; }
  else
    echo "🩺 看门狗: 未装 → ./run.sh watchdog"
  fi
  local lg jid
  jid="$(loop_jobs)"; jid="${jid%% *}"; jid="${jid// /}"
  lg="$(active_loop_log "$jid")"
  if [[ -n "$jid" && "$lg" != "$RUNS_DIR/loop_${jid}.out" ]]; then
    echo "   ⏳ 作业 ${jid} 尚在排队, 日志未生成; 下方为上一个作业的日志"
  fi
  [[ -n "$lg" ]] && { echo "📄 作业日志: $lg"; tail -12 "$lg"; }
  echo
  squeue -u "$USER" -o "%.8i %.14j %.2t %.10M %.20R" 2>/dev/null | head -8
  echo
  source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
  python -u optimize_peptide_infinite.py --status
}

# ============= 【legacy】登录节点常驻驱动 =============

start_infinite() {
  echo "⚠️  这是【legacy】登录节点常驻驱动形态, 已知脆弱（会话被清理即无声消失,"
  echo "    曾导致 GPU 空转 8h22m）。除非确有必要, 请用默认形态: ./run.sh loop"
  if driver_alive; then
    echo "♾️  无限搜索已在运行 (PID $(cat "$PID_FILE"))。"
    echo "   日志: tail -f \"$RUNS_DIR\"/infinite_*.log | ./run.sh stop 停止"
    exit 0
  fi
  rm -f "$PID_FILE" "$DISABLED_FILE" 2>/dev/null   # 清理陈旧状态, 重新允许自愈
  mkdir -p "$RUNS_DIR"
  local ts; ts=$(date +%Y%m%d_%H%M%S)
  local log="$RUNS_DIR/infinite_${ts}.log"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base
  # --migrate: 把家目录旧结果迁入共享库（幂等, 已迁过会自动跳过）
  # 后台日志由共享库的 runs/ 目录持久化, 计算节点与登录节点都可见。
  # 注: setsid 只能脱离会话/进程组, 逃不出 sshd.service 的 cgroup（实测）,
  #   故仍需 cron 看门狗兵托（见 scripts/driver_watchdog.sh 文件头）。
  #   setsid→nohup→python 为 exec 链, 不换 PID, 故 $! 即真实 python PID。
  setsid nohup python -u optimize_peptide_infinite.py --migrate "$@" \
        </dev/null >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "♾️  无限搜索已后台启动 (PID $pid)，结果持续追加进共享数据库："
  echo "   🗄️  $DB_ROOT"
  echo "   📄 日志: tail -f $log"
  echo "   🛑 停止: ./run.sh stop   （已提交的 GPU 作业不受影响）"
  echo "   📊 状态: ./run.sh legacy-status"
  install_watchdog "$REPO_DIR/scripts/driver_watchdog.sh"
  sleep 3
  echo "------------------------------------------------------ 启动日志前 20 行："
  head -20 "$log" 2>/dev/null
}

stop_infinite() {
  # 先立标, 避免 5 分钟内 cron 看门狗把刚停掉的驱动又拉起来
  mkdir -p "$RUNS_DIR"; date '+%F %T' > "$DISABLED_FILE"
  if ! driver_alive; then
    echo "无限搜索未在运行。"
    rm -f "$PID_FILE" 2>/dev/null
    echo "已写入停用标记（看门狗不会复活）; 重新开跑: ./run.sh infinite"
    exit 0
  fi
  local pid; pid=$(cat "$PID_FILE")
  echo "🛑 停止无限搜索 (PID $pid) …"
  kill -TERM "$pid"
  for _ in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "   未在 12s 内退出, 强制 kill"; kill -9 "$pid"; }
  rm -f "$PID_FILE"
  echo "✅ 已停止。共享数据库保留在 $DB_ROOT; squeue -u $USER 可查看仍在跑的 GPU 作业。"
  echo "   看门狗仍装于 crontab（遇停用标记会跳过）; 彻底移除: ./run.sh no-watchdog"
}

status_infinite() {
  if driver_alive; then
    echo "♾️  无限搜索运行中 (PID $(cat "$PID_FILE"))"
  elif [[ -f "$DISABLED_FILE" ]]; then
    echo "♾️  无限搜索已停用（$(cat "$DISABLED_FILE") 手动停止, 看门狗不会复活）"
  else
    echo "♾️  无限搜索未在运行（且无停用标记 → 看门狗会在 ${WATCHDOG_INTERVAL}min 内拉起）"
  fi
  if crontab -l 2>/dev/null | grep -q "$BLOCK_BEGIN"; then
    echo "🩺 看门狗: 已装 (每 ${WATCHDOG_INTERVAL}min)"
    [[ -s "$WLOG" ]] && { echo "   最近自愈记录:"; tail -4 "$WLOG" | sed 's/^/     /'; }
  else
    echo "🩺 看门狗: 未装 → ./run.sh watchdog"
  fi
  local last; last=$(ls -t "$RUNS_DIR"/infinite_*.log 2>/dev/null | head -1)
  [[ -n "$last" ]] && { echo "📄 最近日志尾部: $last"; tail -15 "$last"; }
  echo
  squeue -u "$USER" -o "%.8i %.14j %.2t %.10M %.20R" 2>/dev/null | head -8
  echo
  source ~/miniconda3/etc/profile.d/conda.sh; conda activate base
  python -u optimize_peptide_infinite.py --status
}

# ---- 本地集群推理后端(有限入口): 编排跑在登录节点, 进度实时输出 ----
run_local() {
  local entry="$1"; shift
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base
  echo "🧬 启动本地集群推理（进度实时输出, Ctrl+C 退出编排; 已提交的 GPU 作业不受影响）"
  echo "------------------------------------------------------"
  python -u "$entry" "$@"
  local rc=$?
  echo "------------------------------------------------------"
  [[ $rc -eq 0 ]] && echo "✅ 本轮编排结束（计算作业若仍在队列, 完成后重跑本命令即可续作）" \
                  || echo "❌ 编排退出（退出码 $rc）"
  exit $rc
}

case "$mode" in
  loop)     shift; start_loop "$@"; exit $? ;;
  stop)     stop_loop; exit $? ;;
  status)   status_loop; exit $? ;;
  restart-loop|restart) restart_loop; exit $? ;;
  watchdog)    rm -f "$LOOP_DISABLED" "$DISABLED_FILE"; mkdir -p "$RUNS_DIR"; install_watchdog; exit $? ;;
  no-watchdog) uninstall_watchdog; exit $? ;;
  infinite) shift; start_infinite "$@"; exit $? ;;
  legacy-stop)   stop_infinite; exit $? ;;
  legacy-status) status_infinite; exit $? ;;
  bo)   run_local optimize_peptide_BO.py ;;
  v100) run_local optimize_peptide_BO.py --targets HTR1A,BIN1 ;;
  mc)   run_local optimize_peptide_local.py ;;
  bo-server) SBATCH=run_peptide_bo.sbatch; NAME="贝叶斯优化（云端后端）" ;;
  mc-server) SBATCH=run_peptide.sbatch;    NAME="蒙特卡洛（云端后端）" ;;
  *) echo "未知模式 '$mode'，可选: 默认(长驻循环作业) | loop | stop | status | watchdog | no-watchdog | infinite(legacy) | bo | v100 | mc | bo-server | mc-server"; exit 1 ;;
esac

if [[ ! -f "$SBATCH" ]]; then
  echo "❌ 找不到作业脚本 $SBATCH"; exit 1
fi

ts=$(date +%Y%m%d_%H%M%S)
out="slurm_watch_${mode}_${ts}.out"
err="slurm_watch_${mode}_${ts}.err"
tailpid=""

cleanup() { [[ -n "$tailpid" ]] && kill "$tailpid" 2>/dev/null; }
trap cleanup EXIT
trap 'echo; echo "👀 已停止进度跟踪，任务仍在后台继续运行。"; exit 130' INT TERM

# ---- 提交（细节对用户隐藏）----
jobid=$(sbatch --parsable --output="$out" --error="$err" "$SBATCH")
if [[ -z "$jobid" ]]; then
  echo "❌ 提交任务失败（分区/账户/配额问题，请联系管理员）"
  exit 1
fi
jobid=${jobid%%;*}
echo "🚀 ${NAME}任务已提交，进入队列…"

final_state() {   # 用 sacct 取作业最终状态(权威), 避免 squeue 竞态
  sacct -j "$jobid" -o State -n -P 2>/dev/null | head -1 | awk '{print $1}'
}
batch_done() {    # .batch 步 COMPLETED 意味着输出已刷盘, 可以安全读取日志
  sacct -j "${jobid}.batch" -o State -n -P 2>/dev/null | head -1 | grep -q COMPLETED
}

# ---- 等待日志出现(作业可能很快跑完, 最长等 10 分钟; 结束后靠 sacct 判定) ----
for i in $(seq 1 300); do
  if [[ -s "$out" ]]; then break; fi
  fs=$(final_state)
  case "$fs" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL)
      # 作业已结束: 若仍无日志, 给一点缓冲再退出
      sleep 1; break ;;
  esac
  sleep 2
done

if [[ -s "$out" ]]; then
  echo "📡 已连接进度日志（Ctrl+C 停止查看）："
  echo "------------------------------------------------------"
  tail -f "$out" &
  tailpid=$!

  # ---- 监视直到任务结束 ----
  while :; do
    fs=$(final_state)
    case "$fs" in
      ""|RUNNING|CONFIGURING|COMPLETING|PENDING) sleep 3 ;;
      *) break ;;
    esac
  done
  sleep 1
  cleanup; wait "$tailpid" 2>/dev/null
  echo "------------------------------------------------------"
else
  # 作业很快结束: 输出经共享文件系统回写有延迟, 耐心等它落盘
  # (.batch 步 COMPLETED 且文件非空 = 输出已收集完毕)
  for i in $(seq 1 30); do
    [[ -s "$out" ]] && batch_done && break
    sleep 3
  done
  echo "------------------------------------------------------"
  if [[ -s "$out" ]]; then
    echo "（作业执行较快, 以下为完整日志）"
    cat "$out"
  else
    echo "（作业无标准输出, 请检查 $err）"
  fi
  echo "------------------------------------------------------"
fi

fs=$(final_state)
case "$fs" in
  COMPLETED)  echo "✅ 任务已完成" ;;
  CANCELLED*) echo "⛔ 任务已被取消" ;;
  "")         echo "⚠️ 未获取到作业状态，可用 sacct -j $jobid 查询" ;;
  *)          echo "❌ 任务异常（状态: $fs），请查看: $err" ;;
esac
