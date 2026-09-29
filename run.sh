#!/bin/bash
# =====================================================
# 一键运行优化任务并实时输出进度（隐藏作业号等中间细节）
#
# 用法:
#   ./run.sh            # 默认: 贝叶斯优化 + 本地集群推理后端（无每日配额）
#   ./run.sh mc         # 蒙特卡洛 + 本地集群推理后端
#   ./run.sh bo-server  # 贝叶斯优化 + AlphaFold Server 云端后端
#   ./run.sh mc-server  # 蒙特卡洛 + AlphaFold Server 云端后端
#
# 说明:
#   - 本地后端: 编排在登录节点运行(实时打印进度), 每个候选的
#     GPU 推理作业由入口自动提交到计算节点, 无需每日配额
#   - 云端后端: 提交轻量编排作业到计算节点, 实时跟踪其日志
#   - Ctrl+C 只是停止查看进度, 已提交的计算任务仍会继续运行
# =====================================================
set -u
cd "$(dirname "$0")"

mode="${1:-bo}"

# ---- 本地集群推理后端: 编排跑在登录节点, 进度实时输出 ----
run_local() {
  local entry="$1"
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate base
  echo "🧬 启动本地集群推理（进度实时输出, Ctrl+C 退出编排; 已提交的 GPU 作业不受影响）"
  echo "------------------------------------------------------"
  python -u "$entry"
  local rc=$?
  echo "------------------------------------------------------"
  [[ $rc -eq 0 ]] && echo "✅ 本轮编排结束（计算作业若仍在队列, 完成后重跑本命令即可续作）" \
                  || echo "❌ 编排退出（退出码 $rc）"
  exit $rc
}

case "$mode" in
  bo)   run_local optimize_peptide_BO.py ;;
  mc)   run_local optimize_peptide_local.py ;;
  bo-server) SBATCH=run_peptide_bo.sbatch; NAME="贝叶斯优化（云端后端）" ;;
  mc-server) SBATCH=run_peptide.sbatch;    NAME="蒙特卡洛（云端后端）" ;;
  *) echo "未知模式 '$mode'，可选: 默认(贝叶斯优化+本地) | mc | bo-server | mc-server"; exit 1 ;;
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
