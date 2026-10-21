#!/usr/bin/env bash
# =============================================================================
# 无限循环作业 —— SLURM 侧包装脚本（单作业形态，计算节点上运行）
#
# 由 小肽优化_本地集群.submit_loop_job() 提交（资源指令走命令行参数，作业配置走 --export 环境变量，与仓库既有约定一致）。
#
# ⭐ 为什么需要这个形态（2026-10-04 定案）：
#   旧形态 = 登录节点常驻 Python 驱动 + 反复 sbatch 小包作业。它脆弱在：
#   登录节点把所有交互进程塞进同一个 cgroup /system.slice/sshd.service，setsid 逃不出去 → 会话被清理时驱动无声消失 → GPU 空转、结果滞留未入库（已发生两次，其中一次空转 8h22m）。
#   新形态 = **一个长驻作业内部自带无限循环**（见 小肽优化_节点内.py）：
#   作业跑满分区时限 MaxTime（5 天）才退出，期间反复 {挑候选→跑AF3→收割入库}。
#   于是既不需要登录节点常驻进程，也不需要“计算节点里 sbatch 下一包”（那会与 QoS MaxJobsPerUser=1 的运行槽互斥）。
#   登录节点只留 cron 看门狗，职责退化为“队列里没这个作业了就再交一个”。
#
# 环境分工（实测：af3_old **不含** sklearn/pandas，故两套环境必须并存）：
#   conda base   → 本脚本与 小肽优化_节点内.py（编排 / GP / 入库）
#   conda af3_old→ AF3 推理那一步（由 作业脚本/AF3打包推理.sh 自行激活）
#
# 可选环境变量：
#   PEPOPT_TARGETS=HTR1A+BIN1   待评估靶标（V100 放不下 UNC13C）
#                               ⭐ 分隔符必须是 '+'：该变量经 SLURM --export 下发，而 --export 列表以逗号分隔，值内含逗号会被截断（PEPOPT_TARGETS=HTR1A,BIN1 → 只收到 HTR1A）。
#                               本脚本与 python 侧均已兼容 ',' 与 '+' 两种写法。
#   PEPOPT_CYCLE_SEQS=24        每轮序列数
#   PEPOPT_MAX_HOURS=118        时间预算（= MaxTime 减去续交余量）
#   PEPOPT_RESULTS_DIR          结果目录（须等于共享库的 af3_results）
# =============================================================================
set -uo pipefail

REPO_DIR="${PEPOPT_WORK_DIR:-$SLURM_SUBMIT_DIR}"
cd "$REPO_DIR" || { echo "❌ 无法进入 $REPO_DIR" >&2; exit 2; }

echo "=============================================================="
echo " 无限循环作业启动：$(date '+%F %T') | 节点 $(hostname)"
echo "   作业号   ：${SLURM_JOB_ID:-?}"
echo "   时限请求 ：${PEPOPT_TIME_REQUESTED:-?}（循环内部预算 ${PEPOPT_MAX_HOURS:-118}h）"
echo "   GPU      ：${PEPOPT_GPUS_PER_JOB:-1} 张 | CPU ${SLURM_CPUS_ON_NODE:-?}"
echo "   靶标（下发）：${PEPOPT_TARGETS:-HTR1A+BIN1}   ← 务必核对！应含 BIN1"
echo "   每轮序列 ：${PEPOPT_CYCLE_SEQS:-24}"
echo "   仓库     ：$REPO_DIR"
echo "   数据库   ：${PEPOPT_DB_ROOT:-/public_bme2/Share200T/管吉松/peptide_opt_db}"
echo "=============================================================="
# 提交时就拦住靶标被截断（SLURM --export 以逗号分隔，值内含逗号会被静默截断）
case ",${PEPOPT_TARGETS:-HTR1A+BIN1}," in
  *",,"*|*BIN1*|*UNC13C*) : ;;
  *) echo "❌ 靶标下发异常：'${PEPOPT_TARGETS:-}' 未含 BIN1 → 拒绝启动以免白烧 5 天机时"
     echo "   （请确认 build_loop_job_cmd 用 '+' 而非 ',' 拼接靶标）"; exit 4 ;;
esac

# ---- 激活编排环境（base：含 sklearn/pandas/matplotlib）----
CONDA_SH="${PEPOPT_CONDA_SH:-}"
if [ -z "$CONDA_SH" ]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh"; do
        [ -f "$c" ] && { CONDA_SH="$c"; break; }
    done
fi
if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate base || { echo "❌ 激活 base 环境失败" >&2; exit 2; }
else
    echo "⚠️ 未定位到 conda.sh，沿用节点默认 python（可能缺 sklearn）"
fi

# 依赖自检：缺 GP 依赖就**立刻**退出，别把 5 天的 GPU 槽浪费在报错循环上
python -c "import sklearn, scipy, pandas, numpy" 2>/dev/null || {
    echo "❌ base 环境缺 sklearn/scipy/pandas/numpy → 安装后重交：" >&2
    echo "   pip install scikit-learn scipy pandas matplotlib openpyxl" >&2
    exit 3
}

# ---- 任务中间层（HyperQueue server+worker）**托管**编排进程 ----
# 守护成为作业主体（exec），“换代码”就只需 `任务控制.sh restart-service`：
# 重启受管进程 → 重新 import 新代码，而守护与 SLURM 分配都不动→不必 scancel 后重新排队（分区满载时重排可能要等数小时，已实测踩过）。编排进程只是提交者/等待者，计算跑在 HQ worker 里 → 重启它不丢在跑的预测，未完成部分由新进程 drain_backlog 按结果目录与台账接管。taskd随作业退出而清理（SLURM 杀进程组 + 自身 trap 双保险）。
TASKD="$REPO_DIR/../任务中间层/任务守护.sh"
# -u：关闭输出缓冲，使 SLURM 日志可实时 tail
PY_CMD="python -u 小肽优化_节点内.py \
    --targets '${PEPOPT_TARGETS:-HTR1A+BIN1}' \
    --cycle-seqs '${PEPOPT_CYCLE_SEQS:-24}' \
    --max-hours '${PEPOPT_MAX_HOURS:-118}'"

if [ -f "$TASKD" ]; then
    # --gpus 要的是**卡号列表**而非数量：SLURM 已把本作业真实分到的卡注入 CUDA_VISIBLE_DEVICES，直接透传才正确。误传数量（如 1 张卡传 "1"）会让 worker 声明错卡号 → 任务经 任务包装.sh 拿到 CUDA_VISIBLE_DEVICES=1 → jax 直接 FAILED_PRECONDITION：No visible GPU devices（2026-10-21 实测）。
    GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
    if [ -z "$GPU_IDS" ]; then
        # 未注入时退回探测：取可见卡的 **index 列表**，不是计数
        GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                  | tr '\n' ',' | sed 's/,$//')
    fi
    echo "🧭 任务守护托管编排进程；客户端：任务中间层/任务控制.sh"
    echo "   GPU 卡号：${GPU_IDS:-（无 → 纯 CPU worker）}"
    echo "   换代码不释放节点：./任务控制.sh restart-service"
    echo "   周期边界优雅重启：./运行.sh restart-loop"
    exec bash "$TASKD" --gpus "${GPU_IDS:-none}" \
        --cpus "${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}" \
        --reload-code 42 \
        --run "$PY_CMD"
    # exec 不会返回；守护的退出码即作业退出码（重载码 42 在守护内部消化）
fi

echo "⚠️ 未找到 $TASKD，AF3 将走整包直跑路径"
export PEPOPT_HQ_MODE=0
eval "$PY_CMD"
exit $?
