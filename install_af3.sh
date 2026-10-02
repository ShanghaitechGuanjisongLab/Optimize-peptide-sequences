#!/usr/bin/env bash
# =====================================================================
# install_af3.sh — 全自动安装 AlphaFold3 到本地 conda 环境（无任何手动步骤）
#
# 设计原则：所有版本号均在运行时动态探测，不硬编码：
#   - C++ 标准       ← 解析 AF3 源码 CMakeLists.txt 的 CMAKE_CXX_STANDARD
#   - Python 下限    ← 解析 pyproject.toml 的 requires-python
#   - numpy pin      ← 解析 pyproject.toml 中的 numpy==x.y.z（如有）
#   - cmake 版本     ← 解析 pyproject.toml 中 build-system 的 cmake 约束
#   - sysroot 版本   ← `ldd --version` 读系统 glibc，再对 `conda search`
#                      结果取「≤ 系统 glibc 的最高版本」
#   - gcc 版本       ← `conda search` 全部候选，降序逐个安装并自检
#                      （按探测到的 C++ 标准编译+链接+运行），失败自动降级重试
#
# 用法：
#   bash install_af3.sh          # 前台运行
#   nohup bash install_af3.sh > install_af3.log 2>&1 &   # 后台运行
# =====================================================================
set -euo pipefail

# ------------------------------- 配置 --------------------------------
AF3_SRC="/public_bme2/Share200T/管吉松/AlphaFold3"
CONDA_ROOT="$HOME/anaconda3"
ENV_NAME="af3"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$HOME/af3_install_${STAMP}.log"

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] ❌ 错误: $*" >&2; exit 1; }

# 全部输出同时写入日志文件
exec > >(tee -a "$LOG_FILE") 2>&1
log "开始安装，完整日志: $LOG_FILE"

# ----------------------------- 预检查 --------------------------------
[[ -d "$AF3_SRC" ]] || die "找不到 AF3 代码目录: $AF3_SRC"
[[ -f "$AF3_SRC/pyproject.toml" ]] || die "AF3 代码目录不完整（缺 pyproject.toml）"
source "$CONDA_ROOT/etc/profile.d/conda.sh" || die "无法加载 conda"
conda activate "$ENV_NAME" || die "无法激活环境 $ENV_NAME（如不存在请先 conda create -n $ENV_NAME python=<项目要求版本>）"

# ------------- 动态探测 1: 项目自身要求（从 AF3 源码解析） -------------
CXX_STD="$(sed -nE 's/.*set[(]CMAKE_CXX_STANDARD[[:space:]]+([0-9]+)[)].*/\1/p' "$AF3_SRC/CMakeLists.txt" | head -1)"
[[ -n "$CXX_STD" ]] || die "无法从 $AF3_SRC/CMakeLists.txt 解析 CMAKE_CXX_STANDARD"
REQ_PY_SPEC="$(sed -nE 's/^requires-python[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$AF3_SRC/pyproject.toml" | head -1)"
NPY_PIN="$(grep -oE 'numpy==[0-9]+\.[0-9]+\.[0-9]+' "$AF3_SRC/pyproject.toml" | head -1 || true)"
CMAKE_SPEC="$(grep -oE 'cmake[><=~]+[0-9.]+' "$AF3_SRC/pyproject.toml" | head -1 || true)"
log "项目要求: C++ 标准=c++$CXX_STD | Python: ${REQ_PY_SPEC:-未声明} | numpy: ${NPY_PIN:-无pin} | cmake: ${CMAKE_SPEC:-默认}"

PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "当前环境: $ENV_NAME (Python $PYVER)"
if [[ -n "$REQ_PY_SPEC" ]]; then
  # 用解析出的 requires-python 约束动态校验当前 Python 版本
  python - "$PYVER" "$REQ_PY_SPEC" <<'PYEOF' || die "环境 Python $PYVER 不满足项目要求 $REQ_PY_SPEC"
import re, sys
cur_s, spec = sys.argv[1], sys.argv[2]
m = re.search(r'(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)*)', spec)
if not m:
    sys.exit(f"无法解析 requires-python: {spec}")
def tup(v): return [int(x) for x in v.split('.')]
cur, req = tup(cur_s), tup(m.group(2))
n = max(len(cur), len(req))
cur += [0]*(n-len(cur)); req += [0]*(n-len(req))
ops = {'>=': cur>=req, '<=': cur<=req, '==': cur==req,
       '>': cur>req, '<': cur<req, '!=': cur!=req}
sys.exit(0 if ops[m.group(1)] else 1)
PYEOF
  log "✅ Python $PYVER 满足 $REQ_PY_SPEC"
fi

# 数据与权重可读性（安装前快速确认，跑不了也没意义）
for f in "/public_bme2/Share200T/管吉松/weights/af3.bin" \
         "/public_bme2/Share200T/管吉松/databases/alphafold3/uniref90_2022_05.fa"; do
  head -c 1 "$f" >/dev/null 2>&1 || die "关键文件不可读: $f"
done
log "✅ 权重与数据库可读性预检通过"

# ------------- 动态探测 2: 系统 glibc → 选 conda sysroot -------------
SYS_GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)"
[[ -n "$SYS_GLIBC" ]] || die "无法探测系统 glibc 版本"
# 在 conda 可用的 sysroot 中取「≤ 系统 glibc 的最高版本」，保证链接符号本机存在
# （若 sysroot 过高，产物会引用本机 glibc 没有的符号，运行/链接时报 undefined reference）
SYSROOT_VER="$(conda search -c conda-forge sysroot_linux-64 2>/dev/null \
  | awk '/^sysroot_linux-64/ {print $2}' | sort -Vu | uniq \
  | awk -v g="$SYS_GLIBC" '{split($1,a,"."); split(g,b,".");
      if (a[1]<b[1] || (a[1]==b[1] && a[2]<=b[2])) print}' | tail -1)"
[[ -n "$SYSROOT_VER" ]] || die "conda-forge 无 ≤ 系统 glibc($SYS_GLIBC) 的 sysroot_linux-64"
log "系统 glibc: $SYS_GLIBC → 选定 sysroot_linux-64=$SYSROOT_VER"

# ------------- 动态探测 3: gcc 候选（降序逐个自检, 失败自动降级）-------------
GCC_MAJORS="$(conda search -c conda-forge gcc_linux-64 2>/dev/null \
  | awk '/^gcc_linux-64/ {print $2}' | sort -Vru | awk -F. '{print $1}' | uniq)"
[[ -n "$GCC_MAJORS" ]] || die "conda-forge 无可用 gcc_linux-64"
log "gcc 候选(降序尝试): $(echo $GCC_MAJORS | tr '\n' ' ')"

# 自检源码按解析到的 C++ 标准生成；验证 编译+链接+运行 全链路
TEST_SRC="$(mktemp /tmp/cxxtest_XXXXXX.cc)"
{
  echo '#include <vector>'
  if (( CXX_STD >= 20 )); then
    echo '#include <concepts>'
    echo 'template <std::integral T> T add(T a, T b) { return a + b; }'
    echo 'int main() { std::vector<int> v{1,2,3}; return add(1,2)==3 ? 0 : 1; }'
  elif (( CXX_STD >= 17 )); then
    echo '#include <optional>'
    echo 'int main() { std::optional<int> o{1}; std::vector<int> v{1,2,3}; return o && v.size()==3 ? 0 : 1; }'
  else
    echo 'int main() { std::vector<int> v{1,2,3}; return v.size()==3 ? 0 : 1; }'
  fi
} > "$TEST_SRC"

TMP_STEP_LOG="$(mktemp /tmp/af3_step_XXXXXX.log)"
cmake_spec="${CMAKE_SPEC:-cmake}"   # 用 pyproject 解析出的约束；无则不限版本
toolchain_ready=false
for MAJOR in $GCC_MAJORS; do
  log "▶ 尝试工具链 gcc/gxx $MAJOR.* + sysroot $SYSROOT_VER ..."
  if ! conda install -y -q -c conda-forge \
      "gcc_linux-64=$MAJOR.*" "gxx_linux-64=$MAJOR.*" \
      "sysroot_linux-64=$SYSROOT_VER" \
      libstdcxx-ng "$cmake_spec" ninja pkg-config >"$TMP_STEP_LOG" 2>&1; then
    log "  conda 安装/解算失败，降级尝试下一版本："
    tail -10 "$TMP_STEP_LOG" | sed 's/^/    | /'
    continue
  fi
  # 重新激活，让编译器包的 activate.d 脚本设置 CC/CXX/FLAGS
  conda deactivate; conda activate "$ENV_NAME" || die "重新激活环境失败"
  CUR_SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
  [[ -d "$CUR_SYSROOT" ]] || { log "  sysroot 目录缺失，降级尝试"; continue; }
  TRY_CXX="$(ls "$CONDA_PREFIX"/bin/x86_64-conda*-g++ 2>/dev/null | head -1)"
  [[ -n "$TRY_CXX" ]] || { log "  未找到 g++，降级尝试"; continue; }
  if "$TRY_CXX" -std=c++"$CXX_STD" --sysroot="$CUR_SYSROOT" \
      -L"$CONDA_PREFIX/lib" -Wl,-rpath,"$CONDA_PREFIX/lib" \
      "$TEST_SRC" -o "${TEST_SRC%.cc}" >"$TMP_STEP_LOG" 2>&1 \
     && "${TEST_SRC%.cc}" >>"$TMP_STEP_LOG" 2>&1; then
    log "✅ gcc $MAJOR 通过 c++$CXX_STD 编译+链接+运行自检: $($TRY_CXX --version | head -1)"
    GCC_CHOSEN="$MAJOR"
    CONDA_BUILD_SYSROOT="$CUR_SYSROOT"
    toolchain_ready=true
    break
  fi
  log "  自检失败，降级尝试下一版本："
  tail -8 "$TMP_STEP_LOG" | sed 's/^/    | /'
done
rm -f "$TEST_SRC" "${TEST_SRC%.cc}" "$TMP_STEP_LOG"
$toolchain_ready || die "所有 gcc 候选均未通过 c++$CXX_STD 自检"
export CONDA_BUILD_SYSROOT

# conda 编译器带平台三元组前缀（兜底：若激活脚本未设置则手动指定）
CC_BIN="${CC:-$(ls "$CONDA_PREFIX"/bin/x86_64-conda*-gcc | head -1)}"
CXX_BIN="${CXX:-$(ls "$CONDA_PREFIX"/bin/x86_64-conda*-g++ | head -1)}"
export CC="$CC_BIN" CXX="$CXX_BIN"
# 显式把 sysroot 加进编译/链接参数（防止 pip 构建隔离子进程丢失激活变量）
export CFLAGS="${CFLAGS:-} --sysroot=$CONDA_BUILD_SYSROOT"
export CXXFLAGS="${CXXFLAGS:-} --sysroot=$CONDA_BUILD_SYSROOT"
# 让产物带 rpath 指向环境 lib（运行时找到新版 libstdc++），并保留 sysroot 链接参数
export LDFLAGS="${LDFLAGS:-} --sysroot=$CONDA_BUILD_SYSROOT -Wl,-rpath,$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
log "最终工具链: gcc $GCC_CHOSEN | sysroot $SYSROOT_VER | 编译器 $CXX_BIN"

# ------------------------ 安装 Python 依赖 ----------------------------
# numpy pin 从 pyproject.toml 动态解析；未声明则交给 pip 自行解算
if [[ -n "$NPY_PIN" ]]; then
  log "按 pyproject.toml 的 pin 安装 $NPY_PIN ..."
  pip install "$NPY_PIN" || die "numpy 安装失败"
else
  log "pyproject.toml 未 pin numpy，跳过预安装"
fi

log "构建并安装 alphafold3（含 C++ 扩展编译，耗时约 10-30 分钟）..."
cd "$AF3_SRC"
# pip 构建隔离会按 build-system 拉取 scikit_build_core/pybind11/cmake/ninja wheel
pip install --no-cache-dir . || die "alphafold3 构建安装失败"

# ------------------------------ 验证 ----------------------------------
log "验证安装..."
python - <<'EOF' || die "alphafold3 导入验证失败"
import alphafold3
print('✅ import alphafold3:', alphafold3.__file__)
from alphafold3.common import folding_input   # 触发 C++ 扩展加载
print('✅ folding_input (含 alphafold3.cpp) 加载成功')
import jax
print('✅ JAX:', jax.__version__, '| devices(登录节点无GPU属正常):', jax.devices())
EOF

log "验证 run_alphafold.py 入口..."
python "$AF3_SRC/run_alphafold.py" --help >/dev/null 2>&1 \
  && log "✅ run_alphafold.py --help 可执行" \
  || log "⚠️ run_alphafold.py --help 异常（不一定致命，跑任务时再看）"

cat <<SUMMARY

============================================================
🎉 AlphaFold3 安装完成！
------------------------------------------------------------
环境      : conda activate $ENV_NAME
包位置    : $CONDA_PREFIX/lib/python$PYVER/site-packages
日志      : $LOG_FILE

关键路径（运行任务时使用）：
  代码      : $AF3_SRC/run_alphafold.py
  模型权重  : /public_bme2/Share200T/管吉松/weights
  数据库    : /public_bme2/Share200T/管吉松/databases/alphafold3

GPU 推理示例（需在 GPU 节点上跑，登录节点无 GPU）：
  python $AF3_SRC/run_alphafold.py \\
    --json_path=input.json \\
    --model_dir=/public_bme2/Share200T/管吉松/weights \\
    --db_dir=/public_bme2/Share200T/管吉松/databases \\
    --output_dir=输出目录
============================================================
SUMMARY
log "全部完成 ✅"
