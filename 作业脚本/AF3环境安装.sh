#!/usr/bin/env bash
# 在当前用户已有 af3 环境中构建/验证 AlphaFold3，不重复 conda 依赖解算。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AF3_ENV_PREFIX="${PEPOPT_AF3_PYTHON:-/public_bme2/Share200T/管吉松/envs/af3/bin/python}"
AF3_ENV_PREFIX="${AF3_ENV_PREFIX%/bin/python}"
export CONDA_PREFIX="$AF3_ENV_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="${PEPOPT_AF3_PYTHON:-$CONDA_PREFIX/bin/python}"
export CONDA_BUILD_SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
CPP_SRC=/public_bme2/Share200T/管吉松/envs/af3-build-cache/cpp-src
export CMAKE_ARGS="-DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_SYSROOT=$CONDA_BUILD_SYSROOT -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DZLIB_INCLUDE_DIR=$CONDA_PREFIX/include -DZLIB_LIBRARY=$CONDA_PREFIX/lib/libz.so.1.3.2 -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DFETCHCONTENT_SOURCE_DIR_ABSEIL-CPP=$CPP_SRC/abseil-cpp -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$CPP_SRC/pybind11 -DFETCHCONTENT_SOURCE_DIR_PYBIND11_ABSEIL=$CPP_SRC/pybind11_abseil -DFETCHCONTENT_SOURCE_DIR_CIFPP=$CPP_SRC/cifpp -DFETCHCONTENT_SOURCE_DIR_DSSP=$CPP_SRC/dssp -DFETCHCONTENT_SOURCE_DIR_MY-EIGEN3=$CPP_SRC/eigen -DFETCHCONTENT_SOURCE_DIR_CATCH2=$CPP_SRC/Catch2 -DFETCHCONTENT_SOURCE_DIR_LIBMCFP=$CPP_SRC/libmcfp"
export CFLAGS="--sysroot=$CONDA_BUILD_SYSROOT -I$CONDA_PREFIX/include ${CFLAGS:-}"
export CXXFLAGS="--sysroot=$CONDA_BUILD_SYSROOT -I$CONDA_PREFIX/include ${CXXFLAGS:-}"
export LDFLAGS="--sysroot=$CONDA_BUILD_SYSROOT -Wl,-rpath,$CONDA_PREFIX/lib ${LDFLAGS:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR=/tmp/pepopt-pip-cache
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_RETRIES=2
export PIP_TIMEOUT=30

cd /public_bme2/Share200T/管吉松/AlphaFold3 || exit 2
"$PYTHON" -m pip install --no-cache-dir --no-build-isolation --no-deps -v . || exit $?
"$PYTHON" -m pip install --no-index --find-links=/public_bme2/Share200T/管吉松/envs/af3-build-cache/runtime-wheels \
    absl-py chex 'dm-haiku==0.0.13' dm-tree 'jax==0.4.34' 'jax[cuda12]==0.4.34' \
    'jax-triton==0.2.0' 'jaxtyping==0.2.34' 'rdkit==2024.3.2' 'triton==3.1.0' \
    tqdm 'typeguard==2.13.3' zstandard || exit $?
"$PYTHON" - <<'PY' || exit $?
import alphafold3
from alphafold3.common import folding_input
import jax
import pyhmmer
print('AF3_IMPORT', alphafold3.__file__)
print('JAX_VERSION', jax.__version__)
print('JAX_DEVICES', jax.devices())
print('PYHMMER', pyhmmer.__version__)
PY
"$PYTHON" /public_bme2/Share200T/管吉松/AlphaFold3/run_alphafold.py --help >/dev/null || exit $?
cd "$ROOT" || exit 2
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" 小肽模板.py check || exit $?
echo 'AF3_ENV_READY=1'
