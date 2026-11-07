"""跨进程 AF3/JAX 编译与 autotune 缓存配置（导入零副作用）。

版本/显卡型号/CUDA 驱动隔离缓存命名空间，避免不同运行时复用不兼容的条目。缓存是优化而非结果真源：损坏时可删除重建，AF3 必须仍能正常运行。
注意：Python 子进程不能回写父 shell 环境。runner 必须调用 --emit-env 后source生成的 env.sh，才能把 JAX/XLA 配置传给真正的 AF3 进程。
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess

DEFAULT_ROOT = "/public_bme2/Share200T/管吉松/peptide_opt_db/af3_compile_cache"


def _gpu():
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader"], text=True, timeout=20)
        return out.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return os.environ.get("PEPOPT_GPU_TYPE", "unknown_gpu")


def configure(override_root=None):
    """在 Python/JAX 导入前调用；返回持久化缓存配置。"""
    prefix = pathlib.Path(os.environ.get("AF3_ENV_PREFIX",
        "/public_bme2/Share200T/管吉松/envs/af3"))
    python = pathlib.Path(os.environ.get("AF3_PYTHON", str(prefix / "bin" / "python")))
    try:
        jax_version = subprocess.check_output([
            str(python), "-c",
            "import jax,jaxlib; print(jax.__version__+'+'+jaxlib.__version__)"],
            text=True, timeout=60).strip()
    except (OSError, subprocess.SubprocessError):
        jax_version = "unknown_jax"
    profile = "|".join((jax_version, _gpu(),
                        os.environ.get("AF3_FLASH_ATTN", "xla"),
                        os.environ.get("AF3_NEED_XLA_7X_FLAG", "0")))
    profile_hash = hashlib.sha256(profile.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile)[:100]
    root = pathlib.Path(override_root or os.environ.get(
        "PEPOPT_COMPILE_CACHE", DEFAULT_ROOT))
    cache = root / f"{safe}-{profile_hash}"
    cache.mkdir(parents=True, exist_ok=True)
    xla = cache / "xla"
    xla.mkdir(exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache / "jax")
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "1"
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
    # 缓存不可用时安全退回重编译，不能因缓存故障使整个 AF3 任务失败。
    os.environ["JAX_RAISE_PERSISTENT_CACHE_ERRORS"] = "0"
    os.environ["JAX_EXPLAIN_CACHE_MISSES"] = os.environ.get(
        "PEPOPT_CACHE_DEBUG", "0")
    flags = os.environ.get("XLA_FLAGS", "")
    flags += f" --xla_gpu_kernel_cache_file={xla / 'kernels.bin'}"
    flags += f" --xla_gpu_per_fusion_autotune_cache_dir={xla}"
    flags += " --xla_gpu_experimental_autotune_cache_mode=AUTOTUNE_CACHE_MODE_UPDATE"
    os.environ["XLA_FLAGS"] = flags.strip()
    os.environ["PEPOPT_COMPILE_CACHE_ACTIVE"] = str(cache)
    info = {"profile": profile, "root": str(root), "cache": str(cache)}
    with open(cache / "profile.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"♻️ AF3 持久化编译/autotune 缓存：{cache}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--emit-env", action="store_true",
                        help="只写可由父 shell source 的环境文件；shell 随后必须 source")
    args = parser.parse_args()
    info = configure(args.root)
    if args.emit_env:
        env_path = pathlib.Path(info["cache"]) / "env.sh"
        values = {
            key: os.environ[key] for key in (
                "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                "JAX_RAISE_PERSISTENT_CACHE_ERRORS", "JAX_EXPLAIN_CACHE_MISSES",
                "XLA_FLAGS", "PEPOPT_COMPILE_CACHE_ACTIVE")
        }
        with open(env_path, "w") as f:
            f.write("# 由父 shell source，不能作为独立 Python 子进程使用\n")
            for key, value in values.items():
                f.write(f"export {key}='{value}'\n")
        print(f"CACHE_ENV_FILE={env_path}", flush=True)
    else:
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
