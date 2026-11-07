"""JAX 跨进程持久化缓存 A/B 自检；仅在独立 GPU 测试作业中调用。"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "af3适配"))
import 编译缓存


def main():
    mode = sys.argv[1]
    cache_root = Path(os.environ["PEPOPT_COMPILE_CACHE"])
    if mode == "cold":
        import shutil
        shutil.rmtree(cache_root, ignore_errors=True)
    info = 编译缓存.configure()
    import time
    import jax
    import jax.numpy as jnp
    from functools import partial

    @partial(jax.jit, static_argnames=("n",))
    def work(x, y, n):
        a = x @ y
        for _ in range(8):
            a = jax.nn.gelu(a) @ y.T
        return a.sum()

    # 固定较大的矩阵与形状，产生可观察的编译与 autotune 活动。
    x = jnp.arange(256 * 256, dtype=jnp.float32).reshape(256, 256) / 1000
    y = jnp.arange(256 * 256, dtype=jnp.float32).reshape(256, 256).T / 2000
    start = time.monotonic()
    value = float(work(x, y, 8))
    print(f"CACHE_TEST {mode} result={value:.6f} elapsed={time.monotonic()-start:.3f}s",
          flush=True)
    print(f"CACHE_TEST {mode} devices={jax.devices()} cache={info['cache']}", flush=True)
    files = [p for p in cache_root.rglob("*") if p.is_file()]
    print(f"CACHE_TEST {mode} files={len(files)} bytes={sum(p.stat().st_size for p in files)}",
          flush=True)
    if mode == "warm":
        assert len(files) > 1, "没有生成持久化缓存"
        assert any(p.name != "profile.json" for p in files), "只有 profile，没有缓存条目"


if __name__ == "__main__":
    main()
