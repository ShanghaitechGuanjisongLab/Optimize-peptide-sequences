"""编译缓存配置与 HQ 快照的离线测试，不执行真实 AF3。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompileCacheOfflineTest(unittest.TestCase):
    def test_version_isolated_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PEPOPT_COMPILE_CACHE=tmp,
                       PEPOPT_GPU_TYPE="NVIDIAA10080GBPCIe")
            proc = subprocess.run([
                "/public_bme2/Share200T/管吉松/envs/af3/bin/python",
                str(ROOT / "af3适配" / "编译缓存.py")],
                env=env, text=True, capture_output=True, check=True)
            info = json.loads(proc.stdout[proc.stdout.index("{"):])
            self.assertTrue(Path(info["cache"]).is_dir())
            profile = json.loads((Path(info["cache"]) / "profile.json").read_text())
            self.assertIn("0.4.34", profile["profile"])
            self.assertIn("NVIDIAA10080GBPCIe", profile["profile"])

    def test_runner_config_and_hq_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "af3").mkdir()
            fake_af3 = root / "af3" / "run_alphafold.py"
            fake_af3.write_text("print('FAKE_AF3_OK')\n")
            job = root / "input.json"
            job.write_text(json.dumps({"sequences": []}))
            env = dict(os.environ,
                       AF3_CODE_DIR=str(root / "af3"),
                       AF3_BUNDLE_DIR=str(root),
                       AF3_OUT_DIR=str(root / "out"),
                       AF3_MODEL_DIR=str(root),
                       AF3_DB_DIR=str(root),
                       AF3_WORK_DIR=str(ROOT),
                       AF3_ENV_PREFIX="/public_bme2/Share200T/管吉松/envs/af3",
                       AF3_PYTHON="/public_bme2/Share200T/管吉松/envs/af3/bin/python",
                       AF3_FLASH_ATTN="triton",
                       AF3_NEED_XLA_7X_FLAG="0",
                       PEPOPT_COMPILE_CACHE=str(root / "cache"))
            subprocess.run(["bash", str(ROOT / "作业脚本" / "AF3打包推理.sh")],
                           env=env, text=True, capture_output=True, check=True,
                           timeout=120)
            self.assertTrue(any((root / "cache").rglob("profile.json")))

            spec = importlib.util.spec_from_file_location("af3任务", ROOT / "af3适配" / "af3任务.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            tdir = mod.gen_task(str(job), env, str(root / "hq"))
            self.assertTrue((Path(tdir) / "_编译缓存快照.py").is_file())
            self.assertTrue((Path(tdir) / "_msa缓存快照.py").is_file())
            launcher = (Path(tdir) / "_启动器.sh").read_text()
            self.assertIn("_编译缓存快照.py", launcher)
            self.assertIn("_msa缓存快照.py", launcher)
            self.assertNotIn(str(ROOT / "af3适配" / "编译缓存.py"), launcher)


if __name__ == "__main__":
    unittest.main()
