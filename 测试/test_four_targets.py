"""四靶标扫描的离线回归测试；不提交作业、不访问生产数据库写路径。"""

import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class FourTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="pepopt-four-test-")
        cls.root = Path(cls.tmp.name)
        cls.原靶标配置 = os.environ.get("PEPOPT_TARGET_CONFIG")
        cls.测试配置 = cls.root / "靶标配置.json"
        cls.测试配置.write_text(json.dumps({"启用靶标": ["HTR1A", "ASIC1A"]}), encoding="utf-8")
        os.environ["PEPOPT_TARGET_CONFIG"] = str(cls.测试配置)
        os.environ["PEPOPT_DB_ROOT"] = str(cls.root / "newdb")
        os.environ["PEPOPT_RESULTS_DIR"] = str(cls.root / "newdb" / "af3_results")
        os.environ["PEPOPT_MSA_CACHE"] = str(cls.root / "msa")
        os.environ["PEPOPT_PARTITION"] = "bme_gpu"
        os.environ["PEPOPT_ACCOUNT"] = "liuzhen1"
        import 小肽公共库 as common
        import 小肽数据库 as db
        import 小肽优化_无限 as inf
        import 小肽优化_本地集群 as loc
        import 小肽优化_节点内 as node
        import 小肽模板 as templates
        cls.common, cls.db, cls.inf = common, db, inf
        cls.loc, cls.node, cls.templates = loc, node, templates

    @classmethod
    def tearDownClass(cls):
        if cls.原靶标配置 is None:
            os.environ.pop("PEPOPT_TARGET_CONFIG", None)
        else:
            os.environ["PEPOPT_TARGET_CONFIG"] = cls.原靶标配置
        cls.tmp.cleanup()

    def test_template_alignment_and_single_chain(self):
        payload = self.templates.load()
        align = payload["alignment"]
        self.assertEqual(align["aligned_query_residues"], 526)
        self.assertEqual(align["identical_residues"], 517)
        template = payload["template"]
        self.assertEqual(len(template["queryIndices"]), 526)
        tables = self.templates._parse_cif(template["mmcif"])
        fields, atoms = tables["_atom_site"]
        chains = {r[fields.index("_atom_site.label_asym_id")] for r in atoms}
        self.assertEqual(chains, {"A"})
        self.assertGreater(len(atoms), 1000)

    def test_乘积只计算HTR1A和ASIC1A(self):
        row = {"iptm_htr1a": 0.9, "iptm_bin1": 0.8,
               "iptm_unc13c": 0.7, "iptm_asic1a": 0.6}
        self.assertEqual(self.db.PRODUCT_TARGETS, ("HTR1A", "ASIC1A"))
        self.assertAlmostEqual(self.db.objective_product(row), 0.9 * 0.6)
        del row["iptm_bin1"], row["iptm_unc13c"]
        self.assertAlmostEqual(self.db.objective_product(row), 0.9 * 0.6)
        del row["iptm_asic1a"]
        self.assertIsNone(self.db.objective_product(row))

    def test_靶标配置是唯一名单且无效时拒绝运行(self):
        with tempfile.TemporaryDirectory() as 目录:
            配置路径 = Path(目录) / "靶标配置.json"
            for 名单 in (["HTR1A", "ASIC1A"], ["HTR1A", "BIN1", "UNC13C", "ASIC1A"]):
                配置路径.write_text(json.dumps({"启用靶标": 名单}), encoding="utf-8")
                self.assertEqual(self.db.读取启用靶标(str(配置路径)), tuple(名单))
                环境 = dict(os.environ, PEPOPT_TARGET_CONFIG=str(配置路径), PEPOPT_TARGETS="HTR1A,UNC13C")
                结果 = subprocess.run([os.sys.executable, str(ROOT / "小肽数据库.py"), "靶标"], env=环境, capture_output=True, text=True, check=True)
                self.assertEqual(结果.stdout.strip(), ",".join(名单))
            for 名单 in ([], ["HTR1A", "HTR1A"], ["HTR1A", "UNKNOWN"], ["ASIC1A"], "HTR1A", ["HTR1A", 1]):
                with self.subTest(名单=名单):
                    配置路径.write_text(json.dumps({"启用靶标": 名单}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.db.读取启用靶标(str(配置路径))

    def test_切换口径备份旧统计并保留暂停靶标(self):
        with tempfile.TemporaryDirectory() as 目录, patch.dict(os.environ, {"PEPOPT_DB_ROOT": 目录, "PEPOPT_RESULTS_DIR": str(Path(目录) / "af3_results")}):
            for 靶标, 评分 in (("HTR1A", 0.9), ("ASIC1A", 0.6), ("BIN1", 0.8), ("UNC13C", 0.7)):
                self.db.upsert_result("s000", 靶标, 评分, peptide=self.common.SEQ_202)
            self.db.upsert_result("s001", "HTR1A", 0.8, peptide=self.common.SEQ_202)
            self.db.upsert_result("s002", "HTR1A", 0.8, peptide=self.common.SEQ_202)
            self.db.upsert_result("s002", "ASIC1A", 0.5, peptide=self.common.SEQ_202)
            旧记录 = self.db._read_csv(self.db.paths()["master_csv"], self.db.MASTER_FIELDS)
            for 行 in 旧记录:
                行.update({"obj_n": "99", "obj_sum": "12", "obj_max": "0.3024", "obj_pool": "[0.3]", "obj_climb": "1"})
            self.db._rewrite_master(旧记录)
            因子 = ({"ASIC1A": [0.4, 0.5]}, {"HTR1A": 0.9})
            所有因子 = {"s000": 因子, "s002": ({"HTR1A": [0.9, 0.6], "ASIC1A": [0.6, 0.9]}, {})}
            with patch.object(self.inf.seq_sample, "collect_product_factors", side_effect=lambda 标签, 行: 所有因子.get(标签)):
                结果 = self.db.重建乘积口径()
            现值 = self.db.load_database()
            self.assertEqual(现值["s000"]["obj_n"], 3)
            self.assertAlmostEqual(现值["s000"]["obj_max"], 0.54)
            self.assertAlmostEqual(现值["s000"]["obj_sum"], 1.35)
            self.assertEqual(现值["s000"]["iptm_bin1"], 0.8)
            self.assertEqual(现值["s000"]["iptm_unc13c"], 0.7)
            self.assertIsNone(现值["s001"]["objective_product"])
            self.assertIsNone(现值["s001"]["obj_n"])
            self.assertAlmostEqual(现值["s002"]["objective_product"], 0.54)
            self.assertAlmostEqual(现值["s002"]["score_htr1a"], 0.1)
            备份 = self.db._read_csv(str(Path(结果["备份目录"]) / "peptide_database.csv"), self.db.MASTER_FIELDS)
            self.assertEqual(备份[0]["obj_n"], "99")
            self.assertEqual(json.loads((Path(目录) / "乘积口径.json").read_text())["靶标"], ["HTR1A", "ASIC1A"])

    def test_启动检查只在名单变化时重建统计(self):
        with tempfile.TemporaryDirectory() as 目录, patch.dict(os.environ, {"PEPOPT_DB_ROOT": 目录}):
            with patch.object(self.db, "重建乘积口径") as 重建:
                self.assertIsNone(self.db.确保乘积口径())
                重建.assert_not_called()
                Path(self.db.paths()["master_csv"]).write_text("tag\n", encoding="utf-8")
                标记路径 = Path(目录) / "乘积口径.json"
                标记路径.write_text(json.dumps({"靶标": list(self.db.PRODUCT_TARGETS)}), encoding="utf-8")
                self.assertIsNone(self.db.确保乘积口径())
                重建.assert_not_called()
                标记路径.write_text(json.dumps({"靶标": ["HTR1A", "BIN1", "UNC13C", "ASIC1A"]}), encoding="utf-8")
                self.db.确保乘积口径()
                重建.assert_called_once_with()

    def test_只改一份配置即可同步Python作业和排名(self):
        with tempfile.TemporaryDirectory() as 目录:
            根目录 = Path(目录)
            项目 = 根目录 / "项目"
            项目.mkdir()
            (项目 / "小肽数据库.py").symlink_to(ROOT / "小肽数据库.py")
            配置路径 = 项目 / "靶标配置.json"
            假环境 = 根目录 / "conda.sh"
            假环境.write_text("conda() { :; }\n", encoding="utf-8")
            工具目录 = 根目录 / "bin"
            工具目录.mkdir()
            (工具目录 / "python").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            (工具目录 / "python").chmod(0o755)
            守护目录 = 根目录 / "任务中间层"
            守护目录.mkdir()
            (守护目录 / "任务守护.sh").write_text('#!/bin/bash\n"$TASKCTL_PYTHON" -c \'import json,os,sys; print(json.dumps({"靶标":os.environ["PEPOPT_TARGETS"],"参数":sys.argv[1:]}))\' "$@"\n', encoding="utf-8")
            数据库目录 = 根目录 / "数据库"
            with patch.dict(os.environ, {"PEPOPT_DB_ROOT": str(数据库目录)}):
                self.db._rewrite_master([{"tag": "s000", "peptide": "SEQ_A", "iptm_htr1a": "0.9", "iptm_asic1a": "0.6"}, {"tag": "s001", "peptide": "SEQ_B", "iptm_htr1a": "0.8", "iptm_asic1a": "0.7", "iptm_bin1": "0.5", "iptm_unc13c": "0.5"}])
            探针 = "import json; import 小肽数据库 as 数据库; import 小肽优化_无限 as 无限; import 小肽优化_节点内 as 节点; import 小肽优化_BO as 贝叶斯; import 小肽序贯采样 as 采样; print(json.dumps([list(数据库.PRODUCT_TARGETS),无限.DEFAULT_TARGETS,节点.DEFAULT_TARGETS,贝叶斯.ACTIVE_TARGETS,list(采样.PRODUCT_TARGETS)]))"
            for 名单 in (["HTR1A", "ASIC1A"], ["HTR1A", "BIN1", "UNC13C", "ASIC1A"]):
                with self.subTest(名单=名单):
                    配置路径.write_text(json.dumps({"启用靶标": 名单}), encoding="utf-8")
                    环境 = dict(os.environ, PEPOPT_TARGET_CONFIG=str(配置路径), PEPOPT_TARGETS="HTR1A,UNKNOWN", PEPOPT_DB_ROOT=str(数据库目录), PEPOPT_WORK_DIR=str(项目), PEPOPT_CONDA_SH=str(假环境), TASKCTL_PYTHON=os.sys.executable, CUDA_VISIBLE_DEVICES="0", PATH=str(工具目录) + os.pathsep + os.environ["PATH"])
                    结果 = subprocess.run([os.sys.executable, "-c", 探针], cwd=ROOT, env=环境, capture_output=True, text=True, check=True)
                    self.assertEqual(json.loads(结果.stdout), [名单, ",".join(名单), ",".join(名单), 名单, 名单])
                    作业 = subprocess.run(["bash", str(ROOT / "作业脚本" / "AF3循环作业.sh")], cwd=ROOT, env=环境, capture_output=True, text=True, check=True)
                    启动 = json.loads(作业.stdout.splitlines()[-1])
                    self.assertEqual(启动["靶标"], ",".join(名单))
                    self.assertNotIn("--targets", 启动["参数"][-1])
                    展示 = subprocess.run([os.sys.executable, str(ROOT / "查看前五.py")], cwd=ROOT, env=环境, capture_output=True, text=True, check=True)
                    self.assertIn(" × ".join(名单) + " 乘积", 展示.stdout)
                    self.assertIn(f"靶标齐全记录：{2 if len(名单) == 2 else 1} 条", 展示.stdout)
                    表头 = next(行 for 行 in 展示.stdout.splitlines() if 行.startswith("排名"))
                    for 靶标 in self.db.TARGETS_ALL:
                        self.assertEqual(靶标 in 表头, 靶标 in 名单)

    def test_four_target_package_and_template(self):
        cycle = self.root / "cycle"
        targets = ["ASIC1A", "HTR1A", "BIN1", "UNC13C"]
        n, skipped = self.node.prepare_cycle([self.common.SEQ_202], targets, str(cycle))
        self.assertEqual(n, 4, skipped)
        self.assertFalse(skipped)
        jobs = {p.stem.rsplit("_", 1)[-1].upper(): json.loads(p.read_text())
                for p in cycle.glob("*.json")}
        self.assertEqual(set(jobs), set(targets))
        asic = jobs["ASIC1A"]["sequences"][1]["protein"]
        self.assertEqual(asic["id"], ["B", "C", "D"])
        self.assertEqual(asic["templates"], [self.templates.load()["template"]])
        self.assertEqual(jobs["UNC13C"]["sequences"][1]["protein"]["id"], ["B"])
        for name, job in jobs.items():
            path = cycle / (job["name"] + ".json")
            tokens = self.loc.estimate_tokens(str(path))
            self.assertLessEqual(tokens, self.loc.GPU_MAX_TOKENS)
            if name == "ASIC1A":
                self.assertEqual(tokens, 1597)
            if name == "UNC13C":
                self.assertGreater(tokens, 2048)
        self.assertEqual(self.inf.DEFAULT_TARGETS, "HTR1A,ASIC1A")

    def test_生产输入不生成暂停靶标(self):
        with tempfile.TemporaryDirectory() as 目录:
            n, skipped = self.node.prepare_cycle([self.common.SEQ_202], self.db.PRODUCT_TARGETS, 目录)
            self.assertEqual(n, 2, skipped)
            self.assertEqual({路径.stem.rsplit("_", 1)[-1] for 路径 in Path(目录).glob("*.json")}, {"htr1a", "asic1a"})

    def test_msa_preserves_explicit_template(self):
        spec = importlib.util.spec_from_file_location("msa缓存", ROOT / "af3适配" / "msa缓存.py")
        msa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msa)
        sequence = self.common.ASIC1A_SEQ
        msa.save(sequence, ">q\n" + sequence, "", [{"mmcif": "automatic"}])
        job = self.common.create_af3_json(
            self.common.SEQ_202, sequence, "asic",
            **self.inf.target_af3_options("ASIC1A"))
        path = self.root / "msa.json"
        path.write_text(json.dumps(job))
        hit, miss = msa.embed([str(path)])
        self.assertEqual((hit, miss), (1, 1))  # 肽链不在测试缓存中
        protein = json.loads(path.read_text())["sequences"][1]["protein"]
        self.assertEqual(protein["unpairedMsa"], ">q\n" + sequence)
        self.assertEqual(protein["templates"], [self.templates.load()["template"]])

    def test_custom_template_missing_msa_fails(self):
        spec = importlib.util.spec_from_file_location("msa缓存2", ROOT / "af3适配" / "msa缓存.py")
        msa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msa)
        job = self.common.create_af3_json(
            self.common.SEQ_202, self.common.ASIC1A_SEQ, "asic-no-msa",
            **self.inf.target_af3_options("ASIC1A"))
        path = self.root / "no-msa.json"
        path.write_text(json.dumps(job))
        with self.assertRaisesRegex(RuntimeError, "显式结构模板缺少"):
            msa.embed([str(path)])

    def test_slurm_command_requests_a100_80gb(self):
        cmd, _ = self.loc.build_loop_job_cmd(self.inf.PRODUCT_TARGETS, 2)
        self.assertIn("--partition=bme_gpu", cmd)
        self.assertIn("--account=liuzhen1", cmd)
        self.assertIn("--gres=gpu:NVIDIAA10080GBPCIe:1", cmd)
        export = next(x for x in cmd if x.startswith("--export="))
        self.assertIn("PEPOPT_TARGETS=HTR1A+ASIC1A", export)
        self.assertIn("PEPOPT_GPU_MAX_TOKENS=5120", export)
        self.assertIn("PEPOPT_FLASH_ATTN=triton", export)

    def test_isolated_bootstrap_excludes_old_asic_and_stats(self):
        archive = self.root / "archive"
        archive.mkdir()
        old_fields = self.db.MASTER_FIELDS
        old = {c: "" for c in old_fields}
        old.update({"tag": "sold", "peptide": self.common.SEQ_202,
                    "var_region": self.common.ORIG_TAIL,
                    "iptm_htr1a": "0.9", "iptm_bin1": "0.8",
                    "iptm_asic1a": "0.99", "obj_n": "5", "obj_max": "0.7"})
        with (archive / "peptide_database.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old_fields)
            w.writeheader()
            w.writerow(old)
        with (archive / "sequences.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.db.SEQREG_FIELDS)
            w.writeheader()
            w.writerow({"tag": "sold", "var_region": self.common.ORIG_TAIL,
                        "peptide": self.common.SEQ_202, "first_seen": "old"})
        result = self.db.initialize_four_target_campaign(str(archive))
        self.assertTrue(result["initialized"])
        rows = self.db.load_database()
        self.assertEqual(len(rows), 1)
        row = next(iter(rows.values()))
        self.assertEqual(row["iptm_htr1a"], 0.9)
        self.assertEqual(row["iptm_bin1"], 0.8)
        self.assertIsNone(row["iptm_asic1a"])
        self.assertIsNone(row["obj_n"])
        self.assertIsNone(self.db.objective_product(row))
        self.assertEqual(self.db.refresh_best(), [])

    def test_summary_fallback_and_backlog_limit(self):
        old_root = os.environ["PEPOPT_DB_ROOT"]
        archive = self.root / "summary-archive"
        result_root = archive / "af3_results"
        result_root.mkdir(parents=True)
        with (archive / "sequences.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.db.SEQREG_FIELDS)
            w.writeheader()
            w.writerow({"tag": "sold2", "var_region": self.common.ORIG_TAIL,
                        "peptide": self.common.SEQ_202, "first_seen": "old"})
        for target in ("htr1a", "bin1", "asic1a"):
            directory = result_root / f"sold2_{target}"
            directory.mkdir()
            (directory / f"sold2_{target}_summary_confidences.json").write_text(json.dumps({
                "iptm": 0.99, "ranking_score": 0.8,
                "chain_pair_iptm": [[0.0, 0.75, 0.65, 0.55],
                                    [0.75, 0.0, 0.9, 0.9]]}))
        try:
            os.environ["PEPOPT_DB_ROOT"] = str(self.root / "fallback-new")
            result = self.db.initialize_four_target_campaign(str(archive))
            self.assertEqual(result["archived_rows"], 1)
            self.assertIn("summary_confidences", result["value_source"])
            row = next(iter(self.db.load_database().values()))
            self.assertEqual(row["iptm_htr1a"], 0.75)
            self.assertEqual(row["iptm_bin1"], 0.75)
            self.assertIsNone(row["iptm_asic1a"])  # 旧 ASIC1A 明确不复用
            cycle = self.root / "limited-backlog"
            nseq, ntask = self.node.drain_backlog(
                self.inf.PRODUCT_TARGETS, str(cycle), 1)
            self.assertEqual((nseq, ntask), (1, 1))
            self.assertEqual({p.stem.rsplit("_", 1)[-1] for p in cycle.glob("*.json")},
                             {"asic1a"})
        finally:
            os.environ["PEPOPT_DB_ROOT"] = old_root

    def test_backlog_prioritizes_existing_target_product(self):
        目标靶标 = ("HTR1A", "BIN1", "UNC13C", "ASIC1A")
        cases = (
            ("gap_beats_low_product_leftover",
             (("s000", (0.99, 0.1), True), ("s100", (0.8, 0.8), False)), "s100"),
            ("high_product_leftover",
             (("s000", (0.99, 0.1), False), ("s100", (0.8, 0.8), True)), "s100"),
            ("tie_by_tag",
             (("s020", (0.5, 0.8), False), ("s010", (0.8, 0.5), False)), "s010"),
            ("zero_before_unscored",
             (("s000", (None, None), False), ("s100", (0.0, 0.8), False)), "s100"),
            ("multiply_only_existing_factors",
             (("s000", (0.8, 0.8), False), ("s100", (0.75, None), False)), "s100"),
        )
        for label, records, expected_tag in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.dict(os.environ, {"PEPOPT_DB_ROOT": str(root / "db")}):
                    self.db.paths(ensure=True)
                    for index, (tag, scores, leftover) in enumerate(records):
                        variable = self.common.ORIG_TAIL[:-1] + "ACDEFG"[index]
                        peptide = self.common.FIXED_A + variable
                        self.db.register_sequence(tag, variable, peptide)
                        for target, score in zip(("HTR1A", "BIN1"), scores):
                            if score is not None:
                                self.db.upsert_result(tag, target, score,
                                                      peptide=peptide, var=variable)
                        if leftover:
                            canonical = self.db.canonical_dir(tag, "ASIC1A")
                            os.makedirs(canonical, exist_ok=True)
                            self.loc.write_input_json(
                                peptide, self.common.ASIC1A_SEQ, f"{tag}_asic1a",
                                canonical, **self.inf.target_af3_options("ASIC1A"))
                    selected_row = self.db.load_database().get(expected_tag, {})
                    missing = sum(selected_row.get(f"iptm_{target.lower()}") is None for target in 目标靶标)
                    全库缺口 = {"HTR1A": sum(scores[0] is None for _, scores, _ in records), "BIN1": sum(scores[1] is None for _, scores, _ in records), "UNC13C": len(records), "ASIC1A": len(records)}
                    cycle = root / "cycle"
                    priorities = {}
                    selected = self.node.drain_backlog(目标靶标, str(cycle), 1, task_priorities=priorities)
                    self.assertEqual(selected, (1, missing))
                    files = list(cycle.glob("*.json"))
                    self.assertEqual({path.stem.split("_", 1)[0] for path in files},
                                     {expected_tag})
                    self.assertEqual(priorities,
                                     {path.stem: self.node.RESAMPLE_PRIORITY + 全库缺口[path.stem.rsplit("_", 1)[1].upper()]
                                      for path in files})
                    for tag, scores, leftover in records:
                        if tag != expected_tag:
                            self.assertFalse((Path(self.db.canonical_dir(tag, "UNC13C"))
                                              / "input.json").exists())

    def test_af3_maintenance_pause_reports_safe_boundary(self):
        spec = importlib.util.spec_from_file_location("af3_maintenance", ROOT / "af3适配" / "af3任务.py")
        adapter = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"TASKCTL_STATE": directory}):
            spec.loader.exec_module(adapter)
            requested = Path(directory) / "af3-drain.requested"
            requested.write_text("maintenance")

            def release_pause(seconds):
                ready = json.loads(Path(str(requested) + ".ready").read_text())
                self.assertEqual(ready["pid"], os.getpid())
                self.assertEqual(ready["host"], os.uname().nodename)
                requested.unlink()

            with patch.object(adapter.time, "sleep", side_effect=release_pause) as wait:
                adapter.pause_at_batch_boundary()
            wait.assert_called_once_with(1)

    def test_af3_submission_declares_gpu_memory(self):
        spec = importlib.util.spec_from_file_location("af3_tasks", ROOT / "af3适配" / "af3任务.py")
        tasks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tasks)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target, memory in (("HTR1A", "5:10"), ("BIN1", "6.5:13"),
                                   ("ASIC1A", "22:44"), ("UNC13C", "32.5:65")):
                with self.subTest(target=target):
                    path = root / f"{target.lower()}.json"
                    path.write_text(json.dumps(self.common.create_af3_json(
                        self.common.SEQ_202, self.inf.TARGET_SEQS[target], target.lower(),
                        target_copies=self.inf.TARGET_COPIES.get(target, 1))))
                    with patch.object(tasks, "gen_task", return_value=str(root)), \
                            patch.object(tasks.subprocess, "run") as submit:
                        submit.return_value.stdout = "ID=17"
                        submit.return_value.stderr = ""
                        submit.return_value.returncode = 0
                        self.assertEqual(tasks.submit(str(path), {}, str(root)), 17)
                        arguments = submit.call_args.args[0]
                        self.assertEqual(arguments[arguments.index("--gpu-memory") + 1], memory)
                        下限, 上限 = tasks.显存需求范围(str(path), {})
                        self.assertEqual(下限 * 2, 上限)
            invalid = root / "invalid.json"
            invalid.write_text('{"sequences": []}')
            with self.assertRaises(ValueError):
                tasks.minimum_gpu_memory(str(invalid))

    def test_已有长驻排队仍会调用逐档补齐入口(self):
        with patch.object(self.loc, "PREFLIGHT_OK", True), patch.object(self.loc, "loop_job_active", return_value=["old-large"]), patch.object(self.loc, "逐档启动循环", return_value={"plan": [{"gpu_memory": 40}]}) as 启动:
            self.assertEqual(self.loc.submit_loop_job(list(self.db.PRODUCT_TARGETS), 8), {"plan": [{"gpu_memory": 40}]})
            启动.assert_called_once_with(list(self.db.PRODUCT_TARGETS), 8, None)

    def test_启动已在线时不重建离线需求(self):
        with patch.object(self.loc.subprocess, "run", return_value=Mock(returncode=0)), patch.object(self.node, "drain_backlog") as 全量补齐:
            self.assertEqual(self.loc.逐档启动循环(list(self.db.PRODUCT_TARGETS), 8), {"主服务在线": True, "plan": []})
            全量补齐.assert_not_called()

    def test_af3_memory_estimate_tracks_size_not_gpu_models(self):
        spec = importlib.util.spec_from_file_location("af3_memory", ROOT / "af3适配" / "af3任务.py")
        tasks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tasks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"

            def estimate(length, copies=1, **settings):
                path.write_text(json.dumps(self.common.create_af3_json(
                    self.common.SEQ_202, "A" * length, "memory-test",
                    target_copies=copies)))
                return tasks.minimum_gpu_memory(str(path), settings)

            self.assertEqual(estimate(422), estimate(423))
            self.assertLess(estimate(422), estimate(593))
            self.assertLess(estimate(526), estimate(526, copies=3))
            self.assertEqual(estimate(526, copies=3), estimate(526 * 3))
            self.assertEqual(estimate(593, PEPOPT_GPU_TYPE="V100-32GB"),
                             estimate(593, PEPOPT_GPU_TYPE="A100-80GB"))
            self.assertEqual(estimate(593, AF3_MEMORY_FIXED_GB="2",
                                      AF3_MEMORY_PAIR_GB="4",
                                      AF3_MEMORY_MARGIN="1.25"), 6)
            self.assertGreater(estimate(6000), estimate(5000))
            for setting, value in (("AF3_MEMORY_FIXED_GB", "-1"),
                                   ("AF3_MEMORY_PAIR_GB", "nan"),
                                   ("AF3_MEMORY_MARGIN", "0.5")):
                with self.subTest(setting=setting), self.assertRaises(ValueError):
                    estimate(593, **{setting: value})
            path.write_text(json.dumps({"sequences": [
                {"protein": {"id": "A", "sequence": "AAA"}},
                {"ligand": {"id": "B", "ccdCodes": ["ATP"]}},
            ]}))
            with self.assertRaisesRegex(ValueError, "仅支持蛋白体系"):
                tasks.minimum_gpu_memory(str(path))

    def test_queue_refill_keeps_each_target_waiting_without_duplicates(self):
        spec = importlib.util.spec_from_file_location("af3_queue", ROOT / "af3适配" / "af3任务.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"PEPOPT_DB_ROOT": str(root / "db")}):
                self.db.paths(ensure=True)
                for tag, variable, score in (("s000", "SPVDAVCS", 0.1),
                                              ("s999", "SPVDVVAS", 0.8)):
                    peptide = self.common.FIXED_A + variable
                    self.db.register_sequence(tag, variable, peptide)
                    for target in ("HTR1A", "BIN1"):
                        self.db.upsert_result(tag, target, score,
                                              peptide=peptide, var=variable)
                waiting = [
                    {"id": 1, "name": "s000_bin1", "task_stats": {"waiting": 1}},
                    {"id": 2, "name": "s999_asic1a", "task_stats": {"running": 1}},
                ]
                submitted = []

                def submit(path, env, task_root, priority=0):
                    job = json.loads(Path(path).read_text())
                    submitted.append(job)
                    waiting.append({"id": len(waiting) + 1, "name": job["name"],
                                    "task_stats": {"waiting": 1}})
                    return waiting[-1]["id"]

                adapter.submit = Mock(side_effect=submit)
                new_peptide = self.common.FIXED_A + "SPVDVVCA"
                with patch.object(self.inf, "sweep_pick", return_value=[new_peptide]) as pick:
                    refiller = self.node.HqQueueRefiller(
                        self.db.PRODUCT_TARGETS, time.time() + 86400, adapter)
                    refiller.refill(waiting)
                    names = {job["name"] for job in submitted}
                    self.assertIn("s000_asic1a", names)
                    self.assertNotIn("s999_asic1a", names)
                    self.assertFalse(any(name.endswith(("_bin1", "_unc13c")) for name in names))
                    self.assertEqual(adapter.waiting_target_counts(waiting, self.db.PRODUCT_TARGETS),
                                     dict.fromkeys(self.db.PRODUCT_TARGETS, 1))
                    for call in adapter.submit.call_args_list:
                        name = Path(call.args[0]).stem
                        self.assertEqual(call.kwargs["priority"],
                                         self.node.RESAMPLE_PRIORITY + 2
                                         if name == "s000_asic1a" else 0)
                    self.assertEqual(pick.call_count, 1)
                    previous = adapter.submit.call_count
                    refiller.refill(waiting)
                    self.assertEqual(adapter.submit.call_count, previous)
                    self.assertEqual(pick.call_count, 1)
                    refiller.deadline = 0
                    refiller.refill([])
                    self.assertEqual(adapter.submit.call_count, previous)

    def test_hq_wait_polls_refill_while_owned_tasks_run(self):
        spec = importlib.util.spec_from_file_location("af3_wait", ROOT / "af3适配" / "af3任务.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        callback = Mock()
        running = [{"id": 7, "name": "s000_htr1a", "task_stats": {"running": 1}}]
        finished = [{"id": 7, "name": "s000_htr1a", "task_stats": {"finished": 1}}]
        with patch.object(adapter, "queue_jobs", side_effect=[running, finished]), \
                patch.object(adapter.time, "sleep") as sleep:
            adapter.wait([7], on_poll=callback, poll_seconds=1)
            self.assertEqual(callback.call_count, 2)
            self.assertEqual(sleep.call_count, 1)

    def test_harvest_does_not_move_active_worker_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PEPOPT_DB_ROOT": directory}):
                result_root = Path(self.db.paths(ensure=True)["results"])
                self.db.register_sequence("s000", self.common.ORIG_TAIL, self.common.SEQ_202)
                sibling = result_root / "s000_asic1a_20261110_120000"
                sibling.mkdir()
                summary = sibling / "s000_asic1a_summary_confidences.json"
                summary.write_text(json.dumps({"iptm": 0.2}))
                self.assertEqual(self.db.harvest_results(exclude_jobs={"s000_asic1a"})[0], 0)
                self.assertTrue(summary.is_file())
                self.assertFalse(Path(self.db.canonical_dir("s000", "ASIC1A")).exists())
                self.assertEqual(self.db.harvest_results()[0], 1)

    def test_hq_error_never_falls_back_to_direct_execution(self):
        with patch.object(self.node, "run_af3_cycle_hq", side_effect=RuntimeError("HQ unavailable")), \
                patch.object(self.node.subprocess, "run") as direct:
            with self.assertRaisesRegex(RuntimeError, "HQ unavailable"):
                self.node.run_af3_cycle("unused")
            direct.assert_not_called()

    def test_slurm_entry_has_no_direct_execution_fallback(self):
        script = (ROOT / "作业脚本" / "AF3循环作业.sh").read_text()
        self.assertNotIn("PEPOPT_HQ_MODE=0", script)
        self.assertNotIn('eval "$PY_CMD"', script)
        self.assertIn("不允许直跑回退", script)

    def test_partial_hq_submission_fails_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "s000_htr1a.json").write_text("{}")
            (root / "s000_bin1.json").write_text("{}")
            adapter = Mock()
            adapter.queue_jobs.return_value = []
            adapter.submit.side_effect = [17, RuntimeError("rejected")]
            with patch.object(self.node, "load_hq_adapter", return_value=adapter):
                with self.assertRaisesRegex(RuntimeError, "rejected"):
                    self.node.run_af3_cycle_hq(str(root))
            adapter.wait.assert_not_called()

    def test_补算优先级按全库各靶标缺失序列数计算(self):
        注册表 = {f"s{编号:04d}": {"peptide": "SEQ"} for 编号 in range(1000)}
        评分表 = {标签: {"iptm_htr1a": 0.7, "iptm_asic1a": None} for 标签 in 注册表}
        评分表["s0000"]["iptm_htr1a"] = None
        评分表["s0001"]["iptm_asic1a"] = 0.0
        评分表["s0002"]["iptm_asic1a"] = float("nan")
        评分表["row_only"] = {"iptm_htr1a": 0.7, "iptm_asic1a": None}
        缺口 = self.node.全库靶标缺口(["HTR1A", "ASIC1A"], 注册表, 评分表)
        self.assertEqual(缺口, {"HTR1A": 1, "ASIC1A": 1000})
        self.assertEqual(self.node.resample_priority(缺口["ASIC1A"]), self.node.RESAMPLE_PRIORITY + 1000)
        评分表["s0002"]["iptm_asic1a"] = 0.4
        self.assertEqual(self.node.全库靶标缺口(["ASIC1A"], 注册表, 评分表), {"ASIC1A": 999})
        self.assertEqual(self.node.全库靶标缺口(["ASIC1A"], {}, {}), {"ASIC1A": 0})

    def test_af3_resample_priority_grows_with_deficit(self):
        priority = self.node.resample_priority
        self.assertGreater(priority(1), 0)
        self.assertGreater(priority(4), priority(1))
        self.assertGreater(priority(9), priority(4))
        spec = importlib.util.spec_from_file_location("af3_resample_wait", ROOT / "af3适配" / "af3任务.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with patch.object(adapter, "queue_jobs", return_value=[
                {"id": 7, "task_stats": {"running": 1}}]), \
                self.assertRaisesRegex(RuntimeError, "refill failed"):
            adapter.wait([7], on_poll=Mock(side_effect=RuntimeError("refill failed")))

    def test_hq_failed_or_missing_task_is_not_success(self):
        spec = importlib.util.spec_from_file_location("af3_strict_wait", ROOT / "af3适配" / "af3任务.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        for state in ("failed", "canceled", "aborted", "unknown"):
            with self.subTest(state=state), \
                    patch.object(adapter, "queue_jobs", return_value=[
                        {"id": 7, "task_stats": {state: 1}}]), \
                    self.assertRaises(RuntimeError):
                adapter.wait([7])
        with patch.object(adapter, "queue_jobs", return_value=[]), \
                self.assertRaisesRegex(RuntimeError, "缺少"):
            adapter.wait([7])

    def test_hq_submission_propagates_per_task_priorities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["s000_htr1a", "s001_htr1a", "s002_htr1a"]
            for name in names:
                (root / f"{name}.json").write_text("{}")
            priorities = {names[1]: self.node.resample_priority(4),
                          names[2]: self.node.resample_priority(1)}
            adapter = Mock()
            adapter.queue_jobs.return_value = []
            adapter.submit.side_effect = [17, 18, 19]
            with patch.object(self.node, "load_hq_adapter", return_value=adapter):
                self.node.run_af3_cycle(str(root), priority=priorities)
            self.assertEqual([call.kwargs["priority"]
                              for call in adapter.submit.call_args_list],
                             [priorities[names[1]], priorities[names[2]], 0])
            self.assertEqual([Path(call.args[0]).stem
                              for call in adapter.submit.call_args_list],
                             [names[1], names[2], names[0]])

    def test_resample_consolidation_passes_sample_deficits(self):
        module = self.inf.seq_sample
        fits = {f"s{index:03d}": Mock(max_x=0.8 - index / 10, n=1)
                for index in range(4)}
        fits["s000"].prob_gt.return_value = 1.0
        fits["s001"].prob_gt.return_value = 0.5
        fits["s002"].prob_gt.return_value = 0.75

        def request_samples(records, sampler):
            sampler("s001")
            sampler("s002")
            sampler("s000")
            return {"added": 0, "rounds": 1, "capped": 0}

        sampler = Mock(return_value={})
        with patch.object(self.db, "load_database", return_value=dict.fromkeys(fits, {})), \
                patch.object(self.db, "load_obj_fit", side_effect=fits.get), \
                patch.object(self.db, "save_obj_stats"), \
                patch.object(module, "consolidation_pass", side_effect=request_samples), \
                patch.dict(os.environ, {"PEPOPT_RESAMPLE_TAGS": "2"}):
            module.run_consolidation(sampler)
        sampler.assert_called_once_with(["s000", "s002"], n={"s000": 3, "s002": 2})

    def test_resample_job_labels_use_each_records_sample_deficit(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = {tag: {"peptide": self.common.SEQ_202} for tag in ("s000", "s001")}
            factors = ({"HTR1A": [0.5]}, {"ASIC1A": 0.5})
            with patch.object(self.db, "load_database", return_value=rows), \
                    patch.object(self.db, "results_dir", return_value=directory), \
                    patch.object(self.node.seq_sample, "collect_product_factors", return_value=factors), \
                    patch.object(self.db, "extract_sample_iptms", return_value=[0.6]), \
                    patch.object(self.node, "run_af3_cycle", return_value=0) as run:
                result = self.node.make_resample_fn(time.time() + 86400)(
                    ["s000", "s001"], n={"s000": 7, "s001": 1})
            self.assertEqual(set(result), {"s000", "s001"})
            labels = run.call_args.kwargs["priority"]
            self.assertEqual({name.split("_", 1)[0]: value for name, value in labels.items()},
                             {"s000": self.node.resample_priority(7),
                              "s001": self.node.resample_priority(1)})

    def test_bo_and_sequential_use_same_product(self):
        bo = self.inf.bo
        bo.set_active_targets(self.db.PRODUCT_TARGETS)
        row = {"iptm_htr1a": 0.9, "iptm_bin1": 0.8,
               "iptm_unc13c": 0.7, "iptm_asic1a": 0.6}
        self.assertAlmostEqual(bo.objective_of(row), 0.54)
        samples = self.inf.seq_sample.build_product_samples(
            {"ASIC1A": [0.6, 0.5]}, {"HTR1A": 0.9})
        self.assertEqual(len(samples), 2)
        for got, expected in zip(samples, [0.54, 0.45]):
            self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
