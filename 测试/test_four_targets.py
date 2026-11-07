"""四靶标扫描的离线回归测试；不提交作业、不访问生产数据库写路径。"""

import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FourTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="pepopt-four-test-")
        cls.root = Path(cls.tmp.name)
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

    def test_product_requires_all_four(self):
        row = {"iptm_htr1a": 0.9, "iptm_bin1": 0.8,
               "iptm_unc13c": 0.7, "iptm_asic1a": 0.6}
        self.assertEqual(self.db.PRODUCT_TARGETS,
                         ("HTR1A", "BIN1", "UNC13C", "ASIC1A"))
        self.assertAlmostEqual(self.db.objective_product(row), 0.9 * 0.8 * 0.7 * 0.6)
        del row["iptm_asic1a"]
        self.assertIsNone(self.db.objective_product(row))

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
        self.assertEqual(self.inf.DEFAULT_TARGETS, "HTR1A,BIN1,UNC13C,ASIC1A")

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
        self.assertIn("PEPOPT_TARGETS=HTR1A+BIN1+UNC13C+ASIC1A", export)
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
            self.assertEqual((nseq, ntask), (1, 2))
            self.assertEqual({p.stem.rsplit("_", 1)[-1] for p in cycle.glob("*.json")},
                             {"unc13c", "asic1a"})
        finally:
            os.environ["PEPOPT_DB_ROOT"] = old_root

    def test_bo_and_sequential_use_same_product(self):
        bo = self.inf.bo
        bo.set_active_targets(self.db.PRODUCT_TARGETS)
        row = {"iptm_htr1a": 0.9, "iptm_bin1": 0.8,
               "iptm_unc13c": 0.7, "iptm_asic1a": 0.6}
        self.assertAlmostEqual(bo.objective_of(row), 0.3024)
        samples = self.inf.seq_sample.build_product_samples(
            {"UNC13C": [0.7, 0.6], "ASIC1A": [0.6, 0.5]},
            {"HTR1A": 0.9, "BIN1": 0.8})
        self.assertEqual(len(samples), 2)
        for got, expected in zip(samples, [0.3024, 0.216]):
            self.assertAlmostEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
