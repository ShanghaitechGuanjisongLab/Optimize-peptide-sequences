"""
小肽序列优化 — AlphaFold Server 云端预测后端库。

仅当使用官方 AlphaFold Server 作为预测后端时导入本模块
（入口: optimize_peptide_AF3.py；BO 入口加 --server）。

云端专属内容：
  - 任务 JSON 生成与批量分批逻辑（af3_jobs/）
  - 结果 zip 评分缓存（af3_results/）
  - 可选的 af3_automator 浏览器自动化（缺失时静默降级为手动模式）
  - 云端版蒙特卡洛编排（自动化可用性检查、待处理任务处理）

通用逻辑（序列、突变规则、评分提取、MC 主循环）在 peptide_common.py。
"""

import os
import json
import glob
import shutil
import random

import numpy as np

from peptide_common import (SEQ_202, FIXED_A, ORIG_TAIL, ROOT_DIR,
                            MC_STEPS, TEMPERATURE,
                            HTR1A_SEQ, UNC13C_SEQ, BIN1_SEQ,
                            generate_mutant, create_af3_json,
                            extract_scores_from_zip, run_monte_carlo)

# 尝试导入自动化模块（静默；本库只在云端入口被导入，缺失属正常情况）
try:
    from af3_automator import (
        submit_batch_and_collect,
        submit_multiple_batches,
        collect_results,
        check_playwright_installed,
        check_auth_exists,
        get_remaining_quota,
        DAILY_LIMIT,
        JSON_DIR as AUTO_JSON_DIR,
        RESULTS_DIR as AUTO_RESULTS_DIR,
    )
    HAS_AUTOMATOR = True
except ImportError:
    HAS_AUTOMATOR = False

    # 降级占位：保证模块接口始终完整，调用点在 HAS_AUTOMATOR 判断下不会触达
    def submit_batch_and_collect(*args, **kwargs):
        raise RuntimeError("af3_automator 模块未安装，无法自动提交")

    def submit_multiple_batches(*args, **kwargs):
        raise RuntimeError("af3_automator 模块未安装，无法自动提交")

    def collect_results(*args, **kwargs):
        return []

    def check_playwright_installed():
        return False

    def check_auth_exists():
        return False

    def get_remaining_quota():
        return 0

    DAILY_LIMIT = 30

    # 占位实现: 保证本库的公共接口在缺少 af3_automator 时依然完整，
    # 调用方无需做条件判断。
    def submit_batch_and_collect(json_file, wait=True, max_wait_minutes=120):
        print("  ⚠️  af3_automator 不可用，无法自动提交，请手动上传 JSON")
        return 0

    def check_playwright_installed():
        return False

    def check_auth_exists():
        return False

    def get_remaining_quota():
        return 0

# ===================== 云端后端配置 =====================

JSON_OUTPUT_DIR = "af3_jobs"        # 生成的 JSON 文件目录
RESULTS_DIR = "af3_results"         # 下载的结果 zip 目录

# AlphaFold3 Web Server 每日限制
MAX_JOBS_PER_BATCH = 100            # 单次 JSON 上传最大任务数
DAILY_LIMIT = 30                    # 每日免费配额（af3_automator 存在时被其覆盖）

# 自动化模式开关
AUTO_MODE = True                    # True=自动提交到AF3服务器, False=手动模式

# =====================================================================


def predict_and_score(peptide_seq, target_seq, target_name, out_dir, auto_submit=True):
    """
    预测 peptide_seq (小肽) 与 target_seq 的复合物，返回 (score, iptm, best_cif)。

    工作流程：
      1. 生成 AF3 JSON 文件 → af3_jobs/
      2. 检查 af3_results/ 中是否已有对应结果 zip
      3. 如有结果则直接提取评分
      4. 如无结果且 auto_submit=True: 自动提交到 AF3 Server，等待并下载
      5. 如无结果且 auto_submit=False: 提示用户手动上传

    评分策略：score = 1 - iptm（值越小代表结合越强，与原版一致）
    同时记录 ranking_score 作为辅助参考。

    target_name: 靶标名称（如 "HTR1A", "UNC13C", "BIN1"）
    auto_submit: 是否尝试自动提交 (需要 af3_automator 模块和认证)
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    job_name = os.path.basename(out_dir)

    # ---- 第1步：检查是否已有结果 zip ----
    result_zip = os.path.join(RESULTS_DIR, f"{job_name}.zip")
    if os.path.exists(result_zip):
        print(f"  ⏭️  [{target_name}] 发现已有结果 {job_name}.zip，直接提取评分")
        scores_dict = extract_scores_from_zip(result_zip)
        if scores_dict:
            iptm = scores_dict["iptm"]
            ranking_score = scores_dict["ranking_score"]
            score = round(1.0 - iptm, 4)
            print(f"  📊 [{target_name}] 评分: score={score} (iptm={iptm:.3f}, ranking={ranking_score:.3f}, "
                  f"AB-iptm={scores_dict['chain_pair_iptm_AB']:.3f}, "
                  f"disorder={scores_dict['fraction_disordered']:.3f}, clash={scores_dict['has_clash']})")
            return score, iptm, result_zip
        else:
            print(f"  ⚠️  [{target_name}] 结果 zip 解析失败，将重新生成 JSON")

    # ---- 第2步：生成 AF3 JSON 文件 ----
    json_file = os.path.join(JSON_OUTPUT_DIR, f"{job_name}.json")
    job = create_af3_json(peptide_seq, target_seq, job_name)
    with open(json_file, "w") as f:
        json.dump(job, f, indent=2)
    print(f"  📝 [{target_name}] JSON 已生成: {json_file}")

    # ---- 第3步：尝试自动提交 ----
    if auto_submit and HAS_AUTOMATOR and AUTO_MODE:
        if check_auth_exists():
            remaining = get_remaining_quota()
            if remaining <= 0:
                print(f"  ⚠️  [{target_name}] 今日配额已用完 ({DAILY_LIMIT}/{DAILY_LIMIT})，等待手动处理")
                print(f"  ⚠️  明天重新运行脚本可自动继续")
                return None, None, None

            print(f"  🤖 [{target_name}] 自动提交到 AlphaFold3 Server...")
            downloaded = submit_batch_and_collect(
                json_file,
                wait=True,
                max_wait_minutes=120
            )

            if downloaded > 0:
                # 重新检查结果
                result_zip = os.path.join(RESULTS_DIR, f"{job_name}.zip")
                if os.path.exists(result_zip):
                    scores_dict = extract_scores_from_zip(result_zip)
                    if scores_dict:
                        iptm = scores_dict["iptm"]
                        score = round(1.0 - iptm, 4)
                        print(f"  📊 [{target_name}] 自动评分: score={score} (iptm={iptm:.3f})")
                        return score, iptm, result_zip
                else:
                    # 尝试从下载中匹配
                    for f in os.listdir(RESULTS_DIR):
                        if f.endswith('.zip') and job_name in f:
                            result_zip = os.path.join(RESULTS_DIR, f)
                            scores_dict = extract_scores_from_zip(result_zip)
                            if scores_dict:
                                iptm = scores_dict["iptm"]
                                score = round(1.0 - iptm, 4)
                                print(f"  📊 [{target_name}] 自动评分: score={score} (iptm={iptm:.3f})")
                                return score, iptm, result_zip
            else:
                print(f"  ⚠️  [{target_name}] 自动提交未获取到结果，可能需要手动处理")
        else:
            print(f"  ⚠️  [{target_name}] 未完成认证，无法自动提交")
            print(f"  ⚠️  请先运行 setup_auth.py 完成首次登录")

    # ---- 第4步：提示用户手动操作（当自动提交不可用时） ----
    print(f"  ⚠️  [{target_name}] 请在 AlphaFold Server 上传此 JSON 并运行预测")
    print(f"  ⚠️  登录 https://alphafoldserver.com/ → Upload JSON → 选择 {json_file}")
    print(f"  ⚠️  完成后下载结果 zip，重命名为 {job_name}.zip 放入 {RESULTS_DIR}/")
    print(f"  ⚠️  然后重新运行脚本即可自动提取评分")

    # 返回占位值，表示该任务尚未完成
    return None, None, None


def generate_all_json_files(peptide_seq, step_label=""):
    """
    为给定肽序列生成所有需要的 AF3 JSON 文件（HTR1A、UNC13C、BIN1）。
    方便一次性批量上传。
    """
    jobs = []

    for target_name, target_seq, label in [
        ("HTR1A", HTR1A_SEQ, "htr1a"),
        ("UNC13C", UNC13C_SEQ, "unc13c"),
        ("BIN1", BIN1_SEQ, "bin1")
    ]:
        job_name = f"{step_label}_{label}" if step_label else label
        json_file = os.path.join(JSON_OUTPUT_DIR, f"{job_name}.json")
        job = create_af3_json(peptide_seq, target_seq, job_name)
        with open(json_file, "w") as f:
            json.dump(job, f, indent=2)
        jobs.append(job)
        print(f"  📝 JSON 已生成: {json_file} (小肽 vs {target_name})")

    return jobs


def collect_pending_jobs():
    """收集所有已生成 JSON 但尚未有结果的作业"""
    json_files = glob.glob(os.path.join(JSON_OUTPUT_DIR, "*.json"))
    pending = []
    for jf in json_files:
        job_name = os.path.splitext(os.path.basename(jf))[0]
        result_zip = os.path.join(RESULTS_DIR, f"{job_name}.zip")
        if not os.path.exists(result_zip):
            pending.append((job_name, jf))
    return pending


def batch_generate_all_json():
    """
    批量生成当前 MC 步骤需要的所有 JSON 文件。
    同时尝试自动提交到 AlphaFold Server。

    工作流程:
      1. 预生成所有突变序列
      2. 为每个序列×靶标生成 JSON
      3. 自动分批提交到 AlphaFold Server (如果可用)
      4. 等待结果并下载
    """
    print("\n" + "=" * 60)
    print("📋 批量生成 AlphaFold3 JSON 任务")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # 先生成所有突变，收集去重后的序列
    mutant_set = set()
    mutant_set.add(SEQ_202)  # 初始序列

    current_tail = ORIG_TAIL
    for step in range(MC_STEPS):
        new_tail = generate_mutant(current_tail)
        new_A = FIXED_A + new_tail
        mutant_set.add(new_A)

    total_jobs = len(mutant_set) * 3  # 每个序列3个靶标
    print(f"🔬 共 {len(mutant_set)} 个唯一肽序列需要预测")
    print(f"📊 每个序列需要3个靶标(HTR1A, UNC13C, BIN1) = 共 {total_jobs} 个预测任务")
    print(f"⚠️  每日免费配额: {DAILY_LIMIT} 个任务")
    print(f"📅 预计需要 {total_jobs / DAILY_LIMIT:.1f} 天完成\n")

    # 收集所有 job
    all_jobs = []
    for i, seq in enumerate(sorted(mutant_set)):
        if seq == SEQ_202:
            step_label = "step_init"
        else:
            step_label = f"seq_{i:03d}"

        for target_name, target_seq, label in [
            ("HTR1A", HTR1A_SEQ, "htr1a"),
            ("UNC13C", UNC13C_SEQ, "unc13c"),
            ("BIN1", BIN1_SEQ, "bin1")
        ]:
            job_name = f"{step_label}_{label}"
            job = create_af3_json(seq, target_seq, job_name)
            all_jobs.append(job)

    # 分批保存（每 MAX_JOBS_PER_BATCH 个一组，对应单次上传）
    for batch_idx in range(0, len(all_jobs), MAX_JOBS_PER_BATCH):
        batch = all_jobs[batch_idx:batch_idx + MAX_JOBS_PER_BATCH]
        batch_num = batch_idx // MAX_JOBS_PER_BATCH + 1
        batch_file = os.path.join(JSON_OUTPUT_DIR, f"batch_{batch_num:03d}.json")
        with open(batch_file, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"✅ 批次 {batch_num}: {len(batch)} 个任务 → {batch_file}")

    # 保存序列索引文件
    index = {}
    for i, seq in enumerate(sorted(mutant_set)):
        step_label = f"seq_{i:03d}" if seq != SEQ_202 else "step_init"
        index[step_label] = seq
    with open(os.path.join(JSON_OUTPUT_DIR, "sequence_index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n📦 全部 JSON 已生成到 {JSON_OUTPUT_DIR}/")

    # ---- 尝试自动提交 ----
    if HAS_AUTOMATOR and AUTO_MODE and check_auth_exists():
        print(f"\n🤖 自动提交模式已启用")
        remaining = get_remaining_quota()
        if remaining > 0:
            print(f"🚀 开始自动提交第1批 (今日剩余配额: {remaining})...")
            batch_file = os.path.join(JSON_OUTPUT_DIR, "batch_001.json")
            if os.path.exists(batch_file):
                submit_batch_and_collect(batch_file, wait=True, max_wait_minutes=180)
        else:
            print(f"⚠️  今日配额已用完，请明天重新运行以提交下一批")
    else:
        print(f"\n📤 登录 https://alphafoldserver.com/ 上传 batch 文件")
        print(f"📥 下载结果 zip 后放入 {RESULTS_DIR}/ 目录")
        print(f"🔄 然后重新运行脚本即可自动提取评分并完成优化\n")

    return all_jobs


def check_results_status():
    """检查当前结果完成状态"""
    json_files = glob.glob(os.path.join(JSON_OUTPUT_DIR, "*.json"))
    result_zips = glob.glob(os.path.join(RESULTS_DIR, "*.zip"))

    # 从 JSON 中提取 job 名称
    json_jobs = set()
    for jf in json_files:
        if os.path.basename(jf).startswith("batch_"):
            continue  # 跳过批量文件
        json_jobs.add(os.path.splitext(os.path.basename(jf))[0])

    # 从 zip 中提取 job 名称
    zip_jobs = set()
    for zf in result_zips:
        zip_jobs.add(os.path.splitext(os.path.basename(zf))[0])

    completed = json_jobs & zip_jobs
    pending_jobs = json_jobs - zip_jobs

    print(f"\n📊 任务状态报告")
    print(f"   JSON 任务总数: {len(json_jobs)}")
    print(f"   已完成 (有结果): {len(completed)}")
    print(f"   待处理: {len(pending_jobs)}")

    if pending_jobs:
        print(f"\n   待处理任务 (前20):")
        for j in sorted(pending_jobs)[:20]:
            print(f"     - {j}")
        if len(pending_jobs) > 20:
            print(f"     ... 还有 {len(pending_jobs) - 20} 个")

    return len(completed), len(pending_jobs)


def optimize_with_server(mc_steps=MC_STEPS, temperature=TEMPERATURE):
    """
    云端版蒙特卡洛编排：检查自动化可用性与待处理任务，
    然后调用通用 MC 主循环（预测函数 = 云端 predict_and_score）。
    """
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 检查自动化可用性
    auto_available = HAS_AUTOMATOR and AUTO_MODE and check_auth_exists()
    if AUTO_MODE and not auto_available:
        print("\n⚠️  自动模式已启用但不可用:")
        if not HAS_AUTOMATOR:
            print("   - af3_automator 模块未找到")
        elif not check_auth_exists():
            print("   - 未完成认证 (请运行 setup_auth.py)")
        print("   - 将降级为手动模式\n")

    # 检查待处理任务
    pending = collect_pending_jobs()
    if pending:
        print(f"\n⚠️  发现 {len(pending)} 个待处理任务（已生成 JSON 但无结果 zip）")
        if auto_available:
            print(f"   🤖 将自动提交这些任务...")
            for name, jf in pending:
                print(f"   📤 提交: {name}")
                submit_batch_and_collect(jf, wait=True, max_wait_minutes=120)
        else:
            print("   请先在 AlphaFold Server 提交这些任务并下载结果。")
            print("   待处理任务列表:")
            for name, jf in pending[:10]:
                print(f"     - {name}")
            if len(pending) > 10:
                print(f"     ... 还有 {len(pending) - 10} 个")
        print()

    return run_monte_carlo(predict_and_score, ROOT_DIR,
                           mc_steps=mc_steps, temperature=temperature,
                           on_not_ready=lambda seq, label: generate_all_json_files(seq, label))
