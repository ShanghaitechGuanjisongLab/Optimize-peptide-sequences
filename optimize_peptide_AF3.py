"""
蒙特卡洛优化小肽序列 — AlphaFold3 Web Server 版本 (全自动)
=============================================================
改动说明：
  1. ColabFold 输入（双链 FASTA + Singularity）→ AlphaFold3 Web Server JSON 提交
  2. ColabFold 评分（_scores_rank_001_*.json 中的 iptm）→ AlphaFold3 评分（summary_confidences.json 中的 ranking_score / iptm）
  3. 新增: af3_automator 模块实现浏览器全自动提交/等待/下载

使用方式 (全自动模式):
  1. 首次运行 setup_auth.py 完成 Google 登录认证
  2. 运行本脚本 → 自动生成 JSON → 自动提交 AF3 → 自动等待 → 自动下载 → 自动评分 → MC 优化

使用方式 (手动模式, 当自动化不可用时):
  - 脚本生成 JSON 文件到 af3_jobs/ 目录
  - 登录 https://alphafoldserver.com/ 手动上传 JSON 并运行
  - 下载结果 zip 包到 af3_results/ 目录
  - 脚本自动提取评分
"""

import random
import math
import os
import json
import glob
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import time

# 尝试导入自动化模块
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
    print("⚠️  af3_automator 模块未找到，将使用手动模式")

# ===================== 配置区域 =====================

# 小肽 202（仅优化后8位可变区）
SEQ_202 = "YGRKKRRQRRRSPVDVVCS"
FIXED_A = SEQ_202[:-8]          # 固定部分 "YGRKKRRQRRR"
ORIG_TAIL = SEQ_202[-8:]        # 原始可变区 "SPVDVVCS"

# 靶标蛋白 Htr1a (5HT1A)
HTR1A_SEQ = "MDVLSPGQGNNTTSPPAPFETGGNTTGISDVTVSYQVITSLLLGTLIFCAVLGNACVVAAIALERSLQNVANYLIGSLAVTDLMVSVLVLPMAALYQVLNKWTLGQVTCDLFIALDVLCCTSSILHLCAIALDRYWAITDPIDYVNKRTPRRAAALISLTWLIGFLISIPPMLGWRTPEDRSDPDACTISKDHGYTIYSTFGAFYIPLLLMLVLYGRIFRAARFRIRKTVKKVEKTGADTRHGASPAPQPKKSVNGESGSRNWRLGVESKAGGALCANGAVRQGDDGAALEVIEVHRVGNSKEHLPLPSEAGPTPCAPASFERKNERNAEAKRKMALARERKTVKTLGIIMGTFILCWLPFFIVALVLPFCESSCHMPTLLGAIINWLGYSNSLLNPVIYAYFNKDFQNAFKKIIKCKFCRQ"

# 2个底线蛋白
UNC13C_SEQ = "MVANFFKSLILPYIHKLCKGMFTKKLGNTNKNKEYRQQKKDQDFPTAGQTKSPKFSYTFKSTVKKIAKCSSTHNLSTEEDEASKEFSLSPTFSYRVAIANGLQKNAKVTNSDNEDLLQELSSIESSYSESLNELRSSTENQAQSTHTMPVRRNRKSSSSLAPSEGSSDGERTLHGLKLGALRKLRKWKKSQECVSSDSELSTMKKSWGIRSKSLDRTVRNPKTNALEPGFSSSGCISQTHDVMEMIFKELQGISQIETELSELRGHVNALKHSIDEISSSVEVVQSEIEQLRTGFVQSRRETRDIHDYIKHLGHMGSKASLRFLNVTEERFEYVESVVYQILIDKMGFSDAPNAIKIEFAQRIGHQRDCPNAKPRPILVYFETPQQRDSVLKKSYKLKGTGIGISTDILTHDIRERKEKGIPSSQTYESMAIKLSTPEPKIKKNNWQSPDDSDEDLESDLNRNSYAVLSKSELLTKGSTSKPSSKSHSARSKNKTANSSRISNKSDYDKISSQLPESDILEKQTTTHYADATPLWHSQSDFFTAKLSRSESDFSKLCQSYSEDFSENQFFTRTNGSSLLSSSDRELWQRKQEGTATLYDSPKDQHLNGGVQGIQGQTETENTETVDSGMSNGMVCASGDRSHYSDSQLSLHEDLSPWKEWNQGADLGLDSSTQEGFDYETNSLFDQQLDVYNKDLEYLGKCHSDLQDDSESYDLTQDDNSSPCPGLDNEPQGQWVGQYDSYQGANSNELYQNQNQLSMMYRSQSELQSDDSEDAPPKSWHSRLSIDLSDKTFSFPKFGSTLQRAKSALEVVWNKSTQSLSGYEDSGSSLMGRFRTLSQSTANESSTTLDSDVYTEPYYYKAEDEEDYTEPVADNETDYVEVMEQVLAKLENRTSITETDEQMQAYDHLSYETPYETPQDEGYDGPADDMVSEEGLEPLNETSAEMEIREDENQNIPEQPVEITKPKRIRPSFKEAALRAYKKQMAELEEKILAGDSSSVDEKARIVSGNDLDASKFSALQVCGGAGGGLYGIDSMPDLRRKKTLPIVRDVAMTLAARKSGLSLAMVIRTSLNNEELKMHVFKKTLQALIYPMSSTIPHNFEVWTATTPTYCYECEGLLWGIARQGMKCLECGVKCHEKCQDLLNADCLQRAAEKSSKHGAEDKTQTIITAMKERMKIREKNRPEVFEVIQEMFQISKEDFVQFTKAAKQSVLDGTSKWSAKITITVVSAQGLQAKDKTGSSDPYVTVQVGKNKRRTKTIFGNLNPVWDEKFYFECHNSTDRIKVRVWDEDDDIKSRVKQHFKKESDDFLGQTIVEVRTLSGEMDVWYNLEKRTDKSAVSGAIRLKINVEIKGEEKVAPYHIQYTCLHENLFHYLTEVKSNGGVKIPEVKGDEAWKVFFDDASQEIVDEFAMRYGIESIYQAMTHFSCLSSKYMCPGVPAVMSTLLANINAFYAHTTVSTNIQVSASDRFAATNFGREKFIKLLDQLHNSLRIDLSKYRENFPASNTERLQDLKSTVDLLTSITFFRMKVLELQSPPKASMVVKDCVRACLDSTYKYIFDNCHELYSQLTDPSKKQDIPREDQGPTTKNLDFWPQLITLMVTIIDEDKTAYTPVLNQFPQELNMGKISAEIMWTLFALDMKYALEEHENQRLCKSTDYMNLHFKVKWFYNEYVRELPAFKDAVPEYSLWFEPFVMQWLDENEDVSMEFLHGALGRDKKDGFQQTSEHALFSCSVVDVFAQLNQSFEIIKKLECPNPEALSHLMRRFAKTINKVLLQYAAIVSSDFSSHCDKENVPCILMNNIQQLRVQLEKMFESMGGKELDSEASTILKELQVKLSGVLDELSVTYGESFQVIIEECIKQMSFELNQMRANGNTTSNKNSAAMDAEIVLRSLMDFLDKTLSLSAKICEKTVLKRVLKELWKLVLNKIEKQIVLPPLTDQTGPQMIFIAAKDLGQLSKLKEHMIREDARGLTPRQCAIMEVVLATIKQYFHAGGNGLKKNFLEKSPDLQSLRYALSLYTQTTDALIKKFIDTQTSQSRSSKDAVGQISVHVDITATPGTGDHKVTVKVIAINDLNWQTTAMFRPFVEVCILGPNLGDKKRKQGTKTKSNTWSPKYNETFQFILGKENRPGAYELHLSVKDYCFAREDRIIGMTVIQLQNIAEKGSYGAWYPLLKNISMDETGLTILRILSQRTSDDVAKEFVRLKSETRSTEESA"
BIN1_SEQ = "MAEMGSKGVTAGKIASNVQKKLTRAQEKVLQKLGKADETKDEQFEQCVQNFNKQLTEGTRLQKDLRTYLASVKAMHEASKKLNECLQEVYEPDWPGRDEANKIAENNDLLWMDYHQKLVDQALLTMDTYLGQFPDIKSRIAKRGRKLVDYDSARHHYESLQTAKKKDEAKIAKPVSLLEKAAPQWCQGKLQAHLVAQTNLLRNQAEEELIKAQKVFEEMNVDLQEELPSLWNSRVGFYVNTFQSIAGLEENFHKEMSKLNQNLNDVLVGLEKQHGSNTFTVKAQPSDNAPAKGNKSPSPPDGSPAATPEIRVNHEPEPAGGATPGATLPKSPSQLRKGPPVPPPPKHTPSKEVKQEQILSLFEDTFVPEISVTTPSQFEAPGPFSEQASLLDLDFDPLPPVTSPVKAPTPSGQSIPWDLWEPTESPAGSLPSGEPSAAEGTFAVSWPSQTAEPGPAQPAEASEVAGGTQPAAGAQEPGETAASEAASSSLPAVVVETFPATVNGTVEGGSGAGRLDLPPGFMFKVQAQHDYTATDTDELQLKAGDVVLVIPFQNPEEQDEGWLMGVKESDWNQHKELEKCRGVFPENFTERVP"

# 蒙特卡洛参数
MC_STEPS = 100
TEMPERATURE = 0.7
ROOT_DIR = "output_af3/202_htr1a"
JSON_OUTPUT_DIR = "af3_jobs"        # 生成的 JSON 文件目录
RESULTS_DIR = "af3_results"         # 下载的结果 zip 目录

# AlphaFold3 Web Server 每日限制
MAX_JOBS_PER_BATCH = 100            # 单次 JSON 上传最大任务数
DAILY_LIMIT = 30                    # 每日免费配额

# 自动化模式开关
AUTO_MODE = True                    # True=自动提交到AF3服务器, False=手动模式

# =====================================================================

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def generate_mutant(orig_tail):
    """生成编辑距离≤2的突变序列（模式A/B/C）"""
    base = list(orig_tail)
    mode = random.choice(["A", "B", "C"])
    cur = base.copy()
    operations = 0

    if mode == "A":   # 点突变1-2个
        n_mut = random.choice([1, 2])
        positions = random.sample(range(len(cur)), n_mut)
        for pos in positions:
            old = cur[pos]
            new = random.choice([aa for aa in AA_LIST if aa != old])
            cur[pos] = new
        operations = n_mut

    elif mode == "B": # 仅长度变化（±1或±2）
        delta = random.choice([1, 2, -1, -2])
        new_len = len(cur) + delta
        if new_len < 6 or new_len > 10:
            return generate_mutant(orig_tail)
        if delta > 0:
            for _ in range(delta):
                cur.insert(random.randint(0, len(cur)), random.choice(AA_LIST))
            operations = delta
        else:
            positions = random.sample(range(len(cur)), -delta)
            for pos in sorted(positions, reverse=True):
                del cur[pos]
            operations = -delta

    else: # 模式C: 长度变化+点突变
        delta = random.choice([1, -1])
        new_len = len(cur) + delta
        if new_len < 6 or new_len > 10:
            return generate_mutant(orig_tail)
        if delta == 1:
            cur.insert(random.randint(0, len(cur)), random.choice(AA_LIST))
        else:
            del cur[random.randint(0, len(cur)-1)]
        operations = 1
        max_sub = 1
        n_mut = random.randint(1, max_sub)
        positions = random.sample(range(len(cur)), n_mut)
        for pos in positions:
            old = cur[pos]
            new = random.choice([aa for aa in AA_LIST if aa != old])
            cur[pos] = new
        operations += n_mut

    if operations > 2 or len(cur) < 6 or len(cur) > 10:
        return generate_mutant(orig_tail)
    return "".join(cur)


def create_af3_json(peptide_seq, target_seq, job_name, model_seeds=[1]):
    """
    创建 AlphaFold3 Web Server 格式的 JSON 任务。
    预测肽链(peptide_seq)与靶标(target_seq)的蛋白-蛋白复合物。

    返回: dict (AF3 JSON 格式)
    """
    job = {
        "name": job_name,
        "modelSeeds": model_seeds,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],           # 小肽链
                    "sequence": peptide_seq
                }
            },
            {
                "protein": {
                    "id": ["B"],           # 靶标蛋白链
                    "sequence": target_seq
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    return job


def extract_scores_from_zip(zip_path):
    """
    从 AlphaFold3 结果 zip 包中提取评分。

    AF3 结果 zip 结构:
      <job_name>/
        <job_name>_summary_confidences.json   ← 汇总评分
        <job_name>_confidences.json
        <job_name>_model_0.cif
        ...

    summary_confidences.json 关键字段:
      - iptm: 界面 TM-score (0~1), 衡量蛋白间界面置信度
      - ptm: 整体结构 TM-score (0~1)
      - ranking_score: 综合排名分 (0.8*iptm + 0.2*ptm + ...)
      - chain_pair_iptm: 链对 ipTM 矩阵, [0,1] 是 A-B 链界面
      - chain_pair_pae_min: 链对间最低 PAE
      - fraction_disordered: 无序比例
      - has_clash: 是否有严重原子冲突

    返回: dict {
        "iptm": float,
        "ptm": float,
        "ranking_score": float,
        "chain_pair_iptm_AB": float,   # A链-B链界面ipTM
        "fraction_disordered": float,
        "has_clash": bool
    }
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 查找 summary_confidences.json
            summary_files = [f for f in zf.namelist() if f.endswith('_summary_confidences.json')]

            if not summary_files:
                print(f"  ⚠️  {zip_path} 中未找到 summary_confidences.json")
                return None

            with zf.open(summary_files[0]) as f:
                conf = json.load(f)

            iptm = conf.get("iptm", 0.0)
            ptm = conf.get("ptm", 0.0)
            ranking_score = conf.get("ranking_score", 0.0)
            fraction_disordered = conf.get("fraction_disordered", 0.0)
            has_clash = conf.get("has_clash", False)

            # 提取 A-B 链界面 ipTM (chain_pair_iptm[0][1])
            chain_pair_iptm = conf.get("chain_pair_iptm", [])
            chain_pair_iptm_AB = 0.0
            if len(chain_pair_iptm) > 1 and len(chain_pair_iptm[0]) > 1:
                chain_pair_iptm_AB = chain_pair_iptm[0][1]

            # 提取链对最低 PAE
            chain_pair_pae = conf.get("chain_pair_pae_min", [])
            chain_pair_pae_AB = None
            if len(chain_pair_pae) > 1 and len(chain_pair_pae[0]) > 1:
                chain_pair_pae_AB = chain_pair_pae[0][1]

            return {
                "iptm": iptm,
                "ptm": ptm,
                "ranking_score": ranking_score,
                "chain_pair_iptm_AB": chain_pair_iptm_AB,
                "fraction_disordered": fraction_disordered,
                "has_clash": has_clash,
                "chain_pair_pae_AB": chain_pair_pae_AB
            }

    except Exception as e:
        print(f"  ❌ 解析 {zip_path} 失败: {e}")
        return None


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

    # 分批保存（每 DAILY_LIMIT 个一组，对应每天一批）
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


def optimize_peptide():
    """
    对小肽进行蒙特卡洛优化，寻找与 Htr1a 结合最强且对底线蛋白选择性好的变体。

    使用 AlphaFold3 Web Server：
      - 自动模式: JSON 生成 → 自动提交 → 等待 → 下载 → 评分 → MC
      - 手动模式: JSON 生成 → 手动上传/下载 → 评分 → MC
    """
    opt_dir = os.path.join(ROOT_DIR, "HTR1A_best")
    best_dir = os.path.join(opt_dir, "BEST_STRUCTURE")
    os.makedirs(best_dir, exist_ok=True)
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

    energy_history = []
    reject_count = 0

    current_A = SEQ_202
    current_E, _, _ = predict_and_score(current_A, HTR1A_SEQ, "HTR1A",
                                         os.path.join(opt_dir, "step_init"),
                                         auto_submit=auto_available)

    # 检查初始评分是否可用
    if current_E is None:
        print("\n❌ 初始序列评分未就绪。")
        if auto_available:
            print("   自动提交已触发，请等待完成后重新运行 optimize_peptide()")
        else:
            print("   请先在 AlphaFold Server 提交任务并下载结果。")
            print("   提示：运行 batch_generate_all_json() 可批量生成所有 JSON。")
        # 生成 JSON 文件
        generate_all_json_files(current_A, "step_init")
        return None, None

    best_A = current_A
    best_E = current_E
    best_result_path = None

    print(f"\n===== 优化 Htr1a 与小肽 202 | 初始分数: {current_E} =====")

    for step in range(MC_STEPS):
        new_tail = generate_mutant(ORIG_TAIL)
        new_A = FIXED_A + new_tail

        # Step A: 预测新突变对 Htr1a 的结合
        step_dir_htr1a = os.path.join(opt_dir, f"step_{step+1}_HTR1A")
        new_E, iptm_htr1a, result_path = predict_and_score(
            new_A, HTR1A_SEQ, "HTR1A", step_dir_htr1a, auto_submit=auto_available
        )

        # 检查结果是否就绪
        if new_E is None:
            print(f"  ⚠️  Step {step+1} 的 HTR1A 预测结果未就绪，跳过")
            energy_history.append(None)
            generate_all_json_files(new_A, f"step_{step+1}")
            continue

        # Step B: 预测新突变对两个底线蛋白的结合
        step_dir_unc13c = os.path.join(opt_dir, f"step_{step+1}_UNC13C")
        _, iptm_unc13c, _ = predict_and_score(
            new_A, UNC13C_SEQ, "UNC13C", step_dir_unc13c, auto_submit=auto_available
        )

        step_dir_bin1 = os.path.join(opt_dir, f"step_{step+1}_BIN1")
        _, iptm_bin1, _ = predict_and_score(
            new_A, BIN1_SEQ, "BIN1", step_dir_bin1, auto_submit=auto_available
        )

        if iptm_unc13c is None or iptm_bin1 is None:
            print(f"  ⚠️  Step {step+1} 的底线蛋白预测结果未就绪，跳过")
            energy_history.append(None)
            continue

        # 过滤条件：两个底线蛋白的ipTM均 ≥ 0.8，则放弃该突变
        if iptm_unc13c >= 0.8 and iptm_bin1 >= 0.8:
            reject_count += 1
            print(f"Step {step+1:2d} | ❌ 拒绝 (UNC13C ipTM={iptm_unc13c:.3f}, BIN1 ipTM={iptm_bin1:.3f} 均≥0.8, 选择性差)")
            energy_history.append(None)
            continue

        energy_history.append(new_E)

        dE = new_E - current_E
        if dE < 0 or random.random() < math.exp(-dE / TEMPERATURE):
            current_A, current_E = new_A, new_E
            if current_E < best_E:
                best_E, best_A = current_E, new_A
                if result_path and os.path.exists(result_path):
                    dest = os.path.join(best_dir, "BEST_peptide_HTR1A.zip")
                    shutil.copy(result_path, dest)
                    best_result_path = dest

        status = "接受" if current_A == new_A else "放弃"
        print(f"Step {step+1:2d} | {status} | HTR1A: {new_E:.4f} | UNC13C iptm={iptm_unc13c:.3f} | BIN1 iptm={iptm_bin1:.3f} | 当前分数: {current_E:.4f} | 最佳: {best_E:.4f}")

    # 绘制能量曲线
    valid_indices = [i+1 for i, e in enumerate(energy_history) if e is not None]
    valid_energies = [e for e in energy_history if e is not None]
    if valid_energies:
        plt.figure(figsize=(10, 5))
        plt.plot(valid_indices, valid_energies, 'b-', label='Score (1 - iptm)')
        plt.axhline(best_E, color='red', linestyle='--', label='Best')
        plt.xlabel("Step (valid mutations only)")
        plt.ylabel("Score (lower = better)")
        plt.title(f"Monte Carlo Optimization for HTR1A Binding Peptide (AlphaFold3)\n(Rejected {reject_count} mutations due to off-target binding)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(opt_dir, "optimization_curve_af3.png"))
        plt.close()

    # 保存最终结果到Excel
    df = pd.DataFrame([[best_A, best_E, reject_count]],
                      columns=["最优序列", "最优分数(1-iptm)", "因底线蛋白拒绝次数"])
    df.to_excel(os.path.join(opt_dir, "result_af3.xlsx"), index=False)

    print("\n" + "="*60)
    print("✅ 优化完成 (AlphaFold3)")
    print(f"🏆 最佳序列: {best_A}")
    print(f"📊 最佳分数: {best_E}  (ipTM = {1-best_E:.3f})")
    print(f"🚫 因底线蛋白(ipTM≥0.8)被拒绝: {reject_count} 次")
    print(f"📁 最佳结果: {best_result_path}")
    print("="*60)
    return best_A, best_E


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


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("🧬 AlphaFold3 Web Server — 蒙特卡洛肽优化")
    print("=" * 60)

    # 检查自动化状态
    if HAS_AUTOMATOR:
        auto_ready = check_playwright_installed() and check_auth_exists()
        mode_str = "🤖 全自动模式" if (AUTO_MODE and auto_ready) else "🖐️  半自动/手动模式"
        print(f"运行模式: {mode_str}")
        if not check_playwright_installed() and AUTO_MODE:
            print("  ⚠️  playwright-cli 未安装，无法使用自动模式")
            print("  安装: npm install -g @playwright/cli@latest")
        if not check_auth_exists() and AUTO_MODE:
            print("  ⚠️  未完成认证，请先运行 setup_auth.py")
    else:
        print("运行模式: 🖐️  手动模式 (af3_automator 未安装)")
    print()

    print("使用说明:")
    if HAS_AUTOMATOR and AUTO_MODE and check_auth_exists():
        print("  🤖 自动模式: 脚本将自动提交到 AlphaFold Server")
        print("  1. batch_generate_all_json() → 生成 JSON + 自动提交")
        print("  2. optimize_peptide() → 自动评分 + MC 优化")
        print("  3. check_results_status() → 查看进度")
    else:
        print("  🖐️  手动模式:")
        print("  1. batch_generate_all_json() → 批量生成 JSON")
        print("  2. 登录 https://alphafoldserver.com/ → 上传 JSON 并运行")
        print("  3. 下载结果 zip → 放入 af3_results/ 目录")
        print("  4. optimize_peptide() → 自动提取评分并完成优化")
    print()

    # 第一步：批量生成 JSON (并尝试自动提交)
    batch_generate_all_json()

    # 检查结果并尝试优化
    completed, pending = check_results_status()
    if completed > 0:
        print("\n🚀 已有结果可用，尝试运行蒙特卡洛优化...")
        best_seq, best_score = optimize_peptide()
    else:
        print("\n" + "=" * 60)
        if HAS_AUTOMATOR and AUTO_MODE and check_auth_exists():
            print("📋 任务已自动提交，等待 AlphaFold Server 完成预测...")
            print("   完成后重新运行脚本即可自动提取评分并完成优化")
        else:
            print("📋 下一步操作:")
            print("  1. 登录 https://alphafoldserver.com/")
            print("  2. 点击 Upload JSON，上传 af3_jobs/batch_001.json")
            print("  3. 等待预测完成，下载所有结果 zip")
            print("  4. 将 zip 文件放入 af3_results/ 目录")
            print("  5. 重新运行脚本即可自动提取评分并完成 MC 优化")
        print("=" * 60)
