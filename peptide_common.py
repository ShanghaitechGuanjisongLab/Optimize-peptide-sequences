"""
小肽序列优化 — 公共库。

内容（纯库，导入零副作用，不依赖 af3_automator 与任何预测后端）：
  - 序列常量: SEQ_202 / HTR1A_SEQ / UNC13C_SEQ / BIN1_SEQ
  - 突变规则: generate_mutant()（编辑距离 ≤ 2）
  - AF3 任务格式: create_af3_json()
  - 评分提取: extract_scores_from_zip()
  - 蒙特卡洛主循环: run_monte_carlo(predict_fn, opt_root)
    后端无关，predict_fn 由各入口（云端/本地）注入

评分说明: score = 1 - ipTM 是界面置信度的代理指标，并非真实亲和度。
"""

import random
import math
import os
import json
import zipfile
import shutil

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
ROOT_DIR = "output_af3/202_htr1a"   # 优化结果输出根目录



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




def run_monte_carlo(predict_fn, opt_root, mc_steps=100, temperature=0.7,
                    off_threshold=0.8, on_not_ready=None):
    """
    后端无关的蒙特卡洛优化主循环。

    predict_fn(peptide_seq, target_seq, target_name, out_dir, auto_submit=True)
        → (score, iptm, result_path)；结果未就绪时返回 (None, None, None)。
    opt_root: 输出根目录（最佳结构与结果表写入其中）。
    """
    opt_dir = os.path.join(opt_root, "HTR1A_best")
    best_dir = os.path.join(opt_dir, "BEST_STRUCTURE")
    os.makedirs(best_dir, exist_ok=True)

    energy_history = []
    reject_count = 0

    current_A = SEQ_202
    current_E, _, _ = predict_fn(current_A, HTR1A_SEQ, "HTR1A",
                                 os.path.join(opt_dir, "step_init"))
    if current_E is None:
        print("\n❌ 初始序列评分未就绪，请等待预测结果后重跑本脚本。")
        return None, None

    best_A, best_E = current_A, current_E
    best_result_path = None
    print(f"\n===== 优化 Htr1a 与小肽 202 | 初始分数: {current_E} =====")

    for step in range(mc_steps):
        new_tail = generate_mutant(ORIG_TAIL)
        new_A = FIXED_A + new_tail

        step_dir_htr1a = os.path.join(opt_dir, f"step_{step+1}_HTR1A")
        new_E, iptm_htr1a, result_path = predict_fn(
            new_A, HTR1A_SEQ, "HTR1A", step_dir_htr1a)
        if new_E is None:
            print(f"  ⚠️  Step {step+1} 的 HTR1A 预测结果未就绪，跳过")
            energy_history.append(None)
            if on_not_ready is not None:
                on_not_ready(new_A, f"step_{step+1}")
            continue

        step_dir_unc13c = os.path.join(opt_dir, f"step_{step+1}_UNC13C")
        _, iptm_unc13c, _ = predict_fn(new_A, UNC13C_SEQ, "UNC13C", step_dir_unc13c)
        step_dir_bin1 = os.path.join(opt_dir, f"step_{step+1}_BIN1")
        _, iptm_bin1, _ = predict_fn(new_A, BIN1_SEQ, "BIN1", step_dir_bin1)
        if iptm_unc13c is None or iptm_bin1 is None:
            print(f"  ⚠️  Step {step+1} 的底线蛋白预测结果未就绪，跳过")
            energy_history.append(None)
            continue

        if iptm_unc13c >= off_threshold and iptm_bin1 >= off_threshold:
            reject_count += 1
            print(f"Step {step+1:2d} | ❌ 拒绝 (UNC13C ipTM={iptm_unc13c:.3f}, "
                  f"BIN1 ipTM={iptm_bin1:.3f} 均≥{off_threshold}, 双双越过结合红线)")
            energy_history.append(None)
            continue

        energy_history.append(new_E)
        dE = new_E - current_E
        if dE < 0 or random.random() < math.exp(-dE / temperature):
            current_A, current_E = new_A, new_E
            if current_E < best_E:
                best_E, best_A = current_E, new_A
                if result_path and os.path.exists(result_path):
                    dest = os.path.join(best_dir, "BEST_peptide_HTR1A.zip")
                    shutil.copy(result_path, dest)
                    best_result_path = dest

        status = "接受" if current_A == new_A else "放弃"
        print(f"Step {step+1:2d} | {status} | HTR1A: {new_E:.4f} | "
              f"UNC13C iptm={iptm_unc13c:.3f} | BIN1 iptm={iptm_bin1:.3f} | "
              f"当前分数: {current_E:.4f} | 最佳: {best_E:.4f}")

    # 绘制能量曲线与保存结果表
    valid_indices = [i+1 for i, e in enumerate(energy_history) if e is not None]
    valid_energies = [e for e in energy_history if e is not None]
    if valid_energies:
        plt.figure(figsize=(10, 5))
        plt.plot(valid_indices, valid_energies, 'b-', label='Score (1 - iptm)')
        plt.axhline(best_E, color='red', linestyle='--', label='Best')
        plt.xlabel("Step (valid mutations only)")
        plt.ylabel("Score (lower = better)")
        plt.title("Monte Carlo Optimization for HTR1A Binding Peptide")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(opt_dir, "optimization_curve_af3.png"))
        plt.close()

    df = pd.DataFrame([[best_A, best_E, reject_count]],
                      columns=["最优序列", "最优分数(1-iptm)", "因底线蛋白拒绝次数"])
    df.to_excel(os.path.join(opt_dir, "result_af3.xlsx"), index=False)

    print("\n" + "=" * 60)
    print("✅ 蒙特卡洛优化完成")
    print(f"🏆 最佳序列: {best_A}")
    print(f"📊 最佳分数: {best_E}  (ipTM = {1-best_E:.3f})")
    print(f"🚫 因底线蛋白(ipTM≥{off_threshold})被拒绝: {reject_count} 次")
    print(f"📁 最佳结果: {best_result_path}")
    print("=" * 60)
    return best_A, best_E

