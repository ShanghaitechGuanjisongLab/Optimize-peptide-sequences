**小肽序列优化**：基于 AlphaFold3 的小肽序列优化脚本仓库（蒙特卡洛 / 贝叶斯优化两种搜索策略，支持官方服务器与本地集群推理两种预测后端）。

# 简介

本仓库提供围绕肽 **202**（`YGRKKRRQRRRSPVDVVCS`）的序列优化脚本：仅对 C 端 8 位可变区做突变搜索，目标是找到与靶标蛋白 **HTR1A（5-HT1A 受体）** **结合更强**、同时对两个底线蛋白 **UNC13C** 和 **BIN1** **结合更弱**的变体。（两者均以各复合物预测的 ipTM 作代理指标：肽–HTR1A 的 ipTM **越高越好**，肽–UNC13C / 肽–BIN1 的 ipTM **越低越好**——"底线"指后两者的结合红线（上限）而非保底下限：若变体对两者仍双高结合（均 ≥ 0.8），视为结合谱未完成重定向，不予接受；且均为预测意义，最终需实验验证。）共三个入口：

| 入口文件 | 搜索策略 | 预测后端 |
| --- | --- | --- |
| `optimize_peptide_AF3.py` | 蒙特卡洛（原版） | AlphaFold Server（手动上传/下载） |
| `optimize_peptide_BO.py` | **贝叶斯优化（推荐）** | 默认本地集群 GPU 推理；`--server` 切换为 AlphaFold Server |
| `optimize_peptide_local.py` | 蒙特卡洛 | 本地集群 GPU 推理（自动提交 SLURM 作业，重跑续作） |
| `peptide_common.py` | —（公共库） | 序列常量、突变规则、AF3 JSON 格式、评分提取、后端无关的蒙特卡洛主循环；导入零副作用 |
| `peptide_server.py` | —（云端后端库） | 云端 JSON 批量/缓存/可选浏览器自动化编排，仅云端入口导入 |

# 研究背景

**疾病与动物模型**：阿尔茨海默症（AD）发病率随年龄陡增（65 岁以上约 15%、85 岁以上接近 50%），核心症状为学习记忆损伤，常伴语言/运动功能下降与情绪异常；病理特征以 β 淀粉样蛋白（Aβ）沉积形成的老年斑为代表，同时伴随神经可塑性损伤、神经元群体兴奋性异常与海马/皮层环路进行性退化。本研究采用的 **5xFAD** 模型小鼠转入 3 个人源 APP 突变（瑞典 K670N/M671L、佛罗里达 I716V、伦敦 V717I）与 2 个人源 PSEN1 突变（M146L、L286V），1.5 月龄即出现胞内 Aβ42 聚集、2 月龄起形成斑块、4 月龄出现记忆损害，较好模拟了 AD 病程。电生理层面，AD 的工作记忆损伤与 **θ-γ 跨频耦合紊乱**密切相关，且这种节律异常甚至早于 Aβ 斑块出现，可能作为早期病理标志。

**QD202 是什么**：QD202 是上海魁特迪生物科技有限公司（英文名 QuietD，"QD" 即其缩写，"202" 为化合物编号，故组内简称"202"）研发的 1 类新药（19 个氨基酸的小肽），拟用于轻中度 AD 治疗，2023 年进入临床试验阶段（另有急性缺血性卒中适应症），作用机制为**调控囊泡型质子泵（V-ATPase）功能**——囊泡型质子泵功能减退是 AD 等退行性神经疾病的重要发病机制之一。序列结构上，前 11 位是穿膜肽（TAT 47-57：`YGRKKRRQRRR`，源自 HIV-1 Tat 蛋白转导域，负责携带入胞），后 8 位是主要结合片段（即本仓库优化的可变区）。前期实验已证明：QD202 靶点之一位于海马的**突触囊泡和溶酶体**，通过增强 V-ATPase 氢离子内流提高神经元溶酶体酸度，膜片钳记录显示其能提高 EPSC 频率。

**靶点初筛与三个蛋白的来历**：前期用 AlphaFold3 预测了 QD202 与小鼠突触囊泡蛋白的结合，初筛 ipTM>0.6 的命中者共 13 个（BTBD8、BIN1、PRKCG、PRKCB、PACSIN1、GIT1、GPR151、RIMBP2、SCRIB、UNC13C、OTOF、GIT2、ARPC2），多集中于突触囊泡的胞吞/胞吐及其调控；且 ATP 酶各亚基与 QD202 **不直接结合**，提示 QD202 可能通过间接/竞争性方式调控 V-ATPase。命中者中与 QD202 预测结合最强的两个正是本仓库的"底线蛋白"：

- **UNC13C**（ipTM=0.81）：突触囊泡 priming 因子，活性区核心蛋白，富集于海马/皮层/小脑；
- **BIN1**（ipTM=0.82）：BAR+SH3 结构域衔接蛋白，介导突触囊泡内吞；同时 BIN1 是继 APOE 之后最强的晚发型 AD 遗传风险基因，与 tau 病理强相关，功能受扰动可能加剧病理。

**本仓库的角色**：在上述基础上做序列工程——保持 TAT 穿膜段固定，只改造后 8 位结合片段，目标是**与 HTR1A（5-HT1A 受体）结合更强、同时对 UNC13C / BIN1 结合更弱**。**研究假设**：QD202（或其优化变体）经 TAT 段入胞后结合**海马**的 5-HT1A 受体并**发挥激活（激动）作用**；受体被激活后反过来**抑制** AD 早期海马 CA1 锥体神经元的过度兴奋，从而缓解 θ-γ 节律紊乱与记忆损害，并与 QD202 已知的 V-ATPase/溶酶体酸化机制形成互补。

选择 HTR1A 的依据：5-HT1A 是 Gi/o 偶联 GPCR（421 aa，7 次跨膜），异源受体广泛分布于海马等前脑边缘系统并参与记忆过程；AD 早期海马锥体神经元过度兴奋与 θ-γ 节律紊乱是公认病理特征（Busche & Konnerth 2015）。**海马方向的直接支持**来自 Wang et al.（2023, *Cell Reports*, doi:10.1016/j.celrep.2023.112152，浙江大学孙秉贵组，hAPP-J20 小鼠）：该研究发现 AD 模型海马的 5-HT/5-HT1aR 信号**减弱**，而化学遗传学激活中缝正中核（MRN）5-HT 神经元可经 5-HT3aR/5-HT1aR **降低** CA1 锥体神经元活动并**改善**记忆，直接给予 5-HT1aR 选择性**激动剂**同样改善记忆——即在海马，5-HT1A 被激活的净效应是抑制锥体细胞，**激动**方向具治疗价值。（一处未证实的限定：该文摘要仅说 5-HT 信号 "impaired"，未区分是递质水平还是受体表达下降；正文受 Elsevier 版权限制未能核到分子层面数据。）

**需注意受体极性高度依赖其所在的细胞类型**：当 5-HT1A 位于中间神经元时，其激活会抑制 GABA 释放、使锥体细胞去抑制而**抬高**兴奋性（Puig et al., 2011, *Cereb Cortex*，主要基于皮层；*Pharmacol Rev*, 2024）。Wu et al.（2024, *Aging Cell*, doi:10.1111/acel.14187，APP/PS1 小鼠）在**杏仁核基底外侧区（BLA）**证实了这条通路：AD 早期（认知损害前）5-HT1AR 在 **PV 中间神经元**中表达上调 → PV 放电减少、GABA 释放减弱 → BLA 锥体神经元基础态过度活跃，双敲低 5-HT1AR/5-HT2AR 可恢复兴奋–抑制平衡并逆转情绪与认知缺陷（组会材料《5-HT1A 背景介绍》所引"过度表达 → 中间神经元被过度抑制 → 环路过度兴奋"即对应这条 BLA 链条；需注意其中过度兴奋为 5-HT1A↑ 与 5-HT2A↑ 双受体协同，单独敲低 5-HT1AR 无效）。可见同一受体在 BLA 与海马 CA1 的净极性不同，而两地的 AD 改变方向恰好相反（BLA 上调、海马减弱），各自把受体推向放大过度兴奋的一侧，最终收敛到同一疾病表型。BLA 侧另两点不确定性：5-HT1A 基线主要分布于谷氨酸能神经元（仅病理增量在 PV，且原文只敲低了 PV 上的受体）；GRAB_5-HT 探针显示 AD 与 WT 的 5-HT 递质水平无差异、异常在受体表达。另，5-HT1A 至少有四个作用位点（中缝自体受体、谷氨酸能末梢突触前异受体、锥体细胞突触后异受体、中间神经元异受体）且符号各异，经典的"5-HT1A 拮抗剂促认知"（Dijk 1995、Schechter 2005）针对的是突触前末梢位点。另需注意两文的模型与时间窗不同（APP/PS1、周龄 8–12 vs hAPP-J20、4–5 月龄）。在人脑层面，主流 PET/尸检研究显示 AD 中 5-HT1A 结合总体是**减少**的（如 PMC9285435、PMC6493011；轻中度病理期海马无显著变化、重度 Braak V/VI 期 CA1 减少，见 PMC3112246），与上述小鼠海马"信号减弱"的方向一致；综述 Verdurand & Zimmer 2017（*Neuropharmacology*）则指出，该受体在不同脑区、不同病程中究竟"该激动还是该限制"尚无定论。因此本仓库采用的海马激动假设，是当前证据下更贴合本项目定位（前期实验在海马）的**工作假设**，而非已确立的结论。

**本仓库的能力边界**：AlphaFold3/ipTM 只能回答一件事——该变体肽与 HTR1A **是否可能形成稳定界面、结合是否可能成立**。结合成立之后究竟是激动还是拮抗、效力强弱如何，**不是本仓库代码要解决的问题**，须由湿实验裁决；上述海马激动方向同理，只是据现有文献与前期定位（实验在海马）取定的前提，而非脚本可判定的量。一个附带的工程约束：TAT 穿膜肽经全身给药会同时到达 BLA 与海马，而可变区仅 8 位的小肽不具备细胞类型选择性，因而无法在两地产生相反效力——这也意味着脑区/细胞类型层面的效力问题只能留给后续体内实验。

> 资料来源：组会汇报《QD202 的药理药效验证及可能靶点的筛选》与《5-HT1A 背景介绍》（喻鸿博）；上海魁特迪生物官网与公开临床试验登记；AD 遗传学与 5-HT1A 相关文献（*Trends in Neurosciences* 2025 "BIN1 and Alzheimer's disease: the tau connection"；Wu et al. 2024, *Aging Cell*, doi:10.1111/acel.14187〔杏仁核 BLA，5-HT1A 在 PV 中间神经元上调〕；Wang et al. 2023, *Cell Reports*, doi:10.1016/j.celrep.2023.112152〔海马 CA1，5-HT1A 信号减弱、激动剂有益〕；Verdurand & Zimmer 2017, *Neuropharmacology*〔"激动 vs 限制 5-HT1A"的对立假说综述〕；Puig et al. 2011, *Cereb Cortex*）。

# 工作原理

1. **序列拆分**：`固定区 = "YGRKKRRQRRR"`，`可变区 = "SPVDVVCS"`（后 8 位）。
2. **突变生成**（编辑距离 ≤ 2，长度限制 6–10 个残基），每轮等概率随机选用模式 A / B / C，且只作用于可变区：
   - 模式 A：1–2 个点突变（替换残基，长度不变）；
   - 模式 B：仅长度变化（随机位置插入/删除 1 或 2 个残基）；
   - 模式 C：±1 长度变化 + 1 个点突变。

   名词解释（以原始可变区 `SPVDVVCS` 为例）：

   - **编辑距离**：把一个序列变成另一个序列所需的最少编辑操作次数，操作仅含三种——替换、插入、删除。脚本限定 ≤ 2，即每个候选与原始可变区最多相差 2 次编辑，保证搜索始终停留在原始序列的"近邻"区域内做局部优化。例如 `SPVDVVAS`（C→A，距离 1）、`SPVDGVVCS`（插入 G，距离 1）、`SPVVVS`（删去 D 和 C，距离 2）。
   - **长度限制**：突变后的可变区长度必须保持在 6–10 个残基（原始 8 位，即允许 ±2 以内）。越界的样本会被丢弃并重新抽取。原因是可变区太短难以形成有效结合界面，太长则合成成本与结构不确定性都会上升，该限制为搜索空间划定了物理合理的边界。
   - 注意：每轮突变均从**原始**可变区 `ORIG_TAIL` 生成（不做跨步累积突变），配合固定随机种子（42）保证候选集可复现、支持断点续跑。
3. **结构预测**：对每个候选肽分别生成 3 个 AlphaFold3 JSON 任务（肽为 A 链，靶标为 B 链），提交到 AlphaFold Server。
4. **评分**：从结果包 `*_summary_confidences.json` 读取各指标。三个靶标（HTR1A、UNC13C、BIN1）各自独立预测、各自有一个 ipTM（候选必须三者齐备才进入评估表），字段含义：

   | 字段 | 含义 | 用途 |
   | --- | --- | --- |
   | `score` | $1 - \text{ipTM}$，优化目标，越小越好 | MC/BO 最小化它（本质是界面 ipTM 的线性变换） |
   | `iptm` | 该靶标界面的 TM-score（0~1，界面置信度） | 结合界面可信度的代理指标（非真实亲和度） |
   | `ranking` | AF3 综合排名分：$0.8\,\text{ipTM} + 0.2\,\text{pTM} + 0.5 f_{disordered} - 100\,\mathbb{1}_{clash}$ | 仅作参考，不参与优化 |
   | `AB-iptm` | A 链（肽）-B 链（靶标）链对 ipTM | 双链体系下与 ipTM 一致，供核对 |

   > 名词：TM-score（模板建模分数，结构相似度指标）$= \max_{\text{叠合}} \frac{1}{L}\sum_i \frac{1}{1+(d_i/d_0)^2}$，其中 $d_i$ 为叠合后第 $i$ 对残基距离，$d_0 = 1.24(L-15)^{1/3}-1.8$ 按长度归一（故与蛋白大小无关）；$1$ 为完全一致，$>0.5$ 视为同折叠，$\approx 0.17$ 为随机基线。pTM / ipTM 是其"预测版"：以模型输出的 PAE 误差分布替代真实距离（ipTM 再限定到跨链接触面残基对）。比较对象注意：TM-score 比较"预测结构"与"真实结构"——后者尚未测定，pTM/ipTM 是模型用训练所得的误差预测（PAE）自我估计"若与实验结构比对预期得多少分"，属于可信度而非亲和度。
5. **蒙特卡洛接受**（Metropolis，温度 $T = 0.7$，默认 100 步）：若两个底线蛋白的 ipTM **均 ≥ 0.8**（双双越过结合红线，结合谱未完成重定向），直接拒绝。
6. **结果保存**：记录最佳序列、最佳结构的 zip 包、优化曲线图与 Excel 结果表。

# 运行模式

| 模式 | 条件 | 流程 |
| --- | --- | --- |
| 🤖 全自动 | 提供 `af3_automator` 模块 + Playwright，且已通过 `setup_auth.py` 完成 Google 登录 | 生成 JSON → 自动提交 → 等待 → 自动下载 → 评分 → MC 优化 |
| 🖐️ 手动 | 缺少自动化模块或认证时自动降级 | 生成 JSON 到 `af3_jobs/` → 人工上传到 <https://alphafoldserver.com/> → 下载结果 zip 到 `af3_results/` → 重跑脚本自动提取评分 |

> 注：`af3_automator.py` 与 `setup_auth.py` 不包含在本仓库中；缺少时脚本会打印提示并自动进入手动模式，不影响使用。

# 依赖安装

```bash
pip install numpy pandas matplotlib openpyxl
pip install scikit-learn scipy      # 仅贝叶斯优化入口需要（本地推理环境已有）
```

仅在使用全自动模式时需要额外安装 Playwright 及浏览器（以 `af3_automator` 模块的提示为准），并预先运行 `setup_auth.py` 完成一次性登录认证。

使用本地集群推理（默认后端）时，另需推理环境 `af3_old` 中的 `hmmer` 套件（若已有则跳过）：

```bash
conda install -n af3_old -c bioconda hmmer
```

# 使用方法

```bash
python optimize_peptide_AF3.py
```

脚本执行流程：

1. `batch_generate_all_json()`：以固定随机种子（42）预生成全部突变序列（初始序列 + 100 步突变，去重），为每个序列 × 3 个靶标生成单个任务 JSON，同时按每 100 个任务一组输出 `batch_001.json`、`batch_002.json` …… 便于分批上传；另生成 `sequence_index.json` 记录 `任务标签 → 序列` 映射。
2. `check_results_status()`：比对 `af3_jobs/` 与 `af3_results/`，报告已完成/待处理任务数。
3. `optimize_peptide()`：对已有结果的候选进行评分与蒙特卡洛优化；结果未就绪的任务会打印提示并跳过，**之后重跑脚本即可断点续作**（已下载的 zip 会被缓存复用）。

手动模式的典型工作方式：

1. 首次运行脚本，得到 `af3_jobs/batch_001.json` 等批次文件；
2. 登录 <https://alphafoldserver.com/>，点击“上传 JSON”上传批次文件并运行；
3. 待预测完成后下载所有结果 zip，按 `<任务名>.zip` 命名放入 `af3_results/`；
4. 重新运行脚本，自动评分并继续优化；
5. 重复步骤 2–4，直到所有批次完成（脚本会打印预计所需天数）。

# 贝叶斯优化入口（推荐）

蒙特卡洛版每步随机抽候选，在“每天只能评估约 10 个候选”的昂贵评估场景下样本效率低。`optimize_peptide_BO.py` 改用**贝叶斯优化**：

1. **候选库**：按相同突变规则预生成约 1500 个编辑距离 ≤ 2 的候选（固定种子，可复现）；
2. **冷启动**：评估初始序列 + 8 个随机候选；
3. **代理模型**：拟合 3 个高斯过程（目标 `score = 1 - ipTM(HTR1A)`；约束 `ipTM(UNC13C)`、`ipTM(BIN1)`）；
4. **采集函数**：约束期望改进 $\text{cEI}(x) = \text{EI}(x) \times P(\text{ipTM}_{UNC13C} < 0.8) \times P(\text{ipTM}_{BIN1} < 0.8)$，将评估机会集中在“对 HTR1A 改进潜力大、且两个底线蛋白 ipTM 双低（不越红线）概率高”的候选上（比蒙特卡洛版的硬阈值过滤更平滑）；
5. **批量提交**：每轮选 8 个候选（带特征空间多样性约束），按轮批量提交；
6. **断点续跑**：已评估候选追加写入 `output_bo/bo_evaluations.csv`，重跑即续作。

```bash
python optimize_peptide_BO.py                # 默认：本地集群推理后端（自动提交 SLURM GPU 作业，无每日配额）
python optimize_peptide_BO.py --server       # 切换为 AlphaFold Server 云端后端（手动上传/下载，受配额限制）
python optimize_peptide_BO.py --budget 120 --batch 8 --pool 1500   # 自定义预算/批量/候选库
```

输出位于 `output_bo/`：`bo_evaluations.csv`（全部评估记录）、`bo_curve.png`（最优分改进曲线）、`BEST_STRUCTURE/`（最佳序列结构包）、`result_bo.xlsx`（最终结果表）。

# 本地集群推理入口（蒙特卡洛）

`optimize_peptide_local.py` 是与主脚本相同逻辑的蒙特卡洛版本，但预测后端换为**本地 GPU 推理**：对每个候选自动写入输入 JSON 与 `.sbatch` 作业（`bme_gpupub` 分区、`v-jiamh` 账户、1 GPU），提交到集群计算节点运行，完成后重跑脚本即可提取评分、继续优化（支持续跑）。前置条件（详见文件头说明）：

- 本地 AlphaFold3 代码与权重：仓库外已具备（`/public/slst/home/v-jiamh/alphafold3` + `~/yhb/af3.bin`）；
- 序列数据库：脚本会**自动检测磁盘空间并把解压作为 SLURM 作业提交到计算节点**（目标 `LOCAL_DB_DIR = /public_bme2/Share200T/v-jiamh_af3_databases/alphafold3`，家目录有 ~500G 配额放不下）；检测到解压作业运行中会**转入实时进度跟踪**，完成后自动继续；
- 推理环境 `af3_old` 已具备，若缺 `jackhmmer` 需 `conda install -n af3_old -c bioconda hmmer`。

# 在 SLURM 集群上运行

在 HPC 登录节点（如 `bme_login1`）上，**推荐直接运行优化入口，本地集群推理是默认后端**——入口会自动为每个候选提交独立的 `.sbatch` GPU 作业，无需手动包装，也没有每日配额：

```bash
python optimize_peptide_BO.py       # 贝叶斯优化 + 本地集群推理（推荐，默认行为）
python optimize_peptide_local.py    # 蒙特卡洛 + 本地集群推理
squeue -u $USER                     # 查看自动提交的推理作业状态
```

仅当确需改用 **AlphaFold Server 云端后端**（受每日约 30 任务的配额限制，且当前为手动上传/下载模式）时，才需要显式切换：

```bash
python optimize_peptide_BO.py --server    # 贝叶斯优化 + 云端后端
python optimize_peptide_AF3.py            # 蒙特卡洛 + 云端后端（原版）
```

云端后端的编排工作极轻，若不想在登录节点长时间运行，可用仓库自带的委托脚本把它提交到计算节点。最省事的方式是一键脚本（自动提交 + 实时滚动进度 + 结束报告，不暴露作业号细节）：

```bash
./run.sh               # 一键提交：贝叶斯优化（云端后端）并实时跟踪进度
./run.sh mc            # 蒙特卡洛版本
```

或手动等价操作：

```bash
sbatch run_peptide_bo.sbatch   # BO + 云端后端
tail -f pepopt_bo_<作业号>.out # 实时查看进度（作业号由提交信息给出）
```

若使用全自动模式（需自备 `af3_automator` 模块、Playwright 及浏览器），请在 `.sbatch` 脚本中相应安装这些组件，并注意计算节点需可访问外网。

# 可调参数

均位于脚本顶部“配置区域”：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `SEQ_202` | `YGRKKRRQRRRSPVDVVCS` | 起始小肽序列（自动拆分为固定区 + 后 8 位可变区） |
| `HTR1A_SEQ` / `UNC13C_SEQ` / `BIN1_SEQ` | — | 靶标与两个底线蛋白的全长序列 |
| `MC_STEPS` | `100` | 蒙特卡洛迭代步数 |
| `TEMPERATURE` | `0.7` | Metropolis 接受温度 |
| `ROOT_DIR` | `output_af3/202_htr1a` | 优化结果输出根目录 |
| `JSON_OUTPUT_DIR` | `af3_jobs` | 生成的 AF3 JSON 任务目录 |
| `RESULTS_DIR` | `af3_results` | 下载的结果 zip 目录 |
| `MAX_JOBS_PER_BATCH` | `100` | 单次 JSON 上传的最大任务数 |
| `DAILY_LIMIT` | `30` | AlphaFold Server 免费日配额（用于估算天数） |
| `AUTO_MODE` | `True` | 是否尝试自动提交（缺条件时自动降级为手动） |

替换靶标或起始肽时，直接修改对应序列常量即可。

# 目录与输出

```
├── peptide_common.py            # 公共库: 序列/突变/JSON/评分/蒙特卡洛主循环（导入零副作用）
├── peptide_server.py            # 云端后端库: 批量JSON/结果缓存/可选浏览器自动化（仅云端入口导入）
├── optimize_peptide_AF3.py      # 入口: 蒙特卡洛 + AlphaFold Server 云端后端（瘦壳）
├── optimize_peptide_BO.py       # 入口: 贝叶斯优化（推荐，默认本地后端）
├── optimize_peptide_local.py    # 入口: 蒙特卡洛 + 本地集群推理
├── run_peptide.sbatch           # SLURM 委托脚本（服务器版蒙特卡洛）
├── run_peptide_bo.sbatch        # SLURM 委托脚本（服务器版贝叶斯优化）
├── run.sh                       # 一键运行（默认本地后端 + 实时进度）
├── af3_jobs/                    # 生成的 AF3 JSON 任务、批次与序列索引（运行后产生）
├── af3_results/                 # AlphaFold Server 结果 zip（按 <任务名>.zip 命名）
├── af3_local_results/           # 本地推理任务目录与结果（运行后产生）
├── output_af3/202_htr1a/HTR1A_best/   # 蒙特卡洛版输出（运行后产生）
│   ├── BEST_STRUCTURE/BEST_peptide_HTR1A.zip   # 最佳序列的 AF3 结果包
│   ├── optimization_curve_af3.png              # score 随优化步数变化曲线
│   └── result_af3.xlsx                         # 最优序列、最优分数、拒绝次数汇总
└── output_bo/                   # 贝叶斯优化版输出（运行后产生）
    ├── bo_evaluations.csv                      # 全部已评估候选与三靶标评分（续跑依据）
    ├── bo_curve.png                            # 最优分随评估数改进曲线
    ├── BEST_STRUCTURE/BEST_peptide_HTR1A.zip   # 最佳序列的 AF3 结果包
    └── result_bo.xlsx                          # 最优序列与三靶标评分汇总
```

# 常用函数速查

| 函数 | 所属模块 | 作用 |
| --- | --- | --- |
| `generate_mutant(orig_tail)` | `peptide_common` | 生成编辑距离 ≤ 2 的可变区突变 |
| `create_af3_json(peptide_seq, target_seq, job_name)` | `peptide_common` | 生成单条 AlphaFold3 JSON 任务 |
| `extract_scores_from_zip(zip_path)` | `peptide_common` | 从结果 zip 中提取 `iptm` / `ptm` / `ranking_score` 等评分 |
| `run_monte_carlo(predict_fn, opt_root, ...)` | `peptide_common` | 后端无关的蒙特卡洛主循环（预测函数由入口注入） |
| `batch_generate_all_json()` | `peptide_server` | 批量生成全部序列 × 全部靶标的 JSON 并分批 |
| `check_results_status()` | `peptide_server` | 检查云端已完成 / 待处理任务 |
| `optimize_with_server()` | `peptide_server` | 云端版蒙特卡洛编排（自动化检查 + 待处理任务处理） |
| `optimize_peptide_BO.run_bo(predict_fn, pool_size, batch_size, budget)` | `optimize_peptide_BO` | 贝叶斯优化主循环（冷启动 → 拟合 3 个 GP → 约束期望改进采集 → 批量评估） |

# 常见问题

- **提示 `af3_automator 模块未找到`？** 属正常现象，脚本自动切换为手动模式，按提示人工上传/下载即可。
- **提示今日配额已用完？** AlphaFold Server 免费配额为每日 30 个任务，次日重跑脚本即可自动继续。
- **如何知道还差哪些任务？** 运行 `check_results_status()`（或直接重跑脚本），它会列出所有已生成 JSON 但没有结果 zip 的任务。
- **中断后如何续跑？** 直接重新运行脚本：蒙特卡洛版用固定随机种子重新生成同一候选集；贝叶斯优化版从 `output_bo/bo_evaluations.csv` 恢复；两版均复用已有的结果缓存。
- **贝叶斯优化版报“缺少依赖: ['sklearn', 'scipy']”？** 按“依赖安装”一节补装 `scikit-learn scipy` 后重跑即可。
- **本地推理版提示缺数据库？** 脚本会自动检查磁盘空间后把解压提交为计算节点异步作业并实时跟踪进度（无需保持 SSH 连接），完成后重跑即继续；若磁盘不足会打印所需容量并中止，可改 `LOCAL_DB_DIR` 到更大目录后重跑。
