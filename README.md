**小肽序列优化**：基于 AlphaFold3 Web Server 的小肽序列蒙特卡洛优化脚本仓库。

# 简介

本仓库提供一个基于 **AlphaFold3 Web Server** 的小肽序列蒙特卡洛优化脚本 `optimize_peptide_AF3.py`。它以肽 **202**（`YGRKKRRQRRRSPVDVVCS`）为起点，仅对 C 端 8 位可变区进行突变搜索，目标是找到与靶标蛋白 **HTR1A（5-HT1A 受体）** 结合更强、同时对两个底线蛋白 **UNC13C** 和 **BIN1** 选择性更好的变体。

# 工作原理

1. **序列拆分**：`固定区 = "YGRKKRRQRRR"`，`可变区 = "SPVDVVCS"`（后 8 位）。
2. **突变生成**（编辑距离 ≤ 2，长度限制 6–10 个残基）：
   - 模式 A：1–2 个点突变；
   - 模式 B：仅长度变化（插入/删除 ±1 或 ±2）；
   - 模式 C：±1 长度变化 + 1 个点突变。
3. **结构预测**：对每个候选肽分别生成 3 个 AlphaFold3 JSON 任务（肽为 A 链，靶标为 B 链），提交到 AlphaFold Server。
4. **评分**：从结果包 `*_summary_confidences.json` 读取 `iptm`、`ptm`、`ranking_score`、A-B 链界面 ipTM、`fraction_disordered`、`has_clash` 等，定义 $\text{score} = 1 - \text{ipTM}$，越小越好。
5. **蒙特卡洛接受**（Metropolis，温度 $T = 0.7$，默认 100 步）：若两个底线蛋白的 ipTM **均 ≥ 0.8**，视为选择性差，直接拒绝。
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
```

仅在使用全自动模式时需要额外安装 Playwright 及浏览器（以 `af3_automator` 模块的提示为准），并预先运行 `setup_auth.py` 完成一次性登录认证。

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
├── optimize_peptide_AF3.py      # 主脚本
├── af3_jobs/                    # 生成的 AF3 JSON 任务与批次文件、序列索引
├── af3_results/                 # AlphaFold Server 结果 zip（按 <任务名>.zip 命名）
└── output_af3/202_htr1a/HTR1A_best/
    ├── BEST_STRUCTURE/BEST_peptide_HTR1A.zip   # 最佳序列的 AF3 结果包
    ├── optimization_curve_af3.png              # score 随优化步数变化曲线
    └── result_af3.xlsx                         # 最优序列、最优分数、拒绝次数汇总
```

# 常用函数速查

| 函数 | 作用 |
| --- | --- |
| `generate_mutant(orig_tail)` | 生成编辑距离 ≤ 2 的可变区突变 |
| `create_af3_json(peptide_seq, target_seq, job_name)` | 生成单条 AlphaFold3 JSON 任务 |
| `extract_scores_from_zip(zip_path)` | 从结果 zip 中提取 `iptm` / `ptm` / `ranking_score` 等评分 |
| `batch_generate_all_json()` | 批量生成全部序列 × 全部靶标的 JSON 并分批 |
| `check_results_status()` | 检查已完成 / 待处理任务 |
| `optimize_peptide()` | 执行蒙特卡洛优化主流程 |

# 常见问题

- **提示 `af3_automator 模块未找到`？** 属正常现象，脚本自动切换为手动模式，按提示人工上传/下载即可。
- **提示今日配额已用完？** AlphaFold Server 免费配额为每日 30 个任务，次日重跑脚本即可自动继续。
- **如何知道还差哪些任务？** 运行 `check_results_status()`（或直接重跑脚本），它会列出所有已生成 JSON 但没有结果 zip 的任务。
- **中断后如何续跑？** 直接重新运行脚本：脚本用固定随机种子重新生成同一候选集，并复用 `af3_results/` 中已有的结果。
