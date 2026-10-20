af3适配 — AF3 任务的任务中间层适配层
=====================================

本目录是 **AF3 专属**的任务适配逻辑（放在 AF3 自己的项目里，通用调度工具 `../../任务中间层/` 不含任何任务适配）。

# 组成

```
af3适配/
├── msa缓存.py   # AF3 MSA 磁盘缓存: 提取/内嵌/自愈收割
└── af3任务.py   # AF3 JSON → 任务中间层任务（生成启动器+提交+等待）
```

# MSA 磁盘缓存（msa缓存.py）

单任务粒度的最大障碍是 MSA 摊销失效（靶标链搜索 ≈78min/进程）。解法：
**内嵌 MSA** —— AF3 输入 JSON 的每条链可带 `unpairedMsa`/`pairedMsa`/
`templates` 三件套，三件齐全时 data pipeline **完全跳过搜索**。缓存值全部取自 AF3 自己输出的 `_data.json`（pipeline 的真实搜索结果），回放即特征逐字节等价 —— 前提是数据库与 max_template_date 配置不变（本项目一直如此）。

- 靶标链（恒定序列）：播种自历史结果，永久免搜索；
- 肽链：首次任务现场搜索，完成时自动收割入缓存（自愈），同序列后续任务复用。

缓存根：`peptide_opt_db/msa_cache/`（可用 `PEPOPT_MSA_CACHE` 覆盖）。

# 任务流（af3任务.py）

`小肽优化_节点内.run_af3_cycle` 默认走此路径（`PEPOPT_HQ_MODE=0` 可强制回退整包直跑）：每个 AF3 JSON 一个独立任务（一任务一 job → `任务控制.sh cancel <id>` 精确取消单个预测），生成的启动器在任务开始时内嵌缓存 MSA、结束时收割本任务的 MSA 入缓存（同序列后续任务自动免搜索）。

手动补算/试算：`任务中间层/任务控制.sh add --name seqX_htr1a --gpus 1
--priority 10 -- bash……`（priority 越大越先跑）。
