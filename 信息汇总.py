"""只读汇总前五序列、计算中的靶标和分区、Slurm 等待分配及 HQ 等待任务。"""

import argparse
import csv
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import 查看前五 as 排名
from 小肽公共库 import FIXED_A

项目目录 = Path(__file__).resolve().parent
中间层目录 = Path(os.environ.get("TASKCTL_HOME", 项目目录.parent / "任务中间层")).resolve()
sys.path.insert(0, str(中间层目录))
import 查看队列 as HQ队列
from 自动扩容 import gpu_memory, resource_amount, slurm_fields, worker_capacity


def 表格文本(表头, 数据行, 空提示):
    表格 = [[str(值) for 值 in 表头], *[[str(值) for 值 in 行] for 行 in 数据行]]
    if not 数据行:
        return 空提示

    def 显示宽度(文本):
        return sum(2 if unicodedata.east_asian_width(字符) in ("W", "F") else 1 for 字符 in 文本)

    列宽 = [max(显示宽度(行[列]) for 行 in 表格) for 列 in range(len(表头))]
    return "\n".join("  ".join(值 + " " * (列宽[列] - 显示宽度(值)) for 列, 值 in enumerate(行)).rstrip() for 行 in 表格)


def 读取前五():
    记录 = 排名.读取排名()
    前五 = [{"排名": 编号, "标签": 标签, "序列": 行.get("peptide") or "", "分量": dict(zip(排名.TARGETS, 分量)), "乘积": 乘积, "样本数": int(行.get("obj_n") or 0)} for 编号, (乘积, 标签, 行, 分量) in enumerate(记录[:5], 1)]
    return {"数据库": 排名.db.db_root(), "启用靶标": list(排名.TARGETS), "齐全记录数": len(记录), "记录": 前五}


def 读取运行靶标(作业详情, 工作者列表, 查询分区=None):
    工作者索引 = {工作者["id"]: 工作者 for 工作者 in 工作者列表 if 工作者.get("ended") is None}
    记录列表 = []
    for 作业 in 作业详情:
        GPU作业 = any(资源.get("resource") == "gpus" and resource_amount(资源) > 0 for 提交 in 作业.get("submits", []) for 方案 in (提交.get("array") or {}).get("resources", []) for 资源 in 方案.get("resources", []))
        if not GPU作业:
            continue
        名称 = 作业["info"]["name"]
        靶标 = next((靶标 for 靶标 in 排名.db.TARGETS_ALL if re.search(rf"_{re.escape(靶标.lower())}(?:_s\d+)?$", 名称.lower())), "其他GPU任务")
        for 任务 in 作业.get("tasks", []):
            if 任务.get("state") != "running":
                continue
            工作者 = 工作者索引.get(任务.get("worker"))
            if 工作者 is None:
                raise RuntimeError(f"运行任务 {作业['info']['id']} 的 worker 信息缺失，请重新查询")
            容量 = worker_capacity(工作者)
            记录列表.append({"节点": 工作者["configuration"]["hostname"], "每卡显存GiB": 容量.get("gpumem", 0) / max(1, 容量.get("gpus", 0)), "靶标": 靶标, "HQ作业号": 作业["info"]["id"], "任务名": 名称, "Slurm作业号": str((工作者.get("allocation") or {}).get("id", "未关联"))})
    分配分区 = (查询分区 or 读取分配分区)() if 记录列表 else {}
    for 记录 in 记录列表:
        记录["分区"] = 分配分区.get(记录["Slurm作业号"], "未关联" if 记录["Slurm作业号"] == "未关联" else "未知")
    return sorted(记录列表, key=lambda 记录: (记录["分区"], 记录["HQ作业号"]))


def 读取HQ快照(查询, 查询分区=None):
    作业列表 = 查询("job", "list", "--all")
    活跃编号 = [作业["id"] for 作业 in 作业列表 if any((作业.get("task_stats") or {}).get(状态, 0) for 状态 in ("running", "waiting"))]
    详情 = 查询("job", "info", ",".join(map(str, 活跃编号))) if 活跃编号 else []
    if set(活跃编号) - {作业["info"]["id"] for 作业 in 详情}:
        raise RuntimeError("HQ 未返回全部活跃作业详情，请重新查询")
    工作者 = 查询("worker", "list")

    def 快照查询(*参数):
        if 参数 == ("job", "list", "--all"):
            return 作业列表
        if 参数[:2] == ("job", "info"):
            编号 = {int(值) for 值 in 参数[2].split(",")}
            return [作业 for 作业 in 详情 if 作业["info"]["id"] in 编号]
        raise ValueError(f"未支持的快照查询：{参数}")

    return {"运行靶标": 读取运行靶标(详情, 工作者, 查询分区), "等待任务": HQ队列.读取等待任务(快照查询)}


def 运行命令(参数):
    结果 = subprocess.run(参数, capture_output=True, text=True, timeout=60)
    if 结果.returncode:
        raise RuntimeError(f"{参数[0]} 查询失败：{结果.stderr.strip() or 结果.stdout.strip()}")
    return 结果.stdout.strip()


def 读取分配分区(运行=运行命令):
    输出 = 运行(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%P"])
    分配分区 = {}
    for 行 in csv.reader(输出.splitlines(), delimiter="|"):
        if len(行) != 2:
            raise ValueError("Slurm 作业分区列数异常，无法准确显示运行分区")
        分配分区[行[0]] = 行[1]
    return 分配分区


def 读取Slurm等待(运行=运行命令):
    输出 = 运行(["squeue", "--start", "-h", "-u", getpass.getuser(), "-t", "PENDING", "-o", "%i|%j|%P|%b|%C|%m|%l|%R|%S"])
    if not 输出:
        return []
    节点详情 = [slurm_fields(行) for 行 in 运行(["scontrol", "-o", "show", "node"]).splitlines() if 行.strip()]
    记录列表 = []
    for 行 in csv.reader(输出.splitlines(), delimiter="|"):
        if len(行) != 9:
            raise ValueError("Slurm 等待队列列数异常，无法准确显示节点范围")
        编号, 名称, 分区, GPU请求, CPU, 主机内存, 时限, 原因, 预计开始 = 行
        明细 = slurm_fields(运行(["scontrol", "-o", "show", "job", 编号]))

        def 展开节点(值):
            return set(运行(["scontrol", "show", "hostnames", 值]).splitlines()) if 值 and 值 not in ("(null)", "None") else set()

        包含节点 = 展开节点(明细.get("ReqNodeList"))
        排除节点 = 展开节点(明细.get("ExcNodeList"))
        型号 = re.search(r"(?:^|,)gpu:([^:,]+):\d+", GPU请求)
        数量 = sum(map(int, re.findall(r"(?:^|,)gpu(?::[^:,]+)?:(\d+)", GPU请求)))
        候选节点 = []
        显存档位 = set()
        显存未知 = False
        for 节点 in 节点详情:
            节点名 = 节点.get("NodeName")
            if not 节点名 or not set(分区.split(",")).intersection(节点.get("Partitions", "").split(",")) or 节点名 in 排除节点 or (包含节点 and 节点名 not in 包含节点):
                continue
            显卡 = re.findall(r"(?:^|,)gpu:([^:,()]+):(\d+)", 节点.get("Gres", ""))
            if 数量 and sum(int(张数) for 类型, 张数 in 显卡 if 型号 is None or 类型 == 型号.group(1)) < 数量:
                continue
            候选节点.append(节点名)
            if 数量:
                for 类型, _ in 显卡:
                    if 型号 is not None and 类型 != 型号.group(1):
                        continue
                    显存 = gpu_memory(类型)
                    if 显存 is None:
                        显存未知 = True
                    else:
                        显存档位.add(显存)
        if 型号 is not None and gpu_memory(型号.group(1)) is not None:
            显存档位 = {gpu_memory(型号.group(1))}
            显存未知 = False
        记录列表.append({"Slurm作业号": 编号, "作业名": 名称, "分区": 分区, "GPU请求": GPU请求 if 数量 else "CPU", "GPU数": 数量, "每卡显存档位GiB": sorted(显存档位), "显存档位未知": 显存未知 or (数量 > 0 and not 显存档位), "CPU": CPU, "主机内存": 主机内存, "时限": 时限, "候选节点": sorted(候选节点), "预计开始": 预计开始 if 预计开始 not in ("N/A", "Unknown", "None") else "未知", "等待原因": 原因})
    return 记录列表


def 格式化Slurm等待(记录列表):
    显示GPU数 = any(记录["GPU数"] != 1 for 记录 in 记录列表)
    表头 = ["Slurm ID", "作业名", "分区", *(["GPU数"] if 显示GPU数 else []), "申请显存(GiB/卡)", "候选节点范围", "预计开始", "等待原因"]
    数据行 = []
    for 记录 in 记录列表:
        档位 = [str(显存) for 显存 in 记录["每卡显存档位GiB"]]
        if 记录["显存档位未知"]:
            档位.append("未知")
        显存文本 = "/".join(档位) if 记录["GPU数"] else "CPU"
        if len(档位) > 1:
            显存文本 += "(未限定单档)"
        数据行.append([记录["Slurm作业号"], 记录["作业名"], 记录["分区"], *([记录["GPU数"]] if 显示GPU数 else []), 显存文本, ",".join(记录["候选节点"]) or "未确定", 记录["预计开始"], 记录["等待原因"]])
    return 表格文本(表头, 数据行, "当前没有 Slurm 排队作业。")


def 生成汇总(查询HQ=None, 查询Slurm=None, 查询排名=None):
    查询HQ = 查询HQ or HQ队列.查询HQ
    查询Slurm = 查询Slurm or 读取Slurm等待
    查询排名 = 查询排名 or 读取前五
    结果 = {"查询时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "前五": None, "运行靶标": None, "Slurm等待": None, "HQ等待": None, "错误": {}}
    try:
        结果["前五"] = 查询排名()
    except (OSError, ValueError, RuntimeError) as 错误:
        结果["错误"]["前五"] = str(错误)
    try:
        快照 = 读取HQ快照(查询HQ)
        结果["运行靶标"], 结果["HQ等待"] = 快照["运行靶标"], 快照["等待任务"]
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as 错误:
        结果["错误"]["HQ"] = str(错误)
    try:
        结果["Slurm等待"] = 查询Slurm()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as 错误:
        结果["错误"]["Slurm"] = str(错误)
    return 结果


def 格式化汇总(汇总):
    文本 = ["查询时间：" + 汇总["查询时间"], "", "一、当前前五"]
    前五 = 汇总["前五"]
    if 前五 is None:
        文本.append("查询失败：" + 汇总["错误"]["前五"])
    else:
        文本.append(f"数据库：{前五['数据库']} | 排序：{' × '.join(前五['启用靶标'])} | 齐全记录：{前五['齐全记录数']}")
        文本.append("固定区：" + FIXED_A)
        数据行 = [[行["排名"], 行["序列"].removeprefix(FIXED_A), *[f"{行['分量'][靶标]:.2f}" for 靶标 in 前五["启用靶标"]], f"{行['乘积']:.6f}", 行["样本数"]] for 行 in 前五["记录"]]
        文本.append(表格文本(["排名", "可变区", *前五["启用靶标"], "乘积", "样本"], 数据行, "没有评分齐全的序列。"))
    文本.extend(["", "二、计算中的靶标及分区"])
    if 汇总["运行靶标"] is None:
        文本.append("查询失败：" + 汇总["错误"]["HQ"])
    else:
        数据行 = [[行["分区"], f"{行['每卡显存GiB']:g}", 行["靶标"], 行["HQ作业号"], 行["任务名"], 行["Slurm作业号"]] for 行 in 汇总["运行靶标"]]
        文本.append(表格文本(["分区", "显存(GiB/卡)", "靶标", "HQ ID", "任务名", "Slurm ID"], 数据行, "当前没有运行中的 GPU 计算任务。"))
    文本.extend(["", "三、Slurm 层等待分配"])
    if 汇总["Slurm等待"] is None:
        文本.append("查询失败：" + 汇总["错误"]["Slurm"])
    else:
        文本.append(格式化Slurm等待(汇总["Slurm等待"]))
        文本.append("候选节点尚未分配，仅按分区、GPU 型号和包含/排除约束筛选；预计开始时间不是保证。")
    文本.extend(["", "四、HQ 层等待任务"])
    文本.append("查询失败：" + 汇总["错误"]["HQ"] if 汇总["HQ等待"] is None else HQ队列.格式化队列(汇总["HQ等待"]))
    return "\n".join(文本)


def 主程序():
    解析器 = argparse.ArgumentParser(description=__doc__)
    解析器.add_argument("--状态目录", type=Path, help="HQ 共享状态目录")
    解析器.add_argument("--json", dest="机器可读", action="store_true", help="输出结构化汇总")
    参数 = 解析器.parse_args()
    汇总 = 生成汇总(查询HQ=lambda *查询参数: HQ队列.查询HQ(*查询参数, 状态目录=参数.状态目录))
    print(json.dumps(汇总, ensure_ascii=False, indent=2) if 参数.机器可读 else 格式化汇总(汇总))
    return 1 if 汇总["错误"] else 0


if __name__ == "__main__":
    raise SystemExit(主程序())