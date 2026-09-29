"""
蒙特卡洛优化小肽序列 — AlphaFold Server 云端入口
=============================================================
本文件只是流程入口，逻辑拆分在两个库中：

  - peptide_common.py  公共逻辑（序列常量、突变规则、AF3 JSON 格式、
                       评分提取、蒙特卡洛主循环）。导入零副作用。
  - peptide_server.py  云端后端库（JSON 生成/批量分批、结果缓存、
                       可选的 af3_automator 浏览器自动化、云端编排）。

只有走云端后端时才需要运行本脚本；本地集群推理请用
optimize_peptide_local.py，贝叶斯优化请用 optimize_peptide_BO.py。

使用方式:
  - 全自动模式: 先运行 setup_auth.py 完成 Google 登录，再运行本脚本
    （需自备 af3_automator 模块与 Playwright）
  - 手动模式: 脚本生成 JSON 到 af3_jobs/，人工上传到
    https://alphafoldserver.com/ 并把结果 zip 放入 af3_results/ 后重跑
"""

import random

import numpy as np

from peptide_server import (HAS_AUTOMATOR, AUTO_MODE,
                            check_playwright_installed, check_auth_exists,
                            batch_generate_all_json, check_results_status,
                            optimize_with_server)


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
        print("运行模式: 🖐️  手动模式 (af3_automator 未安装，将使用手动模式)")
    print()

    print("使用说明:")
    if HAS_AUTOMATOR and AUTO_MODE and check_auth_exists():
        print("  🤖 自动模式: 脚本将自动提交到 AlphaFold Server")
        print("  1. batch_generate_all_json() → 生成 JSON + 自动提交")
        print("  2. optimize_with_server() → 自动评分 + MC 优化")
        print("  3. check_results_status() → 查看进度")
    else:
        print("  🖐️  手动模式:")
        print("  1. batch_generate_all_json() → 批量生成 JSON")
        print("  2. 登录 https://alphafoldserver.com/ → 上传 JSON 并运行")
        print("  3. 下载结果 zip → 放入 af3_results/ 目录")
        print("  4. optimize_with_server() → 自动提取评分并完成优化")
    print()

    # 第一步：批量生成 JSON (并尝试自动提交)
    batch_generate_all_json()

    # 检查结果并尝试优化
    completed, pending = check_results_status()
    if completed > 0:
        print("\n🚀 已有结果可用，尝试运行蒙特卡洛优化...")
        best_seq, best_score = optimize_with_server()
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
