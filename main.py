"""
训练完成后的评估入口
====================

一键运行论文 Fig.6/7 评估与绘图。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def main():
    steps = [
        ([sys.executable, "evaluate_paper.py"], "Figure 6/7 评估与绘图"),
    ]
    for cmd, desc in steps:
        print(f"=== {desc} ===")
        subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
