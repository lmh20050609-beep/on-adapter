"""独立汇总脚本: 不做任何训练, 只读 CSV 并打印/保存汇总报告。

用于多 backbone 并行跑完 (scripts/run_all_backbones.sh) 之后统一调用一次,
避免每个 backbone 进程各自在退出时并发调用 report_averages()/report_by_backbone()/
paired_seed_report() 而互相竞争同一份 CSV/打印。
"""
import argparse

from adapt_z_multimodal import report_averages, report_by_backbone, paired_seed_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="汇总 finally_backbone.csv 的平均 MSE/MAE")
    ap.add_argument("--csv", default="finally_backbone.csv")
    a = ap.parse_args()
    report_averages(csv_path=a.csv)
    report_by_backbone(csv_path=a.csv)
    paired_seed_report(csv_path=a.csv)
