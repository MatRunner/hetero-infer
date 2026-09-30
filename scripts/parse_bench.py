#!/usr/bin/env python3
"""parse_bench.py — 解析 bench.sh 落盘的原始结果，生成结构化 CSV（最终产物）

用法:
    python3 scripts/parse_bench.py [--raw-root data/raw] [--out data/results.csv]

行为:
    - 扫描 <raw-root>/<exp_id>/<run_ts>/ 目录（bench.sh 的产物）
    - 跳过失败运行（FAILED 标记 / exit_code != 0 / results.json 缺失或损坏）
    - 全量重建输出 CSV（按运行时间升序），幂等可重复执行
    - llama-bench 输出为每测试一行（pp 行 n_gen=0，tg 行 n_prompt=0），
      本脚本将同一次运行的 pp/tg 两行合并为一行，并聚合内存采样峰值

CSV 列（CLAUDE.md 约定：实验 ID、完整命令行、日期、pp、tg、显存峰值、host 内存）:
    exp_id, run_ts, command, model, model_size_gib,
    ngl, ncmoe, tensor_buft_overrides, flash_attn, n_threads, use_mmap, type_k, type_v, split_mode,
    pp_n, pp_tok_s, pp_avg_ms, pp_stddev_tok_s,
    tg_n, tg_tok_s, tg_avg_ms, tg_stddev_tok_s,
    gpu_mem_peak_mib, host_rss_peak_gib, host_sys_used_peak_gib,
    backends, gpu_info, cpu_info, build_commit, duration_s, exit_code, raw_dir
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FIELDS = [
    "exp_id", "run_ts", "command",
    "model", "model_size_gib",
    "ngl", "ncmoe", "tensor_buft_overrides", "flash_attn", "n_threads",
    "use_mmap", "type_k", "type_v", "split_mode",
    "pp_n", "pp_tok_s", "pp_avg_ms", "pp_stddev_tok_s",
    "tg_n", "tg_tok_s", "tg_avg_ms", "tg_stddev_tok_s",
    "gpu_mem_peak_mib", "host_rss_peak_gib", "host_sys_used_peak_gib",
    "backends", "gpu_info", "cpu_info", "build_commit",
    "duration_s", "exit_code", "raw_dir",
]

FA_MAP = {-1: "auto", 0: "off", 1: "on"}


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"解析 {path} 失败: {e}")
        return None


def col_max(path: Path, col: str):
    """读 CSV 某列的最大数值；文件缺失/列缺失/无有效值返回 None"""
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            vals = []
            for row in csv.DictReader(f):
                v = (row.get(col) or "").strip()
                if v:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            return max(vals) if vals else None
    except Exception as e:
        warn(f"读取 {path} 失败: {e}")
        return None


def fmt(v, nd=3):
    return "" if v is None else round(float(v), nd)


def pick_test(rows, kind: str):
    """从 llama-bench results.json 的行中取 pp（n_gen=0）或 tg（n_prompt=0）行"""
    if kind == "pp":
        cand = [r for r in rows if int(r.get("n_gen", 0)) == 0 and int(r.get("n_prompt", 0)) > 0]
    else:
        cand = [r for r in rows if int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0]
    if not cand:
        return None
    if len(cand) > 1:
        warn(f"{kind} 测试多于一行（可能传了多值参数），取第一行")
    return cand[0]


def parse_run(run_dir: Path):
    meta_p = run_dir / "meta.json"
    res_p = run_dir / "results.json"
    if not meta_p.is_file() or not res_p.is_file():
        warn(f"跳过 {run_dir.name}: 缺少 meta.json 或 results.json")
        return None
    meta = load_json(meta_p)
    if meta is None:
        return None
    if (run_dir / "FAILED").exists() or int(meta.get("exit_code", 1)) != 0:
        warn(f"跳过 {run_dir.name}: 失败的运行（exit_code={meta.get('exit_code')}）")
        return None
    rows = load_json(res_p)
    if not isinstance(rows, list) or not rows:
        warn(f"跳过 {run_dir.name}: results.json 为空或格式异常")
        return None

    pp = pick_test(rows, "pp")
    tg = pick_test(rows, "tg")
    if pp is None or tg is None:
        warn(f"跳过 {run_dir.name}: 未找到 pp/tg 测试行")
        return None

    # 内存峰值（任一基准行都行，两次取相同环境）
    base = pp
    gpu_peak = col_max(run_dir / "gpu_mem.csv", "gpu_mem_used_mib")
    rss_peak = col_max(run_dir / "host_mem.csv", "proc_rss_hwm_mib")
    sys_peak = col_max(run_dir / "host_mem.csv", "sys_used_mib")

    model_size = base.get("model_size")

    def ms(r):
        return fmt(float(r["avg_ns"]) / 1e6, 2) if r.get("avg_ns") is not None else ""

    return {
        "exp_id": meta.get("exp_id", run_dir.parent.name),
        "run_ts": meta.get("start", run_dir.name),
        "command": meta.get("command", ""),
        "model": base.get("model_filename", ""),
        "model_size_gib": fmt(float(model_size) / 2**30, 2) if model_size is not None else "",
        "ngl": base.get("n_gpu_layers", ""),
        "ncmoe": base.get("n_cpu_moe", ""),
        "tensor_buft_overrides": base.get("tensor_buft_overrides", ""),
        "flash_attn": FA_MAP.get(base.get("flash_attn"), base.get("flash_attn", "")),
        "n_threads": base.get("n_threads", ""),
        "use_mmap": base.get("use_mmap", ""),
        "type_k": base.get("type_k", ""),
        "type_v": base.get("type_v", ""),
        "split_mode": base.get("split_mode", ""),
        "pp_n": pp.get("n_prompt", ""),
        "pp_tok_s": fmt(pp.get("avg_ts")),
        "pp_avg_ms": ms(pp),
        "pp_stddev_tok_s": fmt(pp.get("stddev_ts")),
        "tg_n": tg.get("n_gen", ""),
        "tg_tok_s": fmt(tg.get("avg_ts")),
        "tg_avg_ms": ms(tg),
        "tg_stddev_tok_s": fmt(tg.get("stddev_ts")),
        "gpu_mem_peak_mib": int(gpu_peak) if gpu_peak is not None else "",
        "host_rss_peak_gib": fmt(rss_peak / 1024, 2) if rss_peak is not None else "",
        "host_sys_used_peak_gib": fmt(sys_peak / 1024, 2) if sys_peak is not None else "",
        "backends": base.get("backends", ""),
        "gpu_info": base.get("gpu_info", ""),
        "cpu_info": base.get("cpu_info", ""),
        "build_commit": base.get("build_commit", ""),
        "duration_s": meta.get("duration_s", ""),
        "exit_code": meta.get("exit_code", ""),
        "raw_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=str(REPO_ROOT / "data" / "raw"), help="原始结果根目录（默认 data/raw）")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "results.csv"), help="输出 CSV 路径（默认 data/results.csv）")
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_path = Path(args.out)
    if not raw_root.is_dir():
        print(f"ERROR: 原始结果目录不存在: {raw_root}", file=sys.stderr)
        sys.exit(1)

    run_dirs = []
    for exp_dir in sorted(raw_root.iterdir()):
        if exp_dir.is_dir():
            for run_dir in sorted(exp_dir.iterdir()):
                if run_dir.is_dir():
                    run_dirs.append(run_dir)

    records = []
    for run_dir in run_dirs:
        rec = parse_run(run_dir)
        if rec is not None:
            records.append(rec)

    if not records:
        print(f"ERROR: {raw_root} 下没有有效运行（全部失败或为空），未生成 CSV", file=sys.stderr)
        sys.exit(1)

    records.sort(key=lambda r: str(r["run_ts"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)

    # 摘要
    print(f"== {len(records)} 次有效运行 -> {out_path}")
    print(f"{'exp_id':<24} {'run_ts':<26} {'pp_tok_s':>9} {'tg_tok_s':>9} {'gpu_MiB':>9} {'rss_GiB':>8}")
    for r in records:
        print(f"{str(r['exp_id']):<24} {str(r['run_ts'])[:26]:<26} "
              f"{str(r['pp_tok_s']):>9} {str(r['tg_tok_s']):>9} "
              f"{str(r['gpu_mem_peak_mib']):>9} {str(r['host_rss_peak_gib']):>8}")


if __name__ == "__main__":
    main()
