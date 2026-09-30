#!/usr/bin/env python3
"""parse_bench.py — 解析 bench.sh 落盘的原始结果，生成结构化 CSV（最终产物）

用法:
    python3 scripts/parse_bench.py [--raw-root data/raw] [--out data/results.csv]

行为:
    - 扫描 <raw-root>/<exp_id>/<run_ts>/ 目录（bench.sh 的产物）
    - 跳过失败运行（FAILED 标记 / exit_code != 0 / results.json 缺失或损坏）
    - 全量重建输出 CSV（按运行时间升序），幂等可重复执行
    - 支持批量多值运行（-ngl 0,16,32 / -t a,b,c 等）：llama-bench 一次进程输出
      多个测试配置，本脚本按配置分组，每个配置输出一行（pp/tg 两测试合并）
    - 内存峰值按 test_time 分段归属到各配置：第 i 个测试的区间为
      (前一测试结束时刻, 本测试结束时刻]，区间内采样最大值即该测试的峰值。
      近似值（含前一测试的尾部与 warmup），单配置运行时等价于全程峰值

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
from datetime import datetime
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

# 区分同一 results.json 内多个测试配置的全部字段（组合为分组键）
CFG_FIELDS = [
    "model_filename", "n_gpu_layers", "n_cpu_moe", "tensor_buft_overrides",
    "flash_attn", "n_threads", "cpu_mask", "cpu_strict", "use_mmap",
    "type_k", "type_v", "split_mode", "main_gpu", "no_kv_offload",
    "n_batch", "n_ubatch", "devices", "tensor_split", "use_direct_io",
    "embeddings", "no_op_offload", "no_host", "fit_target", "fit_min_ctx",
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


def to_epoch(ts):
    """ISO 时间字符串 -> epoch 秒；支持 'Z' 与 '+08:00' 两种后缀"""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def load_samples(path: Path, col: str):
    """读采样 CSV 的 (timestamp, value) 列表；文件/列缺失返回空表"""
    if not path.is_file():
        return []
    try:
        out = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    e = float(row.get("timestamp") or "")
                    v = float((row.get(col) or "").strip())
                except (TypeError, ValueError):
                    continue
                out.append((e, v))
        return out
    except Exception as e:
        warn(f"读取 {path} 失败: {e}")
        return []


def peaks_per_test(run_start, rows, samples):
    """按 test_time 把采样分段，返回与 rows 等长的每测试峰值列表。

    第 i 个测试的区间 = (第 i-1 个测试的 test_time, 第 i 个测试的 test_time]，
    首个区间下界为运行开始时刻（meta.start）。
    """
    n = len(rows)
    if not samples:
        return [None] * n
    bounds = []
    prev = run_start
    for r in rows:
        t = to_epoch(r.get("test_time"))
        if t is None:
            t = prev  # test_time 缺失时区间退化为空
        bounds.append((prev, t))
        prev = t
    out = []
    for lo, hi in bounds:
        vals = [v for e, v in samples if e is not None and lo <= e <= hi]
        out.append(max(vals) if vals else None)
    return out


def fmt(v, nd=3):
    return "" if v is None else round(float(v), nd)


def pick_test(rows, kind: str):
    """从（同一配置的）测试行中取 pp（n_gen=0）或 tg（n_prompt=0）行"""
    if kind == "pp":
        cand = [r for r in rows if int(r.get("n_gen", 0)) == 0 and int(r.get("n_prompt", 0)) > 0]
    else:
        cand = [r for r in rows if int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0]
    return cand[0] if cand else None


def config_key(row):
    return tuple((f, row.get(f)) for f in CFG_FIELDS)


def group_max(idxs, peaks):
    vals = [peaks[i] for i in idxs if peaks[i] is not None]
    return max(vals) if vals else None


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

    # 内存采样按 test_time 分段归属到每个测试行
    run_start = to_epoch(meta.get("start"))
    gpu_peaks = peaks_per_test(run_start, rows, load_samples(run_dir / "gpu_mem.csv", "gpu_mem_used_mib"))
    rss_peaks = peaks_per_test(run_start, rows, load_samples(run_dir / "host_mem.csv", "proc_rss_mib"))
    sys_peaks = peaks_per_test(run_start, rows, load_samples(run_dir / "host_mem.csv", "sys_used_mib"))

    # 按配置分组（保持首次出现顺序）
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(config_key(r), []).append(i)

    records = []
    for idxs in groups.values():
        grp_rows = [rows[i] for i in idxs]
        pp = pick_test(grp_rows, "pp")
        tg = pick_test(grp_rows, "tg")
        if pp is None or tg is None:
            warn(f"跳过 {run_dir.name} 的一个配置组: 未同时找到 pp/tg 测试行")
            continue
        base = pp  # 组内配置字段相同，任取一行
        model_size = base.get("model_size")

        def ms(r):
            return fmt(float(r["avg_ns"]) / 1e6, 2) if r.get("avg_ns") is not None else ""

        records.append({
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
            "gpu_mem_peak_mib": int(v) if (v := group_max(idxs, gpu_peaks)) is not None else "",
            "host_rss_peak_gib": fmt(v / 1024, 2) if (v := group_max(idxs, rss_peaks)) is not None else "",
            "host_sys_used_peak_gib": fmt(v / 1024, 2) if (v := group_max(idxs, sys_peaks)) is not None else "",
            "backends": base.get("backends", ""),
            "gpu_info": base.get("gpu_info", ""),
            "cpu_info": base.get("cpu_info", ""),
            "build_commit": base.get("build_commit", ""),
            "duration_s": meta.get("duration_s", ""),
            "exit_code": meta.get("exit_code", ""),
            "raw_dir": str(run_dir.relative_to(REPO_ROOT)) if run_dir.is_relative_to(REPO_ROOT) else str(run_dir),
        })
    return records


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
        recs = parse_run(run_dir)
        if recs:
            records.extend(recs)

    if not records:
        print(f"ERROR: {raw_root} 下没有有效运行（全部失败或为空），未生成 CSV", file=sys.stderr)
        sys.exit(1)

    records.sort(key=lambda r: (str(r["run_ts"]), str(r["exp_id"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(records)

    # 摘要
    print(f"== {len(records)} 行（配置级） -> {out_path}")
    print(f"{'exp_id':<24} {'ngl':>4} {'t':>4} {'pp_tok_s':>9} {'tg_tok_s':>9} {'gpu_MiB':>9} {'rss_GiB':>8}")
    for r in records:
        print(f"{str(r['exp_id']):<24} {str(r['ngl']):>4} {str(r['n_threads']):>4} "
              f"{str(r['pp_tok_s']):>9} {str(r['tg_tok_s']):>9} "
              f"{str(r['gpu_mem_peak_mib']):>9} {str(r['host_rss_peak_gib']):>8}")


if __name__ == "__main__":
    main()
