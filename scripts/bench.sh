#!/usr/bin/env bash
# bench.sh — 统一封装 llama-bench：跑一次基准，原始结果全量落盘
#
# 用法:
#   bench.sh -e <exp_id> -m <model.gguf> [其余参数原样透传给 llama-bench]
#   透传示例: -ngl 0 | -ngl 999 -ot "per_layer_token_embd\.weight=CPU" | -fa on | -t 64
#   支持批量多值（llama-bench 原生语法，一次进程跑完整批）:
#     -ngl 0,16,32,48        离散多值
#     -ngl 0-48+8            范围 0..48 步长 8
#     -t 64,128,256          多值可组合，产出 = 各参数值的笛卡尔积
#   透传参数不得包含 -m/-p/-n/-r/-o/-oe/--progress（口径与输出格式由本脚本固定）
#
# 环境变量:
#   BENCH_BIN              llama-bench 路径（默认取 PATH）
#   BENCH_PREFIX           命令前缀，如 'numactl --cpunodebind 0 --membind 0'
#   BENCH_SAMPLE_INTERVAL  内存采样间隔秒（默认 0.5）
#
# 统一口径（CLAUDE.md 实验纪律，勿改）: -p 512 -n 128 -r 3
#   注: 当前 llama-bench 无 -c/--ctx-size 参数，ctx 按 p+n 自动分配
#
# 产物目录 data/raw/<exp_id>/<YYYYmmdd_HHMMSS>/:
#   meta.json    运行元数据（完整命令行、起止时间、退出码、主机信息）
#   results.json llama-bench -o json 原始结果（含每次重复的 samples）
#   bench.log    完整 stderr（warning / 进度 / md 汇总表，人类可读）
#   gpu_mem.csv  GPU 显存采样（无 nvidia-smi 时仅表头）
#   host_mem.csv 进程当前 RSS / 累计 RSS 峰值(HWM) / 系统已用内存采样
#                （多配置批量运行时，parse_bench.py 按 test_time 分段归属到各配置）
#   FAILED       退出码非 0 时创建（parse_bench.py 跳过该目录）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${BENCH_RAW_ROOT:-$REPO_ROOT/data/raw}"

usage() {
    cat <<'EOF'
bench.sh — 统一封装 llama-bench：跑一次基准（支持多值批量），原始结果全量落盘

用法:
  bench.sh -e <exp_id> -m <model.gguf> [其余参数原样透传给 llama-bench]
  透传示例: -ngl 0 | -ngl 999 -ot "per_layer_token_embd\.weight=CPU" | -fa on | -t 64
  批量示例: -ngl 0,16,32,48 | -ngl 0-48+8 | -t 64,128,256（llama-bench 原生多值语法）
  透传参数不得包含 -m/-p/-n/-r/-o/-oe/--progress（口径与输出格式由本脚本固定）

环境变量:
  BENCH_BIN              llama-bench 路径（默认取 PATH）
  BENCH_PREFIX           命令前缀，如 'numactl --cpunodebind 0 --membind 0'
  BENCH_SAMPLE_INTERVAL  内存采样间隔秒（默认 0.5）

统一口径: -p 512 -n 128 -r 3（ctx 由 llama-bench 按 p+n 自动分配，无 -c 参数）
产物: data/raw/<exp_id>/<时间戳>/{meta.json,results.json,bench.log,gpu_mem.csv,host_mem.csv}
EOF
    exit "${1:-0}"
}

# ---------- 参数解析（-e/-m 之外的参数全部透传） ----------
EXP_ID=""
MODEL=""
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e) EXP_ID="${2:?"-e 需要参数"}"; shift 2 ;;
        -m) MODEL="${2:?"-m 需要参数"}"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) PASS+=("$1"); shift ;;
    esac
done

[[ -n "$EXP_ID" ]] || { echo "ERROR: 缺少 -e <exp_id>" >&2; usage 1; }
[[ -n "$MODEL" ]] || { echo "ERROR: 缺少 -m <model.gguf>" >&2; usage 1; }
[[ -f "$MODEL" ]] || { echo "ERROR: 模型文件不存在: $MODEL" >&2; exit 1; }
[[ "$EXP_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: exp_id 只允许 [A-Za-z0-9._-]: $EXP_ID" >&2; exit 1; }

BENCH_BIN="${BENCH_BIN:-llama-bench}"
command -v "$BENCH_BIN" >/dev/null 2>&1 || { echo "ERROR: 找不到 llama-bench，请设 BENCH_BIN=/path/to/llama-bench" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: 需要 python3" >&2; exit 1; }

INTERVAL="${BENCH_SAMPLE_INTERVAL:-0.5}"
P=512; N=128; R=3   # 统一口径（CLAUDE.md 实验纪律，勿改）

TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$RAW_ROOT/$EXP_ID/$TS"
mkdir -p "$OUTDIR"

# ---------- 组装完整命令 ----------
CMD=()
if [[ -n "${BENCH_PREFIX:-}" ]]; then
    read -ra PREFIX_ARR <<< "$BENCH_PREFIX"
    CMD+=("${PREFIX_ARR[@]}")
fi
CMD+=("$BENCH_BIN" -m "$MODEL" -p "$P" -n "$N" -r "$R")
CMD+=("${PASS[@]+"${PASS[@]}"}")
CMD+=(-o json -oe md --progress)

CMD_STR="$(printf '%q ' "${CMD[@]}")"
BENCH_BIN_ABS="$(command -v "$BENCH_BIN")"

echo "== exp_id: $EXP_ID"
echo "== cmd:    $CMD_STR"
echo "== out:    $OUTDIR"

echo "timestamp,gpu_mem_used_mib" > "$OUTDIR/gpu_mem.csv"
echo "timestamp,proc_rss_mib,proc_rss_hwm_mib,sys_used_mib" > "$OUTDIR/host_mem.csv"

# ---------- 启动基准 + 内存采样 ----------
START_TS="$(date -Is)"; START_EPOCH="$(date +%s)"
"${CMD[@]}" > "$OUTDIR/results.json" 2> "$OUTDIR/bench.log" &
MAIN_PID=$!

if command -v nvidia-smi >/dev/null 2>&1; then
    (
        while kill -0 "$MAIN_PID" 2>/dev/null; do
            used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s}')" || true
            if [[ -n "${used:-}" ]]; then
                echo "$(date +%s),${used}" >> "$OUTDIR/gpu_mem.csv"
            fi
            sleep "$INTERVAL"
        done
    ) &
    GPU_PID=$!
fi

(
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        rh="$(awk '/^VmRSS/{r=$2}/^VmHWM/{h=$2}END{print r+0, h+0}' "/proc/$MAIN_PID/status" 2>/dev/null)" || true
        sysmib="$(awk '/^MemTotal/{t=$2}/^MemAvailable/{a=$2}END{if(t&&a)print int((t-a)/1024)}' /proc/meminfo 2>/dev/null)" || true
        rss_mib=0; hwm_mib=0
        IFS=' ' read -r rss_mib hwm_mib <<< "${rh:-0 0}" || true
        echo "$(date +%s),$((rss_mib/1024)),$((hwm_mib/1024)),${sysmib:-}" >> "$OUTDIR/host_mem.csv"
        sleep "$INTERVAL"
    done
) &
HOST_PID=$!

# ---------- 元数据落盘 ----------
write_meta() {
    local code="$1"
    END_TS="$(date -Is)"; END_EPOCH="$(date +%s)"
    META_OUT="$OUTDIR/meta.json" M_EXP_ID="$EXP_ID" M_CMD="$CMD_STR" \
    M_BIN="$BENCH_BIN_ABS" M_MODEL="$MODEL" M_PASS="${PASS[*]+${PASS[*]}}" M_PREFIX="${BENCH_PREFIX:-}" \
    M_START="$START_TS" M_END="$END_TS" M_SE="$START_EPOCH" M_EE="$END_EPOCH" M_CODE="$code" \
    M_P="$P" M_N="$N" M_R="$R" M_DIR="$OUTDIR" \
    python3 - <<'PYEOF'
import json, os, platform, socket
meta = {
    "exp_id": os.environ["M_EXP_ID"],
    "command": os.environ["M_CMD"],
    "bench_bin": os.environ["M_BIN"],
    "model": os.environ["M_MODEL"],
    "llama_bench_extra_args": os.environ["M_PASS"],
    "prefix": os.environ["M_PREFIX"],
    "unified": {"p": int(os.environ["M_P"]), "n": int(os.environ["M_N"]), "r": int(os.environ["M_R"])},
    "start": os.environ["M_START"],
    "end": os.environ["M_END"],
    "duration_s": int(os.environ["M_EE"]) - int(os.environ["M_SE"]),
    "exit_code": int(os.environ["M_CODE"]),
    "hostname": socket.gethostname(),
    "uname": " ".join(platform.uname()),
    "raw_dir": os.environ["M_DIR"],
}
with open(os.environ["META_OUT"], "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
PYEOF
}

trap 'write_meta 130; kill "$MAIN_PID" "${GPU_PID:-}" "$HOST_PID" 2>/dev/null || true; exit 130' INT TERM

# ---------- 等待结束 ----------
EXIT_CODE=0
wait "$MAIN_PID" || EXIT_CODE=$?
wait "$HOST_PID" 2>/dev/null || true
if [[ -n "${GPU_PID:-}" ]]; then
    wait "$GPU_PID" 2>/dev/null || true
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
    write_meta "$EXIT_CODE"
    touch "$OUTDIR/FAILED"
    tail -5 "$OUTDIR/bench.log" >&2 || true
    echo "ERROR: llama-bench 退出码 $EXIT_CODE，原始数据保留在 $OUTDIR（sweep 应中止）" >&2
    exit "$EXIT_CODE"
fi

write_meta 0

# ---------- 摘要 ----------
python3 - "$OUTDIR/results.json" <<'PYEOF' || true
import json, sys
try:
    rows = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"WARN: 摘要解析失败: {e}")
    sys.exit(0)
for r in rows:
    tag = f"ngl={r.get('n_gpu_layers')} t={r.get('n_threads')} ot={r.get('tensor_buft_overrides')}"
    if int(r.get("n_gen", 0)) == 0 and int(r.get("n_prompt", 0)) > 0:
        print(f"pp{r['n_prompt']}: {float(r['avg_ts']):.2f} tok/s (+/-{float(r['stddev_ts']):.2f})  [{tag}]")
    elif int(r.get("n_prompt", 0)) == 0 and int(r.get("n_gen", 0)) > 0:
        print(f"tg{r['n_gen']}: {float(r['avg_ts']):.2f} tok/s (+/-{float(r['stddev_ts']):.2f})  [{tag}]")
PYEOF

echo "== done: 原始结果已存 $OUTDIR"
echo "== 解析: python3 $REPO_ROOT/scripts/parse_bench.py"
