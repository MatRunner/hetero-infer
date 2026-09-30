# hetero-infer

ARM + A100 异构单机 LLM 推理性能上限研究。目标产出：可复现的 benchmark 脚本 + 实验数据（CSV）+ 系统性分析报告 + gap 瀑布图，形成量化结论。

## 主问题

110GB 的 Qwen3.8-Flash-Next 在 80GB A100 PCIe + 500GB ARM 内存服务器上，性能上限在哪？

- **U_ideal**：全 GPU 常驻的虚上限（decode 理论 ~540 tok/s = 1939 GB/s ÷ 3.6 GB/token），不可达（110 > 80），仅定义总 gap
- **U_machine**：本机可达上限（放置策略最优解）
- **B0**：`-ngl 0` 纯 host 无优化基线

```
Gap_total = U_ideal − B0 = gap_物理 + gap_实现 + gap_残差
```

| gap 成分 | 来源 | 对应手段 |
| --- | --- | --- |
| gap_物理 | 显存不足被迫在 CPU 计算 | L1 层放置策略 |
| gap_实现 | attention 实现 / NUMA 远端访问 / mmap / 线程 / launch 间隙 | L2 FlashAttention、L3 NUMA 绑定、L4 内存布局 |
| gap_残差 | dequant 计算 / MoE 路由 / 采样 / 同步等待 | 不可消除，需归因 |

另有 **L5 KV 量化**（腾显存 → 少卸载层）与 **MTP 投机解码**（一次权重读取验证多 token，突破带宽 roofline，不属于 gap 填补）。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | A100 80GB PCIe（HBM 1939 GB/s，BF16 312 TFLOPS） |
| Host | ARM 服务器，500GB 系统内存 |
| 模型 | Qwen3.8-Flash-Next（qwen4exp）UD-Q4_K_XL，110GB，~131B 总参 / ~6B 激活，48 层 / 512 专家，含 51B 参数 PLE n-gram 表 |
| 引擎 | llama.cpp（已构建，支持 qwen4exp；MTP 需按 PR #28243 另行构建） |

## 项目结构

```
.
├── CLAUDE.md                  # 项目约定与分析框架（agent 工作规范）
├── 本地部署Qwen3-8.md          # 实验设计原始文档
├── docs/
│   └── experiment-design.md   # 实验设计与读数判据（唯一事实来源，持续回填实测结果）
├── scripts/                   # 实验脚本，幂等可重跑
│   ├── env_check.sh           # D1 核查（PCIe/STREAM/NUMA/tensor 清单/MTP 支持）（待实现）
│   ├── bench.sh               # 跑一次基准：原始结果全量落盘 data/raw/<exp_id>/<时间戳>/
│   ├── parse_bench.py         # 解析 data/raw/ → data/results.csv（结构化 CSV）
│   └── sweep_*.sh             # 放置策略 / L2-L5 手段 / MTP 扫描（待实现）
├── data/
│   ├── raw/                   # 原始结果（meta.json / results.json / bench.log / 显存与内存采样）
│   └── results.csv            # 结构化 CSV（一行一次运行：完整命令、pp/tg、显存峰值、host 内存峰值）
└── report/                    # 分析 notebook / 图（瀑布图、斜率曲线）
```

## 使用

统一 benchmark 口径 `-p 512 -n 128 -r 3` 由 [scripts/bench.sh](scripts/bench.sh) 固定执行（ctx 按 p+n 自动分配），原始结果全量落盘后解析为结构化 CSV：

```bash
BENCH_BIN=<path/to/llama-bench> ./scripts/bench.sh -e b0 -m <model.gguf> -ngl 0    # 基线 B0
./scripts/bench.sh -e l1_ple_cpu -m <model.gguf> -ngl 999 \
  -ot "per_layer_token_embd\.weight=CPU"                                            # 最优放置（PLE→CPU）
BENCH_PREFIX="numactl --cpunodebind 0 --membind 0" ./scripts/bench.sh ...           # NUMA 绑定
python3 scripts/parse_bench.py                                                      # 解析 → data/results.csv
```

约定：

- 每个 CSV 行必须含：实验 ID、完整命令行、日期、pp512、tg128、显存峰值、host 内存
- 脚本失败不静默：llama-bench 非零退出码直接中止 sweep
- 报告图表可由 `data/` 的 CSV 重新生成，不存手工编辑的数字
- 报告数字必须来自实测数据，禁止编造；理论值/预测值明确标注；未实测数字保留 `__` 占位符

## 参考

- [unsloth Qwen3.8-Flash-Next-GGUF MTP README（B200 实测 1.67x / 66%）](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
- [DGX Spark 部署参考（PLE 留 host，25 tok/s）](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720)
- [llama.cpp PR #28243（qwen4exp MTP）](https://github.com/ggml-org/llama.cpp/pull/28243)
