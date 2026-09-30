# CLAUDE.md

## 项目定位

ARM+A100 异构单机 LLM 推理性能上限研究。目标产出：可复现的 benchmark 脚本 + 实验数据（CSV）+ 系统性分析报告 + gap 瀑布图，最终形成量化结论（简历项目级质量）。

**主问题：110GB 的 Qwen3.8-Flash-Next 在 80G A100 + 500G ARM 内存服务器上，性能上限在哪？无优化基线距上限多少？gap 由什么构成？每个优化手段收回多少？**

实验设计文档（唯一事实来源，含完整实验矩阵与读数判据）：`docs/experiment-design.md`。实验执行前先读它。

## 环境事实（已确认，勿重复询问或推导）

- GPU：A100 80GB PCIe（HBM 1939 GB/s，BF16 312 TFLOPS）
- Host：ARM 服务器，500GB 系统内存，远程访问
- 模型：Qwen3.8-Flash-Next（qwen4exp 架构）UD-Q4_K_XL，110GB，~131B 总参 / ~6B 激活，48 层 / 512 专家，含 51B 参数 PLE n-gram 表（tensor 名 `per_layer_token_embd`，推理时稀疏查表）
- 引擎：llama.cpp 已构建且支持 qwen4exp；MTP 投机解码需按 PR #28243 另行构建（可与现有版本并存）
- 待 D1 实测补全的分母：PCIe 代际、host STREAM 带宽、NUMA 拓扑、各 tensor 类型体积——核查结果回写实验设计文档第 0 节

## 核心分析框架（所有实验围绕它）

```
Gap_total = U_ideal − B0 = gap_物理 + gap_实现 + gap_残差
```

- **U_ideal**：全 GPU 常驻的 roofline（decode 理论 ~540 tok/s = 1939 GB/s ÷ 3.6 GB/token）。不可达（110 > 80），仅定义总 gap
- **B0**：`-ngl 0` 纯 host 无优化基线
- **gap_物理**（显存不足被迫在 CPU 计算）→ L1 放置策略最小化
- **gap_实现**（attention 实现 / NUMA 远端访问 / mmap / 线程 / launch 间隙）→ L2-L4 消除
- **gap_残差**（dequant 计算 / MoE 路由 / 采样 / 同步等待）→ 不可消除，需归因
- **MTP 投机解码是突破上限**（一次权重读取验证多 token，改写每 token 读取字节），不属于 gap 填补
- 最终交付物：gap 瀑布图（B0 → +L1 → +L3 → +L2 → +L5 → P_final ‖ 残差归因表）

## 关键物理模型（归因时勿搞错）

1. `-ot ...=CPU` 的语义是**该层在 CPU 上计算**，权重不过 PCIe。PCIe 只承担层间 activation（KB 级）+ 每边界一次同步往返
2. 专家层卸载的瓶颈是 ARM 侧 dequant+GEMM 吞吐，**不是 PCIe 带宽**
3. PLE 表放 host 近无损（每 token 只做 KB 级稀疏 gather）
4. 混合注意力（Gated DeltaNet 固定状态 + QSA 稀疏）：KV 增长远低于 dense 公式；`-fa` 只作用于全注意力层，预测收益远小于 dense 模型经验值
5. MoE decode 每步只读激活权重（~3.6GB/token），launch 间隙（48 层 × ~20 kernel × 3µs，llama.cpp 无 CUDA Graph）可能是同量级——**"launch-bound vs 带宽-bound"是本项目潜在头版发现**

## 项目结构（agent 按此组织代码）

```
.
├── CLAUDE.md
├── docs/
│   └── experiment-design.md   # 实验设计与读数判据（从博客迁入，持续回填实测结果）
├── scripts/                   # 所有实验脚本，幂等可重跑
│   ├── env_check.sh           # D1 核查（PCIe/STREAM/NUMA/tensor 清单/MTP 支持）（待实现）
│   ├── bench.sh               # 跑一次基准：原始结果全量落盘 data/raw/<exp_id>/<时间戳>/，口径由脚本固定
│   ├── parse_bench.py         # 解析 data/raw/ → data/results.csv（结构化 CSV，幂等全量重建）
│   └── sweep_*.sh             # E1 放置 / L2-L5 手段 / MTP 扫描（待实现）
├── data/
│   ├── raw/                   # bench.sh 原始落盘（meta.json/results.json/bench.log/gpu_mem.csv/host_mem.csv，失败带 FAILED）
│   └── results.csv            # 结构化结果（一行一次运行：完整命令、pp/tg、显存峰值、host 内存峰值）
└── report/                    # 分析 notebook / 图（瀑布图、斜率曲线）
```

约定：

- 每个 CSV 行必须含：实验 ID、完整命令行、日期、pp512、tg128、显存峰值、host 内存
- 脚本失败不静默：llama-bench 非零退出码直接中止 sweep
- 报告图表可由 `data/` 的 CSV 重新生成，不存手工编辑的数字

## 实验纪律

- 统一口径：`llama-bench -p 512 -n 128 -r 3`（经 `scripts/bench.sh` 固定执行；当前 llama-bench 无 `-c/--ctx-size`，ctx 按 p+n 自动分配），单流；中途不改口径（改口径需全量重跑）
- **每个实验先在实验设计文档写预测，再跑**；读数按设计文档中的判据表归位；实测与预测的偏差本身是归因线索，需记录
- 报告数字必须来自实测数据，禁止编造；理论值/预测值明确标注
- 未实测的数字保留 `__` 占位符，直到数据产出
- 修改实验设计（新增配置、改口径）必须先更新 experiment-design.md 再执行

## 常用命令速查

```bash
# 基线 B0（口径 p512/n128/r3 由 bench.sh 固定；llama-bench 路径用 BENCH_BIN 环境变量指定）
BENCH_BIN=<path/to/llama-bench> ./scripts/bench.sh -e b0 -m <model.gguf> -ngl 0
# 最优放置（PLE→CPU，其余 GPU）
./scripts/bench.sh -e l1_ple_cpu -m <model.gguf> -ngl 999 -ot "per_layer_token_embd\.weight=CPU"
# 卸载第 a-b 层专家（-ot 参数透传）
./scripts/bench.sh -e l1_off8 -m <model.gguf> -ngl 999 -ot "blk\.{a-b}\.ffn_.*_exps\.weight=CPU"
# NUMA 绑定（命令前缀）
BENCH_PREFIX="numactl --cpunodebind 0 --membind 0" ./scripts/bench.sh ...
# 解析原始结果 → 结构化 CSV
python3 scripts/parse_bench.py
# MTP（需 PR #28243 构建）
-md MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2
# Profiling
nsys profile -o <name> ./llama-cli ... ; nsys stats <name>.nsys-rep
ncu --set basic -k "regex:mul_mat|ggml_cuda" ...
nvidia-smi dmon -s um                 # HBM/PCIe 实时吞吐
pidstat -t 1                           # ARM 侧线程利用率
numactl --cpunodebind N --membind N ...  # NUMA 绑定
```

## 禁止事项

- 禁止在未写预测的情况下直接跑新配置
- 禁止并行跑多个实验（共享 GPU/host 会互相污染数据；一次只跑一个配置）
- 禁止用 root 权限或改服务器系统配置（NUMA 绑定等通过 numactl 前缀实现，不持久化）
- 长任务（llama-bench 单配置 >10min）用 nohup/tmux 后台跑，不要阻塞会话
