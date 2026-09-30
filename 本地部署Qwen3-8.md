# ARM+A100 异构单机部署的性能上限研究

> **主问题：110GB 模型在这台 80GB A100 + 500GB ARM 内存的服务器上，性能上限在哪？无优化基线距上限多少？gap 由什么构成？每个优化手段能收回多少？**
>
> 方法论：理论上限推导 → 基线实测 → gap 三层分解 → 手段逐项验证 → 瀑布图对账。产出物是**一张 gap 瀑布图**：从基线出发，每个手段收回一段，逼近本机可达上限，残差逐项归因。

## 0. 环境事实与 D1 核查

| 项 | 值 |
| --- | --- |
| GPU | A100 80GB PCIe（HBM 1939 GB/s，BF16 312 TFLOPS） |
| Host | ARM 服务器，500GB 内存 |
| 模型 | Qwen3.8-Flash-Next（qwen4exp）UD-Q4_K_XL，110GB，~131B 总参 / ~6B 激活，48 层 / 512 专家 |
| 引擎 | llama.cpp（已构建，支持 qwen4exp） |

D1 核查清单（决定后续所有分母）：

```bash
lscpu | grep -i numa            # NUMA 拓扑：单 socket 还是多路（鲲鹏 2P/4P 常见）→ 决定 L3 的预期收益
numactl -H                      # 各节点内存分布 + GPU 挂在哪个节点
nvidia-smi -q | grep -A4 "GPU Link Info"   # PCIe 代际/宽度
# host 内存带宽实测（STREAM triad），多路机器分"本节点/跨节点"各测一次
# tensor 清单：gguf_dump.py 看 PLE 表 / 专家 / 注意力各占多少 GB，量化类型
./llama-server --help | grep -A2 spec-type   # MTP 支持与否
```

## 1. 问题形式化：上限、基线与 gap 的三层分解

### 1.1 三层上限

```
U_ideal    虚上限：全 GPU 常驻的 roofline（decode ~540 tok/s）        —— 不可达（110>80），仅定义总 gap
U_machine  本机可达上限：U_ideal − 显存不足被迫卸载的最小代价          —— 放置策略的最优解
P_final    实测最优：U_machine − 实现层 gap（可优化）− 残差（不可消除）
```

### 1.2 基线定义

**B0 = "什么都不敢动的跑起来"**：`-ngl 0` 纯 host（全 GPU 放不下，这是唯一无决策的跑法）、默认线程数、默认 mmap、`-fa off`、无 NUMA 绑定。

### 1.3 gap 分解（本项目唯一的分析框架）

```
Gap_total = U_ideal − B0
          = gap_物理    显存不足，被迫在 ARM 上计算部分层     ← 放置策略在"最小化"
          + gap_实现    attention 实现 / NUMA 远端访问 / page fault / 线程配置 / launch 间隙   ← 系统调优在"消除"
          + gap_残差    dequant 计算 / MoE 路由开销 / 采样 / 同步等待                        ← 不可消除，但要归因
```

两个子问题对应：**Q1 = 测定 U_ideal、B0 并把 Gap_total 分解到三项；Q2 = 每个手段回收 gap 的哪个成分、多少 ms/token。**

物理模型关键事实（决定归因方向）：`-ot ...=CPU` 的语义是**该层在 CPU 上计算**，权重不过 PCIe。PCIe 只承担层间 activation（KB 级）+ 每边界一次同步往返。所以"专家放 CPU"的瓶颈是 ARM 侧 dequant+GEMM 吞吐，不是 PCIe 带宽。

## 2. Q1：上限与基线的测定

### 2.1 理论上限推导

**Decode（带宽 roofline）**：

```
每 token 读取 ≈ 6B 激活 × 0.6 B/param(Q4) ≈ 3.6 GB
U_ideal(decode) = 1939 GB/s ÷ 3.6 GB ≈ 540 tok/s（1.9 ms/token）
```

**Prefill（算力 roofline）**：

```
每 token ≈ 2×激活参数 ≈ 12 GFLOP
U_ideal(prefill) = 312 TFLOPS ÷ 12 GFLOP ≈ 26000 tok/s（预期 MFU 10-25%）
```

**Host 侧上限（B0 的理论锚点）**：B0 的 decode 上限 = STREAM 带宽 ÷ 3.6GB（D1 实测后填入；参考 DGX Spark 同模型 25 tok/s，A100 服务器 ARM 侧带宽决定 B0 落点）。

**Step-time 预算（归因用的分解公式）**：

```
T_token = T_HBM + T_launch + T_arm_offload + T_ple + T_sample
预测：T_HBM ~1.9ms | T_launch 1-3ms（48层×~20 kernel×3µs，llama.cpp 无 CUDA Graph）
     | T_arm_offload 0.1-0.5ms×卸载层数 | T_ple 0.1-0.5ms | T_sample ~0.1ms
```

**Q1 的潜在头版发现**：6B 激活让权重读取只要 1.9ms，而 launch 间隙可能是同量级——若实测证实 decode 是 launch-bound 而非带宽-bound，就直接解释了 vLLM/TRT-LLM 为什么上 CUDA Graph。

### 2.2 基线 B0 实测

```bash
llama-bench -m model.gguf -ngl 0 -p 512 -n 128 -r 3 -c 32768   # 默认 -t、mmap、无 fa
# 补：-t 扫描（nproc/2, nproc, nproc×2）找默认线程是否已经最优（本身是 L3 的一部分）
```

读数：pp512 / tg128 / ARM 内存占用。**B0 的 tg × 3.6GB ÷ STREAM 带宽 = host 侧效率**，若 <50% 则 B0 连 host 上限都没跑满（说明 gap_实现 在纯 host 路径就存在，NUMA/线程即回收手段）。

### 2.3 gap 构成的判定规则（profiling 怎么读数）

对最优配置跑 `nsys profile` + `ncu` + `nvidia-smi dmon` + `pidstat`，按规则归位：

| 观测 | 判定 | gap 归属 |
| --- | --- | --- |
| kernel gap 总时长 / 步 >30% | launch-bound | gap_实现 → CUDA Graph 方向（llama.cpp 内只能缓解） |
| HBM 利用率 >70% | 已近带宽极限 | 接近 U_ideal，残差小 |
| nsys 中 CPU 段（mul_mat on ARM）占比高 | 卸载层主导 | gap_物理 → 更优放置/更少卸载 |
| 跨 NUMA 节点流量高（perf c2c / numastat） | 远端访问 | gap_实现 → L3 绑定 |
| PCIe 总量小但等待多 | 同步往返 | gap_残差（+放置策略考虑边界合并） |

## 3. Q2：手段—gap 映射矩阵

统一在"当前最优配置"上叠加测每个手段（顺序：先大头后小头），记录 Δtg、Δpp、Δ显存/host 内存。

### L1 层放置策略（预计最大杠杆，回收 gap_物理）

| 配置 | 命令要点 | 回答 |
| --- | --- | --- |
| B0 | `-ngl 0` | 基线 |
| PLE→CPU 其余 GPU | `-ngl 999 -ot "per_layer_token_embd\.weight=CPU"` | 51B 稀疏查表放 host 的损失（预测 <10%） |
| PLE→CPU + N 层专家→CPU | 上者 + `-ot "blk\.{a-b}\.ffn_.*_exps\.weight=CPU"`，N=8/16/24 | 卸载斜率 ms/层 → 换算"1GB 显存值多少 tok/s" |

- **机制**：PLE 每 token 只稀疏 gather（KB 级）→ 放 host 近无损；专家是激活权重全量读取 → 放 host 每层每 token ~14MB 走 ARM 吞吐
- **读数**：斜率线性 → 符合模型；非线性 → 有重叠/串行化，nsys 细查。斜率 ÷ 每层激活字节 = ARM 侧有效吞吐，对比 STREAM 判定算力/带宽受限
- 显存账：110 − PLE(~27GB) ≈ 83GB > 80 − KV − buffer，预期最优 = PLE + ~10 层专家在 host，此即 U_machine 的实测定位

### L2 Flash Attention（`-fa on`，回收 gap_实现）

- **机制与预测**：decode 本身 GEMM/带宽主导，`-fa` 对 tg 帮助小；收益集中在 pp 和长 ctx。**且混合注意力下 `-fa` 只作用于全注意力层**（DeltaNet/QSA 不走该路径）→ 预测收益远小于 dense 模型的经验值，实测检验本身就是新架构的未知数
- **测**：最优配置上 `-fa on/off` × (pp512, tg128, ctx 128K 的 tg)
- **读数**：tg 持平 + pp 提升 → 符合预测；若 tg 也显著变化 → attention 实现在 decode 路径的占比超预期，回 nsys 看 kernel 分布

### L3 NUMA 绑定与线程（回收 gap_实现）

- **机制与预测**：多路 ARM（鲲鹏 2P/4P）跨节点访问延迟 ~2×，110GB 权重默认 interleaved 分布 → gather/读取大量远端命中。预测纯 host 和重卸载配置提升 10-30%；单 socket 则收益≈0（"诚实零结果"也是结果）
- **测**：`numactl --cpunodebind N --membind N`（含 GPU 所在节点 vs 对侧节点两组）× B0 和 L1 最优配置；配合 `-t` 扫描
- **读数**：绑计算节点+本地内存的增益 = 跨节点代价；GPU 对侧节点绑定的损失 = PCIe 跨节点代价。两个数字都是异构部署的通用结论

### L4 内存布局：`--no-mmap --mlock`（回收 gap_实现，预期≈0）

- **机制与预测**：mmap 首次触碰引发 page fault，影响冷启动与首 token 抖动；steady-state 吞吐预测无感
- **测**：开/关对比 tg 稳定性与首个 100 token 的延迟分布
- **读数**：只影响尾延迟不影响吞吐 → 结论"部署时必开（稳定性），benchmark 时无需（不影响均值）"，一句话的工程结论

### L5 KV 量化（`-ctk/-ctv q8_0/q4_0`，间接回收 gap_物理）

- **机制与预测**：权重读取主导 → KV 量化对 tg 影响预测 <5%；真实收益是**腾显存 → 少卸载专家层**，与 L1 联动
- **测**：KV 档位 × 最优放置重算；长 ctx(128K+) 下再测一次（KV 占比随 ctx 上升）
- **读数**：小 ctx 下纯速度收益≈0 + 显存换算出可少卸载 2-3 层 → 净收益为正，量化"KV 量化是显存手段而非速度手段"

### 汇总：gap 瀑布图（最终交付物）

```
tg(tok/s)
U_ideal 540 ─────────────────────────────────┐(虚上限)
U_machine ~??? ────────┐(放置最优, D2 实测定位)  │
P_final ~??? ──┐        │                       │
B0 ~?? ──┤     │        │                       │
      └─+L1─┴─+L3/L2─┴────────────────────────┘
横轴累积：B0 → +L1放置 → +L3 NUMA → +L2 fa → +L5 → P_final ‖ 残差 = 归因表
```

每段标注 Δtok/s 和归因（物理/实现/残差），配合 nsys 截图。**这张图 + 2.3 的判定规则表 = Q1+Q2 的完整答案。**

## 4. 加时赛：MTP 投机解码——改写上限的手段

MTP 不属于 gap 填补：一次权重读取验证 k 个 draft token，**分母从 3.6GB/token 变成 3.6GB/接受长度**，直接突破带宽 roofline。单列一章：

- 前置：MTP head（shared-Q8_0，2.6GB）+ PR #28243 构建（若现有构建无 `--spec-type draft-mtp`）
- 测：开/关 tg 对比、日志 `draft acceptance`、`--spec-draft-n-max` 扫 2/3/4、temp 0/0.7/1.0 的 acceptance 变化
- **A100 特有视角（带宽经济学）**：draft head 2.6GB 每次 draft 全量读取，A100 PCIe 1.9TB/s 下占 ~1.3ms，相对成本高于 B200(8TB/s) → 预测加速比低于参考值 1.67x。低带宽卡上投机解码收益缩水，这个结论本身有传播价值
- MoE 加成：verify 一步算 k+1 个 token、专家激活取 union，每步权重读取远小于 (k+1) 倍 → MTP 在 MoE 上比 dense 更便宜（预测接受长度增加时衰减更平缓）

## 5. 支线（时间富余再做）：UD 动态量化 vs uniform

社区反馈 UD 在 CPU 卸载场景反而慢（IQ 子层 dequant 复杂），无人系统验证。同模型 Q4_K_M 对照：GPU 路径 vs 重卸载路径 × tg/pp + PPL。结果无论证伪证实，补进[从零学习模型量化](../AI/从零学习模型量化.md)。

## 6. 排期（7 天）

| 天 | 任务 | 交付 |
| --- | --- | --- |
| D1 | 核查清单（第 0 节）+ 理论上限成文 + 预算表初值 | 分母确定 |
| D2 | B0 基线（含 -t 扫描）+ L1 放置矩阵 | U_machine 定位 + 卸载斜率 |
| D3 | L3 NUMA × L2 fa × L4 mmap 扫描 | 系统手段数据 |
| D4 | L5 KV 联动 + 瀑布图初版 | P_final + 各段 Δ |
| D5 | MTP 构建 + 测量（加速比/acceptance/温度） | 突破上限数据 |
| D6 | nsys/ncu 归因：预算表对账 + 残差分解 | 判定规则表填完 |
| D7 | 报告定稿 + 博客互链 + 简历表述 | 本文档 |

## 7. 产出与简历表述

- 本报告：瀑布图 + 预算对账表 + 判定规则表（Q1/Q2 的完整答案）
- GitHub：脚本 + CSV + profiling 截图
- 博客反哺：kvcache.md、从零学习模型量化.md、再战transformer.md（MoE 放置实证）、新篇"投机解码"

> 简历 bullet（数字待填）：在 ARM+A100 80G 异构单机上研究 131B-A6B MoE 模型的推理性能上限：推导带宽 roofline（decode 理论 __ tok/s），将"基线 → 上限"的总 gap 分解为物理约束/实现损耗/固有残差三层；通过放置策略（51B PLE 表驻留 host 近无损）、NUMA 绑定、KV 量化等手段将性能从基线 __ tok/s 提升至 __ tok/s（本机可达上限的 __%），逐项量化各手段贡献；经 nsys 归因定位 decode 为 __-bound（launch 间隙占 __%），实测 MTP 投机解码进一步突破带宽上限 __x（acceptance __%）。

## 参考

1. [unsloth Qwen3.8-Flash-Next-GGUF MTP README（B200 实测 1.67x / 66%）](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
2. [DGX Spark 部署参考（PLE 留 host，25 tok/s）](https://forums.developer.nvidia.com/t/qwen3-8-flash-next-ud-q4-k-xl-gguf-on-dgx-spark-with-llama-cpp-gpu-experts-ple-n-gram-table-streamed-from-disk-25-tok-s-up-to-1m-context/381720)
3. [llama.cpp PR #28243（qwen4exp MTP）](https://github.com/ggml-org/llama.cpp/pull/28243)
4. [动态量化 vs uniform 的社区讨论](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/74)
