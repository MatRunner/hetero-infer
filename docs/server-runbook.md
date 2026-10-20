# 服务器统一测试清单（runbook）

开发机与服务器环境不一致：服务器的所有数字（带宽、利用率、显存水位、OOM 边界）都无法本地预演。本清单**一次性列全所有待测数据项**，服务器 session 按批次顺序统一执行，回传 `data/` 后本地分析。

- 按批次顺序执行（依赖偏序），批内可乱序
- **主线为手动测试**：逐行执行 `data/manual-test-matrix.csv`（预填配置与命令，结果列手动填回，CSV 即最终结构化数据）。`scripts/bench.sh`/`parse_bench.py` 保留备用（自动落盘 + 采样更全），两者不冲突：手动跑的行不进 data/raw，但 pp/tg 口径一致（p512/n128/r3）
- 自动模式（可选）：`./scripts/bench.sh -e <exp_id> ...` 落盘原始结果，回传后 `python3 scripts/parse_bench.py` 重建 results.csv

## 手动测试流程（三终端观测法）

每个测试行执行时开两个伴随终端：

```bash
# 终端 2 — GPU 侧：sm%（SM 利用率）/ mem%（显存带宽利用率）/ 显存用量 MiB
nvidia-smi dmon -s um

# 终端 3 — CPU 侧：进程 CPU%（多核累计，1 逻辑核=100%，满机 25600%）与 RSS(KiB)
pidstat -u -r -p $(pgrep -n llama-bench) 1
```

读数口径（填入矩阵对应列，均取 **decode 稳态段**的值，忽略模型加载阶段）：

| 矩阵列 | 来源 | 说明 |
| --- | --- | --- |
| pp512_tok_s / tg128_tok_s | llama-bench 输出表 | tg 那行的 tok/s |
| gpu_mem_peak_mib | 终端 2 显存用量 | 加载完成后水位恒定，稳态值即峰值 |
| gpu_util_pct | 终端 2 sm% | 稳态均值；<70% → 时间在 host 侧或间隙 |
| gpu_bw_util_pct | 终端 2 mem% | 高（>80%）→ 带宽 roofline 逼近 |
| cpu_util_pct | 终端 3 %CPU | 稳态峰值；对比线程数判定利用率饱和度 |
| host_rss_gib | 终端 3 RSS | KiB ÷ 1048576 |
| eff_bw_gb_s | 反推：`tg128 × 3.6 ÷ 1000` | 仅对 ngl=0 行有意义（host 全量读取）；÷ A2 STREAM = host 效率 |
| notes | 手记 | OOM 报错、异常波动、默认 t 实际值等 |

执行顺序与回填：
1. 批次 A（环境核查）先跑 → 每层 GB / ngl 预测上限 / GPU 所在节点（`<gn>`）
2. T01-T04（B）→ T05-T11（C，ngl* 从 T05-T09 择优回填到 T14+ 的 `<ngl*>`）
3. T12-T16（D）→ T17（L2）→ T18（合成）
4. `<MODEL>` 占位符开头一次性替换为服务器模型路径

## 批次 A：环境事实校准（其余一切的分母）

| ID | 数据项 | 方法 | 产出与判据 |
| --- | --- | --- | --- |
| A1 | 参数账裁决 | gguf tensor 清单（gguf-py 的 dump 工具，或 llama-cli `--verbose` 加载日志逐 tensor 求和） | 总参数量、每层字节数、PLE 体积、实际平均 bitwidth → **裁决 131B vs 174.5B 矛盾**，定每层 GB 与 ngl 上限（预测 20-25 层区间） |
| A2 | STREAM 带宽（4 布局） | STREAM 二进制：默认 interleaved / `numactl --cpunodebind 0 --membind 0` / 节点 1 / 跨节点（cpunodebind 0 + membind 1） | 各布局实测 GB/s → B0 反推分母 + L3 NUMA 预期收益的基准。判据：单节点预期 230-260（307 打 7-8 折） |
| A3 | NUMA 拓扑 | `lscpu`、`numactl -H`、`nvidia-smi topo -m` | GPU 所在 socket（L3 绑定方案的分母）、节点间距离 |
| A4 | PCIe 链路 | `nvidia-smi -q \| grep -A2 "GPU Link"` | Gen3/Gen4 ×16 → 物理模型佐证（瓶颈不在 PCIe，低优先级） |
| A5 | llama-bench 参数集 | 服务器构建的 `llama-bench --help` 与本地对照 | 确认 `-fa` / `-fitt` / `--spec-type` 等选项存在性，防后续命令白跑 |
| A6 | ISA 特性与构建目标 | `/proc/cpuinfo` Features（`grep -o 'sve\|bf16\|asimdbf16\|dotprod\|i8mm' \| sort -u`）+ 服务器构建的 CMakeCache CFLAGS | 算力口径佐证（5.9T 按 NEON FP32，见理论预测性能.md ISA 小节）；`dotprod` 未在 `-march` 激活 = 量化 dot 内核退化 FP32 路径（gap_实现 潜在项） |

## 批次 B：基线 B0 + 全观测

| ID | 配置 | 基线数据 | 伴随观测 |
| --- | --- | --- | --- |
| B1 | `-ngl 0` 全默认 | pp512 / tg128 | CPU 利用率、RSS/系统内存、GPU SM/带宽利用率（应≈0，作采样链路 sanity） |
| B2 | t-scan：`-t 64,128,192,256` | 同上 | 最优线程数、SMT 对 memory-bound GEMV 的收益（128 vs 256） |

B0 三判据（读数归位）：

1. **host 效率**：`tg × bytes/token`（A1 校准后用实测值，暂按 3.6-3.75 GB）÷ tg 时间 = 有效带宽 → ÷ A2 默认布局 STREAM = host 效率。**<50% → L3（线程/NUMA）即直接回收手段，预期提升大**
2. **CPU 利用率饱和度**：满负荷（接近满线程）佐证"带宽约束"判定；不饱和 → 调度/同步存在 gap_实现，直接证据
3. **MFU 反推**：pp512 × 12 GFLOP ÷ 5.9 TFLOPS = prefill 的算力利用率 → 对照预期 30-50%

## CPU 侧 vs GPU 侧归因（贯穿 C/D 批次的核心读数）

时间损耗在 CPU 还是 GPU，按**两层证据**判定：

**第一层：伴随采样的利用率对照（每次 bench 自动获得，秒级粗判）**

| 读数组合 | 判定 | 后续动作 |
| --- | --- | --- |
| GPU util 高（>70%）+ CPU util 低 | GPU-bound | 看 GPU 带宽 util：高 → 带宽 roofline 逼近（U_machine 附近）；低 → 算子/launch 问题 → nsys |
| GPU util 低 + CPU util 高 | host-bound（CPU 层计算是瓶颈） | L3 线程/NUMA 优先；nsys 看 host 段 |
| 两者都低 | 同步/launch 间隙主导 | **头版发现候选**——nsys 精确分解 launch gap |

- ngl 扫描的 **GPU util 曲线本身是归因数据**：随 ngl 增加，GPU util 上升 → GPU 逐步饱和；tg 增速放缓的拐点即 host 侧不再是瓶颈的位置
- 注意秒级采样的局限：decode 每 token 仅几 ms，秒级 util 是"时间平均"（含 host 段与 idle），**低估 GPU 真实 kernel 占空比**——第一层只做方向性判断

**第二层：nsys 时间线（代表性配置单独跑，精确分解）**

对 B0 / ngl* / 中间点各 profile 一次（`nsys profile -o <name> ./llama-cli ...`，非 bench 场景，避免 profiler 开销污染速度数据）。判读三段分解：

```
每步 decode 耗时 = GPU kernel busy + launch 间隙（kernel 之间的 GPU idle） + host 计算段（CPU 层串行）
```

- kernel busy 占比高 → 带宽/算子效率问题（对照第一层带宽 util）
- launch 间隙占比可观（预测 48 层 × ~20 kernel × 3µs 与 T_HBM 同量级）→ **"launch-bound vs 带宽-bound"头版发现的裁决证据**
- host 段主导 → 佐证 host-bound，斜率读数（ms/层）与 nsys host 段交叉验证

## 批次 C：放置扫描（L1，依赖 A1/B1）

| ID | 配置 | 数据项 | 判据 |
| --- | --- | --- | --- |
| C1 | `-ngl 0-20+4` + `-ot PLE=CPU`（6 配置批量） | pp/tg/显存峰值 + **GPU util 曲线**（分段归属） | **斜率 = ms/层** → gap_物理 量化；显存斜率 GB/层 对照 A1；ngl* 定位；GPU util 曲线见归因章节 |
| C2 | 边界探测：单独跑 `-ngl 22`（若 20 通过），再 24… | OOM 边界 | 实测 ngl 上限 vs A1 预测（**注意 OOM 会中止 llama-bench 批量，边界只能单点跑**） |
| C3 | 专家级 vs 整层对照：`-ngl 999` + M 层 `-ot "blk.{a-b}.ffn_.*_exps.weight=CPU"`（按 C1 的显存余量选 M） | 同显存水位下 tg 对比 | 专家级放置（attention 全留 GPU）是否优于整层切换 → 决定 L1 最终策略 |

C1 斜率读数：host 层均摊耗时 ÷ 每层激活字节 = ARM 侧有效吞吐 → 对照 A2 STREAM 判定**算力受限 / 带宽受限**（归因分水岭）。

## 批次 D：调优（L3/L2，依赖 C 定位 ngl*）

| ID | 配置 | 数据项 |
| --- | --- | --- |
| D1 | NUMA：GPU 节点绑定 vs 对侧（`BENCH_PREFIX="numactl ..."`），ngl* 固定 | pp/tg + **t 复扫 64/128**（单节点绑定后最优 t 会变，这是已知交互点） |
| D2 | fa on/off（ngl* 固定） | 预期收益小（混合注意力，QSA 仅 12 层）——零结果也有价值 |
| D3 | 合成：全部最优组合 + 2×2 交互抽查（ngl × NUMA） | 验证收益可加性；不可加 → 交互回补实验 |

## 批次 E：归因 profiling + MTP（可与 C/D 并行排队）

| ID | 配置 | 数据项 |
| --- | --- | --- |
| E1 | nsys：B0 / ngl* / 中间点各一次（见上文归因第二层） | 三段分解（kernel busy / launch gap / host 段）→ **头版发现裁决**（launch-bound vs 带宽-bound） |
| E2 | MTP（PR #28243 构建）：`-md MTP/...Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2` | 加速比 + acceptance；预测 < 1.67x（A100 40G 低带宽下 draft head 读取相对成本更高） |
| E3 | （可选）ncu 单 kernel 深挖：`ncu --set basic -k "regex:mul_mat\|ggml_cuda"` | 仅当 E1 指向具体算子时（ncu 拖慢 10-100x，不用于速度数据） |

## 采样列定义（已实现并本地冒烟验证）

bench.sh 伴随采样 / parse_bench.py 按 llama-bench `test_time` 分段归属后输出到 results.csv：

| CSV 列 | 定义 | 采样来源 |
| --- | --- | --- |
| `gpu_util_avg/max_pct` | GPU SM 利用率 avg/max（多卡取 max） | `nvidia-smi --query-gpu=utilization.gpu`（无 GPU 时空列降级） |
| `gpu_bw_util_avg/max_pct` | GPU 显存带宽利用率 avg/max | `nvidia-smi --query-gpu=utilization.memory` |
| `cpu_util_avg/max_pct` | 进程 CPU 总占用，**1 逻辑核 = 100%**（满 256 逻辑核 = 25600%） | `/proc/<pid>/stat` utime+stime 差分 |
| `gpu_mem_peak_mib` 等 | 显存/RSS/系统内存峰值 | 原有功能不变 |

**内存带宽（host 侧）**：ARM 无通用实时计数器，不采样——统一用 `tg × bytes/token ÷ tg 时间` 反推有效带宽，以 A2 STREAM 为校准分母（判据见各批次）。GPU 侧带宽用上表 `gpu_bw_util` 列。

## 环境事实待回填（跑完 A 后更新 CLAUDE.md）

- [ ] 总参数量（131B vs 174.5B 裁决）与每层 GB → ngl 实测上限
- [ ] STREAM 四布局实测带宽
- [ ] GPU 所在 socket + 跨节点延迟
- [ ] PCIe 代际
- [ ] 服务器 llama-bench 参数集
