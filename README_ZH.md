<div align="center">
<h1>
  Xing4.1-Preview
</h1>
</div>

<p align="center"><strong>—— Advancing Frontier Agentic Models On Domestic Chips And Frameworks</strong></p>

<p align="center">
  <b>8192 张昇腾（Ascend）910C 芯片 + 国产训练框架 MindSpore<br>
  总参数 285B、单次激活 17B 的 MoE 模型，专为智能体编程与长程复杂 Agent 任务打造，原生 1M 上下文。</b>
</p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview" target="_blank">GitHub</a> · <a href="https://huggingface.co/XingChen-AGI/Xing4.1-Preview" target="_blank">HF (Transformers 格式)</a> · <a href="https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore" target="_blank">HF (MindSpore 格式)</a> · <a href="https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview" target="_blank">魔搭 ModelScope</a> · <a href="https://modelers.cn/user/XingChen-AGI" target="_blank">魔乐 Modelers</a> · <a href="https://atomgit.com/XingChen-AGI" target="_blank">AtomGit</a> · <a href="https://www.teleai.com.cn/" target="_blank">天翼 AI 开放平台</a> · <a href="https://huggingface.co/papers/2611.22115" target="_blank">技术报告</a>
</p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Parameters" src="https://img.shields.io/badge/Params-285B%20%2F%2017B%20active-7C3AED"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Context" src="https://img.shields.io/badge/Context-1M%20native-orange"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
</p>

<p align="right"><a href="./README.md">English</a> | 中文</p>

# 目录

- [简介](#简介)
- [最新动态](#最新动态)
- [面向昇腾 910C 的软硬件协同设计](#面向昇腾-910c-的软硬件协同设计)
- [国产算力与训练框架](#国产算力与训练框架)
- [模型训练](#模型训练)
- [评测](#评测)
- [快速开始](#快速开始)
- [推荐参数](#推荐参数)
- [应用场景](#应用场景)
- [免责声明](#免责声明)
- [协议与引用](#协议与引用)

# 简介

Xing4.1-Preview 由中电信人工智能科技有限公司研发，是星辰语义大模型系列（原 TeleChat）的新一代模型，采用稀疏混合专家（MoE）架构，结合 Multi-head Latent Attention（MLA）与 DeepSeek Sparse Attention（DSA），面向长上下文交互、复杂推理和 Agent 工作流：

- **国产算力与框架训练：** 8192 张昇腾 910C 芯片 + 国产训练框架 MindSpore，共同支撑模型训练。
- **面向国产芯片的架构协同设计：** 针对国产芯片"矩阵算力强、向量与带宽相对有限"的特征做正向设计，而非把模型搬到国产卡上跑。
- **智能体与编程能力突出：** 在数据分析、前端网页生成、长文理解等 Agent 任务场景下表现优异，且在 Codex、OpenCode、Pi、Claude Code、WorkBuddy 等多种主流 Agent Harness 下均展现出良好的适配性与稳定的效果。
- **MLA + DSA 注意力，原生 1M 上下文：** 结合压缩 Key-Value 表示与长上下文稀疏注意力机制，原生支持 **1M** 的配置上下文长度。

# 最新动态

- 2026-11-23 开源 **Xing4.1-Preview** [[HuggingFace](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) | [HuggingFace (MindSpore)](https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore) | [魔搭](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) | [魔搭 (MindSpore)](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview-MindSpore/) | [天翼 AI 开放平台](https://www.teleai.com.cn/)]

# 面向昇腾 910C 的软硬件协同设计

**基于昇腾 910C 芯片矩阵算力占比高、向量计算受限的硬件特性，以及超节点统一内存编址、卡间通信高带宽等集群优势，Xing4.1-Preview 以国产训练框架 MindSpore 为底座，在模型架构、并行策略、Token 词表等方面进行了特定设计，实现模型能力、训练效率与集群利用率的协同提升，同时参考 DeepSeek V3.2 的 MLA 和 DSA 技术特性提升推理效率，并适配 SGLang 和 vLLM 等开源生态。**

## 稀疏注意力架构选型：降低 Vector 计算瓶颈

相比 KDA、GDN 等混合线性注意力架构，**DSA 保留了以矩阵乘为核心的计算模式，具有更好的 Cube 算力适配性**。混合线性注意力虽然具备长序列计算复杂度优势，但其门控、状态更新及扫描操作对 Vector 性能和算子融合能力要求较高，在当前 910C 软件栈下难以充分释放理论收益。因此我们优先选择 DSA，在降低 Attention 计算复杂度的同时，更好地发挥硬件矩阵计算优势。

## 矮胖型模型架构：提升 Cube 算力利用率

在模型规模与并行策略约束下，采用**适度增加网络宽度、降低网络深度**的设计（45 层 / Hidden Size 4096 / Dense FFN Intermediate 18432），提高核心矩阵乘算子的计算规模、算术强度与数据复用效率，提升 Cube 单元利用率。同时，减少网络层数可降低逐层串行执行中的归一化、激活函数、算子调度及数据搬运等开销，削弱 Vector 性能瓶颈对整体训练吞吐的影响。

## 并行策略优化：提升集群算力利用率

采用"超节点内分片、超节点间复制"的混合分片策略降低单卡显存占用，替代流水线并行以消除 bubble 开销；同时直接在参数张量上通信，省去数据进出中间缓冲区的冗余拷贝（copy-in/copy-out），并将超节点内的参数聚合（AllGather）、梯度归约分片（ReduceScatter）与跨超节点的融合梯度归约（AllReduce）同反向计算异步重叠，实现三重通信掩盖。

## 新版 Token 词表设计：提升 Token 压缩比

本次优化面向 Agent 及 Coding 场景定制词表，基于 Agent 轨迹、工具调用、代码仓库等目标场景的海量语料做子词统计、频次与互信息计算，优先把高频连续片段进行合并，在词表规模恒定条件下，Agent 场景下实现 Token 压缩率提升 16%，Coding 场景下 Token 压缩率提升 23%，大幅提升上述场景下解决问题的能力。

# 国产算力与训练框架

Xing4.1-Preview 训练基于国产技术体系开展，使用 8192 张昇腾（Ascend）910C 芯片与国产训练框架 MindSpore 支撑模型训练，关键工作包括：

- Owner-based Muon 优化器：保持参数与动量的分片布局不变，将每个二维矩阵的 Newton–Schulz 正交化交给分片组内的指定 Owner 计算卡聚合后算一次再分发（并按矩阵形状批处理），避免各卡重复还原完整矩阵与重复计算。

- 细粒度选择性重计算：综合算子执行耗时与输出显存占用精准筛选保留算子，避免整层一刀切及逐算子框架回调开销，并通过边界输入占位释放与重算恢复机制进一步压缩激活显存。

- 长序列稀疏注意力优化：支持索引器（Indexer）逐层提前反向传播以尽早释放辅助计算图，结合动态序列块配对与文档级 KV 窗口裁剪，降低超长上下文下的计算不均与访存冗余。

- 宿主机侧调度：按设备池或线程角色子集划分训练线程的 CPU 亲和性（保留池内负载均衡），并将选定的系统服务与可迁移中断隔离至独立系统核池，降低大规模集群同步等待。

后训练使用自研 TeleGym 平台，其核心作用是把大模型强化学习里的几个环节：任务执行、结果评估、轨迹构建与强化学习优化串成一条链路，并支持异步执行、有限超额采样与部分轨迹续跑——策略更新时在途的 Agent 任务可以继续跑完。简单说，它既是智能体跑任务的"训练场"，也是产出训练数据的"流水线"。

# 模型训练

**Xing4.1-Preview** 支持基于国产训练框架 MindSpore 及 MindSpore Transformers（MindFormers）套件进行模型训练，覆盖预训练、微调、评测、推理与部署，提供主流 Transformer 模型的实现与配置，支持一键启动已适配模型的单卡或多卡任务，并提供丰富的分布式并行能力，便于开发者开展大模型训练与研发，各模型的具体功能支持范围可参阅官方模型库。

有关环境配置、任务启动及分布式并行策略，或需进一步了解框架特性与源码实现，请参阅 [MindSpore Transformers（MindFormers）官方仓库](https://gitcode.com/mindspore/mindformers)。Xing4.1-Preview 国产化训练请参考 [MindSpore MindFormers — Xing4 训练指南](https://gitcode.com/mindspore/mindformers/blob/master/configs/xing4/README.md)。

# 评测

| 基准 | Xing4.1-Preview | DeepSeek-V4-Flash-0731 | GLM-5.3-Flash | DeepSeek-V4.1-Flash-1017 | GLM-5.3 |
| --- | --- | --- | --- | --- | --- |
| **推理与知识** |  |  |  |  |  |
| MMLU-Pro | 85.2 | 85.18 | 87.58 | 87.83 | 88 |
| GPQA Diamond | 89.34 | 90.7 | 88.8 | 93.91 | 91.88 |
| AIME 2026 | 92 | 98.5 | 88 | 100 | 89.3 |
| **编程** |  |  |  |  |  |
| SWE-Bench Pro V2 (Hard51) | 74.5 | 60.8 | 72.7 | 88.2 | 84.3 |
| Terminal-Bench 2.1 | 80.8 | 82.7 † | 84.3 † | 87.6 § | 88.2 † |
| Terminal-Bench 4.0 | 28.7 | 12.1 ‡ | 32.8 ‡ | 33.8 ‡ | 41.9 ‡ |
| Terminal-Bench-Science 0.1 | 12.9 | 4.3 § | 4.8 ‡ | 12.9 | 9.5 ‡ |
| SciCode | 53.95 | 50 | 52 | 59 | 56 |
| SWE-Atlas v1.0-QA | 51.2 | 48 | 56 | 61 | 58.5 |
| SWE-Atlas v1.0-RF | 41.4 | 47.14 | 40 | 44.3 | 45.7 |
| **智能体** |  |  |  |  |  |
| HLE-diamond (w/ tools, text-only) | 52.3 | 53.5 | 47.3 | 70 | 48.7 |
| AutomationBench v1.0.6 | 52.3 | 38.8 † | 48.8 † | 60.5 † | 53.33 † |
| Agent's Last Exam | 20.2 | 25.2 † | 26.3 † | 35.4 † | 28.5 † |
| WorkspaceBench (text-only) | 56.5 | 56.3 | 57.3 | 64 | 57.1 |
| Claw-Eval (text-only) | 80.58 | 79.49 | 79.24 | — | 80.3 |
| Tau3-Banking | 41.5 | 39.4 ‡ | 47.2 ‡ | — | 50.3 ‡ |
| **长文档推理** |  |  |  |  |  |
| GDP.pdf | 13.2 | 11 ‡ | 15.4 ‡ | 17.2 ‡ | 11.2 ‡ |
| AA.LCR | 79.67 | 79.7 ‡ | 80 ‡ | 82.7 ‡ | 79.7 ‡ |

注：

1. **数据来源：** 未标注为自测结果；† 引自该模型官方技术报告；‡ 引自 Artificial Analysis 排行榜；§ 引自第三方评测报告。
2. **Xing4.1-Preview 评测参数：** 除以下例外外统一使用 `temperature=0.8, top_p=0.95, repetition_penalty=1.05`：MMLU-Pro `temperature=0.6`；GPQA Diamond / AIME 2026 `temperature=1.0`；Agent's Last Exam `temperature=1.0, repetition_penalty=1.0`。对比模型使用各自推荐参数，reasoning effort 设为最大档。
3. **评测框架与设置：**
    - HLE-diamond (w/ tools, text-only)：OpenCode；
    - Terminal-Bench 2.1：Xing4.1-Preview 使用 terminus-2 框架，对比模型使用 Claude Code 框架；
    - Terminal-Bench 4.0：mini-swe-agent；
    - Terminal-Bench-Science 0.1：mini-swe-agent；
    - SWE-Atlas v1.0 系列（QA / RF）：mini-swe-agent；
    - Agent's Last Exam：OpenCode v1.19.5 / ALE v1.0 CLI 模式；
    - WorkspaceBench：Claude Code，Judge Model 使用 Claude-Opus-5.5，超时 3600s，纯文本模型注入 hook 限制仅文本输入；
    - SciCode：官方评测代码与配置；
    - Claw-Eval：官方评测代码与配置，未设置超时限制；
    - AutomationBench：v1.0.6，含 [PR #13](https://github.com/zapier/AutomationBench/pull/13) 修复。

# 快速开始

Xing4.1-Preview 已通过[天翼 AI 开放平台](https://www.teleai.com.cn/)提供 OpenAI 兼容 API。

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api-maas.teleai.com.cn/v1",
)

response = client.chat.completions.create(
    model="Xing4.1-Preview",
    messages=[
        {"role": "user", "content": "用单个 HTML 文件做一个团建费用估算页，勾选活动与人数后实时更新明细、总额与人均。"}
    ],
)

print(response.choices[0].message.content)
```

- **上下文窗口：** 原生支持 1M 上下文窗口；
- **思考模式：** 支持开/关（无分档）；
- **结构化输出与函数调用：** 通过 OpenAI 兼容接口支持。

## 服务化推理与生态

Xing4.1-Preview 与 **Xing4.0 架构不同**（采用 MLA + DSA 设计），但保持与 Xing 4 系列一致的上游接口兼容，因此 Xing4.0 已合入上游的微调与服务化生态可直接复用到 Xing4.1（**llama.cpp 与 KTransformers 仅适用于 Xing4.0 系列，不适用于 Xing4.1-Preview**）：

| 框架 | Xing4.1 支持 | 时间 / 版本 | 使用方式 |
| --- | --- | --- | --- |
| [ms-swift](https://github.com/modelscope/ms-swift) | 原生 —— **v4.6.0+** | 2026-10-01 | 使用已合入的 ms-swift |
| [SGLang](https://github.com/sgl-project/sglang) | 原生 —— **main 分支** | 2026-11-05 | 使用已合入的 SGLang main 分支 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 原生 —— **main 分支** | 2026-10-29 | 使用已合入的 LLaMA-Factory main 分支 |
| [vLLM](https://github.com/vllm-project/vllm) | 官方 out-of-tree 插件 | — | [XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin) |
| [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) | 昇腾 NPU | — | [vLLM-Ascend 部署文档](./tutorial/vLLM-Ascend/README.md) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 上游评审中的 PR | — | 见 GitHub README 中的 PR 分支 |

## 本地部署

> [!NOTE]
> Xing4.1-Preview 沿用 Xing 4 系列的服务化栈，部署时需把 `${MODEL_PATH}` 换成本地路径或 `XingChen-AGI/Xing4.1-Preview`。该模型为 285B 参数的 MoE 模型，BF16 精度下仅权重即需约 **8×80 GB** 显存（`--tensor-parallel-size 8`），上下文更长需相应增加资源。原生上下文为 1M，请按显存/节点把上下文参数（`--max-model-len` / `--context-length`）设到硬件可承载的值。parser 标识与预编译镜像 tag 以本仓库官方部署文档（[vLLM](./tutorial/vLLM/xing4.1-preview-vllm.md) / [SGLang](./tutorial/SGLang/xing4.1-preview-sglang.md) / [FlagOS](./tutorial/FlagOS/xing4.1-preview-flagos.md)）为准。

**vLLM**——官方 out-of-tree 插件（[XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin)）。详细部署步骤、参数说明与调用示例见 [vLLM 部署文档](./tutorial/vLLM/xing4.1-preview-vllm.md)。

**SGLang**——main 分支原生支持，无需额外插件；详细部署步骤、参数说明与调用示例见 [SGLang 部署文档](./tutorial/SGLang/xing4.1-preview-sglang.md)。

**FlagOS**——社区已完成 Xing4.1-Preview 的 Day0 跨芯适配，提供覆盖八个平台的 W8A8 版本，安装、启动与推理方法见 [FlagOS 多芯部署指南](./tutorial/FlagOS/xing4.1-preview-flagos.md)。

**vLLM-Ascend（昇腾 NPU）**——通过 vLLM-Ascend 在昇腾 NPU 上部署；详细部署步骤、参数说明与调用示例见 [vLLM-Ascend 部署文档](./tutorial/vLLM-Ascend/README.md)。

# 推荐参数

| 场景 | `temperature` | `top_p` | `repetition_penalty` |
| --- | --- | --- | --- |
| Agent / Code | 0.8 | 0.95 | 1.05 |
| Reasoning | 1.0 | 0.95 | 1.05 |
| General | 0.6 | 0.95 | 1.05 |

# 应用场景

- **智能体协同办公**：多步骤办公流程、跨工具编排、多源材料交叉核对的结构化报表生成；
- **深度研究**：持续性多源调研，输出结构化、可核对的交付物；
- **智能体编程**：完整前端生成（单文件、零依赖、可离线打开的 Web 应用）、3D 创意编程、数据分析与报表；
- **文档密集型业务**：架构引导的结构化信息抽取，服务智能助理、请求路由与文档抽取。

# 免责声明

与所有生成式语言模型一样，Xing4.1-Preview 可能产生不准确或意外的输出。输出由模型自动生成，不代表中国电信观点。请独立核实重要信息，勿将模型用于违反适用法律法规的用途。

请勿将 Xing4 及其衍生模型用于任何危害国家社会安全或违反法律法规的活动，也请勿将其用于未经安全审查与备案的互联网服务。

# 协议与引用

本仓库的权重与代码以 [Apache 2.0](./LICENSE) 协议开源。

## 相关论文

- **Xing4 技术报告：** [TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks](https://arxiv.org/abs/2611.22115)

## 引用

```bibtex
@misc{lab2026telechatxing4advancingfrontier,
      title={TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks}, 
      author={Xingchen AGI Lab and Baotian Hu and Bingkai Yang and Boyang Wang and Changzai Pan and Chao Lyu and Chao Wang and Chen Li and Chenshuo Pan and Chenyu Lin and Chenyuan Zhang and Chongming Liu and Dawei Fan and Di Wu and Dong Guo and Dong Li and Dongfang Li and Enlin Liu and Fanye Xiao and Fengkai Li and Fubei Yao and Gang Lin and Han Gao and Hanming Wu and Haowei He and Huapeng Liu and Hua Wang and Huinan Xu and Jiahong Xiong and Jiaxin Peng and Jianxiang Ma and Jianxing Zhuge and Jie Zhang and Ji Xiong and Jinglin Chen and Jingqi Li and Jingxuan Li and Jiwang Ye and Junchen Li and Junbiao Li and Junyang Wu and Kaidong Yu and Kaipeng Jia and Kehai Chen and Kuai Li and Laobin Liu and Lei Li and Libo Qin and Lingling Shi and Luwen Pu and Menghao Xu and Mengjiao Bao and Mengqi Chen and Min Zhang and Mingrui Wang and Naying Wang and Peicong Han and Peiming Guo and Pengfei Li and Qiang Feng and Qi Wang and Qingjun Zhu and Qingli Li and Qitai Ding and Qiyi Xie and Quanchao Zhao and Rongbao Han and Rui Xue and Ruiyu Fang and Ruiting Song and Shijie Li and Shiquan Wang and Shuangyong Song and Sishi Xiong and Teng Su and Wankang Bao and Weijie Zhang and Wei Mao and Weiwei Fu and Weiyang Huang and Wenmin Deng and Wenqing Hu and Wujie Zhong and Xi Yu and Xiaohui Hu and Xiaomin Fang and Xiaomeng Huang and Xiaoqiong Shi and Xiaotong Shang and Xiaoyu Li and Xinglei Xu and Xin Jiang and Xin Wang and Xin Zhang and Xinzhang Liu and Xue Liu and Xuefeng Bai and Xuewei He and Yahan Li and Yanhan Zhang and Yanlei Xue and Yanzhang Wu and Yang Yu and Yi Liu and Yin Zhang and Yingyan Li and Yitong Yao and Yixuan Li and Yongqiang Xia and Yongxiang Li and Yinzhou Zhu and Yu Zhao and Yuhao Jiang and Yunpeng Zhang and Yuting Lin and Yuxuan Wu and Yuyao Huang and Yuyi Wang and Zebo Peng and Zhenwu Pei and Zhenxiong Li and Zhao Wang and Zhaoxi Zhang and Zhihao Yang and Zhiqiang Dong and Zhiwei Yan and Zhilong Lu and Zhonghao Guo and Zhongheng Wu and Zhongjiang He and Zhongqiu Li and Zhu Yuan and Zhuoru Jiang and Zhuoyi Wei and Zijia Guo and Zilu Huang and Zhipeng Guo and Zirui Li and Zihan Wang},
      year={2026},
      eprint={2611.22115},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2611.22115}, 
}
```

---

<p align="center">
  <b>中电信人工智能科技出品</b><br>
  <a href="https://github.com/XingChen-AGI">XingChen-AGI 开源组织</a>
</p>
