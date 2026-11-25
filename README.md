<div align="center">
<h1>
  Xing4.1-Preview
</h1>
</div>

<p align="center"><strong>—— Advancing Frontier Agentic Models On Domestic Chips And Frameworks</strong></p>

<p align="center">
  <b>8,192 Ascend 910C chips + MindSpore training framework<br>
  285B total / 17B active MoE model, built for agentic coding and long-horizon complex agent tasks, with native 1M context.</b>
</p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview" target="_blank">GitHub</a> · <a href="https://huggingface.co/XingChen-AGI/Xing4.1-Preview" target="_blank">HF (Transformers)</a> · <a href="https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore" target="_blank">HF (MindSpore)</a> · <a href="https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview" target="_blank">ModelScope</a> · <a href="https://modelers.cn/user/XingChen-AGI" target="_blank">Modelers</a> · <a href="https://atomgit.com/XingChen-AGI" target="_blank">AtomGit</a> · <a href="https://www.teleai.com.cn/" target="_blank">TeleAI Open Platform</a> · <a href="https://huggingface.co/papers/2611.22115" target="_blank">Technical Report</a>
</p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Parameters" src="https://img.shields.io/badge/Params-285B%20%2F%2017B%20active-7C3AED"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Context" src="https://img.shields.io/badge/Context-1M%20native-orange"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
</p>

<p align="right">English | <a href="./README_ZH.md">中文</a></p>

# Table of Contents

- [Introduction](#introduction)
- [Latest Updates](#latest-updates)
- [Hardware-Software Co-design for Ascend 910C](#hardware-software-co-design-for-ascend-910c)
- [Compute and Training Framework](#compute-and-training-framework)
- [Model Training](#model-training)
- [Evaluation Results](#evaluation-results)
- [Quick Start](#quick-start)
- [Recommended Parameters](#recommended-parameters)
- [Use Cases](#use-cases)
- [Disclaimer](#disclaimer)
- [License & Citation](#license--citation)

# Introduction

Xing4.1-Preview is developed by China Telecom Artificial Intelligence Technology (Beijing) Co., Ltd. It is the latest model in the Xing series (formerly TeleChat), adopting a Sparse Mixture-of-Experts (MoE) architecture that combines Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA), targeting long-context interaction, complex reasoning, and agent workflows:

- **Compute and training framework:** 8,192 Ascend 910C chips + MindSpore training framework, jointly supporting model training.
- **Hardware-aware architecture co-design:** The architecture is co-designed to leverage the strengths and address the constraints of Ascend 910C AI accelerators, rather than simply porting an existing model onto the hardware.
- **Strong agentic and coding capabilities:** Demonstrates competitive performance on agentic workloads, including data analysis, frontend generation, and long-document comprehension. The model has also been evaluated with multiple mainstream agent harnesses, including Codex, OpenCode, Pi, Claude Code, and WorkBuddy.
- **MLA + DSA attention with native 1M context:** Combines compressed Key-Value representations with long-context sparse attention, natively supporting a **1M** context length.

# Latest Updates

- 2026-11-23 Open-sourced **Xing4.1-Preview** [[HuggingFace](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) | [HuggingFace (MindSpore)](https://huggingface.co/XingChen-AGI/Xing4.1-Preview-MindSpore) | [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) | [ModelScope (MindSpore)](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview-MindSpore/)| [TeleAI Open Platform](https://www.teleai.com.cn/)]

# Hardware-Software Co-design for Ascend 910C

**Leveraging the Ascend 910C chip's high matrix (Cube) compute ratio and constrained vector compute, together with the super-pod cluster's unified memory addressing and high inter-card communication bandwidth, Xing4.1-Preview uses the MindSpore training framework as its foundation. Targeted designs have been applied across model architecture, parallelism strategy, and token vocabulary to jointly improve model capability, training efficiency, and cluster utilization. The model also adopts MLA and DSA from DeepSeek V3.2 to improve inference efficiency, and is compatible with open-source ecosystems such as SGLang and vLLM.**

## Sparse Attention Architecture Selection: Reducing the Vector Compute Bottleneck

Compared with hybrid linear attention architectures such as KDA and GDN, **DSA relies primarily on matrix-multiplication operations, which are better aligned with the Cube compute units**. While hybrid linear attention offers theoretical complexity advantages for long sequences, its gating, state update, and scan operations place high demands on Vector performance and operator fusion capabilities, making it difficult to fully realize theoretical gains under the current 910C software stack. We therefore prioritize DSA, which reduces attention computation complexity while better leveraging the hardware's matrix compute strengths.

## Wide-and-Shallow Architecture: Improving Cube Utilization

Under the constraints of model scale and parallelism strategy, we adopt a design that **moderately increases network width while reducing depth** (45 layers / Hidden Size 4096 / Dense FFN Intermediate 18432). This increases the computation scale, arithmetic intensity, and data reuse efficiency of core matrix multiplication operators, thereby improving Cube unit utilization. At the same time, reducing the number of layers lowers per-layer sequential overheads such as normalization, activation functions, operator scheduling, and data movement, mitigating the impact of Vector performance bottlenecks on overall training throughput.

## Parallelism Strategy Optimization: Improving Cluster Utilization

We adopt a hybrid sharding strategy — "shard within super-pods, replicate across super-pods" — to reduce per-device memory footprint, replacing pipeline parallelism to eliminate bubble overhead. Communication is performed directly on parameter tensors, avoiding redundant copy-in/copy-out through intermediate buffers. Intra-super-pod AllGather and ReduceScatter for parameters and gradients, together with inter-super-pod fused gradient AllReduce, are overlapped asynchronously with backward computation, enabling overlapping of communication across three levels.

## Optimized Token Vocabulary: Improving Token Compression Ratio

The vocabulary was customized for agent and coding scenarios. Subword statistics, frequency counts, and pointwise mutual information were computed over large-scale corpora from target domains — including agent trajectories, tool calls, and code repositories — and high-frequency contiguous spans were preferentially merged. With the vocabulary size held constant, the customized vocabulary reduces token counts by 16% for agent scenarios and 23% for coding scenarios, substantially boosting problem-solving capability in these domains.

# Compute and Training Framework

Xing4.1-Preview training uses 8,192 Ascend 910C chips and the MindSpore training framework. Key contributions include:

- Owner-based Muon optimizer: Keeps parameter and momentum sharding layout unchanged. The Newton–Schulz orthogonalization of each 2D matrix is performed once by a designated Owner device within the sharding group and then broadcast (batched by matrix shape), avoiding redundant full-matrix reconstruction and duplicate computation across devices.

- Fine-grained selective recomputation: Operators are precisely selected for retention based on execution time and output memory footprint, avoiding coarse whole-layer recomputation and per-operator framework callback overhead. Boundary input placeholder release and recomputation recovery mechanisms further compress activation memory.

- Long-sequence sparse attention optimization: Supports layer-wise early backward propagation of Indexers to release auxiliary computation graphs as soon as possible, combined with dynamic sequence block pairing and document-level KV window pruning, reducing computation imbalance and memory access redundancy under ultra-long contexts.

- Host-side scheduling: Training threads are assigned CPU affinity by device pool or thread role subset (while preserving intra-pool load balancing), and selected system services and migratable interrupts are isolated to a dedicated system core pool, reducing large-scale cluster synchronization stalls.

Post-training uses the in-house TeleGym platform. TeleGym provides an integrated infrastructure for agentic post-training, covering task execution, evaluation, trajectory construction, and policy optimization. Its asynchronous execution, bounded oversampling, and partial rollout mechanisms enable efficient utilization of agent execution resources while maintaining trajectory continuity across policy updates.

# Model Training

**Xing4.1-Preview** supports model training via the MindSpore framework and the MindSpore Transformers (MindFormers) toolkit, covering pre-training, fine-tuning, evaluation, inference, and deployment. It provides implementations and configurations for supported Transformer models, with streamlined single-device and distributed training workflows. For the specific feature support scope of each model, please refer to the official model library.

For environment setup, task launching, and distributed parallelism strategies, or to learn more about framework features and source code, please refer to the [MindSpore Transformers (MindFormers) official repository](https://gitcode.com/mindspore/mindformers). For Xing4.1-Preview training, see the [MindSpore MindFormers — Xing4 Training Guide](https://gitcode.com/mindspore/mindformers/blob/master/configs/xing4/README.md).

# Evaluation Results

| Benchmark | Xing4.1-Preview | DeepSeek-V4-Flash-0731 | GLM-5.3-Flash | DeepSeek-V4.1-Flash-1017 | GLM-5.3 |
| --- | --- | --- | --- | --- | --- |
| **Reasoning & Knowledge** |  |  |  |  |  |
| MMLU-Pro | 85.2 | 85.18 | 87.58 | 87.83 | 88 |
| GPQA Diamond | 89.34 | 90.7 | 88.8 | 93.91 | 91.88 |
| AIME 2026 | 92 | 98.5 | 88 | 100 | 89.3 |
| **Coding** |  |  |  |  |  |
| SWE-Bench Pro V2 (Hard51) | 74.5 | 60.8 | 72.7 | 88.2 | 84.3 |
| Terminal-Bench 2.1 | 80.8 | 82.7 † | 84.3 † | 87.6 § | 88.2 † |
| Terminal-Bench 4.0 | 28.7 | 12.1 ‡ | 32.8 ‡ | 33.8 ‡ | 41.9 ‡ |
| Terminal-Bench-Science 0.1 | 12.9 | 4.3 § | 4.8 ‡ | 12.9 | 9.5 ‡ |
| SciCode | 53.95 | 50 | 52 | 59 | 56 |
| SWE-Atlas v1.0-QA | 51.2 | 48 | 56 | 61 | 58.5 |
| SWE-Atlas v1.0-RF | 41.4 | 47.14 | 40 | 44.3 | 45.7 |
| **Agent** |  |  |  |  |  |
| HLE-diamond (w/ tools, text-only) | 52.3 | 53.5 | 47.3 | 70 | 48.7 |
| AutomationBench v1.0.6 | 52.3 | 38.8 † | 48.8 † | 60.5 † | 53.33 † |
| Agent's Last Exam | 20.2 | 25.2 † | 26.3 † | 35.4 † | 28.5 † |
| WorkspaceBench (text-only) | 56.5 | 56.3 | 57.3 | 64 | 57.1 |
| Claw-Eval (text-only) | 80.58 | 79.49 | 79.24 | — | 80.3 |
| Tau3-Banking | 41.5 | 39.4 ‡ | 47.2 ‡ | — | 50.3 ‡ |
| **Long-Document Reasoning** |  |  |  |  |  |
| GDP.pdf | 13.2 | 11 ‡ | 15.4 ‡ | 17.2 ‡ | 11.2 ‡ |
| AA.LCR | 79.67 | 79.7 ‡ | 80 ‡ | 82.7 ‡ | 79.7 ‡ |

Notes:

1. **Data sources:** Unmarked scores are self-evaluated; † from the model's official technical report; ‡ from the Artificial Analysis leaderboard; § from third-party evaluation reports.
2. **Xing4.1-Preview evaluation parameters:** Default `temperature=0.8, top_p=0.95, repetition_penalty=1.05`, with the following exceptions: MMLU-Pro `temperature=0.6`; GPQA Diamond / AIME 2026 `temperature=1.0`; Agent's Last Exam `temperature=1.0, repetition_penalty=1.0`. Comparison models use their respective recommended parameters with reasoning effort set to maximum.
3. **Evaluation frameworks and settings:**
    - HLE-diamond (w/ tools, text-only): OpenCode;
    - Terminal-Bench 2.1: Xing4.1-Preview uses the terminus-2 harness; comparison models use the Claude Code harness;
    - Terminal-Bench 4.0: mini-swe-agent;
    - Terminal-Bench-Science 0.1: mini-swe-agent;
    - SWE-Atlas v1.0 series (QA / RF): mini-swe-agent;
    - Agent's Last Exam: OpenCode v1.19.5 / ALE v1.0 CLI mode;
    - WorkspaceBench: Claude Code, Judge Model uses Claude-Opus-5.5, timeout 3600s, text-only models use an injected hook restricting input to text only;
    - SciCode: official evaluation code and configuration;
    - Claw-Eval: official evaluation code and configuration, no timeout limit;
    - AutomationBench: v1.0.6, including the [PR #13](https://github.com/zapier/AutomationBench/pull/13) fix.

# Quick Start

Xing4.1-Preview is available via an OpenAI-compatible API through the [TeleAI Open Platform](https://www.teleai.com.cn/).

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api-maas.teleai.com.cn/v1",
)

response = client.chat.completions.create(
    model="Xing4.1-Preview",
    messages=[
        {"role": "user", "content": "Build a single-file HTML page for estimating team-building costs: users check activities and enter headcount, and the page updates the breakdown, total, and per-person cost in real time."}
    ],
)

print(response.choices[0].message.content)
```

- **Context window:** Supports a native 1M-token context window;
- **Thinking mode:** Supports on/off (no tiered levels);
- **Structured output and function calling:** Supported via the OpenAI-compatible API.

## Serving and Ecosystem

Xing4.1-Preview has a **different architecture from Xing4.0** (using MLA + DSA), but maintains upstream API compatibility with the Xing 4 series. Fine-tuning and serving ecosystems already integrated upstream for Xing4.0 can be directly reused for Xing4.1 (**llama.cpp and KTransformers only apply to the Xing4.0 series, not to Xing4.1-Preview**):

| Framework | Xing4.1 Support | Date / Version | Usage |
| --- | --- | --- | --- |
| [ms-swift](https://github.com/modelscope/ms-swift) | Native — **v4.6.0+** | 2026-10-01 | Use the upstream ms-swift |
| [SGLang](https://github.com/sgl-project/sglang) | Native — **main branch** | 2026-11-05 | Use the upstream SGLang main branch |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Native — **main branch** | 2026-10-29 | Use the upstream LLaMA-Factory main branch |
| [vLLM](https://github.com/vllm-project/vllm) | Official out-of-tree plugin | — | [XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Upstream PR under review | — | See the PR branch in the GitHub README |

## Local Deployment

> [!NOTE]
> Xing4.1-Preview reuses the Xing 4 series serving stack. During deployment, replace `${MODEL_PATH}` with a local path or `XingChen-AGI/Xing4.1-Preview`. The model is a 285B-parameter MoE model; under BF16 precision, weights alone require approximately **8×80 GB** GPU memory (`--tensor-parallel-size 8`), with longer contexts requiring proportionally more resources. The native context length is 1M — set the context parameters (`--max-model-len` / `--context-length`) to whatever your GPU memory/nodes can support. For parser identifiers and pre-built image tags, refer to the official deployment documentation in this repository ([vLLM](./tutorial/vLLM/xing4.1-preview-vllm.md) / [SGLang](./tutorial/SGLang/xing4.1-preview-sglang.md) / [FlagOS](./tutorial/FlagOS/xing4.1-preview-flagos.md)).

**vLLM** — official out-of-tree plugin ([XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin)). For detailed deployment steps, parameter descriptions, and call examples, see the [vLLM deployment guide](./tutorial/vLLM/xing4.1-preview-vllm.md).

**SGLang** — natively supported on the main branch, no extra plugin needed; for detailed deployment steps, parameter descriptions, and call examples, see the [SGLang deployment guide](./tutorial/SGLang/xing4.1-preview-sglang.md).

**FlagOS** — the community completed Day0 cross-chip adaptation for Xing4.1-Preview, providing W8A8 builds covering eight platforms; for installation, startup, and inference, see the [FlagOS multi-chip deployment guide](./tutorial/FlagOS/xing4.1-preview-flagos.md).

**vLLM-Ascend (Ascend NPU)** — deployment documentation coming soon.

# Recommended Parameters

| Scenario | `temperature` | `top_p` | `repetition_penalty` |
| --- | --- | --- | --- |
| Agent / Code | 0.8 | 0.95 | 1.05 |
| Reasoning | 1.0 | 0.95 | 1.05 |
| General | 0.6 | 0.95 | 1.05 |

# Use Cases

- **Agentic collaborative office work:** Multi-step office workflows, cross-tool orchestration, and structured report generation with multi-source material cross-verification;
- **Deep research:** Sustained multi-source investigation, producing structured and verifiable deliverables;
- **Agentic coding:** Complete frontend generation (single-file, zero-dependency, offline-ready web apps), 3D creative coding, data analysis and reporting;
- **Document-heavy back-office:** Schema-guided structured information extraction, serving intelligent assistants, request routing, and document extraction.

# Disclaimer

Like all generative language models, Xing4.1-Preview may produce inaccurate or unexpected outputs. Outputs are generated automatically by the model and do not represent the views of China Telecom. Please independently verify important information and do not use the model for purposes that violate applicable laws and regulations.

Do not use Xing4 or its derivative models for any activity that endangers national security or public safety or violates laws and regulations, nor for internet services that have not undergone the required safety review and filing.

# License & Citation

The weights and code in this repository are released under the [Apache 2.0](./LICENSE) license.

## Related Papers

- **Xing4 Technical Report:** [TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks](https://arxiv.org/abs/2611.22115)

## Citation

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
  <b>Developed by China Telecom AI Technology</b><br>
  <a href="https://github.com/XingChen-AGI">XingChen-AGI Open Source Organization</a>
</p>
