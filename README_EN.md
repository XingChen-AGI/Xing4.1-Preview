# Xing4.1-Preview

🦉 [GitHub](https://github.com/XingChen-AGI/Xing4.1-Preview) · 🤗 [Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) · 🤖 [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) · 🪐 [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.1-Preview) · 📖 [中文](./README_ZH.md)

# Table of Contents

- [Introduction](#introduction)
- [Model](#model)
- [Benchmarks](#benchmarks)
- [Quick Start](#quick-start)
  - [Local Inference](#local-inference)
  - [Serving Inference](#serving-inference)
    - [vLLM](#vllm)
    - [SGLang](#sglang)
    - [KTransformers](#ktransformers)
    - [FlagOS](#flagos)
    - [vLLM-Ascend](#vllm-ascend)
  - [Hyperparameter Selection](#hyperparameter-selection)
    - [Inference Parameter Recommendations for Xing4.1-Preview](#inference-parameter-recommendations-for-xing41-preview)
- [Fine-tuning](#fine-tuning)
- [Disclaimer & Citation](#disclaimer--citation)
  - [Disclaimer](#disclaimer)
  - [Citation](#citation)

# Introduction

**Xing 4** is developed by China Telecom Artificial Intelligence Technology Co., Ltd. It is the latest generation of the Xingchen semantic large model family (formerly TeleChat), trained entirely on the domestic Ascend NPU platform, and designed for long-context reasoning, code generation, and agentic applications.

**Xing4.1-Preview** is a model in the Xing 4 family. It uses a sparse Mixture-of-Experts (MoE) architecture that combines Multi-head Latent Attention (MLA) and DeepSeek Sparse Attention (DSA) for long-context interaction, complex reasoning, and agent workflows.

- **Agent-oriented architecture:** A 45-layer MoE backbone activates 8 routed experts per token.
- **MLA + DSA:** Combines compressed key-value representations with sparse attention over long contexts.
- **Long context:** Supports a configured context length of **1M**.

# Model

| Configuration | Xing4.1-Preview |
|---|---:|
| Total parameters | **285B** |
| Activated parameters per token | Approximately 17B |
| Transformer layers | **45** |
| Hidden size | 4096 |
| Attention heads | 128 |
| Dense FFN intermediate size | **18432** |
| Expert intermediate size | 2048 |
| Routed experts | 256 |
| Active routed experts per token | 8 |
| Shared experts | 1 |
| Dense / MoE FFN layers | 3 / 42 |
| Attention | MLA + DSA |
| qk head dimension | 192 (NoPE 128 + RoPE 64) |
| v head dimension | 128 |
| MLA q LoRA rank / kv LoRA rank | 1536 / 512 |
| DSA index heads / head dimension / top-k | 64 / 128 / 2048 |
| Normalization / activation | RMSNorm (eps 1e-5) / SiLU |
| Tokenizer / vocabulary size | BPE / 131072 |
| Maximum context length | **1M** |
| RoPE / YaRN | θ=100000 / factor=32 (original context length 32768) |

The model uses a **45-layer Transformer** backbone. The first **3** layers use Dense FFNs, followed by **42** MoE FFN layers. The attention architecture combines MLA and DSA.

# Benchmarks

Xing4.1-Preview is compared with Flash-class models of the same scale on agent and reasoning benchmarks:

| Benchmark | Xing4.1-Preview | Hy4-flash | GLM-5.3-Flash | Deepseek-v4-flash-0731 |
|---|---:|---:|---:|---:|
| Terminal-Bench 4.0 | **20.6**† | 19.2 | 35.4 | 7.0 |
| AutomationBench v1.0.6 | **45.3** | 52.2 | 48.4 | 37.7 |
| Apex-Agents v1.1 | **50.7** | 63.1 | 51.5 | 55.3 |
| HLE w/ tools | **45.7**‡ | 55.2 | 52.2 | 51.5 |
| Claw-Eval (pass^3) | **49.7** | — | — | 49.3 |

Notes:

1. Except for Xing4.1-Preview, the metrics of all other models are taken from their public reports.
2. † Terminal-Bench 4.0: Xing4.1-Preview skipped all GPU-sandbox tasks.
3. ‡ HLE: Xing4.1-Preview was evaluated on text-only data.

# Quick Start

### Local Inference

Xing4.1-Preview is available through the [TeleAI Open Platform](https://www.teleai.com.cn/) via an OpenAI-compatible API for a quick try:

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

- The public API provides the full **1M** context (up to **896K** input / **128K** output per request).
- Thinking mode on/off, structured output, and function calling are supported.

For private deployment, refer to the vLLM / SGLang deployment documents under "Serving Inference" below.

### Serving Inference

> [!NOTE]
> Xing4.1-Preview uses a different architecture from Xing4.0 but maintains upstream interface compatibility. vLLM provides `xing4_0` reasoning / tool-call parsing through an official plugin; SGLang supports it natively on the main branch. For Ascend NPU and multi-chip W8A8 deployment, refer to the corresponding documents.

#### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput, low-latency LLM inference and serving engine that can launch an OpenAI-compatible API service with a single command. The model body of Xing4.1-Preview (`deepseek_v32`) is loaded natively by vLLM, while the `xing4_0` reasoning and tool-call parsing is provided by the official plugin [xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin).

For detailed deployment steps, parameter descriptions, and call examples, refer to the [Xing4.1-Preview vLLM deployment document](./tutorial/vLLM/xing4.1-preview-vllm_en.md).

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a high-performance serving framework for large language models. Its main branch natively supports Xing4.1-Preview, so no additional plugin installation is required.

For detailed deployment steps, parameter descriptions, and call examples, refer to the [Xing4.1-Preview SGLang deployment document](./tutorial/SGLang/xing4.1-preview-sglang_en.md).

#### KTransformers

> [!NOTE]
> KTransformers (and llama.cpp) currently only apply to the Xing4.0 family and **do not support Xing4.1-Preview**.

#### FlagOS

The [FlagOS](https://flagos.io/) community completed Day0 cross-chip adaptation for Xing4.1-Preview, providing a W8A8 version covering eight platforms. For deployment details, refer to the [FlagOS multi-chip deployment guide](./tutorial/FlagOS/xing4.1-preview-flagos.md).

#### vLLM-Ascend

Ascend NPU deployment documentation will be added soon.

### Hyperparameter Selection

#### Inference Parameter Recommendations for Xing4.1-Preview

| Parameter | Value |
|---|---:|
| `temperature` | 0.8 |
| `top_p` | 1.0 |
| `context_window` | 1M tokens |
| `max_tokens` | 128K |

## Fine-tuning

Xing4.1-Preview maintains the same upstream interface compatibility as the Xing 4 family, so the fine-tuning ecosystem already merged upstream for Xing4.0 can be reused directly:

- [ms-swift](https://github.com/modelscope/ms-swift): natively supported (v4.6.0+, since 2026-10-01), can be used directly for fine-tuning.
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): natively supported (main branch, since 2026-10-29), can be used directly for fine-tuning.
- [FlagOS](https://flagos.io/): multi-chip training and deployment adaptation based on FlagOS; see the [FlagOS multi-chip deployment guide](./tutorial/FlagOS/xing4.1-preview-flagos.md) for details.

# Disclaimer & Citation

### Disclaimer

Do not use Xing4 or its derivative models for any activities that endanger national or social security, or violate applicable laws and regulations. Do not use it for internet services that have not undergone security review and filing.

We have made every effort to ensure the compliance of the training data, but due to the complexity of models and data, unforeseen issues may still arise. We assume no liability for any issues arising from the use of this open-source model — including but not limited to data security risks, public opinion risks, and risks arising from the model being misled, misused, disseminated, or improperly utilized.

### Citation

If you find our work useful, please cite:

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
