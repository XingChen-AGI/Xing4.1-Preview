# 星辰语义大模型 · Xing4.1-Preview

🦉 [GitHub](https://github.com/XingChen-AGI/Xing4.1-Preview) · 🤗 [Hugging Face](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) · 🤖 [ModelScope](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) · 🪐 [Modelers](https://modelers.cn/models/XingChen-AGI/Xing4.1-Preview) · 📖 [English](./README_EN.md)

# 目录

- [介绍](#介绍)
- [模型](#模型)
- [能力评估](#能力评估)
- [快速开始](#快速开始)
  - [本地推理](#本地推理)
  - [服务化推理](#服务化推理)
    - [vLLM](#vllm)
    - [SGLang](#sglang)
    - [KTransformers](#ktransformers)
    - [FlagOS](#flagos)
  - [超参选择](#超参选择)
    - [Xing4.1-Preview 模型推理参数选择](#xing41-preview-模型推理参数选择)
- [微调](#微调)
- [声明与引用](#声明与引用)
  - [声明](#声明)
  - [引用](#引用)

# 介绍

**Xing 4** 由中电信人工智能科技有限公司研发，是星辰语义大模型系列（原 TeleChat）的新一代模型，**完全基于国产昇腾 NPU 平台训练**，面向长上下文推理、代码生成和智能体应用场景。

**Xing4.1-Preview** 是 Xing 4 系列模型，采用稀疏混合专家（MoE）架构，结合 Multi-head Latent Attention（MLA）与 DeepSeek Sparse Attention（DSA），面向长上下文交互、复杂推理和 Agent 工作流。

- **面向 Agent 的架构：** 采用 45 层 MoE 主干网络，每 Token 激活 8 个路由专家。
- **MLA + DSA：** 结合压缩 Key-Value 表示与长上下文稀疏注意力机制。
- **长上下文：** 支持  **1M** 的配置上下文长度。

# 模型

| 配置项                                      | Xing4.1-Preview                     |
| ---------------------------------------- | ----------------------------------: |
| 总参数量                                     | **285B**                            |
| 每 Token 激活参数量                            | 约 17B                               |
| Transformer 层数                           | **45**                              |
| Hidden Size                              | 4096                                |
| Attention Heads                          | 128                                 |
| Dense FFN Intermediate Size              | **18432**                           |
| Expert Intermediate Size                 | 2048                                |
| Routed Experts                           | 256                                 |
| Active Routed Experts / Token            | 8                                   |
| Shared Experts                           | 1                                   |
| Dense / MoE FFN 层数                       | 3 / 42                              |
| Attention                                | MLA + DSA                           |
| qk Head Dimension                        | 192（NoPE 128 + RoPE 64）             |
| v Head Dimension                         | 128                                 |
| MLA q LoRA Rank / kv LoRA Rank           | 1536 / 512                          |
| DSA Index Heads / Head Dimension / Top-K | 64 / 128 / 2048                     |
| Normalization / Activation               | RMSNorm（eps 1e-5）/ SiLU             |
| Tokenizer / Vocabulary Size              | BPE / 131072                        |
| 最大上下文长度                                  | 1M                                  |
| RoPE / YaRN                              | θ=100000 / factor=32（原始上下文长度 32768） |

模型采用 **45 层 Transformer** 主干网络，前 **3** 层使用 Dense FFN，后 **42** 层使用 MoE FFN。注意力架构结合 MLA 与 DSA。

# 能力评估

Xing4.1-Preview 在智能体与推理类基准上与同量级 Flash 模型对比：

| 基准 | Xing4.1-Preview | Hy4-flash | GLM-5.3-Flash | Deepseek-v4-flash-0731 |
|---|---:|---:|---:|---:|
| Terminal-Bench 4.0 | **20.6**† | 19.2 | 35.4 | 7.0 |
| AutomationBench v1.0.6 | **45.3** | 52.2 | 48.4 | 37.7 |
| Apex-Agents v1.1 | **50.7** | 63.1 | 51.5 | 55.3 |
| HLE w/ tools | **45.7**‡ | 55.2 | 52.2 | 51.5 |
| Claw-Eval (pass^3) | **49.7** | — | — | 49.3 |

注：

1. 除 Xing4.1-Preview 外，其余模型各指标均取自其公开报告；
2. † Terminal-Bench 4.0：Xing4.1-Preview 跳过了所有 GPU 沙箱类题目；
3. ‡ HLE：Xing4.1-Preview 仅测试了 text-only 数据。

# 快速开始

### 本地推理

　　Xing4.1-Preview 已通过 [天翼 AI 开放平台](https://www.teleai.com.cn/) 提供 OpenAI 兼容 API，可快速体验：

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

- 公开 API 提供完整 **1M** 上下文（单次请求最大输入 **896K** / 最大输出 **128K**）；
- 支持思考模式开 / 关，以及结构化输出与函数调用。

　　如需私有化部署，请参考下方「服务化推理」中的 vLLM / SGLang 部署文档。

### 服务化推理

> [!NOTE]
> Xing4.1-Preview 与 Xing4.0 架构不同，但保持上游接口兼容。vLLM 通过官方插件提供 `xing4_0` 推理 / 工具调用解析，SGLang 在 main 分支原生支持；昇腾 NPU 与多芯 W8A8 部署请参考对应文档。

#### vLLM

[vLLM](https://github.com/vllm-project/vllm) 是高吞吐、低延迟的 LLM 推理与服务引擎，可一键启动 OpenAI 兼容的 API 服务。Xing4.1-Preview 的模型本体（`deepseek_v32`）由 vLLM 原生加载，`xing4_0` 推理与工具调用解析通过官方插件 [xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin) 提供。

详细部署步骤、参数说明与调用示例请参考 [Xing4.1-Preview vLLM 部署文档](./tutorial/vLLM/xing4.1-preview-vllm.md)。

#### SGLang

[SGLang](https://github.com/sgl-project/sglang) 是面向大语言模型的高性能服务框架，main 分支已原生支持 Xing4.1-Preview，无需额外安装插件。

详细部署步骤、参数说明与调用示例请参考 [Xing4.1-Preview SGLang 部署文档](./tutorial/SGLang/xing4.1-preview-sglang.md)。

#### KTransformers

> [!NOTE]
> KTransformers（以及 llama.cpp）目前仅适用于 Xing4.0 系列，**不适用于 Xing4.1-Preview**。

#### FlagOS

[FlagOS](https://flagos.io/) 社区完成了 Xing4.1-Preview 的 Day0 跨芯适配，提供覆盖八个平台的 W8A8 版本。部署方法详见 [FlagOS 多芯部署指南](./tutorial/FlagOS/xing4.1-preview-flagos.md)。

#### vLLM-Ascend（昇腾）

昇腾 NPU 部署文档即将补充。

### 超参选择

#### Xing4.1-Preview 模型推理参数选择

| 参数 | 数值 |
|---|---:|
| `temperature` | 0.8 |
| `top_p` | 1.0 |
| `context_window` | 1M tokens |
| `max_tokens` | 128K |

## 微调

Xing4.1-Preview 保持与 Xing 4 系列一致的上游接口兼容，Xing4.0 已合入上游的微调生态可直接复用：

- [ms-swift](https://github.com/modelscope/ms-swift)：原生支持（v4.6.0+，2026-10-01 起），可直接用于微调；
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：原生支持（main 分支，2026-10-29 起），可直接用于微调；
- [FlagOS](https://flagos.io/)：基于 FlagOS 的多芯训练与部署适配，详见 [FlagOS 多芯部署指南](./tutorial/FlagOS/xing4.1-preview-flagos.md)。

# 声明与引用

### 声明

请勿将 Xing4 及其衍生模型用于任何危害国家社会安全或违反法律法规的活动，也请勿将其用于未经安全审查与备案的互联网服务。

我们已尽力确保训练数据的合规性，但受限于模型与数据的复杂性，仍可能存在无法预见的问题。对于因使用本开源模型所引发的任何问题——包括但不限于数据安全、舆论风险，以及模型被误导、滥用、传播或不当利用所带来的风险——我们不承担相关责任。

### 引用

如需引用我们的工作，请使用如下参考：

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
