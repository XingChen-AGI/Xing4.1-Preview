<div align="center">
<h1>
  Xing4.1-Preview
</h1>
</div>

<p align="center"><b>全栈国产化训练——8192 张昇腾（Ascend）910C 芯片 + 国产框架 MindSpore + 自研训练系统，端到端完成预训练。<br>主力级能力、轻量级成本——总参数 285B、单次激活 17B 的 MoE 模型，专为智能体编程（Agentic Coding）与长程智能体任务打造，原生 1M 上下文。</b></p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview" target="_blank">GitHub</a> · <a href="https://huggingface.co/XingChen-AGI/Xing4.1-Preview" target="_blank">Hugging Face</a> · <a href="https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview" target="_blank">魔搭 ModelScope</a> · <a href="https://modelers.cn/user/XingChen-AGI" target="_blank">魔乐 Modelers</a> · <a href="https://atomgit.com/XingChen-AGI" target="_blank">AtomGit</a> · <a href="https://www.teleai.com.cn/" target="_blank">天翼 AI 开放平台</a> · <a href="./README.md">English</a>
</p>

<p align="center">
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Parameters" src="https://img.shields.io/badge/Params-285B%20%2F%2017B%20active-7C3AED"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview"><img alt="Context" src="https://img.shields.io/badge/Context-1M%20native-orange"></a>
  <a href="https://github.com/XingChen-AGI/Xing4.1-Preview/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"></a>
</p>

# 目录

- [简介](#简介)
- [最新动态](#最新动态)
- [模型架构](#模型架构)
- [评测](#评测)
- [快速开始](#快速开始)
- [应用场景](#应用场景)
- [免责声明](#免责声明)
- [协议与引用](#协议与引用)

# 简介

**Xing4.1-Preview** 是星辰系列（Xing，前身为 [TeleChat](https://github.com/Tele-AI/TeleChat3)）的最新大语言模型，由中电信人工智能科技研发。

- **全栈国产化训练链路**——8192 张昇腾（Ascend）910C 芯片 + 国产框架 MindSpore + 自研训练系统，端到端完成 285B 模型预训练；模型架构从选型阶段即针对国产芯片“矩阵算力强、向量与带宽相对受限”的硬件特征做正向协同设计，把国产算力的硬件特点转化为能力与效率优势。
- **智能体编程与长程智能体任务**——面向需要持续推进、多轮交互才能完成的复杂任务：长上下文规划、工具调用、结构化输出与多步骤协同执行。
- **总参数 285B / 单次激活 17B（MoE）**——以约 17B 级的推理开销，获得 285B 级模型的能力。
- **原生 1M 上下文**——公开 API 即提供完整 1M（单次最大输入 896K / 最大输出 128K）。
- **深度思考、结构化输出、函数调用**——思考模式支持开/关（无分档）；通过 OpenAI 兼容接口提供结构化输出与函数调用。

**本版本为 Preview（预览）发布**：模型权重与 API 现已开放；评测结果见[评测](#评测)一节，技术报告已发布为 [arXiv 2611.22115](https://arxiv.org/abs/2611.22115)。

# 最新动态

- 2026-11-23 开源 **Xing4.1-Preview** [[HuggingFace](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) | [魔搭](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) | [天翼 AI 开放平台](https://www.teleai.com.cn/)]

# 模型架构

**Xing4.1-Preview** 架构如下：

| | 注意力类型 | 注意力稀疏 | 路由专家 | 单 Token 激活专家 | 共享专家 | 总参/激活参数 | 上下文长度 |
|---|---|---|---|---|---|---|---|
| Xing4.1-Preview | MLA | DSA | 256 | Top-8 | 1 | 285B / 17B | 原生 1M |

关键设计选择：

- **DSA（动态稀疏注意力）**：降低长上下文注意力的计算与访存开销，同时保留以矩阵乘为核心的计算模式——这正是训练加速器最擅长的算子。
- **"矮胖型"网络结构**：在模型规模与并行策略约束下适度增加宽度、降低深度，提升核心矩阵算子的计算规模、算术强度与数据复用效率，同时减少逐层串行开销（归一化、激活函数、算子调度与数据搬运）。
- **MLA + DSA 协同**：MLA 低秩压缩缩小 KV Cache，DSA 减少对其的访问，两者协同缓解访存带宽瓶颈、支撑长序列高效训练。
- **协同设计的训练 Infra**：显存分片与通信优化、分布式 Muon 优化器去冗余、DSA 长序列训练系统优化与细粒度选择性重计算四项协同，带来约 **15%–30%** 的端到端训练吞吐提升，长序列场景收益更突出。

## 全栈国产化训练

Xing4.1-Preview 的预训练在国产技术栈上端到端完成：**8192 张昇腾（Ascend）910C 芯片**、**MindSpore 框架**，以及在其之上自研的 HyperParallel 训练系统。算力、框架、训练系统三层协同、全栈打通：

- **HyperParallel-FSDP（HSDP）**：在每个 SuperPoD 内部做参数全分片，跨 SuperPoD 复制参数——高频的 AllGather / ReduceScatter 收敛在超节点内部的高速互联域，跨节点只做粗粒度梯度 AllReduce。因此整个训练**不需要流水线并行**，也就没有流水线气泡带来的算力浪费。
- **Owner-based Muon 优化器**：把每个矩阵的 Newton–Schulz 正交化交给分片组内的指定 Owner 节点算一次再分发，避免各节点重复还原完整矩阵、重复计算。
- **细粒度选择性重计算**：按算子的执行时间与输出显存占用决定是否保留，而不是整层一刀切。
- **宿主机侧调度**：训练线程绑定独占 CPU 核，系统服务与中断处理移出训练 CPU 资源池。
- **综合效果**：端到端训练吞吐较基线提升约 **15%–30%**，长序列场景收益最大。

后训练使用自研统一智能体平台 **TeleGym**，把任务执行、结果评估、轨迹构建与强化学习优化串成一条链路，并支持异步执行、有限超额采样与部分轨迹续跑。技术细节见[技术报告](https://arxiv.org/abs/2611.22115)。

**开源权重的格式说明：** Xing4.1-Preview 权重的原生格式是 MindSpore 格式。考虑到开源生态中的推理与微调工具链大多以 PyTorch 为中心，我们在对外发布前把权重完整转换为 **PyTorch（Hugging Face Transformers）格式**，以 **safetensors** 分片存储。社区可以直接用 Transformers、SGLang、vLLM 等常见框架加载，无需关心训练侧的框架差异。

# 评测

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

该对比集覆盖智能体执行、工具调用与通用推理，各项结果互有高低，建议按自身业务场景对应的任务判断适配度，而非只看单一综合分。

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

- **上下文窗口**：公开 API 即提供完整 **1M** 上下文（单次最大输入 **896K** / 最大输出 **128K**）。
- **思考模式**：支持开/关（无分档）。
- **结构化输出与函数调用**：通过 OpenAI 兼容接口支持。

**推荐参数：** `temperature=0.8`、`top_p=1.0`、`max_tokens=128K`。

## 本地部署

> [!NOTE]
> Xing4.1-Preview 沿用 Xing 4 系列的服务化栈。把 `${MODEL_PATH}` 换成本地路径或 `XingChen-AGI/Xing4.1-Preview`。该模型为 285B BF16 MoE（权重约 521 GiB），仅权重即需约 **8×80 GB** 显存（`--tensor-parallel-size 8`），上下文更长需相应增加资源。原生上下文为 1M，请按显存/节点把上下文参数（`--max-model-len` / `--context-length`）设到硬件可承载的值。parser 标识与预编译镜像 tag 以本仓库官方部署文档（[vLLM](./tutorial/vLLM/xing4.1-preview-vllm.md) / [SGLang](./tutorial/SGLang/xing4.1-preview-sglang.md) / [FlagOS](./tutorial/FlagOS/xing4.1-preview-flagos.md)）为准。

**vLLM**——官方 out-of-tree 插件（[XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin)）。模型本体（`deepseek_v32`）由 vLLM 原生加载，`xing4_0` 推理 / 工具调用解析由该插件提供；详细部署步骤、参数说明与调用示例见 [vLLM 部署文档](./tutorial/vLLM/xing4.1-preview-vllm.md)。

**SGLang**——main 分支原生支持，无需额外插件；详细部署步骤、参数说明与调用示例见 [SGLang 部署文档](./tutorial/SGLang/xing4.1-preview-sglang.md)。

**FlagOS**——社区已完成 Xing4.1-Preview 的 Day0 跨芯适配，提供覆盖八个平台的 W8A8 版本，安装、启动与推理方法见 [FlagOS 多芯部署指南](./tutorial/FlagOS/xing4.1-preview-flagos.md)。

**vLLM-Ascend（昇腾 NPU）**——部署文档即将补充。

# 应用场景

- **智能体协同办公**——多步骤办公流程、跨工具编排、多源材料交叉核对的报表生成。
- **深度研究**——持续性多源调研，输出结构化、可核对的交付物。
- **智能体编程**——完整前端生成（单文件、零依赖、可离线打开的 Web 应用）、3D 创意编程与数据分析流水线。
- **文档密集型业务**——面向业务系统的架构引导结构化抽取。

# 免责声明

与所有生成式语言模型一样，Xing4.1-Preview 可能产生不准确或意外的输出。输出由模型自动生成，不代表中国电信观点。请独立核实重要信息，勿将模型用于违反适用法律法规的用途。

# 协议与引用

本仓库的权重与代码以 [Apache 2.0](./LICENSE) 协议开源。

## 相关论文

- **Xing4 技术报告：** [TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks](https://arxiv.org/abs/2611.22115)
- **cmHC：** [cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction](https://arxiv.org/abs/2611.27193)

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
