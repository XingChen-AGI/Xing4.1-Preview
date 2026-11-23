### Multi-Chip Deployment with FlagOS

[Xing4.1-Preview](https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) is a Mixture-of-Experts (MoE) language model in China Telecom's Xing 4 series, designed for agentic coding, complex reasoning, and long-horizon agent tasks. It has **285B total parameters, approximately 17B activated per token, and a 1M-token context window**. Its architecture combines Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA) for long-context processing.

The FlagOS community provides **W8A8 deployment versions of Xing4.1-Preview for eight chip platforms**. Developers can select the version for their hardware and use the FlagOS inference plugins, operator libraries, and compiler to deploy the model.

#### About FlagOS

FlagOS is a unified, open-source system software stack for diverse AI chips, developed and maintained by the FlagOS community.

FlagOS includes the FlagGems operator library, FlagTree AI compiler, FlagScale parallel training and inference framework, and FlagCX communication library. These projects connect the model, system, and chip layers, helping developers reuse model code and software components while reducing migration and maintenance costs. For inference deployment, vLLM-plugin-FL and SGLang-plugin-FL extend hardware support while preserving the original framework interfaces and usage patterns.

Website: [flagos.io](https://flagos.io/) | Community: [GitHub](https://github.com/flagos-ai) / [GitCode](https://gitcode.com/flagos-ai)

#### Multi-Chip Support and Downloads

FlagRelease provides model adaptation, evaluation, and release tools for deployment across AI chips. Xing4.1-Preview has **eight W8A8 versions: T-Head, NVIDIA, Moore Threads, Huawei Ascend, MetaX, Hygon, Iluvatar, and Kunlunxin**.

| Chip Vendor / Platform | ModelScope | Hugging Face |
| --- | --- | --- |
| T-Head (Zhenwu) | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-zhenwu-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-zhenwu-FlagOS) |
| NVIDIA | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-nvidia-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-nvidia-FlagOS) |
| Moore Threads | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-mthreads-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-mthreads-FlagOS) |
| Huawei Ascend | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-ascend-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-ascend-FlagOS) |
| MetaX | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-metax-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-metax-FlagOS) |
| Hygon | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-hygon-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-hygon-FlagOS) |
| Iluvatar | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-iluvatar-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-iluvatar-FlagOS) |
| Kunlunxin | [Model repository](https://modelscope.cn/models/FlagRelease/Xing4.1-Preview-W8A8-kunlunxin-FlagOS) | [Model repository](https://huggingface.co/FlagRelease/Xing4.1-Preview-W8A8-kunlunxin-FlagOS) |

#### Installation and Usage

##### Option 1: Use a Platform Release

1. Select your target chip platform from the table above and open the corresponding model repository.
2. Follow its model card to prepare the drivers and runtime, download the model weights, and configure the deployment environment.
3. Run inference using the commands in the model card. Refer to that card for accelerator count, memory requirements, parallelism, framework versions, and startup parameters.

##### Option 2: Install from Source

FlagOS supports Xing4.1-Preview through **vLLM-plugin-FL** and **SGLang-plugin-FL**. The current deployment configurations use vLLM 0.24.0 for NVIDIA, T-Head, MetaX, Hygon, Iluvatar, and Kunlunxin, and SGLang 0.5.18 for Huawei Ascend and Moore Threads.

Official developer guides:

- **vLLM-plugin-FL**: [GitHub](https://github.com/flagos-ai/vllm-plugin-FL/blob/main/README.md) / [GitCode](https://gitcode.com/flagos-ai/vllm-plugin-FL/blob/main/README.md)
- **SGLang-plugin-FL**: [GitHub](https://github.com/flagos-ai/sglang-plugin-FL/blob/main/README.md) / [GitCode](https://gitcode.com/flagos-ai/sglang-plugin-FL/blob/main/README.md)

The following source-installation example for **T-Head** uses **vLLM v0.24.0, vLLM-plugin-FL, and FlagGems v5.3.4**. First prepare the chip drivers, runtime, compatible PyTorch build, and [FlagTree compiler](https://github.com/flagos-ai/FlagTree) environment. Apply the model-specific patches, dependencies, and environment settings specified by the platform model card.

```bash
# 1. Install vLLM v0.24.0
git clone --branch v0.24.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty python -m pip install -v --no-build-isolation --no-deps .
cd ..

# 2. Install vllm-plugin-FL
git clone https://github.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
python -m pip install --no-build-isolation -e .
cd ..

# 3. Install FlagGems v5.3.4
python -m pip install scikit-build-core==0.11 pybind11 ninja cmake
git clone --branch v5.3.4 https://github.com/flagos-ai/FlagGems.git
cd FlagGems
python -m pip install --no-build-isolation -e .
cd ..
```

Configure the [FlagCX communication library](https://github.com/flagos-ai/FlagCX) as required by the platform's distributed inference setup. Use the component versions and backend settings specified in the model card when preparing another platform.

##### Run Inference

The following example uses the **T-Head W8A8 version of Xing4.1-Preview** in a configured vLLM-plugin-FL environment. It sets eight-way tensor parallelism and a 32K-token context limit; adjust both to your hardware and the platform model card.

Save the code as `inference.py`:

```python
from vllm import LLM, SamplingParams


if __name__ == "__main__":
    llm = LLM(
        model="FlagRelease/Xing4.1-Preview-W8A8-zhenwu-FlagOS",
        tensor_parallel_size=8,
        max_model_len=32768,
        trust_remote_code=True,
    )
    messages = [
        {"role": "user", "content": "Please introduce the key features of Xing4.1-Preview."}
    ]
    sampling_params = SamplingParams(max_tokens=512, temperature=0.8)
    outputs = llm.chat(messages, sampling_params=sampling_params)
    for output in outputs:
        print(output.outputs[0].text)
```

```bash
VLLM_PLUGINS=fl python inference.py
```

To use local weights, replace `model` with the downloaded model directory. When changing chip platforms, select the corresponding repository from the table and adjust the runtime, plugin backend, parallelism, and context limit according to its model card. For Huawei Ascend and Moore Threads, follow the SGLang-plugin-FL deployment instructions.
