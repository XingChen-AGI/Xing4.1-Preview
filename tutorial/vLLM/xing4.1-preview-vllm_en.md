# Deploying the Xing4.1-Preview Model with vLLM

　　This document describes how to launch an OpenAI-compatible API service for the Xing4.1-Preview model with vLLM. It covers:

- Preparing the base vLLM image and the model
- Downloading the Xing4.1-Preview model
- Starting the vLLM service (installing the official plugin to enable reasoning-chain / tool-call parsing)
- OpenAI-compatible API call examples
- Startup parameter descriptions and inference parameter recommendations

　　The Xing4.1-Preview model body can be loaded directly by an existing vLLM image; to correctly parse the Xing series' dedicated **reasoning-chain output** and **tool-call format**, this document installs the official out-of-tree plugin [XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin). It only registers the `xing4_0` reasoning parser and tool-call parser with vLLM; it does not modify the vLLM source code, nor does it affect model loading.

　　Xing4.1-Preview is a 285B BF16 MoE model with weights of approximately 521 GiB, requiring about **8×80 GB** of GPU memory just to load the weights. The startup commands in this document use `--tensor-parallel-size 8` as an example, which requires at least 8 GPUs.

## Prerequisites

　　Make sure the host machine meets the following software requirements:

- Docker (24.0 or later recommended)
- NVIDIA driver (535 or later recommended)
- NVIDIA Container Toolkit (the `nvidia-docker` runtime)

　　Verify that the GPU is accessible from a container (using the official vLLM image as an example):

```bash
docker run --rm --gpus all vllm/vllm-openai nvidia-smi
```

> [!NOTE]
> The hardware configuration in this document is used only for **demonstration and validation** and is not a mandatory requirement. You may choose other CUDA-capable NVIDIA GPUs based on your own hardware — simply adjust parameters such as `--tensor-parallel-size`, `--max-model-len`, and `--gpu-memory-utilization` according to the actual memory and number of cards. If memory is limited, you can also use the FlagOS W8A8 version as described below.

## Prepare the Base Image

```bash
docker pull vllm/vllm-openai
```

## Download the Xing4.1-Preview Model

　　The Xing4.1-Preview model can be downloaded from the following platforms; the weights are identical across all three:

| Platform | URL |
|----------|-----|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.1-Preview |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.1-Preview |

　　Using ModelScope as an example, download it to a local directory:

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.1-Preview \
  --local_dir /yourpath/models/Xing4.1-Preview
```

　　To further reduce GPU memory usage, you can use the W8A8 version provided by the FlagOS community. See the [FlagOS multi-chip deployment guide](../FlagOS/xing4.1-preview-flagos.md) for details.

## Start the vLLM Service

　　When deploying via Docker, first start the container and mount the model directory:

```bash
docker run -d \
  --name xing41-vllm \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.1-Preview:/models/Xing4.1-Preview \
  vllm/vllm-openai \
  -c "sleep infinity"
```

　　After entering the container, install the official plugin and start the service by following the steps below.

### Enable Reasoning-Chain / Tool-Call Parsing (Install the Official Plugin)

　　To let vLLM correctly parse the Xing series' reasoning chain and tool-call formats, first install the official plugin (`--no-deps` avoids overwriting the vLLM already installed in the image) and confirm that the plugin is discovered by vLLM:

```bash
pip install --no-deps "git+https://github.com/XingChen-AGI/xing4-vllm-plugin"

python3 -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

　　The output should contain `('xing4_0', 'vllm_xing4_0_plugin:register')`.

　　Then explicitly enable the plugin via `VLLM_PLUGINS=xing4_0` and start the service with the parser parameters:

```bash
VLLM_PLUGINS=xing4_0 vllm serve /models/Xing4.1-Preview \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.1-Preview \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 4 \
    --enable-auto-tool-choice \
    --reasoning-parser xing4_0 \
    --tool-call-parser xing4_0
```

　　Once started, the OpenAI-compatible API is available at `http://localhost:8000/v1`. The current terminal will keep outputting service logs; when a message similar to `Uvicorn running on http://0.0.0.0:8000` appears in the logs, the service is ready to serve requests.

> [!NOTE]
> Xing4.1-Preview's native context is **1M**. When not explicitly specified, vLLM defaults to the model's upper limit. If the GPU memory is insufficient to hold the KV cache for a 1M context, explicitly reduce `--max-model-len` (e.g. `--max-model-len 131072`) and lower `--max-num-seqs` accordingly.

### Startup Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `--host 0.0.0.0` | Listen on all network interfaces so the service is reachable from outside the container |
| `--port 8000` | API service port |
| `--served-model-name` | Model name exposed externally; must match the value used when calling the API |
| `--tensor-parallel-size 8` | Tensor parallelism degree; must match the number of available GPUs (the 285B weights require at least 8 cards) |
| `--trust-remote-code` | Trust the remote code in the model repository |
| `--max-model-len` | Maximum context length; defaults to the model upper limit of 1M. Reduce it when memory is limited |
| `--gpu-memory-utilization 0.90` | Upper bound on the GPU memory fraction to reserve; adjust based on available memory |
| `--max-num-seqs 4` | Maximum number of concurrent requests |
| `--enable-auto-tool-choice` | Enable automatic tool choice (function calling); required when the request carries a `tools` parameter |
| `--reasoning-parser xing4_0` | Parses the reasoning-chain output (the `<|thinking|>` content) of the Xing series, provided by the official plugin |
| `--tool-call-parser xing4_0` | Parses the tool-call format (e.g. `<tool_call>`) of the Xing series, provided by the official plugin |

## Calling the API

　　After the service starts, you can call it through the OpenAI-compatible interface. The `model` field in the examples below must match the `--served-model-name` used at startup.

### Basic Chat (curl)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Xing4.1-Preview",
    "messages": [
      {"role": "user", "content": "Introduce yourself in one sentence."}
    ],
    "temperature": 0.8,
    "top_p": 1.0,
    "chat_template_kwargs": {"enable_thinking": true},
    "skip_special_tokens": false
  }'
```

## Inference Parameter Recommendations

　　Based on the official recommendations for Xing4.1-Preview, use the following sampling parameters:

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.8 |
| `top_p` | 1.0 |
| `max_tokens` | 128K |

　　Thinking mode can be toggled via `chat_template_kwargs.enable_thinking` (`true` outputs the `<|thinking|>` reasoning process, `false` outputs only the final result).

## FAQ

### Out of Memory (OOM)

- Lower `--gpu-memory-utilization` (e.g. 0.85) or explicitly reduce `--max-model-len` (e.g. 131072)
- Decrease `--max-num-seqs` to reduce the peak concurrent memory usage
- If memory is still insufficient, use the FlagOS W8A8 quantized version (see the [FlagOS multi-chip deployment guide](../FlagOS/xing4.1-preview-flagos.md))

### Tensor Parallelism Card Count Mismatch

- `--tensor-parallel-size` must divide the number of visible GPUs evenly; the 285B weights require at least 8×80 GB GPUs, so it can be set to `8`; with more cards, set it to a value that divides the total (e.g. 16)

### GPU Not Accessible Inside the Container

- Confirm that NVIDIA Container Toolkit is installed and that `--gpus all` is used when starting the container
- To use only a subset of cards, use `--gpus '"device=0,1,2,3,4,5,6,7"'` instead

### Service Not Reachable from Outside

- Confirm that `--host` is set to `0.0.0.0` and that `-p 8000:8000` is used to map the port

### Plugin Not Taking Effect / xing4_0 Parser Not Found

- Confirm the plugin is installed in the same Python environment as `vllm` (inside the container, use the image's own python/pip) and that the entry-point list contains `('xing4_0', ...)`
- Explicitly add `VLLM_PLUGINS=xing4_0` at startup to allow loading the plugin
- If the logs show a plugin stand-down notice, it means the current vLLM already has a built-in implementation with the same name and can be ignored

### model_type 'deepseek_v32' is not supported Reported

- The vLLM version in use is too old and does not yet have built-in DeepSeek-V3.2 support; upgrade to a recent stable release or nightly, or install vLLM from the main source
