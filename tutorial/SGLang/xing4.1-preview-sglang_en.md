# Deploying the Xing4.1-Preview Model with SGLang

　　This document describes how to launch an OpenAI-compatible API service for the Xing4.1-Preview model with SGLang. It covers:

- Pulling the official SGLang image
- Downloading the Xing4.1-Preview model
- Starting the SGLang service
- OpenAI-compatible API call examples
- Startup parameter descriptions and inference parameter recommendations

　　Xing4.1-Preview uses a different architecture configuration from Xing4.0 (MLA + DSA, with `model_type` set to `deepseek_v32`). SGLang has **natively supported** Xing4.1-Preview on the **main branch** (since 2026-11-05), so it can be loaded and served without installing any additional plugin.

　　The official image `lmsysorg/sglang:dev` is built continuously from main and already includes the Xing4.1-Preview adaptation. **Just pull and use it — no extra dependency installation is needed inside the container**:

- **Method 1 (recommended):** Use the official `lmsysorg/sglang:dev` image, ready to use out of the box.

## Prerequisites

　　Make sure the host machine meets the following software requirements:

- Docker (24.0 or later recommended)
- NVIDIA driver (the official image is based on CUDA 13, so a compatible driver version is required)
- NVIDIA Container Toolkit (the `nvidia-docker` runtime)

　　Verify that the GPU is accessible from a container:

```bash
docker run --rm --gpus all lmsysorg/sglang:dev nvidia-smi
```

　　Xing4.1-Preview is a 285B BF16 MoE model with weights of approximately 521 GiB, requiring about **8×80 GB** of GPU memory just to load the weights. The startup commands in this document use `--tp-size 8` as an example, which requires at least 8 GPUs; adjust this parameter and the context length according to the actual number of cards and available memory.

> [!NOTE]
> The hardware configuration in this document is used only for **demonstration and validation** and is not a mandatory requirement. You may choose other CUDA-capable NVIDIA GPUs based on your own hardware — simply adjust parameters such as `--tp-size`, `--context-length`, and `--mem-fraction-static` according to the actual memory and number of cards. If memory is limited, you can also use the FlagOS W8A8 version as described below.

## Deploying with the Official dev Image

### Pull the Image

```bash
docker pull lmsysorg/sglang:dev
```

> [!NOTE]
> `lmsysorg/sglang:dev` continuously tracks the SGLang main branch and already includes the `xing4_0` reasoning / tool-call parser required by Xing4.1-Preview, so no `pip install` is needed inside the container. If reproducibility is a concern, you can pick a dated nightly fixed tag from the [Docker Hub tags list](https://hub.docker.com/r/lmsysorg/sglang/tags) to replace `dev`.

### Download the Xing4.1-Preview Model

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

　　To further reduce GPU memory usage, you can use the W8A8 version provided by the FlagOS community (covering eight platforms: T-Head, NVIDIA, Moore Threads, Huawei Ascend, MetaX, Hygon, Iluvatar CoreX, and Kunlunxin). See the [FlagOS multi-chip deployment guide](../FlagOS/xing4.1-preview-flagos.md) for details.

### Start the SGLang Service

　　Deployment via Docker is done in two steps: first start the container, then run the SGLang startup command inside it.

　　Step 1: start the container (using bash as the entrypoint and keeping it running), mount the local model directory, and expose the API port:

```bash
docker run -d \
  --name xing41-sglang \
  --gpus all \
  --shm-size 16g \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  -v /yourpath/models/Xing4.1-Preview:/models/Xing4.1-Preview \
  lmsysorg/sglang:dev \
  -c "sleep infinity"
```

　　Step 2: enter the container and run the following command to start the SGLang service. Logs will be output directly to the current terminal:

```bash
sglang serve --model-path /models/Xing4.1-Preview \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Xing4.1-Preview \
    --tp-size 8 \
    --mem-fraction-static 0.90 \
    --max-running-requests 4 \
    --reasoning-parser xing4_0 \
    --tool-call-parser xing4_0
```

　　Once started, the OpenAI-compatible API is available at `http://localhost:8000/v1`. The current terminal will keep outputting service logs; when a message similar to `The server is fired up and ready to roll!` appears in the logs, the service is ready to serve requests.

> [!NOTE]
> Xing4.1-Preview's native context is **1M**. When not explicitly specified, SGLang defaults to the model's upper limit of `1048576`. If the GPU memory is insufficient to hold the KV cache for a 1M context, explicitly reduce `--context-length` (e.g. `--context-length 131072`) and lower `--max-running-requests` accordingly.

### Startup Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| `--model-path` | Directory containing the model weights |
| `--host 0.0.0.0` | Listen on all network interfaces so the service is reachable from outside the container |
| `--port 8000` | API service port |
| `--served-model-name` | Model name exposed externally; must match the value used when calling the API |
| `--tp-size 8` | Tensor parallelism degree; must match the number of available GPUs (the 285B weights require at least 8 cards) |
| `--trust-remote-code` | Trust the remote code in the model repository (e.g. `modeling_deepseek`) |
| `--context-length` | Maximum context length; defaults to the model upper limit of 1M. Reduce it when memory is limited |
| `--mem-fraction-static 0.90` | Upper bound on the static memory fraction; adjust based on available memory |
| `--max-running-requests 4` | Maximum number of concurrent requests |
| `--reasoning-parser xing4_0` | Parses the reasoning-chain output (the `<|thinking|>` content) of the Xing series |
| `--tool-call-parser xing4_0` | Parses the tool-call format (e.g. `<tool_call>`) of the Xing series |

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

- Lower `--mem-fraction-static` (e.g. 0.85) or explicitly reduce `--context-length` (e.g. 131072)
- Decrease `--max-running-requests` to reduce the peak concurrent memory usage
- If memory is still insufficient, use the FlagOS W8A8 quantized version (see the [FlagOS multi-chip deployment guide](../FlagOS/xing4.1-preview-flagos.md))

### Tensor Parallelism Card Count Mismatch

- `--tp-size` must divide the number of visible GPUs evenly; the 285B weights require at least 8×80 GB GPUs, so it can be set to `8`; with more cards, set it to a value that divides the total (e.g. 16)

### GPU Not Accessible Inside the Container

- Confirm that NVIDIA Container Toolkit is installed and that `--gpus all` is used when starting the container
- To use only a subset of cards, use `--gpus '"device=0,1,2,3,4,5,6,7"'` instead

### Service Not Reachable from Outside

- Confirm that `--host` is set to `0.0.0.0` and that `-p 8000:8000` is used to map the port

### xing4_0 Parser Not Found

- `lmsysorg/sglang:dev` already includes this parser; if an older image does not register the `xing4_0` reasoning / tool-call parser, simply re-pull the latest `dev` image
