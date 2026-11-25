# 使用 vLLM 部署 Xing4.1-Preview 模型

　　本文介绍如何通过 vLLM 启动 Xing4.1-Preview 模型的 OpenAI 兼容 API 服务。主要覆盖以下内容：

- 准备 vLLM 基础镜像与模型
- Xing4.1-Preview 模型下载
- 启动 vLLM 服务（安装官方插件以启用推理链 / 工具调用解析）
- OpenAI 兼容 API 调用示例
- 启动参数说明与推理参数建议

　　Xing4.1-Preview 模型本体可被现有 vLLM 镜像直接加载；为正确解析 Xing 系列专属的**思维链输出**与**工具调用格式**，本文安装官方 out-of-tree 插件 [XingChen-AGI/xing4-vllm-plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin)，它仅向 vLLM 注册 `xing4_0` 推理 parser 与工具调用 parser，不修改 vLLM 源码、也不影响模型加载。

　　Xing4.1-Preview 为 285B BF16 MoE 模型，权重约 521 GiB，仅加载权重即需约 **8×80 GB** 显存。本文启动命令以 `--tensor-parallel-size 8` 为例，需要至少 8 张 GPU。

## 环境准备

　　请确认宿主机满足以下软件要求：

- Docker（建议 24.0 及以上）
- NVIDIA 驱动（建议 535 及以上）
- NVIDIA Container Toolkit（`nvidia-docker` 运行时）

　　校验 GPU 是否可被容器访问（以官方 vLLM 镜像为例）：

```bash
docker run --rm --gpus all vllm/vllm-openai nvidia-smi
```

> [!NOTE]
> 本文中的硬件配置仅为**演示与验证所用**，并非强制要求。用户可根据自身硬件选择其他支持 CUDA 的 NVIDIA GPU，只需根据实际显存与卡数调整 `--tensor-parallel-size`、`--max-model-len`、`--gpu-memory-utilization` 等参数即可；显存有限时也可参考下文使用 FlagOS W8A8 版本。

## 准备基础镜像

```bash
docker pull vllm/vllm-openai
```

## 下载 Xing4.1-Preview 模型

　　Xing4.1-Preview 模型可从以下平台下载，三端权重一致：

| 平台 | 地址 |
|------|------|
| Hugging Face | https://huggingface.co/XingChen-AGI/Xing4.1-Preview |
| ModelScope | https://modelscope.cn/models/XingChen-AGI/Xing4.1-Preview |
| Modelers | https://modelers.cn/models/XingChen-AGI/Xing4.1-Preview |

　　以 ModelScope 为例，下载到本地目录：

```bash
pip install modelscope

modelscope download \
  --model XingChen-AGI/Xing4.1-Preview \
  --local_dir /yourpath/models/Xing4.1-Preview
```

　　如需进一步降低显存占用，可使用众智 FlagOS 社区提供的 W8A8 版本，详见 [FlagOS 多芯部署指南](../FlagOS/xing4.1-preview-flagos.md)。

## 启动 vLLM 服务

　　通过 Docker 部署时，先启动容器并挂载模型目录：

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

　　进入容器后，按以下步骤安装官方插件并启动服务。

### 启用推理链 / 工具调用解析（安装官方插件）

　　如需让 vLLM 正确解析 Xing 系列的思维链与工具调用格式，先安装官方插件（`--no-deps` 避免覆盖镜像内已安装的 vLLM），并确认插件已被 vLLM 发现：

```bash
pip install --no-deps "git+https://github.com/XingChen-AGI/xing4-vllm-plugin"

python3 -c "from importlib.metadata import entry_points; \
print([(e.name, e.value) for e in entry_points(group='vllm.general_plugins')])"
```

　　输出中应包含 `('xing4_0', 'vllm_xing4_0_plugin:register')`。

　　然后通过 `VLLM_PLUGINS=xing4_0` 显式启用插件，并加上 parser 参数启动服务：

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

　　启动后通过 `http://localhost:8000/v1` 即可访问 OpenAI 兼容 API。当前终端会持续输出服务日志，当日志中出现类似 `Uvicorn running on http://0.0.0.0:8000` 的字样时，表示服务已对外可用。

> [!NOTE]
> Xing4.1-Preview 原生上下文为 **1M**，不显式指定时 vLLM 默认取模型上限。若显存不足以承载 1M 上下文的 KV Cache，请显式调小 `--max-model-len`（例如 `--max-model-len 131072`），并相应降低 `--max-num-seqs`。

### 启动参数说明

| 参数 | 说明 |
|------|------|
| `--host 0.0.0.0` | 监听所有网卡，使容器外可访问 |
| `--port 8000` | API 服务端口 |
| `--served-model-name` | 对外暴露的模型名，调用 API 时需与之对应 |
| `--tensor-parallel-size 8` | 张量并行卡数，需与可用 GPU 数一致（285B 权重至少需 8 卡） |
| `--trust-remote-code` | 信任模型仓库内的远程代码 |
| `--max-model-len` | 最大上下文长度，默认取模型上限 1M；显存有限时请调小 |
| `--gpu-memory-utilization 0.90` | 预留显存比例上限，按显存余量调整 |
| `--max-num-seqs 4` | 最大并发请求数 |
| `--enable-auto-tool-choice` | 启用自动工具调用（function calling），请求带 `tools` 参数时必须开启 |
| `--reasoning-parser xing4_0` | 解析 Xing 系列的思维链输出（`<|thinking|>` 内容），由官方插件提供 |
| `--tool-call-parser xing4_0` | 解析 Xing 系列的工具调用格式（`<tool_call>` 等），由官方插件提供 |

## 调用 API

　　服务启动后，可使用 OpenAI 兼容接口进行调用。以下示例中的 `model` 字段需与启动时的 `--served-model-name` 一致。

### 基础对话（curl）

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Xing4.1-Preview",
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己。"}
    ],
    "temperature": 0.8,
    "top_p": 1.0,
    "chat_template_kwargs": {"enable_thinking": true},
    "skip_special_tokens": false
  }'
```

## 推理参数建议

　　参考 Xing4.1-Preview 官方建议，推理采样参数选择如下：

| 参数 | 数值 |
|------|------|
| `temperature` | 0.8 |
| `top_p` | 1.0 |
| `max_tokens` | 128K |

　　思考模式可通过 `chat_template_kwargs.enable_thinking` 开关（`true` 输出 `<|thinking|>` 推理过程，`false` 仅输出最终结果）。

## 常见问题

### 显存不足（OOM）

- 降低 `--gpu-memory-utilization`（如 0.85）或显式调小 `--max-model-len`（如 131072）
- 减少 `--max-num-seqs` 以降低并发显存峰值
- 显存仍不足时，可使用 FlagOS W8A8 量化版本（详见 [FlagOS 多芯部署指南](../FlagOS/xing4.1-preview-flagos.md)）

### 张量并行卡数不匹配

- `--tensor-parallel-size` 必须能整除可见 GPU 数；285B 权重至少需要 8 张 80GB 显卡，可设为 `8`；更多卡时可设为能整除总数的值（如 16）

### 容器内无法访问 GPU

- 确认已安装 NVIDIA Container Toolkit，并在启动时使用 `--gpus all`
- 若只希望使用部分卡，可改用 `--gpus '"device=0,1,2,3,4,5,6,7"'`

### 服务无法从外部访问

- 确认 `--host` 设为 `0.0.0.0`，并使用 `-p 8000:8000` 映射端口

### 插件未生效 / 找不到 xing4_0 parser

- 确认插件安装在与 `vllm` 相同的 Python 环境中（容器内需使用镜像自带的 python/pip），且入口点列表包含 `('xing4_0', ...)`
- 启动时显式加上 `VLLM_PLUGINS=xing4_0` 以允许加载该插件
- 若日志中出现插件 stand-down 提示，说明当前 vLLM 已内置同名实现，可忽略
