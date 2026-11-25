# 使用 SGLang 部署 Xing4.1-Preview 模型

　　本文介绍如何通过 SGLang 启动 Xing4.1-Preview 模型的 OpenAI 兼容 API 服务。主要覆盖以下内容：

- 获取官方 SGLang 镜像
- Xing4.1-Preview 模型下载
- 启动 SGLang 服务
- OpenAI 兼容 API 调用示例
- 启动参数说明与推理参数建议

　　Xing4.1-Preview 采用与 Xing4.0 不同的架构配置。SGLang 已在 **main 分支原生支持** Xing4.1-Preview（2026-11-05 起），无需额外安装插件即可加载与服务。

　　官方镜像 `lmsysorg/sglang:dev` 持续从 main 构建，已包含 Xing4.1-Preview 适配，**直接拉取使用即可，无需在容器内额外安装任何依赖包**：

- **推荐**：使用官方 `lmsysorg/sglang:dev` 镜像，开箱即用。

## 环境准备

　　请确认宿主机满足以下软件要求：

- Docker（建议 24.0 及以上）
- NVIDIA 驱动（官方镜像基于 CUDA 13，需与之兼容的驱动版本）
- NVIDIA Container Toolkit（`nvidia-docker` 运行时）

　　校验 GPU 是否可被容器访问：

```bash
docker run --rm --gpus all lmsysorg/sglang:dev nvidia-smi
```

　　Xing4.1-Preview 为 285B BF16 MoE 模型，权重约 521 GiB，仅加载权重即需约 **8×80 GB** 显存。本文启动命令以 `--tp-size 8` 为例，需要至少 8 张 GPU；可根据实际卡数与显存调整该参数及上下文长度。

> [!NOTE]
> 本文中的硬件配置仅为**演示与验证所用**，并非强制要求。用户可根据自身硬件选择其他支持 CUDA 的 NVIDIA GPU，只需根据实际显存与卡数调整 `--tp-size`、`--context-length`、`--mem-fraction-static` 等参数即可；显存有限时也可参考下文使用 FlagOS W8A8 版本。

## 使用官方 dev 镜像部署

### 拉取镜像

```bash
docker pull lmsysorg/sglang:dev
```

> [!NOTE]
> `lmsysorg/sglang:dev` 持续跟踪 SGLang main 分支，已内置 Xing4.1-Preview 所需的 `xing4_0` 推理 / 工具调用 parser，容器内无需再执行任何 `pip install`。如对可复现性有要求，可在 [Docker Hub tags 列表](https://hub.docker.com/r/lmsysorg/sglang/tags) 选择一个带日期的 nightly 固定 tag 替代 `dev`。

### 下载 Xing4.1-Preview 模型

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

　　如需进一步降低显存占用，可使用众智 FlagOS 社区提供的 W8A8 版本（覆盖平头哥、英伟达、摩尔线程、华为昇腾、沐曦、海光、天数智芯、昆仑芯八个平台），详见 [FlagOS 多芯部署指南](../FlagOS/xing4.1-preview-flagos.md)。

### 启动 SGLang 服务

　　通过 Docker 部署分为两步：先启动容器，再在容器内执行 SGLang 启动命令。

　　第一步，启动容器（以 bash 作为入口点并保持运行），挂载本地模型目录并暴露 API 端口：

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

　　第二步，进入容器后执行以下命令启动 SGLang 服务，日志将直接输出到当前终端：

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

　　启动后通过 `http://localhost:8000/v1` 即可访问 OpenAI 兼容 API。当前终端会持续输出服务日志，当日志中出现类似 `The server is fired up and ready to roll!` 的字样时，表示服务已对外可用。

> [!NOTE]
> Xing4.1-Preview 原生上下文为 **1M**，不显式指定时 SGLang 默认取模型上限 `1048576`。若显存不足以承载 1M 上下文的 KV Cache，请显式调小 `--context-length`（例如 `--context-length 131072`），并相应降低 `--max-running-requests`。

### 启动参数说明

| 参数 | 说明 |
|------|------|
| `--model-path` | 模型权重所在目录 |
| `--host 0.0.0.0` | 监听所有网卡，使容器外可访问 |
| `--port 8000` | API 服务端口 |
| `--served-model-name` | 对外暴露的模型名，调用 API 时需与之对应 |
| `--tp-size 8` | 张量并行卡数，需与可用 GPU 数一致（285B 权重至少需 8 卡） |
| `--trust-remote-code` | 信任模型仓库内的远程代码（自定义 config / modeling 文件） |
| `--context-length` | 最大上下文长度，默认取模型上限 1M；显存有限时请调小 |
| `--mem-fraction-static 0.90` | 静态显存占用比例上限，按显存余量调整 |
| `--max-running-requests 4` | 最大并发请求数 |
| `--reasoning-parser xing4_0` | 解析 Xing 系列的思维链输出（`<|thinking|>` 内容） |
| `--tool-call-parser xing4_0` | 解析 Xing 系列的工具调用格式（`<tool_call>` 等） |

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

- 降低 `--mem-fraction-static`（如 0.85）或显式调小 `--context-length`（如 131072）
- 减少 `--max-running-requests` 以降低并发显存峰值
- 显存仍不足时，可使用 FlagOS W8A8 量化版本（详见 [FlagOS 多芯部署指南](../FlagOS/xing4.1-preview-flagos.md)）

### 张量并行卡数不匹配

- `--tp-size` 必须能整除可见 GPU 数；285B 权重至少需要 8 张 80GB 显卡，可设为 `8`；更多卡时可设为能整除总数的值（如 16）

### 容器内无法访问 GPU

- 确认已安装 NVIDIA Container Toolkit，并在启动时使用 `--gpus all`
- 若只希望使用部分卡，可改用 `--gpus '"device=0,1,2,3,4,5,6,7"'`

### 服务无法从外部访问

- 确认 `--host` 设为 `0.0.0.0`，并使用 `-p 8000:8000` 映射端口

### 找不到 xing4_0 parser

- `lmsysorg/sglang:dev` 已内置该 parser；若镜像过旧未注册 `xing4_0` 推理 / 工具调用 parser，重新拉取最新 `dev` 镜像即可
