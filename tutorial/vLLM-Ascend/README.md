# 使用 vLLM-Ascend 部署 Xing4.1-Preview 模型

　　本方案以昇腾 Atlas A3 系列为例。其他系列产品可参考本方案，结合实际软硬件环境自行适配。

> [!IMPORTANT]
> **部署前，请按现场环境替换示例 IP、网卡名和路径。**
>
> - **IP**：将 `192.0.2.10`（单机或 Prefill）和 `192.0.2.11`（Decode）替换为对应节点的实际通信 IP。
> - **网卡**：将部署脚本中的 `eth0` 替换为承载本机通信 IP 的实际网卡名。
> - **路径**：将 `/yourpath/...` 替换为实际权重存放、挂载和读取路径。检查 Docker 命令中涉及的宿主机路径是否与实际安装位置一致。

　　本文介绍如何在昇腾 NPU 上使用 vLLM-Ascend 提供 Xing4.1-Preview 的 OpenAI 兼容 API 服务，涵盖以下内容。

- 获取 vLLM-Ascend 镜像
- 下载 Xing4.1-Preview 模型权重
- 启动单机混部或 Prefill–Decode（PD）分离服务
- 通过 OpenAI 兼容 API 发起请求
- 查看推理参数建议

　　Xing4.1-Preview 是混合专家模型，总参数量为 285B，每个 token 激活约 17B 参数。模型结合多头潜在注意力（MLA）和稀疏注意力（DSA），支持原生 1M token 上下文和工具调用。[官方模型卡](https://www.modelscope.cn/models/XingChen-AGI/Xing4.1-Preview)

## 环境准备

　　本例部署 BF16 权重并提供工具调用。单机混部使用一台 Atlas A3 服务器，最大上下文为 128K（131072 token）。PD 分离使用两台 Atlas A3 服务器分别执行 Prefill 和 Decode，最大上下文为 256K（262144 token）。

> [!NOTE]
> 本文中的硬件配置仅供**演示与验证**。可选用当前 vLLM-Ascend 版本支持、且已适配 Xing4.1-Preview 的其他昇腾硬件。请按每个节点分配的 NPU 数量设置 `--tensor-parallel-size` 和 `--data-parallel-size`，按单设备可用 HBM 和请求长度调整 `--max-model-len` 和 `--gpu-memory-utilization`。并发请求较多时，再调整 `--max-num-seqs`。[vLLM 启动参数说明](https://docs.vllm.ai/en/latest/configuration/engine_args/)

| 配置项 | 版本 |
|------|------|
| 昇腾 NPU 驱动版本 | `26.0.rc3` |
| CANN 版本 | `cann-9.2.1` |


- 在参与部署的宿主机上确认模型权重、分词器和对话模板齐全。PD 分离时，P、D 两侧使用同一版本。将 `/yourpath/models/Xing4.1-Preview` 替换为实际权重目录。

```bash
# 将 /yourpath/models/Xing4.1-Preview 换成当前宿主机的权重目录
ls -lh /yourpath/models/Xing4.1-Preview/{config.json,model.safetensors.index.json,tokenizer.json,chat_template.jinja}
```

> [!NOTE]
> 模型的原生上下文长度与服务开放的上下文长度可以不同。一次请求的输入与输出 token 总数受服务端 `--max-model-len` 限制。

## 使用 vLLM-Ascend 镜像部署

### 拉取镜像

在参与部署的宿主机上执行。

```bash
# 如果使用本地镜像仓库，将镜像地址改为实际地址，并同步修改下文的 IMAGE
docker pull quay.io/teleai/vllm-ascend-a3:0.32-tokenizer-cache-xing4
```

### 下载 Xing4.1-Preview 模型

　　可从以下任一平台获取官方权重。

| 平台 | 地址 |
|------|------|
| Hugging Face | [XingChen-AGI/Xing4.1-Preview](https://huggingface.co/XingChen-AGI/Xing4.1-Preview) |
| ModelScope | [XingChen-AGI/Xing4.1-Preview](https://www.modelscope.cn/models/XingChen-AGI/Xing4.1-Preview) |
| Modelers | [XingChen-AGI/Xing4.1-Preview](https://modelers.cn/models/XingChen-AGI/Xing4.1-Preview) |

　　以 ModelScope 为例，在能够访问模型仓库的机器上执行以下命令。目标目录已有完整权重时，可直接使用该目录。

```bash
python3 -m pip install modelscope
# 将 --local_dir 改为实际权重保存目录，部署时将容器内对应目录填入脚本的 weights
modelscope download --model XingChen-AGI/Xing4.1-Preview --local_dir /yourpath/models/Xing4.1-Preview
```

### 启动 vLLM-Ascend 服务

**启动容器**

单机混部在一台宿主机上执行。PD 分离在 P、D 两台宿主机上分别执行。将存放权重的上级目录挂载进容器。

```bash
# IMAGE 使用前面拉取的同一镜像
IMAGE=quay.io/teleai/vllm-ascend-a3:0.32-tokenizer-cache-xing4

# 按本机实际设备编号和驱动挂载路径调整 --device 与 -v 的宿主机路径。
# 容器名冲突或共享内存容量需要调整时，修改 --name 或 --shm-size。
# 将挂载参数中的 /yourpath/models 换成实际存放权重的上级目录。冒号左侧为宿主机路径，右侧为容器路径。
docker run -itd --privileged \
    --name xing41-preview \
    --net=host \
    --shm-size=512g \
    --ulimit nofile=65535:65535 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /root/.cache:/root/.cache \
    -v /yourpath/models:/yourpath/models \
    "$IMAGE" bash
```

#### 单机混部

本例采用 DP2、TP8，最大上下文为 131072 token。进入单机容器后，替换示例 IP、网卡名和权重路径，将整段命令粘贴到 Bash 执行。

```bash
# 示例 IP、网卡名和权重路径需按现场环境替换。
local_ip="192.0.2.10"
nic_name="eth0"
weights="/yourpath/models/Xing4.1-Preview"

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=600
export HCCL_CONNECT_TIMEOUT=600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export USE_MULTI_BLOCK_POOL=1
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ASCEND_BUFFER_POOL=0:0
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export CPU_AFFINITY_CONF=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT DIST_INIT_METHOD 2>/dev/null || true

nohup vllm serve "$weights" \
 --host 0.0.0.0 \
 --port 8011 \
 --data-parallel-size 2 \
 --tensor-parallel-size 8 \
 --enable-expert-parallel \
 --seed 1024 \
 --served-model-name Xing4.1-Preview \
 --max-model-len 131072 \
 --max-num-batched-tokens 8192 \
 --max-num-seqs 64 \
 --trust-remote-code \
 --gpu-memory-utilization 0.90 \
 --no-enable-prefix-caching \
 --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
 --cp-kv-cache-interleave-size 128 \
 --attention_config.indexer_kv_dtype bf16 \
 --kv-cache-dtype bfloat16 \
 --enforce-eager \
 --uvicorn-log-level "error" \
 --reasoning-parser xing4_0 \
 --enable-auto-tool-choice \
 --tool-call-parser xing4_0 \
 --chat-template "$weights/chat_template.jinja" \
 --hf-overrides '{"model_type": "deepseek_v32"}' \
 --tokenizer-mode hf \
 --enable-prompt-tokens-details \
 >> "running.log" 2>&1 </dev/null &
```

服务就绪后，通过单机实际 IP 的 `8011` 端口调用 API。启动日志写入当前目录的 `running.log`。

#### PD 分离部署

**准备部署脚本**

在各自容器内选择可写目录存放部署文件。

- P 容器使用 [Prefill 模板](prefill/run_dp_template.sh)，D 容器使用 [Decode 模板](decode/run_dp_template.sh)。两个模板均以 `run_dp_template.sh` 的文件名保存。
- 将 [`launch_online_dp.py`](launch_online_dp.py) 分别复制到 P、D 容器，与各自的 `run_dp_template.sh` 放在同一目录。
- 将 [`load_balance_proxy_server_example.py`](load_balance_proxy_server_example.py) 复制到 P 容器，从该文件所在目录启动代理。

**部署前替换示例 IP、网卡和路径**

| 示例值 | 部署时填写 | 需要同步替换的位置 |
|---------|------------|--------------------|
| `192.0.2.10` | P 节点实际通信 IP | P 侧脚本的 `local_ip`、P 侧启动命令的 `--dp-address`、代理的 `--host` 和 `--prefiller-hosts`、直连代理的 API URL。 |
| `192.0.2.11` | D 节点实际通信 IP | D 侧脚本的 `local_ip`、D 侧启动命令的 `--dp-address`、代理的 `--decoder-hosts`。 |
| `eth0` | 对应节点上承载 `local_ip` 的网卡 | P、D 各自脚本的 `nic_name`。 |
| `/yourpath/models` | 存放权重的上级目录 | Docker 的 `-v` 参数。冒号左侧填宿主机目录，右侧填容器内挂载目录。 |
| `/yourpath/models/Xing4.1-Preview` | 完整权重目录 | 权重检查和下载命令填写宿主机路径。P、D 脚本中的 `weights` 填写挂载后容器内可见的模型目录。 |

**部署 Prefill（P 节点）**

在 P 容器中，进入 `launch_online_dp.py` 与 `run_dp_template.sh` 所在目录后运行。

```bash
# 将 --dp-address 中的示例 IP 换为 P 侧 run_dp_template.sh 中填写的实际 local_ip。
# --dp-rpc-port 需使用本机空闲端口。若修改 --vllm-start-port，代理端口列表也要同步更新。
# 若调整卡数或 DP/TP，同步核对 DCP、两侧 Mooncake 拓扑和代理后端列表。
python launch_online_dp.py \
    --dp-size 4 \
    --tp-size 4 \
    --dp-size-local 4 \
    --dp-rank-start 0 \
    --dp-address 192.0.2.10 \
    --dp-rpc-port 12345 \
    --vllm-start-port 7100
```

**部署 Decode（D 节点）**

在 D 容器中，进入 `launch_online_dp.py` 与 `run_dp_template.sh` 所在目录后运行。

```bash
# 将 --dp-address 中的示例 IP 换为 D 侧 run_dp_template.sh 中填写的实际 local_ip。
# --dp-rpc-port 需使用本机空闲端口。若修改 --vllm-start-port，代理端口列表也要同步更新。
# 若调整卡数或 DP/TP，同步核对 DCP、两侧 Mooncake 拓扑和代理后端列表。
python launch_online_dp.py \
    --dp-size 8 \
    --tp-size 2 \
    --dp-size-local 8 \
    --dp-rank-start 0 \
    --dp-address 192.0.2.11 \
    --dp-rpc-port 12345 \
    --vllm-start-port 7100
```

**启动 PD 代理**

P、D 后端就绪后，在 P 容器中进入代理脚本所在目录并运行。每个 host 与同一位置的 port 组成一个后端地址。客户端使用 P 节点的 8010 作为统一入口。

```bash
unset http_proxy https_proxy
# 将 --host 和 --prefiller-hosts 中的所有示例 IP 192.0.2.10 换为 P 节点实际 local_ip。
# 将 --decoder-hosts 中的所有示例 IP 192.0.2.11 换为 D 节点实际 local_ip。
# 按两侧启动命令填写后端端口。若修改代理的 --port，API 请求地址中的端口也要同步更新。
nohup python load_balance_proxy_server_example.py \
    --host 192.0.2.10 --port 8010 \
    --prefiller-hosts 192.0.2.10 192.0.2.10 192.0.2.10 192.0.2.10 \
    --prefiller-ports 7100 7101 7102 7103 \
    --decoder-hosts 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 192.0.2.11 \
    --decoder-ports 7100 7101 7102 7103 7104 7105 7106 7107 \
    > proxy.log 2>&1 </dev/null &
```

## 调用 API

在能访问服务的机器上执行。按实际部署方式替换示例命令中的 API 地址。

- 单机混部使用单机实际 IP 和 `8011` 端口，例如 `http://192.0.2.10:8011/v1`。
- PD 分离使用 P 节点实际 IP 和代理端口，示例为 `http://192.0.2.10:8010/v1`。通过网关访问时填写实际网关路由。

### 基础对话（curl）

　　部署模板对外公布的模型名为 `Xing4.1-Preview`。若修改了 `--served-model-name`，请同步替换请求中的 `model`。

```bash
# 将示例地址换成实际服务地址。单机直连使用 8011，PD 分离使用代理端口。
curl -sS --fail-with-body "http://192.0.2.10:8010/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Xing4.1-Preview",
    "messages": [{"role": "user", "content": "请只回复 OK。"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

**验证工具调用**

通过同一个代理入口提交工具定义。成功时，检查 `choices[0].message.tool_calls[0].function` 中的函数名和参数；本例只验证模型生成工具调用，实际执行由客户端负责。

```bash
# 将示例地址换成实际服务地址。单机直连使用 8011，PD 分离使用代理端口。
curl -sS --fail-with-body "http://192.0.2.10:8010/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Xing4.1-Preview",
    "messages": [{"role":"user","content":"请调用 get_weather 查询北京天气。"}],
    "tools": [{"type":"function","function":{"name":"get_weather","description":"查询城市天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256
  }'
```

## 推理参数建议

　　下表为官方模型卡给出的通用推理建议。上面的短请求使用较小的输出上限，以便快速确认服务能返回完整答案。[官方推理参数](https://www.modelscope.cn/models/XingChen-AGI/Xing4.1-Preview)

| 参数 | 建议值 | 说明 |
|------|------|------|
| `temperature` | 0.8 | 调节采样随机性。上面的连通性测试取 0，便于检查短回答。 |
| `top_p` | 1.0 | 按累计概率筛选候选 token，1.0 表示保留完整概率范围。 |
| `max_tokens` | 128K | 单次请求允许生成的最大 token 数，需在服务上下文扣除输入后的预算内设置。 |

## 常见问题

### 显存不足（OOM）

- 启动时若提示 KV Cache 无法容纳配置的上下文，先检查 NPU 的剩余显存，再降低 `--max-model-len`。
- 高并发时出现显存不足，可降低 `--max-num-seqs`；若报错发生在预填充阶段，再检查 `--max-num-batched-tokens`。

### 张量并行卡数不匹配

- 用 `npu-smi info` 核对容器可见设备，检查 `ASCEND_RT_VISIBLE_DEVICES` 与实际分配是否一致。
- 在单节点承载全部本地 DP 副本且 PP 为 1 的配置中，本地 DP 副本数乘以 TP 大小，应等于为该角色分配的 NPU 数。DCP 复用 TP 组内设备，DCP 大小须能整除 TP 大小。

### 容器内无法访问 NPU

- 先在宿主机执行 `npu-smi info`。若宿主机可见、容器不可见，核对 Docker 启动命令中的设备映射、驱动库和 npu-smi 挂载。

### 服务无法从外部访问

- 在代理所在节点先向各 P、D 实例的 `/health` 发请求，再通过代理发送上面的基础对话请求。这两步可区分后端就绪问题和代理转发问题。
- 如果代理直连可以生成回答，只有评测机访问失败，再核对评测机使用的网关路由、上游 IP 和端口。

### 找不到 xing4_0 parser

- 出现 `invalid tool call parser: xing4_0` 时，检查报错容器中运行 vLLM 的 Python 环境是否注册了 `xing4_0` 工具解析器。对话模板和解析器分别负责输入格式化与输出解析，两者都需要与服务配置匹配。
