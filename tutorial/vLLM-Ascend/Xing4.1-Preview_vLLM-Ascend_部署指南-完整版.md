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
- 配置启动参数和推理采样参数

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

**容器配置说明**

| 配置 | 用途与替换说明 |
|------|------|
| `--name` | 两台宿主机均可使用 `xing41-preview` 作为本机容器名。 |
| 镜像及 `bash` | P、D 使用同一镜像，容器启动后运行 Bash。 |
| `-itd` | `-i` 保持标准输入打开，`-t` 分配终端，`-d` 在后台运行。 |
| `--privileged` | 按示例以特权模式运行容器，部署账号需具备相应权限。 |
| `--net=host` | 使用宿主机网络。后端和代理直接监听宿主机端口，部署前须确认端口可用。 |
| `--shm-size=512g` | 设置容器内 `/dev/shm` 的容量上限，供多进程通信使用；这是主机共享内存，按主机内存和并行规模调整。 |
| `--ulimit nofile=65535:65535` | 将容器进程的文件描述符软、硬限制均设为 65535。 |
| `--device /dev/davinci0` 至 `/dev/davinci15` | 每项映射一个计算设备。示例列出 16 项，按本机分配的设备编号增减，并同步调整并行参数。 |
| `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` | 分别提供设备管理、设备内存管理和主机与设备通信接口。 |
| `/usr/local/dcmi` | 将宿主机的昇腾设备管理接口挂入容器。 |
| `hccn_tool`、`npu-smi` | 分别挂载设备网络诊断工具和 NPU 状态查询工具。 |
| `driver/lib64`、`driver/version.info`、`/etc/ascend_install.info` | 分别挂载驱动库、驱动版本和安装信息。命令中的 `:ro` 表示容器只读访问。 |
| `/root/.cache` | 将宿主机缓存映射到容器；需要单独保存该实例的缓存时，可替换冒号左侧的宿主机路径。 |
| `-v /yourpath/models:/yourpath/models` | 挂载存放权重的上级目录。实际模型目录填写在各自部署脚本的 `weights` 中。 |

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

在各自的容器内选用任意可写目录存放脚本。

- [`launch_online_dp.py`（附录 A）](#附录-a)。P、D 两侧分别与对应的 `run_dp_template.sh` 放在同一目录。启动器按本地 rank 分配设备和端口，并调用该目录中的部署脚本。
- [`load_balance_proxy_server_example.py`（附录 B）](#附录-b)。放在 P 容器内，从脚本所在目录启动代理。

**部署前替换示例 IP、网卡和路径**

| 示例值 | 部署时填写 | 需要同步替换的位置 |
|---------|------------|--------------------|
| `192.0.2.10` | P 节点实际通信 IP | P 侧脚本的 `local_ip`、P 侧启动命令的 `--dp-address`、代理的 `--host` 和 `--prefiller-hosts`、直连代理的 API URL。 |
| `192.0.2.11` | D 节点实际通信 IP | D 侧脚本的 `local_ip`、D 侧启动命令的 `--dp-address`、代理的 `--decoder-hosts`。 |
| `eth0` | 对应节点上承载 `local_ip` 的网卡 | P、D 各自脚本的 `nic_name`。 |
| `/yourpath/models` | 存放权重的上级目录 | Docker 的 `-v` 参数。冒号左侧填宿主机目录，右侧填容器内挂载目录。 |
| `/yourpath/models/Xing4.1-Preview` | 完整权重目录 | 权重检查和下载命令填写宿主机路径。P、D 脚本中的 `weights` 填写挂载后容器内可见的模型目录。 |

交付镜像内置 `xing4_0` 工具调用解析器。

| 项目 | Prefill 节点 | Decode 节点 |
|------|--------------|-------------|
| 每机并行配置 | DP4 × TP4，DCP4 | DP8 × TP2，PCP1、DCP2 |
| 本地 rank 与设备 | rank 0–3，每个副本使用 4 个连续设备 | rank 0–7，每个副本使用 2 个连续设备 |
| 后端 HTTP 端口 | 7100–7103 | 7100–7107 |
| 本机 DP RPC 端口 | 12345 | 12345 |
| Mooncake 角色与配置的 KV 端口 | `kv_producer`，30000 | `kv_consumer`，30300 |
| 上下文 / 每轮调度 token 预算 / 序列数上限 | 262144 / 8192 / 64 | 262144 / 512 / 64 |
| HBM 预算与执行模式 | 0.90，eager | 0.95，异步调度，纯解码图捕获 |

**部署 Prefill（P 节点）**

部署脚本 `run_dp_template.sh`

```bash
#!/usr/bin/env bash
set -e

# 192.0.2.10 为示例 IP。替换为 P 节点实际通信 IP，并同步修改下方的 --dp-address。
local_ip="192.0.2.10"
# eth0 为示例网卡名。填写本机承载 local_ip 的实际网卡名。
nic_name="eth0"
# /yourpath/models/Xing4.1-Preview 为示例路径。填写挂载后容器内直接包含 config.json 和 chat_template.jinja 的模型目录。
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
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=0:0
export CPU_AFFINITY_CONF=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V2_MODEL_RUNNER=1
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT DIST_INIT_METHOD 2>/dev/null || true

# 若改 --served-model-name，两侧脚本和 API 请求的 model 应一致。
# 若 KV 传输端口冲突，在本节点 --kv-transfer-config 中调整 kv_port。
nohup vllm serve "$weights" \
 --host 0.0.0.0 \
 --port $2 \
 --data-parallel-size $3 \
 --data-parallel-rank $4 \
 --data-parallel-address $5 \
 --data-parallel-rpc-port $6 \
 --tensor-parallel-size $7 \
 --decode-context-parallel-size 4 \
 --enable-expert-parallel \
 --seed 1024 \
 --served-model-name Xing4.1-Preview \
 --max-model-len 262144 \
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
 --additional-config '{"tokenizer_cache_gb": 1, "enable_cpu_binding": true, "enable_flashcomm1": true, "enable_mlapo": true, "multistream_overlap_shared_expert":true, "enable_shared_expert_dp": true,"weight_nz_mode":2}' \
 --uvicorn-log-level "error" \
 --reasoning-parser xing4_0 \
 --enable-auto-tool-choice \
 --tool-call-parser xing4_0 \
 --chat-template "$weights/chat_template.jinja" \
 --tokenizer-mode hf \
 --enable-prompt-tokens-details \
 --hf-overrides '{"model_type": "deepseek_v32"}' \
 --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":8,"tp_size":2}}}' \
 >> "rank-$4-port-$2.log" 2>&1 </dev/null &
```

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

部署脚本 `run_dp_template.sh`

```bash
#!/usr/bin/env bash
set -e

# 192.0.2.11 为示例 IP。替换为 D 节点实际通信 IP，并同步修改下方的 --dp-address。
local_ip="192.0.2.11"
# eth0 为示例网卡名。填写本机承载 local_ip 的实际网卡名。
nic_name="eth0"
# /yourpath/models/Xing4.1-Preview 为示例路径。填写挂载后容器内直接包含 config.json 和 chat_template.jinja 的模型目录。
weights="/yourpath/models/Xing4.1-Preview"

export HCCL_IF_IP="$local_ip"
export GLOO_SOCKET_IFNAME="$nic_name"
export TP_SOCKET_IFNAME="$nic_name"
export HCCL_SOCKET_IFNAME="$nic_name"
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=600
export HCCL_CONNECT_TIMEOUT=600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=2048
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=$1
export ASCEND_BUFFER_POOL=0:0
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:/usr/local/lib:$LD_LIBRARY_PATH
export CPU_AFFINITY_CONF=1
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_FASTOKENS=1
export VLLM_USE_V2_MODEL_RUNNER=1
unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT DIST_INIT_METHOD 2>/dev/null || true

# 若改 --served-model-name，两侧脚本和 API 请求的 model 应一致。
# 若 KV 传输端口冲突，在本节点 --kv-transfer-config 中调整 kv_port。
nohup vllm serve "$weights" \
 --host 0.0.0.0 \
 --port $2 \
 --data-parallel-size $3 \
 --data-parallel-rank $4 \
 --data-parallel-address $5 \
 --data-parallel-rpc-port $6 \
 --tensor-parallel-size $7 \
 --enable-expert-parallel \
 --seed 1024 \
 --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
 --served-model-name Xing4.1-Preview \
 --max-model-len 262144 \
 --max-num-batched-tokens 512 \
 --max-num-seqs 64 \
 --trust-remote-code \
 --gpu-memory-utilization 0.95 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --prefill-context-parallel-size 1 \
 --decode-context-parallel-size 2 \
 --cp-kv-cache-interleave-size 128 \
 --attention_config.indexer_kv_dtype bf16 \
 --kv-cache-dtype bfloat16 \
 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64]}' \
 --additional-config '{"tokenizer_cache_gb": 1, "scheduler_config": {"recompute_scheduler_enable": true}, "finegrained_tp_config": {"lmhead_tensor_parallel_size": 4, "embedding_tensor_parallel_size": 4}, "enable_cpu_binding": true, "enable_flashcomm1": true, "enable_mlapo":true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true}, "multistream_overlap_shared_expert": true, "enable_shared_expert_dp": true, "weight_nz_mode":2}' \
 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
 --uvicorn-log-level "error" \
 --reasoning-parser xing4_0 \
 --enable-auto-tool-choice \
 --tool-call-parser xing4_0 \
 --chat-template "$weights/chat_template.jinja" \
 --tokenizer-mode hf \
 --enable-prompt-tokens-details \
 --hf-overrides '{"model_type": "deepseek_v32"}' \
 --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer","kv_port":"30300","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":8,"tp_size":2}}}' \
 >> "rank-$4-port-$2.log" 2>&1 </dev/null &
```

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

### 启动参数说明

| 参数 | 说明 |
|------|------|
| `vllm serve <模型目录>` | 从指定目录加载权重和模型配置。P、D 使用同一版本。 |
| `--host`、`--port` | 设置单个 vLLM 实例的监听地址和端口。客户端请求发往 PD 代理的统一入口。 |
| `--served-model-name` | 设置 API 对外公布的模型名。调用时将该值填入请求的 `model` 字段。 |
| `--tensor-parallel-size`、`--data-parallel-size` | 分别设置每个副本的张量并行规模和数据并行副本数。 |
| `--decode-context-parallel-size`、`--cp-kv-cache-interleave-size` | 设置解码上下文并行规模和 KV Cache 在上下文并行组内的交错粒度。 |
| `--max-model-len` | 单次请求允许的最大上下文长度，包含输入和输出 token。 |
| `--max-num-seqs`、`--max-num-batched-tokens` | 分别限制一次调度可处理的序列数和 token 数。 |
| `--gpu-memory-utilization` | 控制 vLLM 可以使用的设备显存比例。在昇腾后端也沿用此参数名。 |
| `--kv-transfer-config` | 配置 P、D 之间的 KV Cache 传输。P 侧为 `kv_producer`，D 侧为 `kv_consumer`。 |
| `--hf-overrides` | 将模型配置的 `model_type` 设置为 `deepseek_v32`。 |
| `--trust-remote-code` | 允许加载权重目录随附的模型代码。 |
| `--chat-template` | 指定与权重配套的 `chat_template.jinja`，用于格式化对话和工具描述。 |
| `--enable-auto-tool-choice`、`--tool-call-parser xing4_0` | 允许模型自动选择工具，并将生成的工具调用解析为 OpenAI 兼容的 `tool_calls` 字段。 |

**两侧模板中的运行配置**

| 配置 | 本方案中的作用 |
|------|---------------|
| `--dp-size-local`、`--dp-rank-start`、`--vllm-start-port` | 设置本机副本数、起始 rank 和第一个 HTTP 端口；本例各角色都从 rank 0 和端口 7100 开始。 |
| `--enable-expert-parallel`、`enable_shared_expert_dp`、`multistream_overlap_shared_expert` | 启用专家并行、共享专家数据并行和共享专家多流重叠。 |
| `--prefill-context-parallel-size`、`--kv-cache-dtype`、`indexer_kv_dtype` | D 侧显式设置 PCP1；两端 KV Cache 和索引缓存均使用 BF16，交错粒度均为 128。 |
| `--enforce-eager`、`--async-scheduling`、`--compilation-config` | P 侧使用 eager；D 侧启用异步调度，捕获 1、2、4、8、16、24、32、48、64 这些规模的解码图。 |
| `enable_npugraph_ex`、`enable_static_kernel` | 在 D 侧图模式中启用 NPU 图扩展与静态内核。 |
| `recompute_scheduler_enable`、`finegrained_tp_config` | D 侧启用重计算调度，并为输出头与词嵌入设置独立的并行规模。 |
| `enable_cpu_binding`、`enable_flashcomm1`、`enable_mlapo`、`weight_nz_mode` | 两侧均启用 CPU 绑定、FlashComm1 和 MLA 优化，权重 NZ 布局模式均取 2。 |
| `--no-enable-prefix-caching`、`tokenizer_cache_gb`、`VLLM_USE_FASTOKENS` | 两端关闭本地前缀缓存，各设 1 GB 分词器缓存。D 模板额外启用 FASTOKENS。 |
| `--model-loader-extra-config`、`--seed`、`--profiler-config` | 两端各设 128 个权重加载线程、随机种子 1024 和 Torch profiler 输出目录。 |
| `--tokenizer-mode hf`、`--enable-prompt-tokens-details`、`--uvicorn-log-level` | 使用 Hugging Face 分词器，返回提示词用量明细，将 HTTP 服务日志级别设为 error。 |
| `HCCL_IF_IP`、`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` | 使用脚本顶部的 `local_ip` 和 `nic_name`。D 侧还显式绑定 HCCL socket 网卡。 |
| `VLLM_RPC_TIMEOUT`、`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`、`HCCL_EXEC_TIMEOUT`、`HCCL_CONNECT_TIMEOUT` | 分别控制 vLLM RPC、模型执行、HCCL 算子执行和建链超时。 |
| `HCCL_BUFFSIZE`、`HCCL_OP_EXPANSION_MODE` | 通信缓冲区 P 侧取 1024、D 侧取 2048，两侧均使用 AIV 展开模式。 |
| `OMP_PROC_BIND`、`OMP_NUM_THREADS`、`CPU_AFFINITY_CONF` | 两端配置 OpenMP 线程绑定策略、每进程线程数和昇腾 CPU 亲和性策略。 |
| `ASCEND_RT_VISIBLE_DEVICES`、`ASCEND_BUFFER_POOL`、`USE_MULTI_BLOCK_POOL`、`PYTORCH_NPU_ALLOC_CONF` | 控制每个 rank 的可见设备、缓冲池及 NPU 内存分配；P 侧显式启用多块内存池。 |
| `TASK_QUEUE_ENABLE`、`ASCEND_AGGREGATE_ENABLE`、`ASCEND_TRANSPORT_PRINT`、`ACL_OP_INIT_MODE` | 沿用模板的运行时任务队列、聚合、传输日志和算子初始化设置。 |
| `VLLM_USE_V2_MODEL_RUNNER`、`VLLM_ALLOW_LONG_MAX_MODEL_LEN`、`VLLM_LOGGING_LEVEL` | 两端使用 V2 模型执行器，允许配置长上下文，并设置引擎日志级别。 |
| `LD_LIBRARY_PATH`、`unset LOCAL_RANK …` | D 模板补充 Mooncake 动态库搜索目录。启动前清理继承的分布式变量，由当前启动器指定 rank 和组网地址。 |

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

## 附录

### 附录 A

`launch_online_dp.py`

```python
import argparse
import multiprocessing
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default=12345, help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=9000, help="Starting port for the engine.")
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port

def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        "./run_dp_template.sh",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    template_path = "./run_dp_template.sh"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(i * tp_size, (i + 1) * tp_size))
        process = multiprocessing.Process(target=run_command, args=(visible_devices, dp_rank, vllm_engine_port))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

### 附录 B

`load_balance_proxy_server_example.py`

```python
# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import functools
import heapq
import ipaddress
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.aborted_requests = set()  # Track aborted requests
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.node_listener = NodeListener(self)

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()
        # Removed selection locks - no longer needed for synchronous methods

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, server_index, server_reference)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = [(0.0, i, server) for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0.0, i, server) for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)

    def _update_prefiller_priority(self, server_idx: int):
        """Update the priority of a prefiller server in the heap."""
        server = self.prefillers[server_idx]
        # Priority based on active_tokens and active_kv_cache
        priority = server.active_tokens + server.active_kv_cache * 0.3
        # Remove old entry and add new one
        self.prefiller_heap = [(p, i, s) for p, i, s in self.prefiller_heap if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        """Update the priority of a decoder server in the heap."""
        server = self.decoders[server_idx]
        priority = server.active_tokens
        # Remove old entry and add new one
        self.decoder_heap = [(p, i, s) for p, i, s in self.decoder_heap if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))

    def abort_prefiller_request(self, server_idx: int, request_id):  # Changed to synchronous
        """
        Mark a request as aborted. This will helps to release kv cache in
        prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return
        self.prefillers[server_idx].aborted_requests.add(request_id)

    def acquire_aborted_prefiller_requests(self, server_idx: int):  # Changed to synchronous
        """
        Get the set of aborted requests and clear it.
        This is used to release kv cache in prefiller node.
        """
        # No lock needed - atomic operation
        if server_idx >= len(self.prefillers):
            return set()
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.prefiller_heap:
            raise RuntimeError("No prefiller servers available")

        priority, chosen, server = heapq.heappop(self.prefiller_heap)

        # Update the chosen server atomically
        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count

        # Update priority and re-add to heap
        self._update_prefiller_priority(chosen)

        return chosen

    def release_prefiller(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        self.prefillers[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.prefillers):
            return
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        # Update priority queue after releasing
        self._update_prefiller_priority(idx)

    def select_decoder(self, token_count):  # Changed to synchronous
        # No lock needed - entire function is atomic
        if not self.decoder_heap:
            raise RuntimeError("No decoder servers available")

        priority, chosen, server = heapq.heappop(self.decoder_heap)

        # Update the chosen server atomically
        self.decoders[chosen].active_tokens += token_count

        # Update priority and re-add to heap
        self._update_decoder_priority(chosen)

        return chosen

    def release_decoder(self, idx, token_count):  # Changed to synchronous
        # No lock needed - atomic operation
        if idx >= len(self.decoders):
            return
        self.decoders[idx].active_tokens -= token_count
        # Update priority queue after releasing
        self._update_decoder_priority(idx)

    # Omni_infer's calculate_input_scores function
    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score

    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.waiting_nodes[node] = (instance_type, server, 0)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_prefillers:
                self.tainted_prefillers.remove(server)
                self.prefiller_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.prefiller_heap
                ]
                heapq.heapify(self.prefiller_heap)
            elif server not in self.prefillers:
                self.prefillers.append(server)
                # prefiller_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.prefiller_heap, (0, len(self.prefillers) - 1, server))
        self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState]) -> None:
        for server in instances:
            if server in self.tainted_decoders:
                self.tainted_decoders.remove(server)
                self.decoder_heap = [
                    (0, idx, server) if srv == server else (priority, idx, srv)
                    for priority, idx, srv in self.decoder_heap
                ]
                heapq.heapify(self.decoder_heap)
            elif server not in self.decoders:
                self.decoders.append(server)
                # decoder_heap: [(priority_0, 0, server_0)] -> [(priority_0, 0, server_0), (0, 1, server_1)]
                heapq.heappush(self.decoder_heap, (0, len(self.decoders) - 1, server))
        self.print_status(f"Add decoder instances: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint prefill instances {instances}.")
            self._taint_prefillers(instances)
            return True

        instances_to_remove = set(instances)
        self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
        prefiller_heap_copy = self.prefiller_heap.copy()
        prefiller_heap_copy.sort(key=lambda x: x[1])  # sorted by key: prefiller_idx
        prefiller_heap = []
        idx = 0
        for priority, _, server in prefiller_heap_copy:
            if server not in instances_to_remove:
                prefiller_heap.append((priority, idx, server))
                idx += 1

        # prefiller_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.prefiller_heap = prefiller_heap
        heapq.heapify(self.prefiller_heap)
        self.print_status(f"Remove prefiller instances: {instances}.")
        return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        if self.request_num > 0:
            logger.warning(f"Start to taint decode instances {instances}.")
            self._taint_decoders(instances)
            return True

        instances_to_remove = set(instances)
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        decoder_heap_copy = self.decoder_heap.copy()
        decoder_heap_copy.sort(key=lambda x: x[1])  # sorted by key: decoder_idx
        decoder_heap = []
        idx = 0
        for priority, _, server in decoder_heap_copy:
            if server not in instances_to_remove:
                decoder_heap.append((priority, idx, server))
                idx += 1

        # decoder_heap: [(priority_0, 0, server_0), (priority_1, 1, server_1)] -> [(priority_1, 0, server_1)]
        self.decoder_heap = decoder_heap
        heapq.heapify(self.decoder_heap)
        self.print_status(f"Remove decoder instances: {instances}.")
        return False

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        self.prefiller_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.prefiller_heap
        ]
        heapq.heapify(self.prefiller_heap)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        self.decoder_heap = [
            (TAINT_PRIORITY, idx, srv) if srv in instances_to_taint else (priority, idx, srv)
            for priority, idx, srv in self.decoder_heap
        ]
        heapq.heapify(self.decoder_heap)

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state = None


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            for node, (instance_type, server, check_times) in list(self.waiting_nodes.items()):
                is_valid = asyncio.run(self.check_instance_status(server.client))
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server])
                    self.waiting_nodes.pop(node)
                elif check_times == global_args.max_waiting_retries:
                    print(f"Instance {node} was not added to the proxy.")
                    self.waiting_nodes.pop(node)
                else:
                    self.waiting_nodes[node] = (instance_type, server, check_times)

            if self.proxy_state.tainted_prefillers and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_prefillers(self.proxy_state.tainted_prefillers)
                if not need_waiting:
                    self.proxy_state.tainted_prefillers.clear()

            if self.proxy_state.tainted_decoders and not self.proxy_state.request_num:
                need_waiting = self.proxy_state.remove_decoders(self.proxy_state.tainted_decoders)
                if not need_waiting:
                    self.proxy_state.tainted_decoders.clear()
            time.sleep(global_args.waiting_retry_interval)

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/models"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        try:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    """Return if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller_id: int,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, json=req_data, headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return  # Success, exit after streaming
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise e
        except Exception as e:
            # If any chunk has been sent, do not retry, just log and drop
            if "first_chunk_sent" in locals() and first_chunk_sent:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise e


async def _handle_select_instance(api: str, req_data: Any, request_length: int):
    prefiller_score = proxy_state.calculate_prefill_scores(request_length)
    logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
    request_id = await proxy_state.next_req_id()
    # Select prefiller
    prefiller_idx = proxy_state.select_prefiller(prefiller_score)
    prefiller = proxy_state.prefillers[prefiller_idx]
    # Send request to prefiller
    response = await send_request_to_service(
        prefiller.client,
        prefiller_idx,
        api,
        req_data,
        request_id,
        max_retries=global_args.max_retries,
        base_delay=global_args.retry_delay,
    )
    proxy_state.release_prefiller(prefiller_idx, prefiller_score)
    response_json = response.json()
    kv_transfer_params = response_json.get("kv_transfer_params", {})
    if kv_transfer_params:
        req_data["kv_transfer_params"] = kv_transfer_params
    # Select decoder
    decoder_score = proxy_state.calculate_decode_scores(request_length)
    logger.debug("Decoder score: %f", decoder_score)
    # Use the prefiller's kv_transfer_params to select decoder
    decoder_idx = proxy_state.select_decoder(decoder_score)
    decoder = proxy_state.decoders[decoder_idx]
    logger.debug("Using %s %s", prefiller.url, decoder.url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_idx=prefiller_idx,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder=decoder,
        decoder_idx=decoder_idx,
        decoder_score=decoder_score,
    )


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_idx: int
    prefiller_score: float
    prefiller: ServerState
    decoder_idx: int
    decoder_score: float
    decoder: ServerState


async def _handle_completions(api: str, request: Request):
    try:
        proxy_state.request_num += 1
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        instance_info = await _handle_select_instance(api, req_data, request_length)
        stream_flag = bool(req_data.get("stream", False))
        chat_flag = "messages" in req_data

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        elif chat_flag:
            messages = req_data["messages"]
            origin_prompt = messages[0].get("content", "")
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = req_data.get("max_tokens", 16)

        async def generate_stream():
            nonlocal instance_info
            generated_token = ""
            released_kv = False
            retry_count = 0
            retry = True
            completion_tokens = 0
            # Only one await per chunk, minimal logic in loop
            try:
                while retry:
                    retry = False
                    async for chunk in stream_service_response_with_retry(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            yield chunk
                            continue
                        if not chunk_str:
                            continue
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: ") :]
                        try:
                            chunk_json = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            # if chunk is [done], skip it.
                            logger.debug(f"Skipping chunk: {chunk_str}")
                            yield chunk
                            continue
                        choices = chunk_json.get("choices", [])
                        if not choices:
                            yield chunk
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        message = choice.get("message") or {}
                        content = delta.get("content") or message.get("content") or choice.get("text") or ""
                        generated_token += content

                        stop_reason = choice.get("stop_reason")
                        usage = chunk_json.get("usage", {})
                        completion_tokens = (
                            (completion_tokens + 1)
                            if stream_flag
                            else (completion_tokens + usage.get("completion_tokens"))
                        )
                        if stop_reason == "recomputed":
                            retry = True
                            retry_count += 1
                            if chat_flag:
                                messages[0]["content"] = origin_prompt + generated_token
                            else:
                                req_data["prompt"] = origin_prompt + generated_token
                            req_data["max_tokens"] = origin_max_tokens - completion_tokens + retry_count
                            tmp_request_length = len(json.dumps(req_data).encode("utf-8"))
                            instance_info = await _handle_select_instance(api, req_data, tmp_request_length)
                            break
                        if retry_count > 0 and not stream_flag:
                            if chat_flag:
                                choice["message"]["content"] = generated_token
                            else:
                                choice["text"] = generated_token
                            chunk = json.dumps(chunk_json).encode("utf-8")
                        yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                    f"the aborted request {instance_info.request_id} will be routing to the target "
                    "prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(instance_info.prefiller_idx, instance_info.request_id)
                proxy_state.release_prefiller_kv(instance_info.prefiller_idx, instance_info.prefiller_score)

            # After streaming done, release tokens
            proxy_state.release_decoder(instance_info.decoder_idx, instance_info.decoder_score)

        # Determine the correct media type based on stream flag
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        return StreamingResponse(generate_stream(), media_type=media_type)
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        proxy_state.request_num -= 1


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    try:
        req_data = await request.json()
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise e


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
    }


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
```
