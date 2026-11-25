#!/usr/bin/env bash
set -e

# 192.0.2.11 为示例 IP。替换为 D 节点实际通信 IP，并同步修改指南中对应节点启动命令的 --dp-address。
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
