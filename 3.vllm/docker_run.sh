#!/bin/bash
docker run --rm -p 30080:30080 --gpus all -v /home/llm/vllm:/vllm -h vllm --name vllm vllm \
--model /vllm/model/karakuri-lm-70b-chat-v0.1-AWQ \
--served-model-name karakuri-70b \
--quantization awq \
--max-model-len 4095 \
--gpu-memory-utilization 0.9 \
--host 0.0.0.0 \
--port 30080