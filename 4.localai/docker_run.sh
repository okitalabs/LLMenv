docker run --rm -p 40080:40080 --gpus all -v /home/llm/localai:/localai -h localai --name localai localai run \
--config-file /localai/config.yaml \
--models-path /localai/model \
--address="0.0.0.0:40080"