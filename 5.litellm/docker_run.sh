#!/bin/bash
docker run --rm \
--add-host=host.docker.internal:host-gateway \
-v /home/llm/litellm:/litellm \
-p 10080:10080 \
-h litellm --name litellm \
ghcr.io/berriai/litellm:main-latest \
--config /litellm/config.yaml --detailed_debug --port 10080 --