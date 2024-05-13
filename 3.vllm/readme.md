# LLM実行環境の構築
# llama-cpp-python
vLLMはHuggingFaceモデルの16bit、及び量子化GPTQ、AWQ形式のLLMモデルを実行するためのランタイム。
同梱している`vllm.entrypoints.openai.api_server`を使用すると、OpenAI API互換サーバーとして、実行することが出来る。  
`Continous Batching`による複数リクエストの並列処理が可能。  

[vLLM Documentationページ](https://docs.vllm.ai/en/latest/index.html)



## 構成情報
### LLMモデル
|model名|量子化|対象モデル|
|:----|:----|:----|
|karakuri-70b|4bit AWQ|[masao1211/karakuri-lm-70b-chat-v0.1-AWQ](https://huggingface.co/masao1211/karakuri-lm-70b-chat-v0.1-AWQ) |



### Docker設定
|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|vllm|30080|30080|/home/llm/vllm|/vllm|



<br>
<hr>


# 構築手順
## LLMモデルファイルのダウンロード
vllm用のモデルファイル(AWQ)を`$HOME/vllm/model/`にダウンロードしておく。
モデルはディレクトリで構成されるため、以下のプログラムを使用してHuggingFaceからダウンロードする。

### ダウンロードプログラム
`dl_karakuri-70b.py`
```python
from huggingface_hub import snapshot_download, login
# login(token = "Toke ID") ## 認証が必要な場合

model_name = "masao1211/karakuri-lm-70b-chat-v0.1-AWQ" ## ダウンロードするHuggingFaceのモデル名
save_name = "/home/llm/vllm/model/karakuri-lm-70b-chat-v0.1-AWQ" ## ダウンロード先のディレクトリ

download_path = snapshot_download(
    repo_id = model_name,
    local_dir = save_name,
    local_dir_use_symlinks=False
)
```
ダウンロードの実行
```bash
$ mkdir $HOME/vllm/model ## モデル用ディレクトリ作成
$ cd $HOME/vllm/ ## ここにdl_karakuri-70b.pyを配置
$ pip install huggingface_hub ## pythonモジュールのインストール
$ python dl_karakuri-70b.py
```



<hr>


## Dockerイメージの作成
[vllm-project/vllm](https://github.com/vllm-project/vllm)のGitHubにある、[Dockerfile](https://github.com/vllm-project/vllm/blob/main/Dockerfile)から実行用のコンテナイメージ `vllm`を作成する。
```bash
$ cd $HOME/vllm ## vllmディレクトリに移動
$ git clone https://github.com/vllm-project/vllm 
$ cd vllm
$ docker build -t vllm . ## ビルド
```


### Dockerイメージの確認
```bash
$ docker images	
 REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
llamacpp 
vllm                      latest        7d4aab0485ae   3 days ago   8.29GB
```


<hr>


## vLLMサーバの起動
### configファイルの作成
configファイルは無し

### 起動
```bash
$ docker run --rm -p 30080:30080 --gpus all -v /home/llm/vllm:/vllm -h vllm --name vllm vllm \
--model /vllm/model/karakuri-lm-70b-chat-v0.1-AWQ \
--served-model-name karakuri-70b \
--quantization awq \
--max-model-len 4095 \
--gpu-memory-utilization 0.9 \
--host 0.0.0.0 \
--port 30080
```
起動後、フォアグラウンドで実行。以下メッセージが出たら起動完了。  
`CTRL+C`で終了、Dockerもコンテナも削除される(`--rm`オプション)。
```bash
 :
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:30080 (Press CTRL+C to quit)
INFO 05-12 12:11:25 metrics.py:334] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%
```

起動オプションの詳細は末尾の「参考」を参照  
- --served-model-name  
    OpenAI APIの`model`で指定される名前。`gpt-3.5-turbo`や`text-davinch-003`のように指定することも出来る。
- --quantization  
    量子化方式（GPTQ、AWQ）
- --max-model-len  
    コンテキスト長、4096だとエラーになるため少な目に
- --gpu-memory-utilization  
    使用するGPUのメモリ量の割合（デフォルト0.9で90%確保）



### Prompt Template
モデルファイルに定義されているchat templateを使用する。  
独自に定義したい場合、以下のような`.jinja`ファイルで定義し、` --chat-template`オプションで指定する。

#### vicuna.jinjaの例
```jinja
{% for message in messages %}
    {% if (message['role'] == 'system') %}
        {{ 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + message['content'] }}
    {% elif (message['role'] == 'user') %}
        {{ '### Instruction: ' + message['content'] }}
    {% elif (message['role'] == 'assistant') %}
        {{ '### Response: ' + message['content'] }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
### Assistant:
{% endif %}
```


<hr>


## 起動後の確認
### Chat completionの確認
curlで`karakuri-70b`にChat APIで問い合わせてみる。
```bash
time curl http://localhost:30080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "karakuri-70b",
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}' | jq

```
> 注意： APIパラメータの、"templature"、"top_p"が使用できない。


### Embeddings
> 注意： Embeddings APIは使用できない。

<br>
<hr>

## 参考
### リンク
- [vLLM Documentationページ](https://docs.vllm.ai/en/latest/index.html)
- [jinja template](https://github.com/vllm-project/vllm/tree/main/examples)
- [Jinjaテンプレートの書き方をがっつり調べてまとめてみた](https://qiita.com/simonritchie/items/cc2021ac6860e92de25d)

### 起動オプション
```
usage: api_server.py [-h] [--host HOST] [--port PORT]
                     [--uvicorn-log-level {debug,info,warning,error,critical,trace}]
                     [--allow-credentials] [--allowed-origins ALLOWED_ORIGINS]
                     [--allowed-methods ALLOWED_METHODS] [--allowed-headers ALLOWED_HEADERS]
                     [--api-key API_KEY] [--lora-modules LORA_MODULES [LORA_MODULES ...]]
                     [--chat-template CHAT_TEMPLATE] [--response-role RESPONSE_ROLE]
                     [--ssl-keyfile SSL_KEYFILE] [--ssl-certfile SSL_CERTFILE]
                     [--ssl-ca-certs SSL_CA_CERTS] [--ssl-cert-reqs SSL_CERT_REQS]
                     [--root-path ROOT_PATH] [--middleware MIDDLEWARE] [--model MODEL]
                     [--tokenizer TOKENIZER] [--skip-tokenizer-init] [--revision REVISION]
                     [--code-revision CODE_REVISION] [--tokenizer-revision TOKENIZER_REVISION]
                     [--tokenizer-mode {auto,slow}] [--trust-remote-code]
                     [--download-dir DOWNLOAD_DIR]
                     [--load-format {auto,pt,safetensors,npcache,dummy,tensorizer}]
                     [--dtype {auto,half,float16,bfloat16,float,float32}]
                     [--kv-cache-dtype {auto,fp8}]
                     [--quantization-param-path QUANTIZATION_PARAM_PATH]
                     [--max-model-len MAX_MODEL_LEN]
                     [--guided-decoding-backend {outlines,lm-format-enforcer}] [--worker-use-ray]
                     [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
                     [--tensor-parallel-size TENSOR_PARALLEL_SIZE]
                     [--max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS]
                     [--ray-workers-use-nsight] [--block-size {8,16,32}] [--enable-prefix-caching]
                     [--use-v2-block-manager] [--num-lookahead-slots NUM_LOOKAHEAD_SLOTS]
                     [--seed SEED] [--swap-space SWAP_SPACE]
                     [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
                     [--num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE]
                     [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS]
                     [--max-num-seqs MAX_NUM_SEQS] [--max-logprobs MAX_LOGPROBS]
                     [--disable-log-stats]
                     [--quantization {aqlm,awq,fp8,gptq,squeezellm,gptq_marlin,marlin,None}]
                     [--enforce-eager] [--max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE]
                     [--max-seq_len-to-capture MAX_SEQ_LEN_TO_CAPTURE]
                     [--disable-custom-all-reduce] [--tokenizer-pool-size TOKENIZER_POOL_SIZE]
                     [--tokenizer-pool-type TOKENIZER_POOL_TYPE]
                     [--tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG] [--enable-lora]
                     [--max-loras MAX_LORAS] [--max-lora-rank MAX_LORA_RANK]
                     [--lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE]
                     [--lora-dtype {auto,float16,bfloat16,float32}]
                     [--max-cpu-loras MAX_CPU_LORAS] [--fully-sharded-loras]
                     [--device {auto,cuda,neuron,cpu}]
                     [--image-input-type {pixel_values,image_features}]
                     [--image-token-id IMAGE_TOKEN_ID] [--image-input-shape IMAGE_INPUT_SHAPE]
                     [--image-feature-size IMAGE_FEATURE_SIZE]
                     [--scheduler-delay-factor SCHEDULER_DELAY_FACTOR] [--enable-chunked-prefill]
                     [--speculative-model SPECULATIVE_MODEL]
                     [--num-speculative-tokens NUM_SPECULATIVE_TOKENS]
                     [--speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN]
                     [--speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE]
                     [--ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX]
                     [--ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN]
                     [--model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG]
                     [--served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]]
                     [--engine-use-ray] [--disable-log-requests] [--max-log-len MAX_LOG_LEN]

vLLM OpenAI-Compatible RESTful API server.

options:
  -h, --help            show this help message and exit
  --host HOST           host name
  --port PORT           port number
  --uvicorn-log-level {debug,info,warning,error,critical,trace}
                        log level for uvicorn
  --allow-credentials   allow credentials
  --allowed-origins ALLOWED_ORIGINS
                        allowed origins
  --allowed-methods ALLOWED_METHODS
                        allowed methods
  --allowed-headers ALLOWED_HEADERS
                        allowed headers
  --api-key API_KEY     If provided, the server will require this key to be presented in the
                        header.
  --lora-modules LORA_MODULES [LORA_MODULES ...]
                        LoRA module configurations in the format name=path. Multiple modules can
                        be specified.
  --chat-template CHAT_TEMPLATE
                        The file path to the chat template, or the template in single-line form
                        for the specified model
  --response-role RESPONSE_ROLE
                        The role name to return if `request.add_generation_prompt=true`.
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl module's)
  --root-path ROOT_PATH
                        FastAPI root_path when app is behind a path based routing proxy
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We accept multiple
                        --middleware arguments. The value should be an import path. If a function
                        is provided, vLLM will add it to the server using @app.middleware('http').
                        If a class is provided, vLLM will add it to the server using
                        app.add_middleware().
  --model MODEL         Name or path of the huggingface model to use.
  --tokenizer TOKENIZER
                        Name or path of the huggingface tokenizer to use.
  --skip-tokenizer-init
                        Skip initialization of tokenizer and detokenizer
  --revision REVISION   The specific model version to use. It can be a branch name, a tag name, or
                        a commit id. If unspecified, will use the default version.
  --code-revision CODE_REVISION
                        The specific revision to use for the model code on Hugging Face Hub. It
                        can be a branch name, a tag name, or a commit id. If unspecified, will use
                        the default version.
  --tokenizer-revision TOKENIZER_REVISION
                        The specific tokenizer version to use. It can be a branch name, a tag
                        name, or a commit id. If unspecified, will use the default version.
  --tokenizer-mode {auto,slow}
                        The tokenizer mode. * "auto" will use the fast tokenizer if available. *
                        "slow" will always use the slow tokenizer.
  --trust-remote-code   Trust remote code from huggingface.
  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights, default to the default cache
                        dir of huggingface.
  --load-format {auto,pt,safetensors,npcache,dummy,tensorizer}
                        The format of the model weights to load. * "auto" will try to load the
                        weights in the safetensors format and fall back to the pytorch bin format
                        if safetensors format is not available. * "pt" will load the weights in
                        the pytorch bin format. * "safetensors" will load the weights in the
                        safetensors format. * "npcache" will load the weights in pytorch format
                        and store a numpy cache to speed up the loading. * "dummy" will initialize
                        the weights with random values, which is mainly for profiling. *
                        "tensorizer" will load the weights using tensorizer from CoreWeave which
                        assumes tensorizer_uri is set to the location of the serialized weights.
  --dtype {auto,half,float16,bfloat16,float,float32}
                        Data type for model weights and activations. * "auto" will use FP16
                        precision for FP32 and FP16 models, and BF16 precision for BF16 models. *
                        "half" for FP16. Recommended for AWQ quantization. * "float16" is the same
                        as "half". * "bfloat16" for a balance between precision and range. *
                        "float" is shorthand for FP32 precision. * "float32" for FP32 precision.
  --kv-cache-dtype {auto,fp8}
                        Data type for kv cache storage. If "auto", will use model data type.
                        FP8_E5M2 (without scaling) is only supported on cuda version greater than
                        11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for common
                        inference criteria.
  --quantization-param-path QUANTIZATION_PARAM_PATH
                        Path to the JSON file containing the KV cache scaling factors. This should
                        generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache
                        scaling factors default to 1.0, which may cause accuracy issues. FP8_E5M2
                        (without scaling) is only supported on cuda versiongreater than 11.8. On
                        ROCm (AMD GPU), FP8_E4M3 is instead supported for common inference
                        criteria.
  --max-model-len MAX_MODEL_LEN
                        Model context length. If unspecified, will be automatically derived from
                        the model config.
  --guided-decoding-backend {outlines,lm-format-enforcer}
                        Which engine will be used for guided decoding (JSON schema / regex etc) by
                        default. Currently support https://github.com/outlines-dev/outlines and
                        https://github.com/noamgat/lm-format-enforcer. Can be overridden per
                        request via guided_decoding_backend parameter.
  --worker-use-ray      Use Ray for distributed serving, will be automatically set when using more
                        than 1 GPU.
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Number of pipeline stages.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas.
  --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
                        Load model sequentially in multiple batches, to avoid RAM OOM when using
                        tensor parallel and large models.
  --ray-workers-use-nsight
                        If specified, use nsight to profile Ray workers.
  --block-size {8,16,32}
                        Token block size for contiguous chunks of tokens.
  --enable-prefix-caching
                        Enables automatic prefix caching.
  --use-v2-block-manager
                        Use BlockSpaceMangerV2.
  --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
                        Experimental scheduling config necessary for speculative decoding. This
                        will be replaced by speculative config in the future; it is present to
                        enable correctness tests until then.
  --seed SEED           Random seed for operations.
  --swap-space SWAP_SPACE
                        CPU swap space size (GiB) per GPU.
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model executor, which can
                        range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
                        utilization. If unspecified, will use the default value of 0.9.
  --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
                        If specified, ignore GPU profiling result and use this numberof GPU
                        blocks. Used for testing preemption.
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
                        Maximum number of batched tokens per iteration.
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences per iteration.
  --max-logprobs MAX_LOGPROBS
                        Max number of log probs to return logprobs is specified in SamplingParams.
  --disable-log-stats   Disable logging statistics.
  --quantization {aqlm,awq,fp8,gptq,squeezellm,gptq_marlin,marlin,None}, -q {aqlm,awq,fp8,gptq,squeezellm,gptq_marlin,marlin,None}
                        Method used to quantize the weights. If None, we first check the
                        `quantization_config` attribute in the model config file. If that is None,
                        we assume the model weights are not quantized and use `dtype` to determine
                        the data type of the weights.
  --enforce-eager       Always use eager-mode PyTorch. If False, will use eager mode and CUDA
                        graph in hybrid for maximal performance and flexibility.
  --max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE
                        Maximum context length covered by CUDA graphs. When a sequence has context
                        length larger than this, we fall back to eager mode. (DEPRECATED. Use
                        --max-seq_len-to-capture instead)
  --max-seq_len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence length covered by CUDA graphs. When a sequence has
                        context length larger than this, we fall back to eager mode.
  --disable-custom-all-reduce
                        See ParallelConfig.
  --tokenizer-pool-size TOKENIZER_POOL_SIZE
                        Size of tokenizer pool to use for asynchronous tokenization. If 0, will
                        use synchronous tokenization.
  --tokenizer-pool-type TOKENIZER_POOL_TYPE
                        Type of tokenizer pool to use for asynchronous tokenization. Ignored if
                        tokenizer_pool_size is 0.
  --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
                        Extra config for tokenizer pool. This should be a JSON string that will be
                        parsed into a dictionary. Ignored if tokenizer_pool_size is 0.
  --enable-lora         If True, enable handling of LoRA adapters.
  --max-loras MAX_LORAS
                        Max number of LoRAs in a single batch.
  --max-lora-rank MAX_LORA_RANK
                        Max LoRA rank.
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Maximum size of extra vocabulary that can be present in a LoRA adapter
                        (added to the base model vocabulary).
  --lora-dtype {auto,float16,bfloat16,float32}
                        Data type for LoRA. If auto, will default to base model dtype.
  --max-cpu-loras MAX_CPU_LORAS
                        Maximum number of LoRAs to store in CPU memory. Must be >= than
                        max_num_seqs. Defaults to max_num_seqs.
  --fully-sharded-loras
                        By default, only half of the LoRA computation is sharded with tensor
                        parallelism. Enabling this will use the fully sharded layers. At high
                        sequence length, max rank or tensor parallel size, this is likely faster.
  --device {auto,cuda,neuron,cpu}
                        Device type for vLLM execution.
  --image-input-type {pixel_values,image_features}
                        The image input type passed into vLLM. Should be one of "pixel_values" or
                        "image_features".
  --image-token-id IMAGE_TOKEN_ID
                        Input id for image token.
  --image-input-shape IMAGE_INPUT_SHAPE
                        The biggest image input shape (worst for memory footprint) given an input
                        type. Only used for vLLM's profile_run.
  --image-feature-size IMAGE_FEATURE_SIZE
                        The image feature size along the context dimension.
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
                        Apply a delay (of delay factor multiplied by previousprompt latency)
                        before scheduling next prompt.
  --enable-chunked-prefill
                        If set, the prefill requests can be chunked based on the
                        max_num_batched_tokens.
  --speculative-model SPECULATIVE_MODEL
                        The name of the draft model to be used in speculative decoding.
  --num-speculative-tokens NUM_SPECULATIVE_TOKENS
                        The number of speculative tokens to sample from the draft model in
                        speculative decoding.
  --speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN
                        The maximum sequence length supported by the draft model. Sequences over
                        this length will skip speculation.
  --speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE
                        Disable speculative decoding for new incoming requests if the number of
                        enqueue requests is larger than this value.
  --ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX
                        Max size of window for ngram prompt lookup in speculative decoding.
  --ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN
                        Min size of window for ngram prompt lookup in speculative decoding.
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to the model loader
                        corresponding to the chosen load_format. This should be a JSON string that
                        will be parsed into a dictionary.
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API. If multiple names are provided, the
                        server will respond to any of the provided names. The model name in the
                        model field of a response will be the first name in this list. If not
                        specified, the model name will be the same as the `--model` argument.
                        Noted that this name(s)will also be used in `model_name` tag content of
                        prometheus metrics, if multiple names provided, metricstag will take the
                        first one.
  --engine-use-ray      Use Ray to start the LLM engine in a separate process as the server
                        process.
  --disable-log-requests
                        Disable logging requests.
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers being printed in log.
                        Default: Unlimited
```

<hr>

LLM実行委員会