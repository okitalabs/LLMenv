# LLM実行環境の構築
# llama-cpp-python
llama-cpp-pythonは、量子化GGUF形式のLLMモデルを実行するためのランタイムllama-cppをPythonから使えるようにしたバインディング。  
同梱している`llama_cpp.server`を使用すると、OpenAI API互換サーバーとして、実行することが出来る（llama-cppのHTTP Serverとは別物）。  
複数のモデルを定義しておくと、リクエストの処理時にmodel名によって、実行モデルを切り替えることが出来る。


## 構成情報
### LLMモデル
|model名|量子化|対象モデル|
|:----|:----|:----|
|vicuna-13b|q8_0|[TheBloke/vicuna-13B-v1.5-GGUF](https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF) |
|karakuri-8x7b|q6_K|[mmnga/karakuri-lm-8x7b-chat-v0.1-gguf](https://huggingface.co/mmnga/karakuri-lm-8x7b-chat-v0.1-gguf)|


### Docker設定
|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|llamacpp|20080|20080|/home/llm/llamacpp|/llamacpp|


<br>
<hr>


# 構築手順
## LLMモデルファイルのダウンロード

llama-cpp-python用のモデルファイル(GGUF)を`$HOME/llamacpp/model/`にダウンロードしておく。

```bash
$ mkdir $HOME/llamacpp/model ## モデル用ディレクトリ作成
$ cd $HOME/llamacpp/model

## vicuna-13b-v1.5.Q8_0.gguf
$ wget https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF/resolve/main/vicuna-13b-v1.5.Q8_0.gguf

## karakuri-lm-8x7b-chat-v0.1-Q6_K.gguf
$ wget https://huggingface.co/mmnga/karakuri-lm-8x7b-chat-v0.1-gguf/resolve/main/karakuri-lm-8x7b-chat-v0.1-Q6_K.gguf
```


<hr>


## Dockerイメージの作成
[abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)のGitHubにある、[Dockerfile](https://github.com/abetlen/llama-cpp-python/blob/main/docker/cuda_simple/Dockerfile)から実行用のコンテナイメージ `llamacpp`を作成する。
```bash
$ cd $HOME/llamacpp ## llamma-cppディレクトリに移動
$ git clone https://github.com/abetlen/llama-cpp-python.git
$ cd llama-cpp-python/docker/cuda_simple/ ## CUDA板Dockerimage
$ docker build -t llamacpp . ## ビルド
```


### Dockerイメージの確認
```bash
$ docker images	
 REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
llamacpp     latest    f7da52f907e2   9 seconds ago   8.21GB
```


<hr>


## llama-cpp-pythonサーバの起動
### configファイルの作成

以下のファイルを作成する。  
`$HOME/llamacpp/config.json`  
```json
{
  "host": "0.0.0.0",
  "port": 20080,
  "models": [
    {
      "model": "/llamacpp/model/vicuna-13b-v1.5.Q8_0.gguf",
      "model_alias": "vicuna-13b",
      "chat_format": "vicuna",
      "n_gpu_layers": -1,
      "n_ctx": 4096
    },
    {
      "model": "/llamacpp/model/karakuri-lm-8x7b-chat-v0.1-Q6_K.gguf",
      "model_alias": "karakuri-8x7b",
      "chat_format": "llama-2",
      "n_gpu_layers": -1,
      "n_ctx": 4096
    }
  ]
}
```

設定パラメータの詳細は末尾の「参考」を参照。
- model  
    モデルファイルを指定
- model_alias  
    OpenAI APIの`model`で指定される名前。`gpt-3.5-turbo`や`text-davinch-003`のように指定することも出来る。
- chat_format  
    モデルのPrompt Templateの指定。`llama_chat_format.py`に定義されている必要がある。
- n_gpu_layers  
    GPUが使用するlayers数。`-1`にすると全て割り当てる。  
    モデルが使用しているlayers数は、起動時に`llm_load_tensors: offloaded 41/41 layers to GPU`の形で表示されるので確認できる。
- n_ctx  
    最大コンテキスト長 (default: 2048)


### 起動
llama_cpp.serverは、使用するGPUを指定できないっぽい（全GPUを使ってしまう）ので、複数GPUを搭載している場合、Dockerの`--gpu`で使用するGPUを制限すること。
```bash
$ docker run --rm -p 20080:20080 --gpus device=0 --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v /home/llm/llamacpp:/llamacpp -h llamacpp --name llamacpp llamacpp python3 -m llama_cpp.server --config_file /llamacpp/config.json
```

起動後、フォアグラウンドで実行。以下メッセージが出たら起動完了。  
`CTRL+C`で終了、Dockerもコンテナも削除される(`--rm`オプション)。
```bash
 :
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:20080 (Press CTRL+C to quit)
```


<hr>


## 起動後の確認
### Chat completionの確認
curlで`vicuna-13b`にChat APIで問い合わせてみる。
```bash
$ time curl http://localhost:20080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "vicuna-13b",
  "templature": 0.9,
  "top_p": 1.0,
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}' | jq
```

回答例
> `jq`コマンドを併用すると、jsonがフォーマットされて出力される。
```json
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1270  100   929  100   341    145     53  0:00:06  0:00:06 --:--:--   222
{
  "id": "chatcmpl-aef316e8-2b0e-4d2f-82a1-8baa3b62f808",
  "object": "chat.completion",
  "created": 1715497858,
  "model": "vicuna-13b",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": " ランダムに選んだ都道府県は「岐阜県」です。\n\n1. 岐阜県の魅力の一つは、美しい自然が残されていることです。県内には多くの山や森があり、その中には日本の名水「長良川」も流れています。\n2. 岐阜県は食文化も有名です。代表的なものとして「おだわり」や「かつお節」があり、多くの観光客に人気があります。\n3. 岐阜県は歴史にもあふれています。例えば、美濃国分寺や長良川鉄道沿線の小さな町並みなど、歴史を感じられる場所がたくさんあります。",
        "role": "assistant"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 92,
    "completion_tokens": 253,
    "total_tokens": 345
  }
}

real	0m6.383s
user	0m0.024s
sys	0m0.001s
```


### モデルを変えて問い合わせる
curlで`"model": "karakuri-8x7b"`に変えると、処理をするモデルが変更される。この時、Docker側では、モデルの入れ替えメッセージが表示される。  
初回の読み込みはローディングに時間がかかるが、2回目以降はキャッシュされるので、かなり高速でで切り替えられる。
```bash
$ time curl http://localhost:20080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "karakuri-8x7b",
  "templature": 0.9,
  "top_p": 1.0,
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}' | jq
```


### Embeddings
curlで`vicuna-13b`にEmbeddingsを問い合わせてみる。
```bash
time curl http://localhost:20080/v1/embeddings \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "vicuna-13b",
  "input": "query: 夕飯はお肉です。"
}' | jq |less
```
回答例
```bash
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        [
          0.1092359870672226,
                 :
                省略
                 :
          -0.8064247965812683
        ]
      ],
      "index": 0
    }
  ],
  "model": "vicuna-13b",
  "usage": {
    "prompt_tokens": 18,
    "total_tokens": 18
  }
}
```


<br>
<hr>

## 参考

### chat_format
`--chat_fomat`で指定可能な値は、llama_chat_format.pyを@register_chat_formatでgrepして確認できる。  
```
$ grep @register_chat_format ~/llamacpp/llama-cpp-python/llama_cpp/llama_chat_format.py
---
@register_chat_format("llama-2")
@register_chat_format("llama-3")
@register_chat_format("alpaca")
@register_chat_format("qwen")
@register_chat_format("vicuna")
@register_chat_format("oasst_llama")
@register_chat_format("baichuan-2")
@register_chat_format("baichuan")
@register_chat_format("openbuddy")
@register_chat_format("redpajama-incite")
@register_chat_format("snoozy")
@register_chat_format("phind")
@register_chat_format("intel")
@register_chat_format("open-orca")
@register_chat_format("mistrallite")
@register_chat_format("zephyr")
@register_chat_format("pygmalion")
@register_chat_format("chatml")
@register_chat_format("mistral-instruct")
@register_chat_format("chatglm3")
@register_chat_format("openchat")
@register_chat_format("saiga")
@register_chat_format("gemma")
````

### 起動オプション
```
usage: __main__.py [-h] [--model MODEL] [--model_alias MODEL_ALIAS] [--n_gpu_layers N_GPU_LAYERS]
                   [--split_mode SPLIT_MODE] [--main_gpu MAIN_GPU]
                   [--tensor_split [TENSOR_SPLIT ...]] [--vocab_only VOCAB_ONLY]
                   [--use_mmap USE_MMAP] [--use_mlock USE_MLOCK]
                   [--kv_overrides [KV_OVERRIDES ...]] [--seed SEED] [--n_ctx N_CTX]
                   [--n_batch N_BATCH] [--n_threads N_THREADS] [--n_threads_batch N_THREADS_BATCH]
                   [--rope_scaling_type ROPE_SCALING_TYPE] [--rope_freq_base ROPE_FREQ_BASE]
                   [--rope_freq_scale ROPE_FREQ_SCALE] [--yarn_ext_factor YARN_EXT_FACTOR]
                   [--yarn_attn_factor YARN_ATTN_FACTOR] [--yarn_beta_fast YARN_BETA_FAST]
                   [--yarn_beta_slow YARN_BETA_SLOW] [--yarn_orig_ctx YARN_ORIG_CTX]
                   [--mul_mat_q MUL_MAT_Q] [--logits_all LOGITS_ALL] [--embedding EMBEDDING]
                   [--offload_kqv OFFLOAD_KQV] [--flash_attn FLASH_ATTN]
                   [--last_n_tokens_size LAST_N_TOKENS_SIZE] [--lora_base LORA_BASE]
                   [--lora_path LORA_PATH] [--numa NUMA] [--chat_format CHAT_FORMAT]
                   [--clip_model_path CLIP_MODEL_PATH] [--cache CACHE] [--cache_type CACHE_TYPE]
                   [--cache_size CACHE_SIZE] [--hf_tokenizer_config_path HF_TOKENIZER_CONFIG_PATH]
                   [--hf_pretrained_model_name_or_path HF_PRETRAINED_MODEL_NAME_OR_PATH]
                   [--hf_model_repo_id HF_MODEL_REPO_ID] [--draft_model DRAFT_MODEL]
                   [--draft_model_num_pred_tokens DRAFT_MODEL_NUM_PRED_TOKENS] [--type_k TYPE_K]
                   [--type_v TYPE_V] [--verbose VERBOSE] [--host HOST] [--port PORT]
                   [--ssl_keyfile SSL_KEYFILE] [--ssl_certfile SSL_CERTFILE] [--api_key API_KEY]
                   [--interrupt_requests INTERRUPT_REQUESTS]
                   [--disable_ping_events DISABLE_PING_EVENTS] [--root_path ROOT_PATH]
                   [--config_file CONFIG_FILE]

🦙 Llama.cpp python server. Host your own LLMs!🚀

options:
  -h, --help            show this help message and exit
  --model MODEL         The path to the model to use for generating completions.
  --model_alias MODEL_ALIAS
                        The alias of the model to use for generating completions.
  --n_gpu_layers N_GPU_LAYERS
                        The number of layers to put on the GPU. The rest will be on the CPU. Set
                        -1 to move all to GPU.
  --split_mode SPLIT_MODE
                        The split mode to use. (default: 1)
  --main_gpu MAIN_GPU   Main GPU to use.
  --tensor_split [TENSOR_SPLIT ...]
                        Split layers across multiple GPUs in proportion.
  --vocab_only VOCAB_ONLY
                        Whether to only return the vocabulary.
  --use_mmap USE_MMAP   Use mmap. (default: True)
  --use_mlock USE_MLOCK
                        Use mlock. (default: True)
  --kv_overrides [KV_OVERRIDES ...]
                        List of model kv overrides in the format key=type:value where type is one
                        of (bool, int, float). Valid true values are (true, TRUE, 1), otherwise
                        false.
  --seed SEED           Random seed. -1 for random. (default: 4294967295)
  --n_ctx N_CTX         The context size. (default: 2048)
  --n_batch N_BATCH     The batch size to use per eval. (default: 512)
  --n_threads N_THREADS
                        The number of threads to use. Use -1 for max cpu threads (default: 24)
  --n_threads_batch N_THREADS_BATCH
                        The number of threads to use when batch processing. Use -1 for max cpu
                        threads (default: 48)
  --rope_scaling_type ROPE_SCALING_TYPE
  --rope_freq_base ROPE_FREQ_BASE
                        RoPE base frequency
  --rope_freq_scale ROPE_FREQ_SCALE
                        RoPE frequency scaling factor
  --yarn_ext_factor YARN_EXT_FACTOR
  --yarn_attn_factor YARN_ATTN_FACTOR
  --yarn_beta_fast YARN_BETA_FAST
  --yarn_beta_slow YARN_BETA_SLOW
  --yarn_orig_ctx YARN_ORIG_CTX
  --mul_mat_q MUL_MAT_Q
                        if true, use experimental mul_mat_q kernels (default: True)
  --logits_all LOGITS_ALL
                        Whether to return logits. (default: True)
  --embedding EMBEDDING
                        Whether to use embeddings. (default: True)
  --offload_kqv OFFLOAD_KQV
                        Whether to offload kqv to the GPU. (default: True)
  --flash_attn FLASH_ATTN
                        Whether to use flash attention.
  --last_n_tokens_size LAST_N_TOKENS_SIZE
                        Last n tokens to keep for repeat penalty calculation. (default: 64)
  --lora_base LORA_BASE
                        Optional path to base model, useful if using a quantized base model and
                        you want to apply LoRA to an f16 model.
  --lora_path LORA_PATH
                        Path to a LoRA file to apply to the model.
  --numa NUMA           Enable NUMA support.
  --chat_format CHAT_FORMAT
                        Chat format to use.
  --clip_model_path CLIP_MODEL_PATH
                        Path to a CLIP model to use for multi-modal chat completion.
  --cache CACHE         Use a cache to reduce processing times for evaluated prompts.
  --cache_type CACHE_TYPE
                        The type of cache to use. Only used if cache is True. (default: ram)
  --cache_size CACHE_SIZE
                        The size of the cache in bytes. Only used if cache is True. (default:
                        2147483648)
  --hf_tokenizer_config_path HF_TOKENIZER_CONFIG_PATH
                        The path to a HuggingFace tokenizer_config.json file.
  --hf_pretrained_model_name_or_path HF_PRETRAINED_MODEL_NAME_OR_PATH
                        The model name or path to a pretrained HuggingFace tokenizer model. Same
                        as you would pass to AutoTokenizer.from_pretrained().
  --hf_model_repo_id HF_MODEL_REPO_ID
                        The model repo id to use for the HuggingFace tokenizer model.
  --draft_model DRAFT_MODEL
                        Method to use for speculative decoding. One of (prompt-lookup-decoding).
  --draft_model_num_pred_tokens DRAFT_MODEL_NUM_PRED_TOKENS
                        Number of tokens to predict using the draft model. (default: 10)
  --type_k TYPE_K       Type of the key cache quantization.
  --type_v TYPE_V       Type of the value cache quantization.
  --verbose VERBOSE     Whether to print debug information. (default: True)
  --host HOST           Listen address (default: localhost)
  --port PORT           Listen port (default: 8000)
  --ssl_keyfile SSL_KEYFILE
                        SSL key file for HTTPS
  --ssl_certfile SSL_CERTFILE
                        SSL certificate file for HTTPS
  --api_key API_KEY     API key for authentication. If set all requests need to be authenticated.
  --interrupt_requests INTERRUPT_REQUESTS
                        Whether to interrupt requests when a new request is received. (default:
                        True)
  --disable_ping_events DISABLE_PING_EVENTS
                        Disable EventSource pings (may be needed for some clients).
  --root_path ROOT_PATH
                        The root path for the server. Useful when running behind a reverse proxy.
  --config_file CONFIG_FILE
                        Path to a config file to load.
```

### リンク
- [OpenAI Compatible Server](https://llama-cpp-python.readthedocs.io/en/latest/server/)  


<hr>

LLM実行委員会