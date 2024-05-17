# LLM実行環境の構築
# LocalAI + llama-cpp

LocalAI経由でllama-cppを起動して、GGUFのモデルを実行するための手順。llama-cppのContinuous Batchによる同時処理が可能。


## 構成情報
### 変換モデル情報

あらかじめ、以下のモデルを`$HOME/vllm/model`に配置しておく。
- vicuna-13b-v1.5.Q8_0.gguf  
- karakuri-lm-8x7b-chat-v0.1-Q6_K.gguf
- karakuri-lm-70b-chat-v0.1-q4_K_M.gguf

### Docker設定

Docker `vllm`は以下の設定を使用する。

|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|vllm|30080|30080|/home/llm/vllm|/vllm|


<br>
<hr>


# 構築手順
## Dockerイメージの作成
[mudler/localai](https://github.com/mudler/LocalAI)のGitHubにある、[Dockerfile](https://github.com/mudler/LocalAI/blob/master/Dockerfile)から実行用のコンテナイメージ `localai`を作成する。

```bash
$ cd $HOME/localai 
$ git clone https://github.com/mudler/LocalAI.git
$ cd LocalAI
$ vi .env
```

llama-cppがGPUを使用し、並列処理が可能になるように`.env`を変更しビルドする。  各項目の概要は`.env`のコメントを参照。

`$HOME/LocalAI/.env`
```bash
LOCALAI_THREADS=8
LOCALAI_LOG_LEVEL=debug
LOCALAI_SINGLE_ACTIVE_BACKEND=true
BUILD_TYPE=cublas
PYTHON_GRPC_MAX_WORKERS=8
LLAMACPP_PARALLEL=8
LOCALAI_PARALLEL_REQUESTS=true
```
- BUILD_TYPE: GPU(cublas)を使う
- LOCALAI_SINGLE_ACTIVE_BACKEND: trueだと複数モデルを同時に起動しない（都度入れ替える）
- LOCALAI_PARALLEL_REQUESTS: Continuous Batch可能


### Dockerビルド
`--build-arg`で設定を指定することも可能。
```
$ docker build --build-arg BUILD_TYPE=cublas -t localai .
```

<br>
<hr>


## localaiサーバの設定
yamlの設定項目は、[Advanced configuration with YAML files](https://localai.io/advanced/)を参照。ただし、あまり詳しくは書いていないため、Prompt Template等は、末尾のリンクを参考に想像するしか無い💦

以下は、vicuna-13b、karakuri-8x7b、karakuri-70bの設定。  
`$HOME/config.yaml`
```yaml
- name: vicuna-13b
  backend: llama-cpp
  embeddings: true
  context_size: 4095
  f16: true
  gpu_layers: 41
  mmap: true
  parameters:
    model: vicuna-13b-v1.5.Q8_0.gguf
    temperature: 0.7
    top_k: 40
    top_p: 0.7
    seed: -1
  template:
    chat: |
      A chat between a human and an assistant.

      ### Human: 
      {{.Input }}
      ### Assistant: 

- name: karakuri-8x7b
  backend: llama-cpp
  embeddings: true
  context_size: 4095
  f16: true
  gpu_layers: 33
  mmap: true
  mmlock: true
  batch: 512
  parameters:
    model: karakuri-lm-8x7b-chat-v0.1-Q6_K.gguf
    temperature: 0.7
    top_k: 40
    top_p: 0.7
    seed: -1
  template:
    chat: -
      [INST] {{.Input}} [/INST]
    completion: -
      [INST] {{.Input}} [/INST]

- name: karakuri-70b
  backend: llama-cpp
  embeddings: true
  context_size: 4095
  f16: true
  gpu_layers: 81
  mmap: true
  mmlock: true
  batch: 1024
  parameters:
    model: karakuri-lm-70b-chat-v0.1-q4_K_M.gguf
    temperature: 0.7
    top_k: 40
    top_p: 0.7
    seed: -1
  template:
    chat: |
      {{if eq .RoleName "assistant"}}{{.Content}}{{else}}
      [INST]
      {{if .SystemPrompt}}{{.SystemPrompt}}{{else if eq .RoleName "system"}}<<SYS>>{{.Content}}<</SYS>>

      {{else if .Content}}{{.Content}}{{end}}
      [/INST]
      {{end}}
```

### LocalAIサーバの起動
`-e`で設定を変えることが出来る。  
`LOCALAI_SINGLE_ACTIVE_BACKEND=true`だと、GPUメモリ不足を避けるために、実行時にモデルを入れ替える。`false`にすると、複数のモデルが同時に起動される。  
ログの表示を抑えたい場合、`LOCALAI_LOG_LEVEL=info`にする。
```bash
docker run --rm -p 40080:40080 --gpus all -v /home/llm/localai:/localai \
-e LOCALAI_LOG_LEVEL=debug \
-e LOCALAI_SINGLE_ACTIVE_BACKEND=true \
-e LOCALAI_PARALLEL_REQUESTS=true \
-h localai --name localai \
localai run  \
--config-file /localai/config.yaml \
--models-path /localai//model \
--address="0.0.0.0:40080"
```

<hr>

## 実行の確認
curlでmodel名を`vicuna-13b`、`karakuri-8x7b`、`karakuri-70b`に変えて実行してみる。この時サーバ側ではモデルの入れ替えが起きる。    
また、同じモデルに複数同時問い合わせを行って、ほぼ同時にレスポンスが返るか確認する(Continuous Batchによる同時処理)。  
違うモデルに同時に問い合わせた場合、どちらかが`{"error":{"code":500,"message":"could not load model: rpc error: code = Canceled desc = ","type":""}}`のエラーになった。
```bash
$ time curl http://localhost:40080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "vicuna-13b",
  "templature": 0.1,
  "top_p": 0.1,
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}'
```


<hr>

## 課題
LocalAIを使うことにより、llama-cppを使ったGGUF形式のContinuous Batchによる同時処理や複数モデルの利用がOpenAI API形式で利用可能になる。  
しかし、回答が途中で切れたり、最後に特殊記号が入ってしまう問題が多発するため、実用性は厳しい。これらはllama-cpp-pythonのサーバでは発生しないため、何らかの原因があると考えられるが、資料が少ないため原因の解明には至っていない。もう少しこなれるまで、待つしかないか。。。



<br>
<hr>


## 参考
### リンク
- [Metaの「Llama 3」をOpenAI API互換のサーバーを持つllama-cpp-pythonとLocalAIで試す](https://kazuhira-r.hatenablog.com/entry/2024/04/26/001435)  
- [Advanced configuration with YAML](https://localai.io/advanced/)  
- [BreadcrumbsLocalAI/embedded/models/](https://localai.io/advanced/)  
- [LocalAI/examples/configurations/](https://github.com/mudler/LocalAI/tree/master/examples/configurations)  
- [model-gallery/llama2-7b-chat-gguf.yaml](https://github.com/go-skynet/model-gallery/blob/main/llama2-7b-chat-gguf.yaml)  
- [prompt-templates/llama2-chat-message.tmpl](https://github.com/mudler/LocalAI/blob/master/prompt-templates/llama2-chat-message.tmpl)  
- [prompt-templates/vicuna.tmpl](https://github.com/mudler/LocalAI/blob/master/prompt-templates/vicuna.tmpl)  



<hr>

LLM実行委員会