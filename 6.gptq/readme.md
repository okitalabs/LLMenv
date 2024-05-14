# LLM実行環境の構築
# GPTQ変換

Vicuna-13b-1.5の16bitモデルをGPTQ 8bitに変換する手順。  HuggingFaceに登録されているGPTQのほとんどが4bitのため、8bitで利用したい場合、自前で変換する必要がある。  

ここでは、vLLMのDocker Imageを利用し、GPTQ変換に必要な環境を構築しvLLMで実行する。



## 構成情報
### 変換モデル情報
|model名|ランタイム|変換後モデル名|
|:----|:----|:----|
|[lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)|GPTQ 8bit|vicuna-13b-v1.5-gptq_8bit|


### Docker設定
|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|vllm_dev|30080|30080|/home/llm/vllm|/vllm|
||8888|38888|Juptter Port|


<br>
<hr>


# 構築手順
## LLMモデルファイルのダウンロード
あらかじめ、変換元のモデルファイルをダウンロードしておく。

### ダウンロードプログラム
`dl_vicuna-13b.py`
```python
from huggingface_hub import snapshot_download, login
# login(token = "Toke ID") ## 認証が必要な場合

model_name = "https://huggingface.co/lmsys/vicuna-13b-v1.5" ## ダウンロードするHuggingFaceのモデル名
save_name = "/home/llm/vllm/model/vicuna-13b-v1.5" ## ダウンロード先のディレクトリ

download_path = snapshot_download(
    repo_id = model_name,
    local_dir = save_name,
    local_dir_use_symlinks=False
)
```

ダウンロードの実行
```bash
$ # mkdir $HOME/vllm/model ## モデル用ディレクトリ作成
$ cd $HOME/vllm/ ## ここにdl_vicuna-13b.pyを配置
$ pip install huggingface_hub ## pythonモジュールのインストール
$ python dl_vicuna-13b.py
```


<br>
<hr>

## vLLM GPTQサーバの起動
### Docker起動
`--entrypoint ""`でvLLMを起動せず、`/bin/bash`を起動する。  
Port `3888:8888`はJupyterを使う場合。
```bash
$ docker run --rm -p 30080:30080 -p 38888:8888 --gpus all -it --entrypoint "" -v /home/llm/vllm:/vllm -h vllm_dev --name vllm_dev vllm /bin/bash
```
起動後、Shellになるのでここで作業する。
```bash
root@vllm:/vllm-workspace#
```
さらにコンテナにログインしたい場合、以下のコマンドでDockerコンテナに入る。
```
$ docker exec -it vllm_dev /bin/bash
root@vllm:/vllm-workspace#
```

### ライブラリのインストール
GPTQ変換に必要。
```
root@vllm:/vllm-workspace# cd /vllm ## 作業ディレクトリに移動しておく
root@vllm:/vllm# pip install pip install pip install auto-gptq 
```
Jupyterを使う場合に必要。  
Jupyterの起動後、Hostから`http://localhost:38888`、Token`llm`で接続。
```
root@vllm:/vllm# pip install pip install ipywidgets iprogress jupyterLab
root@vllm:/vllm# jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token="llm"
```

### 変換の実行
以下のPythonプログラムをコマンドまたはJupyterから実行する。  

GPTQConfigの`bits=`で量子化ビット数を、  `dataset=`でキャリブレーション用のデータセットを指定する。 'wikitext2','c4','c4-new','ptb','ptb-new'の5種類から選ぶことが推奨されている。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_name = "/vllm/model/vicuna-13b-v1.5" ## 量子化するモデル
save_file = "/vllm/model/vicuna-13b-gptq-8bit" ## 量子化済みのモデル保存先
tokenizer = AutoTokenizer.from_pretrained(model_name)

## GPTQ設定
gptq_config = GPTQConfig(
bits=8, ## 8bitを指定
dataset='c4-new', ## キャリブレーションデータセット 'wikitext2','c4','c4-new','ptb','ptb-new'
tokenizer=tokenizer,
use_exllama=False,
cache_examples_on_gpu=False,
use_cuda_fp16=True
)

## GPUメモリ節約
my_device_map = {'model.embed_tokens': 'cpu', 'model.layers': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}

## 量子化
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=my_device_map,
torch_dtype=torch.float16, quantization_config=gptq_config)

## 指定しないとエラーになる
model.generation_config.temperature=None
model.generation_config.top_p=None

## safetensors形式で保存
model.to('cpu')
model.save_pretrained(save_file, safe_serialization=True)
tokenizer.save_pretrained(save_file)
```

### 変換モデルの確認
以下のファイルが作成される。
```
root@vllm:/vllm# ls /vllm/model/vicuna-13b-gptq-8bit/
config.json                       model-00003-of-00003.safetensors  tokenizer.model
generation_config.json            model.safetensors.index.json      tokenizer_config.json
model-00001-of-00003.safetensors  special_tokens_map.json
model-00002-of-00003.safetensors  tokenizer.json
```


<br>
<hr>


## vLLMサーバの起動
`vicuna-13b-gptq-8bit`をモデル名`vicuna-13b`としてvLLMで起動し、動作の確認をする。


### プロンプトテンプレートの作成
vLLMはvicuna形式のプロンプトをサポートしていないため、別途、jinja形式のファイルを定義して指定する必要がある。

`/vllm/vicuna.jinja`
```jinja
{% for message in messages %}
    {% if (message['role'] == 'system') %}
        {{ 'A chat between a human and an assistant. ' + message['content'] }}
    {% elif (message['role'] == 'user') %}
        {{ '### Human: ' + message['content'] }}
    {% elif (message['role'] == 'assistant') %}
        {{ '### Assistant: ' + message['content'] }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
### Assistant:
{% endif %}
```

### vLLMの起動
`-chat-template`でプロンプトテンプレートファイルを指定する。
```
root@vllm:/vllm# python3 -m vllm.entrypoints.openai.api_server \
--model /vllm/model/vicuna-13b-gptq-8bit \
--served-model-name vicuna-13b \
--quantization gptq \
--max-model-len 4095 \
--gpu-memory-utilization 0.9 \
--host 0.0.0.0 \
--port 30080 \
--chat-template /vllm/vicuna.jinja
```

Hostから確認。
```
$ time curl http://localhost:30080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "vicuna-13b",
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}'
```



<br>
<hr>


## 参考
### リンク
- [HugginFace Quantize 🤗 Transformers models](https://huggingface.co/docs/transformers/ja/main_classes/quantization)
- [【ローカルLLM】Hugging FaceによるGPTQ量子化ガイド](https://note.com/bakushu/n/n6c560265b994)
- [NVIDIA RTX3060(12GB)でLLMを試す：GPTQ量子化](https://zenn.dev/to2watt/articles/e98cbb5c3231ab)
- [キャリブレーションデータにもっと気を配ろうの話](https://note.com/sakusakumura/n/n7d7abca9b2e4)
- [transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py)


<hr>

LLM実行委員会