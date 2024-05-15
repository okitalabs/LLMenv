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
root@vllm:/vllm# pip install auto-gptq 
```
Jupyterを使う場合に必要。  
Jupyterの起動後、Hostから`http://localhost:38888`、Token`llm`で接続。
```
root@vllm:/vllm# pip install ipywidgets iprogress jupyterLab
root@vllm:/vllm# jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token="llm"
```

### 変換の実行
以下のPythonプログラムをコマンドまたはJupyterから実行する。  

GPTQConfigの`bits=`で量子化ビット数を、  `dataset=`でキャリブレーション用のデータセットを指定する。 データセットは'wikitext2','c4','c4-new','ptb','ptb-new'から選ぶことが推奨されている。

```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
stime = time.perf_counter() ## 開始時間

model_name = "/vllm/model/vicuna-13b-v1.5" ## 量子化するモデル
save_file = "/vllm/model/vicuna-13b-gptq-8bit" ## 量子化済みのモデル保存先
tokenizer = AutoTokenizer.from_pretrained(model_name)

## GPTQ設定
gptq_config = GPTQConfig(
bits=8, ## 量子化ビット数、Only support quantization to [2,3,4,8] bits
dataset='c4-new', ## キャリブレーションデータセット ['wikitext2','c4','c4-new','ptb','ptb-new'] 'wikitext2','c4','c4-new','ptb','ptb-new'
tokenizer=tokenizer,
use_exllama=False,
cache_examples_on_gpu=False,
use_cuda_fp16=True
)

## GPUメモリ節約
my_device_map = {'model.embed_tokens': 'cpu', 'model.layers': 'cpu', 'model.norm': 'cpu', 'lm_head': 'cpu'}

## 量子化
## Auto-GPTQがGPTQConfigの設定に従ってモデルを読み込んでくれる
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=my_device_map,
torch_dtype=torch.float16, quantization_config=gptq_config)

## Vicunaは元のgeneration_configに不要なパラメータがある
## この設定があると保存できないので、Noneにするる
quantized_model.generation_config.temperature=None
quantized_model.generation_config.top_p=None
quantized_model.generation_config.max_length=None

## safetensors形式で保存
quantized_model.to('cpu')
quantized_model.save_pretrained(save_file, safe_serialization=True)
tokenizer.save_pretrained(save_file)

print('time:', time.perf_counter() - stime) ##　処理時間
```

> 処理時間はL40sで20分程度、GPU使用メモリは44GB程度必要。  
> GPUメモリが足りない場合、後述の「CPUオフロードによる変換」を参照。


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
`--chat-template`でプロンプトテンプレートファイルを指定する。
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

## CPUオフロードによる変換
GPUメモリが足りない場合、CPUのメモリも使用して変換する。

```python
import os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map

stime = time.perf_counter() ## 計測開始

model_name = "/vllm/model/vicuna-13b-v1.5" ## 量子化するモデル
save_dir = "/vllm/model/vicuna-13b-gptq-8bit-cpu" ## 量子化済みのモデル保存先


tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

my_device_map = infer_auto_device_map(
	model,
	max_memory={0:"16GiB", "cpu":"128GiB"}, ## GPU, CPUのメモリ量を指定
	dtype=torch.float16
)

gptq_config = GPTQConfig(
    bits=8,	## 量子化ビット数、Only support quantization to [2,3,4,8] bits
    dataset="c4-new", ## キャリブレーションデータセット ['wikitext2','c4','c4-new','ptb','ptb-new']
    tokenizer=tokenizer,
    use_exllama=False, 
    cache_examples_on_gpu=False, 
    use_cuda_fp16=True 
)

## Auto-GPTQがGPTQConfigの設定に従ってモデルを読み込んでくれる
quantized_model = AutoModelForCausalLM.from_pretrained(
	model_name, 
	torch_dtype=torch.float16, 
	do_sample=True, 
	quantization_config=gptq_config
) 

## 変換終了
print('conv Time:', time.perf_counter() - stime) ## 計測終了

## Vicunaは元のgeneration_configに不要なパラメータがある
## この設定があると保存できないので、Noneにする
quantized_model.generation_config.temperature=None
quantized_model.generation_config.top_p=None
quantized_model.generation_config.max_length=None

## quantized_modelの保存
quantized_model.to('cpu') ## 全てCPUメモリに移す
quantized_model.save_pretrained(save_dir, safe_serialization=True) ## safetensors形式で保存
tokenizer.save_pretrained(save_dir)

print('Total Time:', time.perf_counter() - stime) ## 計測終了
```
処理時間は1891秒、GPUメモリは15982MiB

<br>
<hr>


## 参考
### リンク
- [HugginFace Quantize 🤗 Transformers models](https://huggingface.co/docs/transformers/ja/main_classes/quantization)
- [【ローカルLLM】Hugging FaceによるGPTQ量子化ガイド](https://note.com/bakushu/n/n6c560265b994)
- [NVIDIA RTX3060(12GB)でLLMを試す：GPTQ量子化](https://zenn.dev/to2watt/articles/e98cbb5c3231ab)
- [キャリブレーションデータにもっと気を配ろうの話](https://note.com/sakusakumura/n/n7d7abca9b2e4)
- [transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py)
- [gpt-neox-20b を 3090 x 2 で動かすメモ(3090 x 1 でも動く!)](https://qiita.com/syoyo/items/ba2c25a573ab8e338cd5)

<hr>

LLM実行委員会