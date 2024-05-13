# LLM実行環境の構築


## 前提条件

LLMの実行エンジンは動作に必要なライブラリのバージョンがかなりシビアなため共存が難しい。このため、Dockerでの起動を推奨する。  
動作環境として、以下の構成が構築済みであること。

- Ubuntu 22.04
- NVIDIA GPU、Driver/CUDA導入済み
- Docker導入済み


<hr>


## 構成情報

### HW, MW Version
|Name|Version|
|:----|:----|
|OS|Ubuntu 22.04.4|
|GPU|L40s/48G|
|NVIDIA Driver|550.54.15|
|CUDA|12.4|
|Docker|26.0.0|
|NVIDIA Container Toolkit|1.14.6|

### SW環境
|Name|Version|
|:----|:----|
|Python| 3.10.14|
|Conda仮想環境名|llm|

### 実行ユーザ
|Name|Value|
|:----|:----|
|User Name|llm|
|UID|10002|
|Group Name|llm|
|GID|10002|
|Home Directory|/home/llm/|

### LLM Engineディレクトリ
|Name|Base Dir|
|:----|:----|
|llama-cpp-python|$HOME/llamacpp/|
|vLLM|$HOME/vllm/|
|LocalAI|$HOME/localai/|
|LiteLLM|$HOME/litellm/|

### LLM Model
|Engine|Model name|Model|
|:----|:----|:----|
|llama-cpp-python|vicuna-13b|vicuna-13b-v1.5.Q8_0.gguf|
| |karakuri-8x7b|karakuri-lm-8x7b-chat-v0.1-gguf|
|vLLM|karakuri-70b|karakuri-lm-70b-chat-v0.1-AWQ|
|LocalAI|sentence-luke|sentence-luke-japanese-base-lite|


### Port情報
|Name|Port|
|:----|:----|
|llama-cpp-python|20080|
|vLLM|30080|
|LocalAI|40080|
|LiteLLM|10080|
|Jupyter|8888|


<hr>


## ユーザ環境作成

### ユーザ追加
ユーザアカウント作成と権限を付与する。ユーザ名は`llm`とする。
```bash
$ sudo groupadd -g 10002 llm
sudo useradd --uid 10002 --gid 10002 --shell /bin/bash --create-home --home-dir /home/llm llm
$ sudo passwd llm
$ sudo usermod -aG sudo llm
$ sudo usermod -aG docker llm
```
### Pythonインストール
クライアントから動作確認で使う。ここではconda環境を使用。  
Python仮想環境名は`llm`とする。
```bash
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ source ~/.bashrc

$ conda create -n llm python=3.10
$ conda activate  llm
$ echo 'conda activate  llm' >> ~/.bashrc ## ログイン時にactivateしておくため
```

### Jupyterのインストール
動作チェック用クライアントのため、とりあえず入れておく。
```bash
conda install jupyterLab
pip install ipywidgets iprogress
```

#### Jupyterの起動
Port `8888`、token `llm`でJupyterを起動する例。
```bash
## フォアグラウンド起動	
$ jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token="llm"

## バックグラウンド起動	
nohup bash -c 'jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token="llm"' &> /dev/null & disown
```

<hr>

LLM実行委員会