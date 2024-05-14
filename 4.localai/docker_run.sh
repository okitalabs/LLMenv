docker run --rm -it --entrypoint "" -p 40080:40080 --gpus all -v /home/llm/localai:/localai -h localai --name localai localai /bin/bash


docker run --rm -p 30080:30080 -p 38888:8888 --gpus all -it --entrypoint "" -v /home/llm/vllm:/vllm -h vllm --name vllm vllm /bin/bash

pip install ipywidgets iprogress jupyterLab

jupyter-lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token="llm"

http://172.18.18.102:38888/lab

cd /vllm
mkdir jupyter
cd jupyter



https://huggingface.co/lmsys/vicuna-13b-v1.5


from huggingface_hub import snapshot_download, login
# login(token = "Toke ID") ## 認証が必要な場合

model_name = "https://huggingface.co/lmsys/vicuna-13b-v1.5" ## ダウンロードするHuggingFaceのモデル名
save_name = "/home/llm/vllm/model/vicuna-13b-v1.5" ## ダウンロード先のディレクトリ

download_path = snapshot_download(
    repo_id = model_name,
    local_dir = save_name,
    local_dir_use_symlinks=False
)


