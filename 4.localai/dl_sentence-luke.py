from huggingface_hub import snapshot_download, login
# login(token = "Toke ID") ## 認証が必要な場合

model_name = "sonoisa/sentence-luke-japanese-base-lite" ## ダウンロードするHuggingFaceのモデル名
save_name = "/home/llm/localai/model/sentence-luke" ## ダウンロード先のディレクトリ

download_path = snapshot_download(
    repo_id = model_name,
    local_dir = save_name,
    local_dir_use_symlinks=False
)