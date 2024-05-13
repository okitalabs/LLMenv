from huggingface_hub import snapshot_download, login
# login(token = "Toke ID") ## 認証が必要な場合

model_name = "masao1211/karakuri-lm-70b-chat-v0.1-AWQ" ## ダウンロードするHuggingFaceのモデル名
save_name = "/home/llm/vllm/model/karakuri-lm-70b-chat-v0.1-AWQ" ## ダウンロード先のディレクトリ

download_path = snapshot_download(
    repo_id = model_name,
    local_dir = save_name,
    local_dir_use_symlinks=False
)