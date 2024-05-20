# LLM実行環境の構築
# LLaMA.cpp HTTP Server

llama-cppのserverで、GGUFのモデルを実行するための手順。Continuous Batchによる同時処理、Embeddigsが可能。  
単一モデルのみ起動出来る。

## 構成情報
### 変換モデル情報

あらかじめ、以下のモデルを`$HOME/vllm/model`に配置しておく。
- vicuna-13b-v1.5.Q8_0.gguf  

### Docker設定

Docker `llammacpp-server`を作成する、以下の設定を使用する。

|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|llamacpp|10080|20080|/home/llm/llamacpp|/llamacpp|


<br>
<hr>


# 構築手順
## Dockerイメージの作成
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)のGitHubにある、[.devops/main-cuda.Dockerfile](https://github.com/ggerganov/llama.cpp/blob/master/.devops/main-cuda.Dockerfile)から実行用のコンテナイメージ `llamacpp-server`を作成する。  


```bash
$ cd $HOME/llamacpp 
$ git clone https://github.com/ggerganov/llama.cpp.git
$ cd llama.cpp
$ cp .devops/main-cuda.Dockerfile Dockerfile
$ docker build -t llamacpp-server .
```


<br>
<hr>


## llama-cppサーバの起動

```bash
docker run --rm -p 20080:20080 --gpus all -v /home/llm/llamacpp:/llamacpp \
-h llamacpp-server --name llamacpp-server \
llamacpp-server  \
--chat-template vicuna \
--threads-batch 8 \
--threads-http 8 \
--model /llamacpp/model/vicuna-13b-v1.5.Q8_0.gguf \
--ctx-size 4096 \
--embeddings \
--parallel 8 \
--cont-batching \
--flash-attn \
--n-gpu-layers 96 \
--host 0.0.0.0 \
--port 20080 
 ```
- --chat-template: Prompt Template名。  
  パラメータ値は[Templates supported by llama_chat_apply_template](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)参考。 
- --cont-batching: Continuous Batchを使う場合。  
- --n-gpu-layers: GPUを使う場合のレイヤ数。-1は不可。  

`--threads-batch`、`--threads-http`、`--parallel`の関係はよくわからないので、同時処理数から同じ値にしておく。

それ以外のパラメータは、[LLaMA.cpp HTTP Server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)  を参考。
<hr>

## 実行の確認
### Chat Completions
単一モデルの起動のため、model名は指定しなくても動作する。
```
time curl http://localhost:20080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "",
  "templature": 0.9,
  "top_p": 1.0,
  "messages": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の 魅力を5つ詳しく答えてください。"}
  ]
}'
```
レスポンス
```
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":" 都道府県名は、「神奈川県」です。\n\n神奈川県は、東京都の南に位置する県です。以下に、神奈川県の魅力を5つ紹介します。\n\n1. 温暖な気候：神奈川県は温暖な気候で、暖かい季節には海水浴や釣り、冬には温泉巡りが楽しめます。\n2. 美味しい食べ物：神奈川県は、昆布や鮑やカキなどの新鮮な魚介類が有名で、その他にも玄米や蕎麦、鰹節などの美味しい食べ物があります。\n3. 観光地：神奈川県には、江の島や葉山、湘南台などの観光地があり、家族連れやカップルに人気があります。\n4. 歴史や文化：神奈川県には、江ノ島の歴史ある城や、小田原の歴史ある街並みがあり、神奈川県の文化について学ぶことができます。\n5. アクセスの良さ：神奈川県は、東京から近く、東京からのアクセスも非常に良いため、東京市内に住む人々にも人気があります。","role":"assistant"}}],"created":1715942193,"model":"","object":"chat.completion","usage":{"completion_tokens":426,"prompt_tokens":70,"total_tokens":496},"id":"chatcmpl-vrepNPH0QRUch3644pdEtKoY6ey1wBzq"}
real  0m10.158s
user  0m0.000s
sys 0m0.008s
```
生成速度は、41.94 token/s  

3リクエスト同時処理の場合は以下。若干生成速度が落ちるが、Continuous Batchが効いているみたい。

- 40.88 token/s  
- 40.71 token/s  
- 39.30 token/s  


### Embeddings
Embeddingsの確認。問題なく使用可能。
```
time curl http://localhost:20080/v1/embeddings \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "vicuna-13b",
  "input": "query: 夕飯はお肉です。"
}'
```


<br>
<hr>


## 参考
### リンク
- [LLaMA.cpp HTTP Server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)  
- [Templates supported by llama_chat_apply_template](https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template)  


<hr>

LLM実行委員会