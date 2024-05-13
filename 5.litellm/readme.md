# LLM実行環境の構築
# LiteLLM

LiteLLM ProxyはさまざまなLLM APIのエンドポイントをOpenAI APIで一元化するためのProxy。APIのmodel名から実際のLLM APIに振り分けたり、負荷分散や認証、ロギングなどさまざまな機能を提供する。

[LiteLLM Documentページ](https://docs.litellm.ai/)  
[LiteLLM Proxy Documentページ](https://docs.litellm.ai/docs/simple_proxy)  
[LiteLLM GitHub](https://github.com/BerriAI/litellm)  

## 構成情報
### Docker設定
|Docker名|Host Port|Docker Port|Host Dir|Docker Dir|
|:----|:----|:----|:----|:----|
|litellm|10080|10080|/home/llm/litellm|/litellm|


### Proxy設定
|model名|FW Host|FW Port|FW model名|
|:----|:----|:----|:----|
|gpt-3.5-turbo|localhost|20080|vicuna-13b|
|text-embedding-ada-002|localhost|20080|sentence-luke|
|vicuna-13b|localhost|20080|vicuna-13b|
|karakuri-8x7b|localhost|20080|karakuri-8x7b|
|karakuri-70b|localhost|30080|karakuri-70b|
|sentence-luke|localhost|40080|sentence-luke|


<br>
<hr>



# 構築手順
## Dockerイメージ
すでにあるコンテナイメージを使用するため、ビルドは不要。



<hr>


## LiteLLM Proxyサーバの起動
### configファイルの作成



以下のファイルを作成する。  
`$HOME/litellm/config.yaml`  
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/vicuna-13b
      api_base: http://host.docker.internal:20080/v1
      api_key: None
  - model_name: text-embedding-ada-002
    litellm_params:
      model: openai/sentence-luke
      api_base: http://host.docker.internal:40080/v1
      api_key: None
  - model_name: vicuna-13b
    litellm_params:
      model: openai/vicuna-13b
      api_base: http://host.docker.internal:20080/v1
      api_key: None
  - model_name: karakuri-8x7b
    litellm_params:
      model: openai/karakuri-8x7b
      api_base: http://host.docker.internal:20080/v1
      api_key: None
  - model_name: karakuri-70b
    litellm_params:
      model: openai/karakuri-70b
      api_base: http://host.docker.internal:30080/v1
      api_key: None
  - model_name: sentence-luke
    litellm_params:
      model: openai/sentence-luke
      api_base: http://host.docker.internal:40080/v1
      api_key: None
```
> - LiteLLM ProxyをDockerで起動すると、バックエンドのランタイムサーバとの通信がlocalhostでは出来ないため、Dockerの起動オプションに`--add-host=host.docker.internal:host-gateway`を付けて、サーバIPを`host.docker.internal`にすることで、ホストOS側のPortにアクセスし、バックエンドサーバと通信する。  

> - `model: openai/vicuna-13b`の`openai/`はランタイム側のAPIがOpenAI API互換の場合、model名の前に`openai/`付ける。



### 起動
```bash
$ docker run --rm \
--add-host=host.docker.internal:host-gateway \
-v /home/llm/litellm:/litellm \
-p 10080:10080 \
-h litellm --name litellm \
ghcr.io/berriai/litellm:main-latest \
--config /litellm/config.yaml --detailed_debug --port 10080 --
```
> - `--detailed_debug`オプションを付けることにより、詳細なログを取ることが出来る。クライアント〜サーバ間の入出力の生メッセージを観察することが出来るので、デバッグ等に有効。


<br>
<hr>

## 起動後の確認

## Chat Complation
バックエンドのランタイムを起動して、`"model": "gpt-3.5-turbo"でアクセスできるか確認。またそれ以外ののmodel名でもアクセス確認してみる。
```bash
$ time curl http://localhost:10080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "system", "content": "あなたは優秀な観光ガイドです。"},
    {"role": "user", "content": "日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。"}
  ]
}'
```

LiteLLM Proxyでは、以下のようなログが表示され内容が確認できる。
```
03:15:47 - LiteLLM:DEBUG: utils.py:1210 - PRE-API-CALL ADDITIONAL ARGS: {'headers': {'Authorization': 'Bearer None'}, 'api_base': ParseResult(scheme='http', userinfo='', host='host.docker.internal', port=30080, path='/v1/', query=None, fragment=None), 'acompletion': True, 'complete_input_dict': {'model': 'karakuri-70b', 'messages': [{'role': 'system', 'content': 'あなたは優秀な観光ガイドです。'}, {'role': 'user', 'content': '日本の都道府県名をランダムに１つ回答してください。その都道府県名の魅力を3つ答えてください。'}], 'extra_body': {}}}
 :
03:15:54 - LiteLLM Router:DEBUG: router.py:1406 - Async Response: ModelResponse(id='cmpl-6affe23b9f8d4426b84e6e7479af7110', choices=[Choices(finish_reason='stop', index=0, message=Message(content=' 日本人の観光ガイドとして、ランダムに選ばれた都道府県名とその魅力を3つご紹介いたします。\n\n鳥取県：\n鳥取砂丘: 世界最大級の海岸砂丘で、スキーやパラグライダーなどのアクティビティが楽しめます。\n水木しげるロード: 漫画家水木しげる氏の出身地であり、妖怪のブロンズ像が街中に立ち並んでいます。\n鳥取城跡: 山上にそびえる江戸時代の城跡で、天守閣からの眺めは絶景です。\n\n以上が鳥取県の主な魅力です。 ', role='assistant'))], created=1715570147, model='karakuri-70b', object='chat.completion', system_fingerprint=None, usage=Usage(completion_tokens=114, prompt_tokens=101, total_tokens=215))
```

### Embeddings
バックエンドのLocalAIを起動し、`text-embedding-ada-002`でアクセスできるか確認。
```
$time curl http://localhost:0080/v1/embeddings \
-H "Content-Type: application/json" \
-H "Authorization: Bearer None" \
-d '{
  "model": "text-embedding-ada-002",
  "input": "query: 夕飯はお肉です。"
}'
```


<br>
<hr>


## 参考

### 起動オプション
```
Usage: litellm [OPTIONS]

Options:
  --host TEXT                Host for the server to listen on.
  --port INTEGER             Port to bind the server to.
  --num_workers INTEGER      Number of gunicorn workers to spin up
  --api_base TEXT            API base URL.
  --api_version TEXT         For azure - pass in the api version.
  -m, --model TEXT           The model name to pass to litellm expects
  --alias TEXT               The alias for the model - use this to give a
                             litellm model name (e.g.
                             "huggingface/codellama/CodeLlama-7b-Instruct-hf")
                             a more user-friendly name ("codellama")
  --add_key TEXT             The model name to pass to litellm expects
  --headers TEXT             headers for the API call
  --save                     Save the model-specific config
  --debug                    To debug the input
  --detailed_debug           To view detailed debug logs
  --use_queue                To use celery workers for async endpoints
  --temperature FLOAT        Set temperature for the model
  --max_tokens INTEGER       Set max tokens for the model
  --request_timeout INTEGER  Set timeout in seconds for completion calls
  --drop_params              Drop any unmapped params
  --add_function_to_prompt   If function passed but unsupported, pass it as
                             prompt
  -c, --config TEXT          Path to the proxy configuration file (e.g.
                             config.yaml). Usage `litellm --config
                             config.yaml`
  --max_budget FLOAT         Set max budget for API calls - works for hosted
                             models like OpenAI, TogetherAI, Anthropic, etc.`
  --telemetry BOOLEAN        Helps us know if people are using this feature.
                             Turn this off by doing `--telemetry False`
  -v, --version              Print LiteLLM version
  --health                   Make a chat/completions request to all llms in
                             config.yaml
  --test                     proxy chat completions url to make a test request
                             to
  --test_async               Calls async endpoints /queue/requests and
                             /queue/response
  --num_requests INTEGER     Number of requests to hit async endpoint with
  --run_gunicorn             Starts proxy via gunicorn, instead of uvicorn
                             (better for managing multiple workers)
  --ssl_keyfile_path TEXT    Path to the SSL keyfile. Use this when you want
                             to provide SSL certificate when starting proxy
  --ssl_certfile_path TEXT   Path to the SSL certfile. Use this when you want
                             to provide SSL certificate when starting proxy
  --local                    for local debugging
  --help                     Show this message and exit.
```


<hr>

LLM実行委員会