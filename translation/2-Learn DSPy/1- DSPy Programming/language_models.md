# Dil Modelleri

Herhangi bir DSPy kodundaki ilk adım, dil modelinizi kurmaktır. Örneğin, OpenAI'nın GPT-4o-mini modelini varsayılan LM'niz olarak aşağıdaki gibi yapılandırabilirsiniz.

```python linenums="1"
# `OPENAI_API_KEY` ortam değişkeni ile kimlik doğrulayın: import os; os.environ['OPENAI_API_KEY'] = 'buraya'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

!!! info "Birkaç farklı LM örneği"

    === "OpenAI"
        `OPENAI_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Gemini (AI Studio)"
        `GEMINI_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        `ANTHROPIC_API_KEY` ortam değişkenini ayarlayarak veya aşağıdaki `api_key` parametresini ileterek kimlik doğrulaması yapabilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-sonnet-4-5-20250929', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Vertex AI (GCP)"
        Google Cloud'un Vertex AI hizmeti için bir hizmet hesabı (service account) JSON anahtarı veya Uygulama Varsayılan Kimlik Bilgileri (Application Default Credentials) ile kimlik doğrulaması yapın. Kimlik bilgilerini doğrudan koda aktarabilir veya ortam değişkenlerini ayarlayabilirsiniz.

        ```python linenums="1"
        import dspy
        import json

        # Hizmet hesabı JSON dosyasını yükleyin ve bir dizeye dönüştürün
        with open("service_account.json") as f:
            credentials = json.dumps(json.load(f))

        lm = dspy.LM(
            "vertex_ai/gemini-2.0-flash",
            vertex_credentials=credentials,
            vertex_project="your-gcp-project-id",
            vertex_location="us-central1",
        )
        dspy.configure(lm=lm)
        ```

        Alternatif olarak, ortam değişkenlerini ayarlayıp kwargs kısmını atlayabilirsiniz:

        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
        export VERTEXAI_PROJECT="your-gcp-project-id"
        export VERTEXAI_LOCATION="us-central1"
        ```

        ```python linenums="1"
        import dspy
        lm = dspy.LM("vertex_ai/gemini-2.0-flash")
        dspy.configure(lm=lm)
        ```

        !!! warning "Yaygın hatalar"
            - `gemini/` değil, `vertex_ai/` model önekini kullanın. `gemini/` öneki, GCP kimlik bilgileri yerine API anahtarı gerektiren Gemini API'sine yönlendirme yapar.
            - `project` veya `location` değil, `vertex_project` ve `vertex_location` parametrelerini kullanın. `vertex_` öneki olmayan parametreler sessizce yoksayılır ve LiteLLM varsayılanlara döner; bu da isteklerin istenmeyen bir bölgeye gitmesine neden olabilir.

    === "Databricks"
        Databricks platformundaysanız, kimlik doğrulaması SDK'ları aracılığıyla otomatik olarak yapılır. Değilseniz, `DATABRICKS_API_KEY` ve `DATABRICKS_API_BASE` ortam değişkenlerini ayarlayabilir veya aşağıdaki gibi `api_key` ve `api_base` bilgilerini iletebilirsiniz.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "GPU sunucusundaki Yerel LM'ler"
          Öncelikle, [SGLang](https://sgl-project.github.io/start/install.html) kurun ve LM'niz ile sunucusunu başlatın.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i [https://flashinfer.ai/whl/cu121/torch2.4/](https://flashinfer.ai/whl/cu121/torch2.4/) 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
          ```

          Ardından, DSPy kodunuzdan OpenAI uyumlu bir uç nokta (endpoint) olarak ona bağlanın.

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
                           api_base="http://localhost:7501/v1",  # bunun portunuza işaret ettiğinden emin olun
                           api_key="", model_type='chat')
          dspy.configure(lm=lm)
          ```

    === "Dizüstü bilgisayarınızdaki Yerel LM'ler"
          Öncelikle, [Ollama](https://github.com/ollama/ollama) kurun ve LM'niz ile sunucusunu başlatın.

          ```bash
          > curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh
          > ollama run llama3.2:1b
          ```

          Ardından, DSPy kodunuzdan ona bağlanın.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
        ```

    === "Diğer sağlayıcılar"
        DSPy'de, [LiteLLM tarafından desteklenen düzinelerce LLM sağlayıcısından](https://docs.litellm.ai/docs/providers) herhangi birini kullanabilirsiniz. Hangi `{PROVIDER}_API_KEY` değişkenini ayarlamanız ve yapıcıya (constructor) `{provider_name}/{model_name}` bilgisini nasıl iletmeniz gerektiği konusundaki talimatlarını takip etmeniz yeterlidir.

        Bazı örnekler:

        - `ANYSCALE_API_KEY` ile `anyscale/mistralai/Mistral-7B-Instruct-v0.1`
        - `TOGETHERAI_API_KEY` ile `together_ai/togethercomputer/llama-2-70b-chat`
        - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` ve `AWS_REGION_NAME` ile `sagemaker/<uç-nokta-adınız>`
        - Ortam değişkenleri olarak `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION` ve isteğe bağlı `AZURE_AD_TOKEN` ve `AZURE_API_TYPE` ile `azure/<dağıtım_adınız>`. Ortam değişkenlerini ayarlamadan harici modelleri başlatıyorsanız aşağıdakini kullanın:
        `lm = dspy.LM('azure/<dağıtım_adınız>', api_key = 'AZURE_API_KEY' , api_base = 'AZURE_API_BASE', api_version = 'AZURE_API_VERSION')`


        Sağlayıcınız OpenAI uyumlu bir uç nokta sunuyorsa, tam model adınızın başına bir `openai/` öneki eklemeniz yeterlidir.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
        dspy.configure(lm=lm)
        ```

Hatalarla karşılaşırsanız, doğru değişken adlarını kullandığınızdan veya doğru prosedürü izlediğinizden emin olmak için lütfen [LiteLLM Belgelerine](https://docs.litellm.ai/docs/providers) başvurun.

## LM'yi doğrudan çağırma.

Yukarıda yapılandırdığınız `lm` nesnesini doğrudan çağırmak oldukça kolaydır. Bu size birleşik bir API sunar ve otomatik önbelleğe alma (caching) gibi yardımcı araçlardan yararlanmanızı sağlar.

```python linenums="1"       
lm("Say this is a test!", temperature=0.7)  # => ['Bu bir testtir!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['Bu bir testtir!']
``` 

## LM'yi DSPy modülleri ile kullanma.

İdeomatik DSPy kullanımı, bir sonraki kılavuzda ele alacağımız _modülleri_ (modules) içerir.

```python linenums="1" 
# Bir modül tanımlayın (ChainOfThought) ve ona bir imza atayın (bir soru verildiğinde bir cevap döndür).
qa = dspy.ChainOfThought('question -> answer')

# Yukarıdaki `dspy.configure` ile yapılandırılan varsayılan LM ile çalıştırın.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Olası Çıktı:**
```text
The castle David Gregory inherited has 7 floors.
```

## Birden fazla LM kullanma.

Varsayılan LM'yi `dspy.configure` ile küresel olarak veya `dspy.context` ile bir kod bloğu içinde değiştirebilirsiniz.

!!! tip "İpucu"
    `dspy.configure` ve `dspy.context` kullanımı iş parçacığı güvenlidir (thread-safe)!


```python linenums="1" 
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
```
**Olası Çıktı:**
```text
GPT-4o-mini: David Gregory'nin miras aldığı kaledeki kat sayısı verilen bilgilerle belirlenemiyor.
GPT-3.5-turbo: David Gregory'nin miras aldığı kale 7 katlıdır.
```

## LM Üretimini Yapılandırma.

Herhangi bir LM için, başlatma sırasında veya sonraki her çağrıda aşağıdaki niteliklerden herhangi birini yapılandırabilirsiniz.

```python linenums="1" 
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

Varsayılan olarak DSPy'deki LM'ler önbelleğe alınır. Aynı çağrıyı tekrarlarsanız aynı çıktıları alırsınız. Ancak `cache=False` ayarlayarak önbelleğe almayı kapatabilirsiniz.

Önbelleğe almayı etkin tutmak ancak yeni bir istek zorlamak istiyorsanız (örneğin, farklı çıktılar elde etmek için), çağrınızda benzersiz bir `rollout_id` geçirin ve sıfır olmayan bir `temperature` (sıcaklık) değeri ayarlayın. DSPy, bir önbellek girişine bakarken hem girdileri hem de `rollout_id` değerini karma hale (hash) getirir; bu nedenle farklı değerler, aynı girdilere ve `rollout_id` değerine sahip gelecekteki çağrıları hâlâ önbelleğe alırken yeni bir LM isteğini zorunlu kılar.

Bu kimlik (ID) ayrıca `lm.history` içinde de kaydedilir; bu da deneyler sırasında farklı sunumları (rollouts) izlemeyi veya karşılaştırmayı kolaylaştırır. `temperature=0` tutarken yalnızca `rollout_id` değerini değiştirmek LM'nin çıktısını etkilemeyecektir.

```python linenums="1"
lm("Say this is a test!", rollout_id=1, temperature=1.0)
```

Bu LM anahtar kelime argümanlarını (kwargs) doğrudan DSPy modüllerine de aktarabilirsiniz. Bunları başlatma sırasında sağlamak, her çağrı için varsayılanları belirler:

```python linenums="1"
predict = dspy.Predict("question -> answer", rollout_id=1, temperature=1.0)
```

Bunları tek bir çağrı için geçersiz kılmak isterseniz, modülü çağırırken bir ``config`` sözlüğü sağlayın:

```python linenums="1"
predict = dspy.Predict("question -> answer")
predict(question="What is 1 + 52?", config={"rollout_id": 5, "temperature": 1.0})
```

Her iki durumda da ``rollout_id`` temel LM'ye iletilir, önbelleğe alma davranışını etkiler ve her yanıtla birlikte saklanır; böylece belirli sunumları (rollouts) daha sonra yeniden oynatabilir veya analiz edebilirsiniz.


## Çıktı ve kullanım meta verilerini inceleme.

Her LM nesnesi; girdiler, çıktılar, jeton (token) kullanımı (ve maliyet) ile meta veriler dahil olmak üzere etkileşimlerinin geçmişini tutar.

```python linenums="1" 
len(lm.history)  # örn. LM'ye yapılan 3 çağrı

lm.history[-1].keys()  # tüm meta verileriyle birlikte LM'ye yapılan son çağrıya erişin
```

**Çıktı:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost', 'timestamp', 'uuid', 'model', 'response_model', 'model_type])
```

## Yanıtlar (Responses) API'sini Kullanma

Varsayılan olarak DSPy, dil modellerini (LM'ler) çoğu standart model ve görev için uygun olan LiteLLM'nin [Chat Completions API](https://docs.litellm.ai/docs/completion) üzerinden çağırır. Ancak OpenAI'ın akıl yürütme modelleri (örneğin, `gpt-5` veya gelecekteki diğer modeller) gibi bazı gelişmiş modeller; DSPy tarafından da desteklenen [Responses API](https://docs.litellm.ai/docs/response_api) üzerinden erişildiğinde daha iyi kalite veya ek özellikler sunabilir.

**Yanıtlar (Responses) API'sini ne zaman kullanmalısınız?**

- `responses` uç noktasını (endpoint) destekleyen veya gerektiren modellerle (OpenAI'ın akıl yürütme modelleri gibi) çalışıyorsanız.
- Belirli modeller tarafından sağlanan gelişmiş akıl yürütme, çok turlu diyalog veya daha zengin çıktı özelliklerinden yararlanmak istediğinizde.

**DSPy'de Yanıtlar API'si nasıl etkinleştirilir:**

Yanıtlar API'sini etkinleştirmek için, `dspy.LM` örneğini oluştururken `model_type="responses"` parametresini ayarlamanız yeterlidir.

```python
import dspy

# Dil modeliniz için Responses API'sini kullanacak şekilde DSPy'yi yapılandırın
dspy.configure(
    lm=dspy.LM(
        "openai/gpt-5-mini",
        model_type="responses",
        temperature=1.0,
        max_tokens=16000,
    ),
)
```

Lütfen tüm modellerin veya sağlayıcıların Responses API'sini desteklemediğini unutmayın; daha fazla ayrıntı için [LiteLLM belgelerini](https://docs.litellm.ai/docs/response_api) kontrol edin.


## Advanced: Building custom LMs and writing your own Adapters.

Though rarely needed, you can write custom LMs by inheriting from `dspy.BaseLM`. Another advanced layer in the DSPy ecosystem is that of _adapters_, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.

