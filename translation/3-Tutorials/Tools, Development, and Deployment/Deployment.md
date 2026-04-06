# Eğitim: DSPy programınızı dağıtma

Bu kılavuz, DSPy programınızı üretim ortamında dağıtmanın iki olası yolunu gösterir: hafif dağıtımlar için FastAPI ve program sürümlendirme ile yönetimini içeren daha üretim odaklı dağıtımlar için MLflow.

Aşağıda, dağıtmak istediğiniz şu basit DSPy programına sahip olduğunuzu varsayacağız. Bunu daha gelişmiş bir şeyle değiştirebilirsiniz. fileciteturn23file0

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

## FastAPI ile Dağıtım

FastAPI, DSPy programınızı REST API olarak sunmanın doğrudan bir yolunu sağlar. Bu, program kodunuza doğrudan erişiminiz olduğunda ve hafif bir dağıtım çözümüne ihtiyaç duyduğunuzda idealdir. fileciteturn23file0

```bash
> pip install fastapi uvicorn
> export OPENAI_API_KEY="your-openai-api-key"
```

Yukarıda tanımlanan `dspy_program`’ı sunmak için bir FastAPI uygulaması oluşturalım. fileciteturn23file0

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="Bir DSPy Chain of Thought programını sunan basit bir API",
    version="1.0.0"
)

# Daha iyi dokümantasyon ve doğrulama için istek modelini tanımla
class Question(BaseModel):
    text: str

# Dil modelinizi yapılandırın ve DSPy programınızı 'asyncify' edin.
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, async_max_workers=4) # varsayılan 8'dir
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Yukarıdaki kodda, `dspy.asyncify` çağırarak DSPy programını yüksek aktarım kapasiteli FastAPI dağıtımları için async modda çalışacak şekilde dönüştürüyoruz. Şu anda bu, DSPy programını ayrı bir iş parçacığında çalıştırır ve sonucunu bekler. fileciteturn23file0

Varsayılan olarak oluşturulan iş parçacığı sınırı 8’dir. Bunu bir çalışan havuzu gibi düşünebilirsiniz.
Eğer aynı anda çalışan 8 programınız varsa ve bir çağrı daha yaparsanız, 9. çağrı bu 8 çağrıdan biri dönene kadar bekler.
Async kapasiteyi yeni `async_max_workers` ayarıyla yapılandırabilirsiniz. fileciteturn23file0

??? "DSPy 2.6.0+ sürümünde Streaming"

    DSPy 2.6.0+ sürümünde streaming de desteklenir; şu komutla kurulabilir: `pip install -U dspy`.

    DSPy programını streaming moda dönüştürmek için `dspy.streamify` kullanabiliriz. Bu, nihai tahmin hazır olmadan önce ara çıktıları (yani O1 tarzı akıl yürütmeyi) istemciye akıtmak istediğinizde yararlıdır. Bu özellik perde arkasında asyncify kullanır ve aynı yürütme semantiğini devralır. fileciteturn23file0

    ```python
    dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
    streaming_dspy_program = dspy.streamify(dspy_program)

    @app.post("/predict/stream")
    async def stream(question: Question):
        async def generate():
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    data = {"prediction": value.labels().toDict()}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {orjson.dumps(data).decode()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # DSPy programı sonucunu çoğu zaman server-sent events olarak akıtmak isteyeceğiniz için
    # buna eşdeğer yardımcı bir fonksiyon da ekledik.

    from dspy.utils.streaming import streaming_response

    @app.post("/predict/stream")
    async def stream(question: Question):
        stream = streaming_dspy_program(question=question.text)
        return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
    ```

Kodunuzu örneğin `fastapi_dspy.py` adlı bir dosyaya yazın. Ardından uygulamayı şu komutla sunabilirsiniz: fileciteturn23file0

```bash
> uvicorn fastapi_dspy:app --reload
```

Bu komut `http://127.0.0.1:8000/` adresinde yerel bir sunucu başlatacaktır. Aşağıdaki Python koduyla test edebilirsiniz: fileciteturn23file0

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "What is the capital of France?"}
)
print(response.json())
```

Aşağıdakine benzer bir yanıt görmelisiniz: fileciteturn23file0

```json
{
  "status": "success",
  "data": {
    "reasoning": "Fransa'nın başkenti, coğrafya derslerinde yaygın olarak öğretilen ve çeşitli bağlamlarda referans verilen iyi bilinen bir bilgidir. Paris, ülkenin siyasi, kültürel ve ekonomik merkezi olarak dünya çapında Fransa'nın başkenti kabul edilir.",
    "answer": "Fransa'nın başkenti Paris'tir."
  }
}
```

## MLflow ile Dağıtım

Eğer DSPy programınızı paketleyip izole bir ortamda dağıtmak istiyorsanız, MLflow ile dağıtımı öneririz.
MLflow; sürümlendirme, izleme ve dağıtım dahil olmak üzere makine öğrenmesi iş akışlarını yönetmek için popüler bir platformdur. fileciteturn23file0

```bash
> pip install mlflow>=2.18.0
```

DSPy programımızı depolayacağımız MLflow izleme sunucusunu başlatalım. Aşağıdaki komut `http://127.0.0.1:5000/` adresinde yerel bir sunucu başlatacaktır. fileciteturn23file0

```bash
> mlflow ui
```

Ardından DSPy programını tanımlayıp MLflow sunucusuna loglayabiliriz. MLflow’da “log” kelimesi çok anlamlı kullanılır; burada temel olarak program bilgisini ortam gereksinimleriyle birlikte MLflow sunucusunda saklamak anlamına gelir. Bu işlem `mlflow.dspy.log_model()` fonksiyonu ile yapılır; aşağıdaki koda bakın: fileciteturn23file0

> [!NOTE]
> MLflow 2.22.0 itibarıyla, MLflow ile dağıtım yaparken DSPy programınızı özel bir DSPy Module sınıfı içine sarmanız gerekir.
> Bunun nedeni, MLflow’un konumsal argümanlar istemesi; buna karşılık DSPy’nin hazır modüllerinin, örneğin `dspy.Predict`
> veya `dspy.ChainOfThought`, konumsal argümanlara izin vermemesidir. Bunu aşmak için, `dspy.Module` sınıfından türeyen
> bir sarmalayıcı sınıf oluşturun ve program mantığınızı aşağıdaki örnekte gösterildiği gibi `forward()` metodunda uygulayın.

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, messages):
        return self.cot(question=messages[0]["content"])

dspy_program = MyProgram()

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

Dağıtılan programın girdiyi alıp çıktıyı OpenAI chat API ile aynı biçimde üretmesi için `task="llm/v1/chat"` ayarını yapmanızı öneririz; bu, LM uygulamaları için yaygın bir arayüzdür. Yukarıdaki kodu örneğin `mlflow_dspy.py` adlı bir dosyaya yazın ve çalıştırın. fileciteturn23file0

Programı logladıktan sonra, kaydedilen bilgileri MLflow UI içinde görebilirsiniz. `http://127.0.0.1:5000/` adresini açın, `deploy_dspy_program` deneyini seçin, ardından az önce oluşturduğunuz çalıştırmayı seçin. `Artifacts` sekmesi altında loglanan program bilgisini aşağıdaki ekran görüntüsüne benzer şekilde göreceksiniz: fileciteturn23file0

![MLflow UI](./dspy_mlflow_ui.png)

UI’dan run id’nizi alın (veya `mlflow_dspy.py` çalıştırıldığında konsola yazdırılan değeri kullanın). Artık loglanan programı aşağıdaki komutla dağıtabilirsiniz: fileciteturn23file0

```bash
> mlflow models serve -m runs:/{run_id}/model -p 6000
```

Program dağıtıldıktan sonra aşağıdaki komutla test edebilirsiniz: fileciteturn23file0

```bash
> curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json"  --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

Aşağıdakine benzer bir yanıt görmelisiniz: fileciteturn23file0

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"reasoning\": \"Soru 2 ile 2'nin toplamını soruyor. Cevabı bulmak için iki sayıyı toplarız: 2 + 2 = 4.\", \"answer\": \"4\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

DSPy programını MLflow ile nasıl dağıtacağınıza ve dağıtımı nasıl özelleştireceğinize dair tam kılavuz için [MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasına bakın. fileciteturn23file0

### MLflow Dağıtımı için En İyi Uygulamalar

1. **Ortam Yönetimi**: Python bağımlılıklarınızı her zaman bir `conda.yaml` veya `requirements.txt` dosyasında açıkça belirtin.
2. **Sürümlendirme**: Model sürümleriniz için anlamlı etiketler ve açıklamalar kullanın.
3. **Girdi Doğrulama**: Açık girdi şemaları ve örnekleri tanımlayın.
4. **İzleme**: Üretim dağıtımları için uygun loglama ve izleme mekanizmaları kurun. fileciteturn23file0

Üretim dağıtımları için MLflow’u containerization ile birlikte kullanmayı değerlendirin: fileciteturn23file0

```bash
> mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
> docker run -p 6000:8080 dspy-program
```

Üretim dağıtım seçenekleri ve en iyi uygulamalar hakkında tam kılavuz için [MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasına bakın. fileciteturn23file0
