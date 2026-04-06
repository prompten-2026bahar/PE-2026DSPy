# Eğitim: Ajanlar

Birkaç araç kullanan bir `dspy.ReAct` ajanını kurmanın ve onu çok adımlı arama için gelişmiş tarama yapacak şekilde optimize etmenin hızlı bir örneğini adım adım inceleyelim.

En güncel DSPy sürümünü `pip install -U dspy` ile kurup devam edin. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekir.

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ve deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. MLflow’u aşağıdaki dört adımı izleyerek kolayca kurabilirsiniz.

![MLflow Trace](./mlflow-tracing-agent.png)

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow arayüzünü başlatın

```bash id="h29v7s"
mlflow ui --port 5000
```

3. Notebook’u MLflow’a bağlayın

```python id="fd035i"
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.

```python id="o4ynmz"
mlflow.dspy.autolog()
```

Yukarıdaki adımları tamamladıktan sonra, notebook üzerinde her program çalıştırması için izleri görebilirsiniz. Bunlar modelin davranışı üzerinde güçlü bir görünürlük sağlar ve eğitim boyunca DSPy kavramlarını daha iyi anlamanıza yardımcı olur.

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

Bu eğitimde, Meta’nın 3 milyar parametreye sahip son derece küçük bir LM’i olan `Llama-3.2-3B-Instruct` modelini kullanacağız.

Böyle bir model, uzun veya karmaşık ajan döngülerinde kutudan çıktığı haliyle çok güvenilir değildir. Ancak çok az RAM gerektirdiği için barındırması son derece hızlı ve ucuzdur.

3B modeli dizüstü bilgisayarınızda Ollama ile, GPU sunucunuzda SGLang ile ya da Databricks veya Together gibi sizin için barındıran bir sağlayıcı üzerinden çalıştırabilirsiniz.

Aşağıdaki kod parçasında ana LM’imizi `Llama-3.2-3B` olarak yapılandıracağız. Ayrıca küçük modeli eğitmeye yardımcı olması için çok az sayıda çağıracağımız daha büyük bir LM’i, yani `GPT-4o`’yu, öğretmen olarak kuracağız.

```python id="nsrtab"
import dspy

llama3b = dspy.LM('<provider>/Llama-3.2-3B-Instruct', temperature=0.7)
gpt4o = dspy.LM('openai/gpt-4o', temperature=0.7)

dspy.configure(lm=llama3b)
```

Şimdi görevimiz için bir veri kümesi yükleyelim. HoVer çok adımlı görevinden örnekler yükleyeceğiz; burada girdi (gerçekten de!) karmaşık bir iddia, aradığımız çıktı ise bu iddiayı doğrulamak için gereken Wikipedia sayfalarının kümesidir.

```python id="tmszyy"
import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="vincentkoc/hover-parquet", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]
```

Bu görevin bir örneğine bakalım:

```python id="4nflya"
example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)
```

Şimdi Wikipedia içinde arama yapmak için bir fonksiyon tanımlayalım. HoVer’da kullanılan veri olan, 2017 yılında Wikipedia’da var olan her makalenin “özet” kısmını (yani ilk paragraflarını) arayabilen bir ColBERTv2 sunucusunu kullanacağız.

```python id="a5lzq2"
DOCS = {}

def search(query: str, k: int) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=k)
    results = [x['text'] for x in results]

    for result in results:
        title, text = result.split(" | ", 1)
        DOCS[title] = text

    return results
```

Şimdi `search` fonksiyonunu kullanarak ReAct ajanımız için iki araç tanımlayalım:

```python id="s4ftjd"
def search_wikipedia(query: str) -> list[str]:
    """İlk 5 sonucu ve ardından 5. ile 30. sıralar arasındaki sonuçların başlıklarını döndürür."""

    topK = search(query, 30)
    titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]], topK[:5]
    return topK + [f"Getirilen diğer sayfaların başlıkları şunlardır: {', '.join(titles)}."]

def lookup_wikipedia(title: str) -> str:
    """Varsa, Wikipedia sayfasının metnini döndürür."""

    if title in DOCS:
        return DOCS[title]

    results = [x for x in search(title, 10) if x.startswith(title + " | ")]
    if not results:
        return f"Şu başlık için Wikipedia sayfası bulunamadı: {title}"
    return results[0]
```

Şimdi DSPy içinde ReAct ajanını tanımlayalım. Oldukça basit olacak: bir `claim` alacak ve `titles: list[str]` listesi üretecek.

Ona, iddiayı doğrulamak (veya çürütmek) için gerekli tüm Wikipedia başlıklarını bulmasını söyleyeceğiz.

```python id="mzil60"
instructions = "Find all Wikipedia titles relevant to verifying (or refuting) the claim."
signature = dspy.Signature("claim -> titles: list[str]", instructions)
react = dspy.ReAct(signature, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
```

Minik 3B modelimizin bunu yapıp yapamayacağını görmek için gerçekten basit bir iddia ile deneyelim!

```python id="r9oaml"
react(claim="David Gregory was born in 1625.").titles[:3]
```

Harika. Şimdi `top5_recall` adlı bir değerlendirme metriği oluşturalım.

Bu metrik, ajanın döndürdüğü ilk 5 başlık içinde yer alan altın standart sayfaların (bunlar her zaman 3 tanedir) oranını döndürecektir.

```python id="xrqrhi"
def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # Optimizasyon için "bootstrapping" yapıyorsak, yalnızca recall kusursuzsa True döndür.
    if trace is not None:
        return recall >= 1.0
    
    # Yalnızca çıkarım yapıyorsak, recall değerini ölç.
    return recall

evaluate = dspy.Evaluate(devset=devset, metric=top5_recall, num_threads=16, display_progress=True, display_table=5)
```

Kutudan çıktığı haliyle ajanımızı, `Llama-3.2-8B` ile değerlendirip zaten ne kadar ileri gidebildiğimize bakalım.

Bu model küçük olduğu için zaman zaman hata verebilir. Bunları gizlemek için bir try/except bloğuna saralım.

```python id="ger7u2"
def safe_react(claim: str):
    try:
        return react(claim=claim)
    except Exception as e:
        return dspy.Prediction(titles=[])

evaluate(safe_react)
```

<details>
<summary>MLflow Experiment içinde Değerlendirme Sonuçlarını İzleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.

```python id="m89dob"
import mlflow

with mlflow.start_run(run_name="agent_evaluation"):
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=top5_recall,
        num_threads=16,
        display_progress=True,
    )

    # Programı her zamanki gibi değerlendir
    result = evaluate(cot)

    # Toplu skoru kaydet
    mlflow.log_metric("top5_recall", result.score)
    # Ayrıntılı değerlendirme sonuçlarını tablo olarak kaydet
    mlflow.log_table(
        {
            "Claim": [example.claim for example in eval_set],
            "Expected Titles": [example.titles for example in eval_set],
            "Predicted Titles": [output[1] for output in result.results],
            "Top 5 Recall": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

Vay canına. Recall açısından yalnızca %8 alıyor. Pek iyi değil!

Şimdi ajanımızın recall değerini en üst düzeye çıkarmak için `dspy.ReAct` içindeki iki istemi birlikte optimize edelim. Bu işlem yaklaşık 30 dakika sürebilir ve `Llama-3.2-3B` modelini optimize etmek için GPT-4o’ya yaklaşık 5 dolarlık çağrı yapabilir.

```python id="tdigpu"
kwargs = dict(teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o, max_errors=999)

tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **kwargs)
optimized_react = tp.compile(react, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=0)
```

Şimdi optimizasyondan sonra tekrar değerlendirelim.

```python id="0uhizr"
evaluate(optimized_react)
```

Harika. Görünüşe göre sistem recall değerini %8’den yaklaşık %40’a çıkararak ciddi biçimde iyileşti. Bu oldukça doğrudan bir yaklaşımdı, ancak DSPy sana buradan devam ederek yineleme yapman için birçok araç sunuyor.

Sonraki adımda, ne öğrendiğini anlamak için optimize edilmiş istemleri inceleyelim. Bir sorgu çalıştıracağız ve ardından son iki istemi inceleyeceğiz; bu da bize ReAct’in iki alt modülünde kullanılan istemleri gösterecek: ajan döngüsünü yürüten modül ve son sonuçları hazırlayan modül. (Alternatif olarak, yukarıdaki talimatları izleyerek MLflow Tracing’i etkinleştirdiysen, ajan tarafından yapılan tüm adımları; LLM çağrıları, istemler ve araç çalıştırmaları dahil olmak üzere, zengin bir ağaç görünümünde görebilirsin.)

```python id="f0ulcy"
optimized_react(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
```

```python id="pqj02v"
dspy.inspect_history(n=2)
```

Son olarak, optimize edilmiş programımızı daha sonra tekrar kullanabilmek için kaydedelim.

```python id="pwdia3"
optimized_react.save("optimized_react.json")

loaded_react = dspy.ReAct("claim -> titles: list[str]", tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
loaded_react.load("optimized_react.json")

loaded_react(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
```

<details>
<summary>Programları MLflow Experiment içinde kaydetme</summary>

<br/>

Programı yerel bir dosyaya kaydetmek yerine, daha iyi yeniden üretilebilirlik ve iş birliği için MLflow içinde takip edebilirsiniz.

1. **Bağımlılık Yönetimi**: MLflow, yeniden üretilebilirliği sağlamak için dondurulmuş ortam meta verisini programla birlikte otomatik olarak kaydeder.
2. **Deney Takibi**: MLflow ile programın performansını ve maliyetini, programın kendisiyle birlikte takip edebilirsiniz.
3. **İş Birliği**: MLflow deneyini paylaşarak programı ve sonuçları ekip üyelerinizle paylaşabilirsiniz.

Programı MLflow içine kaydetmek için aşağıdaki kodu çalıştırın:

```python id="cw8z2w"
import mlflow

# Bir MLflow Run başlat ve programı kaydet
with mlflow.start_run(run_name="optimized_rag"):
    model_info = mlflow.dspy.log_model(
        optimized_react,
        artifact_path="model", # Programı MLflow içinde kaydetmek için herhangi bir ad
    )

# Programı MLflow’dan tekrar yükle
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>
