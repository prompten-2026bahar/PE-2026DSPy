# Eğitim: Çok Adımlı Erişim

Birden fazla alt modüle sahip bir `dspy.Module` oluşturmanın hızlı bir örneğini adım adım inceleyelim. Bunu çok adımlı arama görevi için yapacağız.

En güncel DSPy sürümünü `pip install -U dspy` ile kurup takip edin. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekir.

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz.

1. MLflow’u kurun

```bash
%pip install mlflow>=2.20
```

2. Ayrı bir terminalde MLflow arayüzünü başlatın
```bash
mlflow ui --port 5000
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```

![MLflow Trace](./mlflow-tracing-multi-hop.png)


Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.
</details>

Bu eğitimde, Meta’nın 8 milyar parametreye sahip küçük bir yerel LM’i olan `Llama-3.1-8B-Instruct` modelini kullanacağız.

8B modeli dizüstü bilgisayarınızda Ollama ile, GPU sunucunuzda SGLang ile ya da Databricks veya Together gibi sizin için barındıran bir sağlayıcı üzerinden çalıştırabilirsiniz.

Aşağıdaki kod parçasında bu küçük modeli ana LM’imiz olarak yapılandıracağız. Ayrıca küçük LM’i eğitmeye yardımcı olması için çok az sayıda çağıracağımız daha büyük bir LM’i, yani `GPT-4o`’yu, öğretmen olarak kuracağız. Teknik olarak bu gerekli değildir; küçük model genellikle DSPy içinde bunun gibi görevleri kendi kendine öğrenebilir. Ancak daha büyük bir öğretmen kullanmak bize biraz iç rahatlığı sağlar; böylece ilk sistem veya optimize edici yapılandırması o kadar da önemli olmaz.

```python
import dspy

lm = dspy.LM('<your_provider>/Llama-3.1-8B-Instruct', max_tokens=3000)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=3000)

dspy.configure(lm=lm)
```

### Bağımlılıkları kurun ve veriyi indirin

Erişimi gerçekleştirmek için oldukça hafif olduğu için güzel BM25S kütüphanesini kullanacağız. Bu bileşenleri istediğiniz herhangi bir şeyle değiştirebilirsiniz.

```shell
> pip install -U bm25s PyStemmer "jax[cpu]"
```

Ardından, 2017 yılı itibarıyla mevcut olan tüm 5.000.000 Wikipedia sayfasının özetlerinin (yani ilk paragraflarının) bir anlık görüntüsünü indireceğiz. Bunu erişim külliyatımız olarak kullanacağız.

Bu dosya sıkıştırılmış halde 500MB boyutundadır, bu nedenle indirme ve açma işlemi 2-3 dakika sürebilir.

```python
from dspy.utils import download

download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
!tar -xzvf wiki.abstracts.2017.tar.gz
```

Şimdi külliyatı yükleyelim.

```python
import orjson
corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = orjson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

len(corpus)
```

Ve ardından bunu BM25 erişimi için indeksleyelim! Bu işlem 2-3 dakika sürecektir.

```python
import bm25s
import Stemmer

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)
```

### HoVer veri kümesini yükleyin.

Görevimiz için bir veri kümesi yükleyelim. HoVer çok adımlı görevinden örnekler yükleyeceğiz; burada girdi (gerçekten de!) karmaşık bir iddia, aradığımız çıktı ise bu iddiayı doğrulamak için gereken Wikipedia sayfalarının kümesidir.

```python
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
trainset, devset, testset = hover[:200], hover[200:500], hover[650:]
```

Bu göreve ait bir örneğe bakalım:

```python
example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)
```

Şimdi Wikipedia içinde arama yapmak için bir fonksiyon tanımlayalım. Bu, BM25 indeksimizi kullanacaktır.

```python
def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return run
```

Şimdi DSPy içinde çok adımlı programı tanımlayalım. Oldukça basit olacak: bir `claim` alacak ve `titles: list[str]` listesi üretecek.

Bunu iki alt modül aracılığıyla yapacak: `generate_query` ve `append_notes`.

```python
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

Harika. Şimdi `top5_recall` adlı bir değerlendirme metriği oluşturalım.

Bu metrik, program tarafından döndürülen ilk 5 başlık içinde bulunan altın standart sayfaların (bunlar her zaman 3 tanedir) oranını döndürecektir.

```python
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

Hazır programımızı şimdi değerlendirelim!

```python
evaluate(Hop())
```

<details>
<summary>MLflow Experiment içinde Değerlendirme Sonuçlarını İzleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.


```python
import mlflow

with mlflow.start_run(run_name="hop_evaluation"):
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=top5_recall,
        num_threads=16,
        display_progress=True,
    )

    # Programı her zamanki gibi değerlendir
    result = evaluate(Hop())

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

Şimdi programımızın recall değerini en üst düzeye çıkarmak için `Hop()` programının içindeki iki istemi birlikte optimize edelim. Bu işlem yaklaşık 35 dakika sürebilir ve Llama-3.1-8B’yi optimize etmek için GPT-4o’ya yaklaşık 5 dolarlık çağrı yapabilir.

```python
models = dict(prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o))
tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **models)

kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4)
optimized = tp.compile(Hop(), trainset=trainset, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)
```

Şimdi optimizasyondan sonra tekrar değerlendirelim.

```python
evaluate(optimized)
```

Harika. Görünüşe göre sistem yaklaşık %30 recall’dan %60’ın biraz altına kadar ciddi biçimde iyileşti. Bu oldukça doğrudan bir yaklaşımdı, ancak DSPy sana buradan devam ederek yineleme yapman için birçok araç sunuyor.

Sonraki adımda, ne öğrendiğini anlamak için optimize edilmiş istemleri inceleyelim. Bir sorgu çalıştıracağız ve ardından son iki istemi inceleyeceğiz; bu da bize `Hop()` programının sonraki yinelemesinde her iki alt modül için kullanılan istemleri gösterecek. (Alternatif olarak, yukarıdaki talimatları izleyerek MLflow Tracing’i etkinleştirdiyseniz, ajan tarafından yapılan tüm adımları; LLM çağrıları, istemler, araç yürütmeleri dahil olmak üzere, zengin bir ağaç görünümünde görebilirsiniz.)

```python
optimized(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
```

```python
dspy.inspect_history(n=2)
```

Son olarak, optimize edilmiş programımızı daha sonra tekrar kullanabilmek için kaydedelim.

```python
optimized.save("optimized_hop.json")

loaded_program = Hop()
loaded_program.load("optimized_hop.json")

loaded_program(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
```

<details>
<summary>Programları MLflow Experiment içinde kaydetme</summary>

<br/>

Programı yerel bir dosyaya kaydetmek yerine, daha iyi yeniden üretilebilirlik ve iş birliği için MLflow içinde takip edebilirsiniz.

1. **Bağımlılık Yönetimi**: MLflow, yeniden üretilebilirliği sağlamak için dondurulmuş ortam meta verisini programla birlikte otomatik olarak kaydeder.
2. **Deney Takibi**: MLflow ile programın performansını ve maliyetini, programın kendisiyle birlikte takip edebilirsiniz.
3. **İş Birliği**: MLflow deneyini paylaşarak programı ve sonuçları ekip üyelerinizle paylaşabilirsiniz.

Programı MLflow içine kaydetmek için aşağıdaki kodu çalıştırın:

```python
import mlflow

# Bir MLflow Run başlat ve programı kaydet
with mlflow.start_run(run_name="optimized"):
    model_info = mlflow.dspy.log_model(
        optimized,
        artifact_path="model", # Programı MLflow içinde kaydetmek için herhangi bir ad
    )

# Programı MLflow’dan tekrar yükle
loaded = mlflow.dspy.load_model(model_info.model_uri)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>
