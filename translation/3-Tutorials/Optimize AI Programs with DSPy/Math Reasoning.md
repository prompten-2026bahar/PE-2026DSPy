# Eğitim: Matematiksel Akıl Yürütme

Cebir sorularını yanıtlamak için bir `dspy.ChainOfThought` modülünü kurmaya ve onu optimize etmeye yönelik hızlı bir örneği adım adım inceleyelim.

En güncel DSPy sürümünü `pip install -U dspy` ile kurup devam edin. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekir.

<details>
<summary>Önerilir: Arka planda neler olduğunu anlamak için MLflow Tracing kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, istemleri ve optimizasyon ilerlemesini izler olarak görselleştirmek için MLflow kullanabilir, böylece DSPy’nin davranışını daha iyi anlayabilirsiniz. MLflow’u aşağıdaki dört adımı izleyerek kolayca kurabilirsiniz.

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

Yukarıdaki adımları tamamladıktan sonra, notebook üzerinde her program çalıştırması için izleri görebilirsiniz. Bunlar modelin davranışı üzerinde güçlü bir görünürlük sağlar ve eğitim boyunca DSPy kavramlarını daha iyi anlamanıza yardımcı olur.

![MLflow Trace](./mlflow-tracing-math.png)

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

Şimdi DSPy’ye modüllerimizde OpenAI’ın `gpt-4o-mini` modelini kullanacağımızı söyleyelim. Kimlik doğrulama için DSPy, `OPENAI_API_KEY` değişkeninize bakacaktır. Bunu kolayca [diğer sağlayıcılar veya yerel modellerle](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) değiştirebilirsiniz.

```python
import dspy

gpt4o_mini = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=2000)
dspy.configure(lm=gpt4o_mini)  # aksi belirtilmedikçe varsayılan LM olarak gpt-4o-mini kullanacağız
```

Ardından, [MATH](https://arxiv.org/abs/2103.03874) kıyaslamasından bazı veri örnekleri yükleyelim. Optimizasyon için eğitim bölümünü kullanacağız ve bunu ayrılmış bir geliştirme kümesi üzerinde değerlendireceğiz.

Lütfen aşağıdaki adımın şunu gerektireceğini unutmayın:
```bash
%pip install git+https://github.com/hendrycks/math.git
```

```python
from dspy.datasets import MATH

dataset = MATH(subset='algebra')
print(len(dataset.train), len(dataset.dev))
```

Eğitim kümesinden bir örneği inceleyelim.

```python
example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)
```

Şimdi modülümüzü tanımlayalım. Son derece basit: yalnızca bir `question` alıp bir `answer` üreten tek bir chain-of-thought adımı.

```python
module = dspy.ChainOfThought("question -> answer")
module(question=example.question)
```

Sıradaki adımda, istem optimizasyonundan önce yukarıdaki zero-shot modül için bir değerlendirici ayarlayalım.

```python
THREADS = 24
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

evaluate(module)
```

<details>
<summary>MLflow Experiment içinde değerlendirme sonuçlarını izleme</summary>

<br/>

Değerlendirme sonuçlarını zaman içinde takip etmek ve görselleştirmek için sonuçları MLflow Experiment içine kaydedebilirsiniz.

```python
import mlflow

# Değerlendirmeyi kaydetmek için bir MLflow Run başlat
with mlflow.start_run(run_name="math_evaluation"):
    kwargs = dict(num_threads=THREADS, display_progress=True)
    evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)

    # Programı her zamanki gibi değerlendir
    result = evaluate(module)

    # Toplu skoru kaydet
    mlflow.log_metric("correctness", result.score)
    # Ayrıntılı değerlendirme sonuçlarını tablo olarak kaydet
    mlflow.log_table(
        {
            "Question": [example.question for example in dataset.dev],
            "Gold Answer": [example.answer for example in dataset.dev],
            "Predicted Answer": [output[1] for output in result.results],
            "Correctness": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.

</details>

Ve son olarak modülümüzü optimize edelim. Güçlü akıl yürütme istediğimiz için büyük GPT-4o modelini öğretmen model olarak kullanacağız (optimizasyon sırasında küçük LM için akıl yürütmeyi bootstrap etmekte kullanılır) ancak bunu prompt modeli olarak (talimatları oluşturmakta kullanılır) ya da görev modeli olarak (eğitilen model) kullanmayacağız.

GPT-4o yalnızca az sayıda kez çağrılacaktır. Doğrudan optimizasyonda ve ortaya çıkan (optimize edilmiş) programda yer alan model GPT-4o-mini olacaktır.

Ayrıca `max_bootstrapped_demos=4` belirteceğiz; bu, prompt içinde en fazla dört bootstrap edilmiş örnek istediğimiz anlamına gelir. `max_labeled_demos=4` ise bootstrap edilmiş ve önceden etiketlenmiş örneklerin toplamının en fazla dört olmasını istediğimiz anlamına gelir.

```python
kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o_mini)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)
```

```python
evaluate(optimized_module)
```

Harika. Burada ayrılmış bir küme üzerinde kaliteyi %74’ten %88’in üzerine çıkarmak oldukça doğrudan oldu.

Bununla birlikte, bunun gibi akıl yürütme görevlerinde genellikle daha gelişmiş stratejileri değerlendirmek isteyebilirsiniz; örneğin:

- Bir hesap makinesi işlevine veya `dspy.LocalSandbox`'a erişimi olan bir `dspy.ReAct` modülü
- Üstte çoğunluk oyu (veya bir Aggregator modülü) ile birden fazla optimize edilmiş prompt’u topluluk haline getirmek

Sadece neyin değiştiğini anlamak için, optimizasyondan sonraki prompt’a bakalım. Alternatif olarak, yukarıdaki talimatları izleyerek MLflow tracing’i etkinleştirdiyseniz, zengin iz arayüzünde optimizasyon öncesi ve sonrası prompt’ları karşılaştırabilirsiniz.

```python
dspy.inspect_history()
```
