# Eğitim: Kurumsal Görevler için Yapılandırılmış Bilgi Çıkarımında GEPA

Bu eğitimde, Meta tarafından yayımlanan [Facility Support Analyzer](https://github.com/meta-llama/llama-prompt-ops/tree/main/use-cases/facility-support-analyzer) veri kümesini kullanarak yapılandırılmış bilgi çıkarımı ve sınıflandırmaya yönelik üç parçalı bir görevi inceleyeceğiz. Tesis bakımı veya destek talepleriyle ilgili olarak kurumsal bir ortamda gönderilen bir e-posta ya da mesaj verildiğinde, amaç bunun aciliyetini çıkarmak, duygu durumunu değerlendirmek ve ilgili tüm hizmet talebi kategorilerini belirlemektir. fileciteturn16file0

Basit bir DSPy programı oluşturacağız ve ardından görev için bunu optimize etmek üzere `dspy.GEPA` optimize edicisini kullanacağız. fileciteturn16file0

<details>
<summary>Önerilir: Perde arkasında neler olduğunu anlamak için MLflow Autologging kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. MLflow’un autologging yeteneği, GEPA optimizasyonunun ilerlemesini otomatik olarak izler; ayrıca DSPy’nin davranışını daha iyi anlamak için istemleri ve modül yürütmelerini izler halinde görselleştirir. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz. fileciteturn16file0

**Modül yürütmelerini izler olarak görselleştirin**

![MLflow Trace](./mlflow-tracing-gepa-support.png)

**Optimizasyon ilerlemesini ve sonuçları otomatik olarak izleyin**

![MLflow Tracking](./mlflow-tracking-gepa-support-optimization.png)

**MLflow’u kurma**

1. MLflow’u yükleyin

```bash
%pip install mlflow>=3.0.0
```

2. Ayrı bir terminalde MLflow UI’ı başlatın
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. Notebook’u MLflow’a bağlayın
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. Autologging’i etkinleştirin.

```python
mlflow.dspy.autolog(
    # Optimizasyon ilerlemesini günlüğe kaydet
    log_compiles=True,
    # Değerlendirme sonuçlarını günlüğe kaydet
    log_evals=True,
    # Modül yürütmelerinden izleri günlüğe kaydet
    log_traces=True
)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin. fileciteturn16file0
</details>

### Dil modelini ayarlama

GEPA ile küçük bir modelin nasıl ayarlanabildiğini göstermek için GPT-4.1 nano kullanıyoruz. fileciteturn16file0

```python
api_key = input("OpenAI API anahtarınızı girin: ")
import dspy
lm = dspy.LM("openai/gpt-4.1-nano", temperature=1, api_key=api_key)
dspy.configure(lm=lm)
```

### Veri kümesini yükleme

```python
import requests
import dspy
import json
import random

def init_dataset():
    # URL üzerinden yükle
    url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
    dataset = json.loads(requests.get(url).text)
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]
    random.Random(0).shuffle(dspy_dataset)
    train_set = dspy_dataset[:int(len(dspy_dataset) * 0.33)]
    val_set = dspy_dataset[int(len(dspy_dataset) * 0.33):int(len(dspy_dataset) * 0.66)]
    test_set = dspy_dataset[int(len(dspy_dataset) * 0.66):]

    return train_set, val_set, test_set
```

```python
train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)
```

Örnek bir görev girdisine bakalım. fileciteturn16file0

```python
print("Girdi Mesajı:")
print(train_set[0]['message'])

print("\n\nAltın Standart Cevap:")
for k, v in json.loads(train_set[0]['answer']).items():
    print(f"{k}: {v}")
```

### Görevi çözmek için bir DSPy programı tanımlama

Program, sırasıyla aciliyet, duygu durumu ve kategori sınıflandırmasını ele alan 3 modüllü bir sistemdir. fileciteturn16file0

```python
from typing import List, Literal


class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """
    Verilen mesajı oku ve aciliyeti belirle.
    """
    message: str = dspy.InputField()
    urgency: Literal['low', 'medium', 'high'] = dspy.OutputField()

class FacilitySupportAnalyzerSentiment(dspy.Signature):
    """
    Verilen mesajı oku ve duygu durumunu belirle.
    """
    message: str = dspy.InputField()
    sentiment: Literal['positive', 'neutral', 'negative'] = dspy.OutputField()

class FacilitySupportAnalyzerCategories(dspy.Signature):
    """
    Verilen mesajı oku ve mesaja uygulanabilir kategori kümesini belirle.
    """
    message: str = dspy.InputField()
    categories: List[Literal["emergency_repair_services", "routine_maintenance_requests", "quality_and_safety_concerns", "specialized_cleaning_services", "general_inquiries", "sustainability_and_environmental_practices", "training_and_support_requests", "cleaning_services_scheduling", "customer_feedback_and_complaints", "facility_management_issues"]] = dspy.OutputField()

class FacilitySupportAnalyzerMM(dspy.Module):
    def __init__(self):
        self.urgency_module = dspy.ChainOfThought(FacilitySupportAnalyzerUrgency)
        self.sentiment_module = dspy.ChainOfThought(FacilitySupportAnalyzerSentiment)
        self.categories_module = dspy.ChainOfThought(FacilitySupportAnalyzerCategories)

    def forward(self, message: str):
        urgency = self.urgency_module(message=message)
        sentiment = self.sentiment_module(message=message)
        categories = self.categories_module(message=message)

        return dspy.Prediction(
            urgency=urgency.urgency,
            sentiment=sentiment.sentiment,
            categories=categories.categories
        )

program = FacilitySupportAnalyzerMM()
```

### Çıktıları değerlendirmek için metriği tanımlama

Bu metrik, üç görevin tamamının çıktısını değerlendirir ve toplu puanı döndürür. fileciteturn16file0

```python
def score_urgency(gold_urgency, pred_urgency):
    """
    Aciliyet modülü için puanı hesapla.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    return score

def score_sentiment(gold_sentiment, pred_sentiment):
    """
    Duygu durumu modülü için puanı hesapla.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    return score

def score_categories(gold_categories, pred_categories):
    """
    Kategoriler modülü için puanı hesapla.
    Puan içinde kategori doğruluğuyla aynı eşleşme/eşleşmeme mantığını kullanır.
    """
    correct = 0
    for k, v in gold_categories.items():
        if v and k in pred_categories:
            correct += 1
        elif not v and k not in pred_categories:
            correct += 1
    score = correct / len(gold_categories)
    return score

def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Kategoriler, duygu durumu ve aciliyet için tahmin ile altın standart arasındaki uyuma göre bir puan hesaplar.
    Puanı (float) döndürür.
    """
    # Altın standardı örnekten ayrıştır
    gold = json.loads(example['answer'])

    # Tüm modüller için puanları hesapla
    score_urgency_val = score_urgency(gold['urgency'], pred.urgency)
    score_sentiment_val = score_sentiment(gold['sentiment'], pred.sentiment)
    score_categories_val = score_categories(gold['categories'], pred.categories)

    # Genel puan: üç doğruluğun ortalaması
    total = (score_urgency_val + score_sentiment_val + score_categories_val) / 3

    return total
```

### Optimize edilmemiş programı değerlendirme (GPT-4.1 nano ile çalıştırma)

```python
import dspy
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

evaluate(program)
```

### GEPA ile optimize etme

GEPA, _yansıtıcı_ bir istem optimize edicisidir. Gücü, DSPy programının yürütme ve değerlendirme hatlarından gelen metinsel geri bildirimi inceleyebilmesinde yatar. Bu, GEPA’ya sistemin neden belirli bir puan aldığını anlaması için daha fazla görünürlük sağlar ve performansı artırmanın yollarını içgözlemle belirlemesini mümkün kılar. fileciteturn16file0

Bu senaryoda, nihai puan üç ayrı görevdeki performansa dayanır. Her kestiricinin genel puanın belirli bir bölümünü ele aldığı açıkça görülebilir. fileciteturn16file0

GEPA, tek tek kestirici düzeyinde geri bildirim sağlamayı destekler (bu zorunlu değildir; buna örnek olarak GEPA PAPILLON eğitimine bakabilirsiniz). Değerlendirme metriğimizi, aynı zamanda metin geri bildirimi de sağlayan bir optimizasyon metriğine dönüştürmek için küçük bir değişiklik yapalım! fileciteturn16file0

```python
import json
import dspy

def feedback_urgency(gold_urgency, pred_urgency):
    """
    Aciliyet modülü için geri bildirim üret.
    """
    score = 1.0 if gold_urgency == pred_urgency else 0.0
    if gold_urgency == pred_urgency:
        feedback = f"Mesajın aciliyetini `{gold_urgency}` olarak doğru sınıflandırdın. Bu mesaj gerçekten `{gold_urgency}` aciliyetindedir."
    else:
        feedback = f"Mesajın aciliyetini `{pred_urgency}` olarak yanlış sınıflandırdın. Doğru aciliyet `{gold_urgency}` olmalıydı. Doğru aciliyet etiketine ulaşmak için nasıl akıl yürütebileceğini düşün."
    return feedback, score

def feedback_sentiment(gold_sentiment, pred_sentiment):
    """
    Duygu durumu modülü için geri bildirim üret.
    """
    score = 1.0 if gold_sentiment == pred_sentiment else 0.0
    if gold_sentiment == pred_sentiment:
        feedback = f"Mesajın duygu durumunu `{gold_sentiment}` olarak doğru sınıflandırdın. Bu mesaj gerçekten `{gold_sentiment}`."
    else:
        feedback = f"Mesajın duygu durumunu `{pred_sentiment}` olarak yanlış sınıflandırdın. Doğru duygu durumu `{gold_sentiment}`. Doğru duygu durumu etiketine ulaşmak için nasıl akıl yürütebileceğini düşün."
    return feedback, score

def feedback_categories(gold_categories, pred_categories):
    """
    Kategoriler modülü için geri bildirim üret.
    Puan içinde kategori doğruluğuyla aynı eşleşme/eşleşmeme mantığını kullanır.
    """
    correctly_included = [k for k, v in gold_categories.items() if v and k in pred_categories]
    incorrectly_included = [k for k, v in gold_categories.items() if not v and k in pred_categories]
    incorrectly_excluded = [k for k, v in gold_categories.items() if v and k not in pred_categories]
    correctly_excluded = [k for k, v in gold_categories.items() if not v and k not in pred_categories]  # doğruluk kontrolünde tamlık için

    # Kategori doğruluğunu yeniden hesapla (puan mantığıyla aynı)
    score = (len(correctly_included) + len(correctly_excluded)) / len(gold_categories)

    if score == 1.0:
        fb_text = f"Kategori sınıflandırması kusursuz. Mesajın şu kategorilere girdiğini doğru belirledin: `{repr(correctly_included)}`."
    else:
        fb_text = f"Kategori sınıflandırması kusursuz değil. Mesajın şu kategorilere girdiğini doğru belirledin: `{repr(correctly_included)}`.\n"
        if incorrectly_included:
            fb_text += f"Ancak, mesajın şu kategorilere girdiğini yanlış belirledin: `{repr(incorrectly_included)}`. Mesaj BU kategorilere girmez.\n"
        if incorrectly_excluded:
            prefix = "Ek olarak, " if incorrectly_included else "Ancak, "
            fb_text += f"{prefix}mesajın aslında girdiği şu kategorileri belirlemedin: `{repr(incorrectly_excluded)}`.\n"
        fb_text += "Doğru kategori etiketlerine ulaşmak için nasıl akıl yürütebileceğini düşün."
    return fb_text, score

def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Kategoriler, duygu durumu ve aciliyet için tahmin ile altın standart arasındaki uyuma göre bir puan hesaplar.
    İsteğe bağlı olarak, aynı karşılaştırma mantığını kullanarak belirli bir kestirici modül için geri bildirim metni sağlar.
    score (float) ve feedback (str) içeren bir dspy.Prediction döndürür.
    """
    # Altın standardı örnekten ayrıştır
    gold = json.loads(example['answer'])

    # Tüm modüller için geri bildirimleri ve puanları hesapla
    fb_urgency, score_urgency = feedback_urgency(gold['urgency'], pred.urgency)
    fb_sentiment, score_sentiment = feedback_sentiment(gold['sentiment'], pred.sentiment)
    fb_categories, score_categories = feedback_categories(gold['categories'], pred.categories)

    # Genel puan: üç doğruluğun ortalaması
    total = (score_urgency + score_sentiment + score_categories) / 3

    if pred_name is None:
        return total

    elif pred_name == 'urgency_module.predict':
        feedback = fb_urgency
    elif pred_name == 'sentiment_module.predict':
        feedback = fb_sentiment
    elif pred_name == 'categories_module.predict':
        feedback = fb_categories

    return dspy.Prediction(score=total, feedback=feedback)
```

Değerlendirme metriğinin, metin geri bildirimi üretmek için ihtiyaç duyduğumuz tüm bilgileri zaten içerdiğine dikkat edin; yalnızca neyin karşılaştırıldığını açıkça belirtecek şekilde bunu değiştirdik. Genel olarak, çoğu görev için metrik işlevleri bu tür geri bildirimleri oluşturmak için gerekli temel bileşenleri sağlar; genellikle yapılması gereken şey, programın performansı üzerinde düşünüp iyileştirebilmesi için hangi unsurların GEPA optimize edicisine görünür kılınacağını belirlemektir. fileciteturn16file0

GEPA’yı çalıştıralım. fileciteturn16file0

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light", # <-- Bu eğitim için hafif bir bütçe kullanacağız. Ancak optimize edilmiş performans için genellikle auto="heavy" öneriyoruz!
    num_threads=32,
    track_stats=True,
    use_merge=False,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
)
```

```python
optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

### Optimize edilmiş istemlere bakalım

```python
for name, pred in optimized_program.named_predictors():
    print("================================")
    print(f"Kestirici: {name}")
    print("================================")
    print("İstem:")
    print(pred.signature.instructions)
    print("*********************************")
```

İstemlerde görev hakkında öğrenilmiş ayrıntı düzeyinin ne kadar yüksek olduğuna dikkat edin! fileciteturn16file0

### Şimdi optimize edilmiş programı değerlendirelim

```python
evaluate(optimized_program)
```

GEPA, `auto="light"` ayarında GPT-4.1 nano’nun performansını **%75 puandan %87 puana** çıkarmayı başardı. fileciteturn16file0

### Bonus: Ayrıntılı Sonuçlar

`track_stats=True` ile yapılan bir GEPA çalıştırması, `detailed_results` niteliğinde ayrıntılı sonuçlar döndürür. fileciteturn16file0

- **candidates**: Önerilen adayların listesi.
- **parents**: Her aday için ebeveyn indisleri listesi veya None.
- **val_aggregate_scores**: Her aday için toplu doğrulama puanı.
- **val_subscores**: Her aday için örnek başına doğrulama puanları.
- **per_val_instance_best_candidates**: Her doğrulama örneği için en iyi adayların indisleri.
- **discovery_eval_counts**: Her adayın keşfi için kullanılan metrik çağrısı/rollout sayısı.
- **best_outputs_valset**: Her görev için üretilen en iyi çıktı (`track_best_outputs=True` ile varsa).
- **best_idx**: En yüksek puanlı adayın indisi.
- **best_candidate**: `best_idx` için program.

GEPA’nın bu görev için izlediği optimizasyon izleğini görselleştirelim. fileciteturn16file0

Özellikle, her aday program için ebeveyn programları tanımlayan `parents` niteliğine erişebiliriz. Bunları Graphviz DOT görselleştirmesi olarak çizmek için basit bir betik kullanıyoruz; bu görselleştirme yerelde Graphviz ile veya [GraphvizOnline](https://is.gd/meuHtO) gibi çevrim içi araçlarla işlenebilir. fileciteturn16file0

```python
def dag_to_dot(parent_program_for_candidate, dominator_program_ids, best_program_idx, full_eval_scores):
    dot_lines = [
        "digraph G {",
        "    node [style=filled, shape=circle, fontsize=50];"
    ]
    n = len(parent_program_for_candidate)
    # Düğümleri renkler ve etiketlerde puanlarla ayarla
    for idx in range(n):
        score = full_eval_scores[idx]
        label = f"{idx}\\n({score:.2f})"
        if idx == best_program_idx:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=cyan, fontcolor=black];')
        elif idx in dominator_program_ids:
            dot_lines.append(f'    {idx} [label="{label}", fillcolor=orange, fontcolor=black];')
        else:
            dot_lines.append(f'    {idx} [label="{label}"];')

    # Kenarları ayarla
    for child, parents in enumerate(parent_program_for_candidate):
        for parent in parents:
            if parent is not None:
                dot_lines.append(f'    {parent} -> {child};')

    dot_lines.append("}")
    return "\n".join(dot_lines)

from gepa.gepa_utils import find_dominator_programs
pareto_front_programs = find_dominator_programs(optimized_program.detailed_results.per_val_instance_best_candidates, optimized_program.detailed_results.val_aggregate_scores)

print(dag_to_dot(
    optimized_program.detailed_results.parents,
    pareto_front_programs,
    optimized_program.detailed_results.best_idx,
    optimized_program.detailed_results.val_aggregate_scores
))
```

![Rendered Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABgsAAAXUCAYAAADwSTw4AAAAAXNSR0IArs4c6QAAIABJREFUeF7s3Qm4TdX7wPHXkKkIKbPKHDI1yJChRBl/pgoZEypkiJThykwhKXNUEoXMypQoZSg/SRF+KmMSIfN0/8+7+p/Tude97jnnnmEP3/U8Perevfd612fte9z2u9d6U8TGxn4qNAQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEHCtQIrY2NhY146egSOAAAIIIIAAAggggAACCCCAAAIIIIAAAggggICQLOAmQAABBBBAAAEEEEAAAQQQQAABBBBAAAEEEEDA5QIkC1x+AzB8BBBAAAEEEEAAAQQQQAABBBBAAAEEEEAAAQRIFnAPIIAAAggggAACCCCAAAIIIIAAAggggAACCCDgcgGSBS6/ARg+AggggAACCCCAAAIIIIAAAggggAACCCCAAAIkC7gHEEAAAQQQQAABBBBAAAEEEEAAAQQQQAABBBBwuQDJApffAAwfAQQQQAABBBBAAAEEEEAAAQQQQAABBBBAAAGSBdwDCCCAAAIIIIAAAggggAACCNhQ4I8//pBjx47JyZMn5fTp03L+/Hm5dOmSxMbGSqpUqSRt2rSSUM... )
