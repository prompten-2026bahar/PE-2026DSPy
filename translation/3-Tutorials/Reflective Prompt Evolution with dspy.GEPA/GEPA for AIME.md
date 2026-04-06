# Eğitim: AIME (Matematik) için GEPA

Bu eğitimde, matematik problemlerini (AIME) çözmek için GPT-4.1 Mini’nin Chain of Thought (`dspy.ChainOfThought`) yapısını `dspy.GEPA` optimize edicisiyle optimize ediyoruz!

<details>
<summary>Önerilir: Perde arkasında neler olduğunu anlamak için MLflow Autologging kurun.</summary>

### MLflow DSPy Entegrasyonu

<a href="https://mlflow.org/">MLflow</a>, DSPy ile doğal olarak entegre olan ve açıklanabilirlik ile deney takibi sunan bir LLMOps aracıdır. MLflow’un autologging özelliği, GEPA optimizasyonunun ilerlemesini otomatik olarak izler; ayrıca istemleri ve modül çalıştırmalarını izler halinde görselleştirerek DSPy’nin davranışını daha iyi anlamanızı sağlar. Aşağıdaki dört adımı izleyerek MLflow’u kolayca kurabilirsiniz.

**Modül çalıştırmalarını izler olarak görselleştirin**

![MLflow Trace](./mlflow-tracing-gepa-aime.png)

**Optimizasyon ilerlemesini ve sonuçları otomatik olarak izleyin**

![MLflow Tracking](./mlflow-tracking-gepa-aime-optimization.png)

**MLflow’u kurun**

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
    # Optimizasyon ilerlemesini kaydet
    log_compiles=True,
    # Değerlendirme sonuçlarını kaydet
    log_evals=True,
    # Modül çalıştırmalarından izleri kaydet
    log_traces=True
)
```

Entegrasyon hakkında daha fazla bilgi edinmek için [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) sayfasını da ziyaret edin.
</details>

```python
api_key = input("OpenAI API anahtarınızı girin: ")
import dspy
lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=api_key, max_tokens=32000)
dspy.configure(lm=lm)
```

### AIME veri kümesini yükleme

AIME sınavı, her yıl için 15 soruluk 2 problem setinden oluşur. Bu eğitimde, optimizasyon için önceki yıllara ait (2022-2024) AIME problem setlerini kullanacağız (toplamda 3 yıl x 2 set x 15 problem = 90 problem; bunlar eğitim ve doğrulama kümeleri arasında eşit biçimde bölünecek) ve performansı AIME 2025 üzerinde test edeceğiz (2 set x 15 problem = 30 problem). AIME 2025 küçük bir küme olduğu için, değerlendirmede istatistiksel kararlılık sağlamak amacıyla bunu 5 kez tekrar ediyoruz.

```python
import dspy
from datasets import load_dataset

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]

    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split * 5

    return train_set, val_set, test_set
```

```python
train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)
```

Bir örnek görev girdisine bakalım

```python
print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])
```

### Programı tanımlayalım: Basit bir `dspy.ChainOfThought`

```python
class GenerateResponse(dspy.Signature):
    """Problemi çöz ve cevabı doğru biçimde ver."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)
```

### Değerlendirme metriğini tanımlama

Sadece tahmin edilen cevap ile doğru cevap arasındaki tam eşleşmeyi kontrol ediyoruz.

```python
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)
```

### Optimize edilmemiş Chain Of Thought’u değerlendirme

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

### Programı `dspy.GEPA` ile optimize etme

GEPA, _yansıtıcı_ bir istem optimize edicisidir ve gücü, DSPy programının yürütme ve değerlendirme hatları gibi ek bilgi kaynaklarından yararlanabilmesinde yatar. Bunlar GEPA’ya sistemin neden belirli bir puanı aldığını anlama konusunda daha fazla görünürlük sağlar; ardından GEPA içgörü geliştirerek puanı nasıl iyileştireceğini belirleyebilir. GEPA bu şekilde sağlanan ek denetimden de yararlanabilir. Örneğin optimizasyon sırasında, programın çözemediği problemlerin doğru çözümlerini geri döndürebiliriz.

Bununla birlikte, bu kadar açık denetimin her senaryoda mevcut olmadığını belirtelim. GEPA, farklı geri bildirim biçimleriyle çok esnek biçimde çalışabilir (örneğin PAPILLON eğitiminde gösterilen LLM-as-a-judge geri bildirimi ya da facility-support eğitiminde gösterildiği gibi yalnızca cevap etiketleriyle).

Şimdi değerlendirme metriğini, GEPA için ek denetim sağlayabilecek bir optimizasyon metriğine dönüştürelim!

```python
def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"Nihai cevap geçerli bir tam sayı olmalı ve başka hiçbir şey içermemelidir. Sen '{prediction.answer}' yanıtını verdin; bu bir Python tam sayısı olarak ayrıştırılamadı. Lütfen cevabının ek metin veya biçimlendirme olmadan geçerli bir tam sayı olduğundan emin ol."
        feedback_text += f" Doğru cevap '{correct_answer}' değeridir."
        if written_solution:
            feedback_text += f" İşte tam adım adım çözüm:\n{written_solution}\n\nBu çözümden hangi çıkarımları öğrenebileceğini düşün; böylece gelecekteki cevaplarını ve benzer problemlere yaklaşımını geliştirebilir ve nihai cevabının geçerli bir tam sayı olmasını sağlayabilirsin."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Cevabın doğru. Doğru cevap '{correct_answer}' değeridir."
    else:
        feedback_text = f"Cevabın yanlış. Doğru cevap '{correct_answer}' değeridir."

    if written_solution:
        feedback_text += f" İşte tam adım adım çözüm:\n{written_solution}\n\nBu çözümden hangi çıkarımları öğrenebileceğini düşün; böylece gelecekteki cevaplarını ve benzer problemlere yaklaşımını geliştirebilirsin."

    return dspy.Prediction(score=score, feedback=feedback_text)
```

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

### Üretilen isteme bakalım

```python
print(optimized_program.predict.signature.instructions)
```

Burada GEPA’nın yaptığı şeyin, gelecekteki görev örnekleri için iyi bir plan oluşturmak amacıyla bazı akıl yürütmeleri önceden hesaplamak olduğu görülebilir. Görülmemiş doğrulama kümesindeki geliştirilmiş performans nedeniyle bu istemin genellenmesini bekliyoruz!

### GEPA ile optimize edilmiş Chain Of Thought’u değerlendirme

```python
evaluate(optimized_program)
```

GEPA, GPT-4.1 Mini’nin AIME 2025 üzerindeki performansını **%46,6 skordan %56,6 skora** yükseltmeyi başardı; üstelik yalnızca `auto="light"` bütçesiyle, yani %10’luk bir iyileşme sağladı!
