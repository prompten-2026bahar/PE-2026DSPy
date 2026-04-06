# Çıktı İyileştirme: BestOfN ve Refine

Hem `BestOfN` hem de `Refine`, önbelleği aşmak için farklı rollout ID’leriyle birden fazla `LM` çağrısı yaparak tahminlerin güvenilirliğini ve kalitesini artırmak üzere tasarlanmış DSPy modülleridir. Her iki modül de `N` denemeye ulaştığında veya `reward_fn`, `threshold` değerinin üzerinde bir ödül döndürdüğünde durur.

## BestOfN

`BestOfN`, verilen modülü farklı rollout ID’leriyle birden fazla kez (`N`’e kadar) çalıştıran bir modüldür. Belirtilen eşiği geçen ilk tahmini ya da hiçbiri eşiği karşılamazsa en yüksek ödüle sahip olan tahmini döndürür.

### Temel Kullanım

Diyelim ki modelden tek kelimelik bir yanıt alma şansımızı en üst düzeye çıkarmak istiyoruz. Birden fazla rollout ID deneyip en iyi sonucu döndürmek için `BestOfN` kullanabiliriz.

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Hata Yönetimi

Varsayılan olarak, modül bir deneme sırasında hatayla karşılaşırsa `N` denemeye ulaşana kadar denemeye devam eder. Bu davranışı `fail_count` parametresiyle ayarlayabilirsiniz:

```python
best_of_3 = dspy.BestOfN(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="What is the capital of Belgium?")
# ilk hatadan sonra hata fırlatır
```

## Refine

`Refine`, otomatik bir geri bildirim döngüsü ekleyerek `BestOfN` işlevselliğini genişletir. Her başarısız denemeden sonra (son deneme hariç), modülün performansı hakkında otomatik olarak ayrıntılı geri bildirim üretir ve bu geri bildirimi sonraki çalıştırmalar için ipuçları olarak kullanır.

### Temel Kullanım

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Hata Yönetimi

`BestOfN` gibi, `Refine` da hata oluşsa bile varsayılan olarak en fazla `N` kez deneme yapar. Bunu `fail_count` parametresiyle kontrol edebilirsiniz:

```python
# İlk hatadan sonra dur
refine = dspy.Refine(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)
```

## Karşılaştırma: BestOfN ve Refine

Her iki modül de benzer amaçlara hizmet eder, ancak yaklaşımları farklıdır:

- `BestOfN`, yalnızca farklı rollout ID’lerini dener ve `reward_fn` tarafından tanımlanan en iyi sonucu seçer.
- `Refine`, bir geri bildirim döngüsü ekler; önceki tahmini ve `reward_fn` içindeki kodu kullanarak modülün kendi performansı hakkında LM ile ayrıntılı geri bildirim üretir. Bu geri bildirim daha sonra sonraki çalıştırmalar için ipuçları olarak kullanılır.

## Pratik Örnekler

### Olgusal Doğruluğu Sağlama

```python
import dspy

class FactualityJudge(dspy.Signature):
    """Bir ifadenin olgusal olarak doğru olup olmadığını belirle."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)
```

### Özetleme - Yanıt Uzunluğunu Kontrol Etme

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Özeti, 75 kelimeye yakın olduğu için ödüllendir; daha uzun özetlerde ödül kademeli olarak azalsın.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[Long text to summarize...]"
)
print(result.summary)
```

## `dspy.Suggest` ve `dspy.Assert` Üzerinden Geçiş

`BestOfN` ve `Refine`, DSPy 2.6 itibarıyla `dspy.Suggest` ve `dspy.Assert` için gelen yerine geçen modüllerdir.
