# Metrikler

DSPy bir makine öğrenmesi çerçevesidir, bu nedenle değerlendirme (ilerlemenizi takip etmek için) ve optimizasyon (DSPy'nin programlarınızı daha etkili hale getirebilmesi için) için **otomatik metrikleriniz** üzerine düşünmelisiniz.


## Metrik nedir ve görevim için nasıl bir metrik tanımlarım?

Metrik, verilerinizden örnekleri ve sisteminizin çıktısını alan ve çıktının ne kadar iyi olduğunu nicelleştiren bir puan döndüren bir fonksiyondur. Sisteminizden gelen çıktıları ne iyi veya kötü yapar?

Basit görevler için bu sadece "doğruluk" (accuracy), "tam eşleşme" (exact match) veya "F1 puanı" olabilir. Bu durum, basit sınıflandırma veya kısa yanıtlı soru-cevap (QA) görevleri için geçerli olabilir.

Ancak çoğu uygulama için sisteminiz uzun formda çıktılar üretecektir. Bu durumda metriğiniz, muhtemelen çıktının birden fazla özelliğini kontrol eden (büyük olasılıkla dil modellerinden gelen yapay zeka geri bildirimini kullanan) daha küçük bir DSPy programı olmalıdır.

Bunu ilk denemede doğru yapmak pek olası değildir; bu yüzden basit bir şeyle başlamalı ve üzerinde yineleme yapmalısınız.


## Basit metrikler

Bir DSPy metriği, `example` (örneğin eğitim veya geliştirme setinizden) ve DSPy programınızdan gelen `pred` çıktısını alan ve bir `float` (veya `int` ya da `bool`) puanı döndüren bir Python fonksiyonudur.

Metriğiniz ayrıca `trace` adı verilen isteğe bağlı bir üçüncü argümanı da kabul etmelidir. Bunu şimdilik görmezden gelebilirsiniz, ancak metriğinizi optimizasyon için kullanmak isterseniz bazı güçlü hilelere olanak tanıyacaktır.

İşte `example.answer` ve `pred.answer` değerlerini karşılaştıran basit bir metrik örneği. Bu özel metrik bir `bool` değeri döndürecektir.


```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

Bazı kişiler şu yerleşik (built-in) yardımcı araçları kullanışlı bulmaktadır:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Metrikleriniz daha karmaşık olabilir; örneğin birden fazla özelliği kontrol edebilir. Aşağıdaki metrik, `trace is None` ise (yani değerlendirme veya optimizasyon için kullanılıyorsa) bir `float` döndürecek, aksi takdirde (yani örnek gösterimleri -demonstrations- türetmek için kullanılıyorsa) bir `bool` döndürecektir.

```python
def validate_context_and_answer(example, pred, trace=None):
    # Doğru etiket ile tahmin edilen cevabın aynı olup olmadığını kontrol edin
    answer_match = example.answer.lower() == pred.answer.lower()

    # Tahmin edilen cevabın getirilen bağlamlardan (contexts) birinden gelip gelmediğini kontrol edin
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # Değerlendirme veya optimizasyon yapıyorsak
        return (answer_match + context_match) / 2.0
    else: # Özyükleme (bootstrapping) yapıyorsak, yani her adımın iyi örneklerini (demonstrations) kendimiz oluşturuyorsak
        return answer_match and context_match
```

İyi bir metrik tanımlamak yinelemeli (iteratif) bir süreçtir; bu nedenle başlangıç değerlendirmeleri yapmak, verilerinize ve çıktılarınıza bakmak temel önem taşır.

## Değerlendirme (Evaluation)

Bir metriğiniz olduğunda, basit bir Python döngüsü içinde değerlendirmeler çalıştırabilirsiniz.

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

Eğer bazı yardımcı araçlara ihtiyaç duyarsanız, yerleşik `Evaluate` aracını da kullanabilirsiniz. Bu araç, paralel değerlendirme (çoklu iş parçacığı - multiple threads) veya girdi/çıktı örnekleri ile metrik puanlarının gösterilmesi gibi konularda yardımcı olabilir.

```python
from dspy.evaluate import Evaluate

# Kodunuzda tekrar kullanılabilen değerlendiriciyi kurun.
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# Değerlendirmeyi başlatın.
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```


## Orta Seviye: Metriğiniz İçin Yapay Zeka Geri Bildirimi Kullanma

Çoğu uygulama için sisteminiz uzun formda çıktılar üretecektir; bu nedenle metriğiniz, dil modellerinden (LM) gelen yapay zeka geri bildirimini kullanarak çıktının birden fazla boyutunu kontrol etmelidir.

Bu basit imza (signature) işinize yarayabilir:

```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

Örneğin, aşağıda üretilen bir tweet'in (1) verilen bir soruyu doğru yanıtlayıp yanıtlamadığını ve (2) ilgi çekici olup olmadığını kontrol eden basit bir metrik bulunmaktadır. Ayrıca (3) `len(tweet) <= 280` karakter sınırını da kontrol ediyoruz.

```python
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

Derleme (compiling) sırasında `trace is not None` durumundadır ve değerlendirme konusunda katı olmak isteriz; bu nedenle yalnızca `score >= 2` ise `True` döndüreceğiz. Aksi takdirde, 1.0 üzerinden bir puan (yani `score / 2.0`) döndürürüz.

## İleri Seviye: Metriğiniz Olarak Bir DSPy Programı Kullanmak

Eğer metriğinizin kendisi bir DSPy programıysa, yineleme yapmanın en güçlü yollarından biri metriğin kendisini derlemek (optimize etmek) olacaktır. Metriğin çıktısı genellikle basit bir değer (örneğin 5 üzerinden bir puan) olduğundan, bu genellikle kolaydır; böylece metriğin kendi metriğini tanımlamak ve birkaç örnek toplayarak onu optimize etmek oldukça basit hale gelir.

### İleri Seviye: `trace` Nesnesine Erişim

Metriğiniz değerlendirme çalışmaları sırasında kullanıldığında, DSPy programınızın adımlarını izlemeye çalışmayacaktır.

Ancak derleme (optimizasyon) sırasında DSPy, Dil Modeli (LM) çağrılarınızı izleyecektir (trace). İzleme kaydı, her bir DSPy tahmincisinin (predictor) girdilerini/çıktılarını içerecektir ve siz de bunu optimizasyon için ara adımları doğrulamak amacıyla kullanabilirsiniz.


```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```
