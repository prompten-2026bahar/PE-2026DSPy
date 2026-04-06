# Eğitim: Retrieval-Augmented Generation (RAG)

DSPy'de bilgi getirme destekli üretim (RAG) ile ve RAG olmadan temel soru yanıtlama üzerine hızlı bir örneği inceleyelim. Spesifik olarak, örneğin Linux veya iPhone uygulamaları hakkındaki Teknoloji sorularını yanıtlamak için bir sistem inşa edelim.

En son DSPy'yi `pip install -U dspy` üzerinden kurun ve adımları takip edin. DSPy'ye kavramsal bir genel bakış arıyorsanız, bu [son ders](https://dspy.ai/learn/programming/intro) başlamak için iyi bir yerdir. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekir.

## DSPy Ortamını Yapılandırma

DSPy'ye modüllerimizde OpenAI'nin `gpt-4o-mini` modelini kullanacağımızı söylelelim. Kimlik doğrulama için DSPy, `OPENAI_API_KEY`'inize bakacaktır. Bunu diğer sağlayıcılar veya yerel modeller için kolayca değiştirebilirsiniz.

*Önerilen: Arka planda neler olduğunu anlamak için MLflow Tracing'i kurun.*

**MLflow DSPy Entegrasyonu**
MLflow, DSPy ile yerel olarak entegre olan, açıklanabilirlik ve deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, DSPy'nin davranışını daha iyi anlamak amacıyla istemleri ve optimizasyon ilerlemesini izler (traces) olarak görselleştirmek için MLflow'u kullanabilirsiniz. Aşağıdaki adımları izleyerek MLflow'u kolayca kurabilirsiniz.

1. MLflow'u kurun.
```bash
%pip install mlflow>=3.0.0
```
2. Ayrı bir terminalde MLflow UI'yi başlatın.
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```
3. Notebook'u MLflow'a bağlayın.
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```
4. İzlemeyi etkinleştirin.
```python
mlflow.dspy.autolog()
```

Yukarıdaki adımları tamamladığınızda, not defterinde her program yürütmesi için izleri (traces) görebilirsiniz. Modelin davranışına dair harika bir görünürlük sağlarlar ve eğitim boyunca DSPy'nin kavramlarını daha iyi anlamanıza yardımcı olurlar. Entegrasyon hakkında daha fazla bilgi edinmek için MLflow DSPy Dokümantasyonunu da ziyaret edebilirsiniz.

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

## Bazı Temel DSPy Modüllerini Keşfetme

LM'e her zaman `lm(prompt="prompt")` veya `lm(messages=[...])` üzerinden doğrudan istem (prompt) gönderebilirsiniz. Ancak DSPy, LM işlevlerinizi tanımlamanın daha iyi bir yolu olarak size Modüller (Modules) sunar.

En basit modül `dspy.Predict`'tir. Bir DSPy İmzası (Signature), yani yapılandırılmış bir girdi/çıktı şeması alır ve size belirttiğiniz davranış için çağrılabilir bir fonksiyon geri verir. Girdi olarak bir soru (`str` türünde) alan ve çıktı olarak bir yanıt üreten bir modül bildirmek için imzaların "satır içi" (in-line) gösterimini kullanalım.

```python
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")
print(response.response)
```

Belirttiğimiz değişken adlarının girdi ve çıktı argüman adlarımızı ve rollerini nasıl tanımladığına dikkat edin. Peki, DSPy bu `qa` modülünü oluşturmak için ne yaptı? Bu örnekte henüz şatafatlı bir şey yok. Modül; imzanızı, LM'i ve girdileri, girdileri yapılandırmayı ve yapılandırılmış çıktıları imzanıza uyacak şekilde ayrıştırmayı (parsing) işleyen bir katman olan bir Adaptöre (Adapter) iletti.

DSPy'nin `dspy.ChainOfThought`, `dspy.ProgramOfThought` ve `dspy.ReAct` gibi çeşitli yerleşik modülleri vardır. Bunlar temel `dspy.Predict` ile birbirinin yerine kullanılabilir: görevinize özel olan imzanızı alırlar ve ona genel amaçlı istemleme tekniklerini ve çıkarım zamanı stratejilerini uygularlar.

Örneğin, `dspy.ChainOfThought`, LM'niz imzanızda istenen çıktılara karar vermeden önce ondan bir akıl yürütme (reasoning) elde etmenin kolay bir yoludur. Diğer alanlar ve türlerle denemeler yapmaktan çekinmeyin, örneğin `topics: list[str]` veya `is_realistic: bool`'u deneyin.

```python
qa = dspy.ChainOfThought('question -> response')
response = qa(question="what are high memory and low memory on linux?")
print(response.response)
```

İlginç bir şekilde, akıl yürütme istemek bu durumda çıktı yanıtını daha kısa hale getirebilir. Bu iyi bir şey mi yoksa kötü bir şey mi? İhtiyacınıza bağlıdır: bedava yemek (free lunch) yoktur, ancak DSPy size farklı stratejilerle son derece hızlı bir şekilde deney yapma araçlarını verir.

## DSPy'yi iyi kullanmak, değerlendirme ve yinelemeli geliştirme gerektirir

DSPy sisteminizin kalitesini ölçmek için, örneğin `question`'lar gibi bir grup (1) girdi değerine ve (2) sisteminizden çıkan çıktının kalitesini puanlayabilecek bir metriğe ihtiyacınız vardır. Metrikler büyük ölçüde değişir. Bazı metrikler, örneğin sınıflandırma veya soru yanıtlama için ideal çıktıların "ground-truth" (gerçek) etiketlerine ihtiyaç duyar.

### DSPy'de Örnekleri Manipüle Etme

Şimdi verileri bölelim:
* **Eğitim (Training) ve Doğrulama (Validation) seti:** Bunlar tipik olarak DSPy optimizer'larına verdiğiniz veri bölümleridir. Optimizer'lar tipik olarak doğrudan eğitim örneklerinden öğrenir ve doğrulama örneklerini kullanarak ilerlemelerini kontrol ederler. Hem eğitim hem doğrulama için 30-300 örneğe sahip olmak iyidir. İstem (prompt) optimizer'ları için özel olarak, eğitimden ziyade daha fazla doğrulama örneği vermek genellikle daha iyidir. Aşağıda, toplamda 200 adet kullanacağız. MIPROv2, eğer bir valset vermezseniz bunları %20 eğitim ve %80 doğrulama olarak bölecektir.
* **Geliştirme ve Test setleri:** Geri kalanı, tipik olarak 30-1000 civarında, şu amaçlar için kullanılabilir: geliştirme (sistem üzerinde yinelemeler yaparken bunları inceleyebilirsiniz) ve test etme (nihai ayrılmış değerlendirme).

```python
import datasets

# TechQA veri setinden bir alt küme yükle
dataset = datasets.load_dataset("the_pile_stack_exchange", "technology", split='train', streaming=True)

data = []
for i, item in enumerate(dataset):
    data.append(dspy.Example(question=item['question'], answer=item['answer']).with_inputs('question'))
    if i >= 199: break

trainset, devset = data[:100], data[100:]
```

### DSPy'de Değerlendirme

Soru yanıtlayan `qa` modülümüzü değerlendirelim. Basit bir `answer_exact_match` metriği kullanacağız (küçük harf duyarlılığı olmadan tam eşleşme arar).

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset, metric=dspy.evaluate.answer_exact_match, num_threads=10, display_progress=True)
evaluate(qa)
```

Şimdiye kadar, soru yanıtlama için çok basit bir düşünce zinciri modülü oluşturduk ve küçük bir veri seti üzerinde değerlendirdik. Daha iyisini yapabilir miyiz? Bu kılavuzun geri kalanında, aynı görev için DSPy'de bir bilgi getirme destekli üretim (RAG) programı oluşturacağız. Bunun skoru nasıl önemli ölçüde artırabileceğini göreceğiz, ardından RAG programımızı daha yüksek kaliteli istemler şeklinde derlemek (compile) için DSPy Optimizer'larından birini kullanarak skorlarımızı daha da yükselteceğiz.

## Temel RAG (Retrieval-Augmented Generation)

İlk olarak, RAG araması için kullanacağımız corpus (derlem) verilerini indirelim. Bu eğitimin eski bir sürümü tam (650.000 belge) corpus kullanıyordu. Bunu çok daha hızlı ve çalıştırması ucuz hale getirmek için, corpus'u yalnızca 28.000 belgeye düşürdük.

### Sisteminizin arama motorunu kurun (Set up your system's retriever)

DSPy söz konusu olduğunda, araçları veya arama motorlarını (retrievers) çağırmak için herhangi bir Python kodunu tak-çalıştır şeklinde kullanabilirsiniz. Burada, sadece kolaylık olsun diye OpenAI Embeddings'i kullanacağız ve yerel olarak top-K araması yapacağız.

*Not: Aşağıdaki adım, faiss'ten kaçınmak için ya `pip install -U faiss-cpu` yapmanızı ya da `dspy.retrievers.Embeddings`'e `brute_force_threshold=30_000` parametresini geçmenizi gerektirecektir.*

```python
import requests

url = "[https://huggingface.co/datasets/dspy/techqa/resolve/main/corpus.json](https://huggingface.co/datasets/dspy/techqa/resolve/main/corpus.json)"
corpus = requests.get(url).json()

search = dspy.retrievers.Embeddings(corpus=corpus, k=3)
```

### İlk RAG Modülünüzü Oluşturun (Build your first RAG Module)

```python
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)

rag = RAG()
evaluate(rag)
```

## RAG isteminizi iyileştirmek için bir DSPy Optimizer Kullanma

RAG modülümüz %55 civarı puan alıyor. Onu daha güçlü hale getirmek için seçeneklerimiz nelerdir? Çeşitli seçeneklerden biri onu bir optimizer ile derlemektir (compile). Optimizer'lar, programınızdaki her modül için en iyi istemleri veya birkaç örnekli (few-shot) örnekleri bulmak için veri setinizi kullanır.

Burada `MIPROv2` kullanacağız. Bu optimizer hem istem metnini (instruction) hem de seçilen örnekleri optimize eder.

```python
from dspy.teleprompt import MIPROv2

def metric(gold, pred, trace=None):
    return dspy.evaluate.answer_exact_match(gold, pred)

optimizer = MIPROv2(metric=metric, auto="light")
optimized_rag = optimizer.compile(rag, trainset=trainset)

evaluate(optimized_rag)
```

## Maliyeti göz önünde bulundurma (Keeping an eye on cost)

DSPy, programlarınızın maliyetini izlemenize (track) olanak tanır:

```python
print(lm.inspect_history(n=1))
```

## Kaydetme ve yükleme (Saving and loading)

```python
optimized_rag.save("optimized_rag.json")

# Geri yüklemek için:
# rag_loaded = RAG()
# rag_loaded.load("optimized_rag.json")
```

## Sırada ne var? (What's next?)

Artık DSPy ile temel bir RAG sistemi kurmayı, değerlendirmeyi ve optimize etmeyi öğrendiniz. Ancak DSPy'nin yetenekleri bununla sınırlı değil:

1. **Özel Metrikler:** Sadece tam eşleşme değil, anlamsal benzerlik veya LLM tabanlı metrikler (LLM-as-a-judge) tanımlayabilirsiniz.
2. **Çok Aşamalı Programlar:** Sadece tek bir arama-cevap değil; sorgu genişletme, çok adımlı doğrulama (self-correction) gibi karmaşık akışlar kurabilirsiniz.
3. **Farklı Optimizer'lar:** Eğer çok az veriniz varsa `BootstrapFewShot`, orta ölçekli veri için `MIPROv2` veya büyük ölçekli optimizasyon için `MIPROv2`'nin "heavy" ayarlarını deneyebilirsiniz.
4. **Yerel Modeller:** OpenAI yerine Llama-3 veya Mistral gibi yerel modelleri `dspy.Ollama` veya `dspy.vLLM` üzerinden bağlayarak maliyeti sıfıra indirebilirsiniz.

Bu yolculuğun devamı için [DSPy Dokümantasyonunu](https://dspy.ai/) ve [Örnek Projeleri](https://github.com/stanfordnlp/dspy) incelemenizi öneririz.