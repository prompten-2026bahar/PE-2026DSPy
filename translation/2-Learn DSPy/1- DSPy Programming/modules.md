# Modüller

Bir **DSPy modülü**, dil modellerini (LM) kullanan programlar için temel bir yapı taşıdır.

- Her yerleşik modül, bir **istemi tekniğini** (Chain of Thought veya ReAct gibi) soyutlar. En önemli özellikleri, herhangi bir imzayı (signature) işleyebilecek şekilde genelleştirilmiş olmalarıdır.

- Bir DSPy modülü, **öğrenilebilir parametrelere** (yani istemi oluşturan küçük parçalar ve LM ağırlıkları) sahiptir ve girdileri işlemek ve çıktıları döndürmek için çağrılabilir (invoked).

- Birden fazla modül, daha büyük modüller (programlar) oluşturacak şekilde bir araya getirilebilir. DSPy modülleri doğrudan PyTorch'taki NN (sinir ağı) modüllerinden esinlenmiştir, ancak bu yapı LM programlarına uygulanmıştır.


## `dspy.Predict` veya `dspy.ChainOfThought` gibi yerleşik bir modülü nasıl kullanırım?

En temel modül olan `dspy.Predict` ile başlayalım. Dahili olarak, diğer tüm DSPy modülleri `dspy.Predict` kullanılarak oluşturulur. DSPy'de kullandığımız herhangi bir modülün davranışını tanımlayan bildirimsel özellikler olan [DSPy imzalarına](signatures.md) en azından biraz aşina olduğunuzu varsayıyoruz.

Bir modülü kullanmak için önce ona bir imza vererek **tanımlarız (declare)**. Ardından modülü girdi argümanlarıyla **çağırırız (call)** ve çıktı alanlarını alırız!

```python
sentence = "it's a charming and often affecting journey."  # SST-2 veri kümesinden bir örnek.

# 1) Bir imza ile tanımlayın (Declare).
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Girdi argüman(lar)ı ile çağırın (Call). 
response = classify(sentence=sentence)

# 3) Çıktıya erişin.
print(response.sentiment)
```
**Çıktı:**
```text
True
```

Bir modülü tanımlarken, ona yapılandırma anahtarları (configuration keys) aktarabiliriz.

Aşağıda, `temperature` gibi basit bir yapılandırma anahtarı aktaracağız. Ayrıca `max_tokens` gibi diğer üretim anahtarlarını da aktarabilirsiniz.

Hadi `dspy.ChainOfThought` modülünü kullanalım. Pek çok durumda, sadece `dspy.Predict` yerine `dspy.ChainOfThought` kullanmak bile kaliteyi artırır.

```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Bir imza ile tanımlayın ve bazı yapılandırmaları (config) aktarın.
classify = dspy.ChainOfThought('question -> answer', temperature=0.7)

# 2) Girdi argümanı ile çağırın.
response = classify(question=question)

# 3) Çıktıya erişin.
response.answer
```
**Olası Çıktı:**
```text
'One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.'
```

Buradaki çıktı nesnesini tartışalım. `dspy.ChainOfThought` modülü, genellikle imzanızdaki çıktı alan(lar)ından önce bir `reasoning` (akıl yürütme) alanı enjekte eder.

Hadi (ilk) akıl yürütmeyi ve cevabı inceleyelim!

```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```
**Possible Output:**
```text
Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
```

## Diğer hangi DSPy modülleri var? Onları nasıl kullanabilirim?

Diğerleri oldukça benzerdir. Temel olarak imzanızın uygulanma şeklindeki dahili davranışı değiştirirler!

1. **`dspy.Predict`**: Temel tahminci. İmzayı değiştirmez. Öğrenmenin temel biçimlerini (yani talimatların ve gösterimlerin saklanması ve LM güncellemeleri) yönetir.

2. **`dspy.ChainOfThought`**: LM'ye, imzanın yanıtını vermeden önce adım adım düşünmeyi öğretir.

3. **`dspy.ProgramOfThought`**: LM'ye kod çıktıları üretmeyi öğretir; bu kodun yürütme sonuçları yanıtı belirler.

4. **`dspy.ReAct`**: Verilen imzayı uygulamak için araçları (tools) kullanabilen bir ajandır.

5. **`dspy.MultiChainComparison`**: Nihai bir tahmin üretmek için `ChainOfThought` modülünden gelen birden fazla çıktıyı karşılaştırabilir.

6. **`dspy.RLM`**: Özyinelemeli alt-LLM çağrılarına sahip, korumalı bir Python REPL aracılığıyla geniş bağlamları keşfeden bir [Özyinelemeli Dil Modelidir (Recursive Language Model)](../../api/modules/RLM.md). Bağlam, istemin içine etkili bir şekilde sığmayacak kadar büyük olduğunda kullanılır.

Ayrıca bazı fonksiyon tarzı modüllerimiz de mevcuttur:

7. **`dspy.majority`**: Bir tahmin kümesinden en popüler yanıtı döndürmek için temel oylama yapabilir.

!!! info "Basit görevlerdeki DSPy modüllerine dair birkaç örnek."
    `lm` yapılandırmanızı yaptıktan sonra aşağıdaki örnekleri deneyin. Dil modelinizin (LM) kutudan çıktığı haliyle hangi görevleri iyi yapabildiğini keşfetmek için alanları (fields) özelleştirin.

    === "Matematik"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="Two dice are tossed. What is the probability that the sum equals two?")
        ```
        
        **Olası Çıktı:**
        ```text
        Prediction(
            reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
            answer=0.0277776
        )
        ```

    === "Erişimle Güçlendirilmiş Üretim (RAG)"

        

        ```python linenums="1"       
        def search(query: str) -> list[str]:
            """Wikipedia'dan özetleri getirir."""
            results = dspy.ColBERTv2(url='[http://20.102.90.50:2017/wiki17_abstracts')(query](http://20.102.90.50:2017/wiki17_abstracts')(query), k=3)
            return [x['text'] for x in results]
        
        rag = dspy.ChainOfThought('context, question -> response')

        question = "What's the name of the castle that David Gregory inherited?"
        rag(context=search(question), question=question)
        ```
        
        **Olası Çıktı:**
        ```text
        Prediction(
            reasoning='The context provides information about David Gregory, a Scottish physician and inventor. It specifically mentions that he inherited Kinnairdy Castle in 1664. This detail directly answers the question about the name of the castle that David Gregory inherited.',
            response='Kinnairdy Castle'
        )
        ```

    === "Sınıflandırma"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Verilen bir cümlenin duygu durumunu sınıflandırın."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="This book was super fun to read, though not the last chapter.")
        ```
        
        **Olası Çıktı:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Bilgi Çıkarımı"

        

        ```python linenums="1"        
        text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

        module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities_and_metadata)
        ```
        
        **Olası Çıktı:**
        ```text
        Apple Unveils iPhone 14
        ['Introduction', 'Key Features', "CEO's Statement"]
        [{'entity': 'Apple Inc.', 'type': 'Organization'}, {'entity': 'iPhone 14', 'type': 'Product'}, {'entity': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Ajanlar"

        

        ```python linenums="1"       
        def evaluate_math(expression: str) -> float:
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str) -> str:
            results = dspy.ColBERTv2(url='[http://20.102.90.50:2017/wiki17_abstracts')(query](http://20.102.90.50:2017/wiki17_abstracts')(query), k=3)
            return [x['text'] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
        print(pred.answer)
        ```
        
        **Olası Çıktı:**

        ```text
        5761.328
        ```


## Birden fazla modülü nasıl daha büyük bir programda birleştiririm?

DSPy, modülleri dilediğiniz herhangi bir kontrol akışı içinde kullanan basit bir Python kodudur; sadece `compile` (derleme) sırasında LM çağrılarınızı izlemek (trace) için arka planda küçük bir sihir barındırır. Bu, modülleri özgürce çağırabileceğiniz anlamına gelir.

Örnek olarak aşağıda yeniden oluşturulan [multi-hop search](https://dspy.ai/tutorials/multihop_search/) (çok adımlı arama) gibi eğitim videolarını inceleyebilirsiniz.

```python linenums="1"        
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

Ardından, özel modül sınıfı `Hop`'un bir örneğini (instance) oluşturabilir ve onu `__call__` yöntemiyle çağırabilirsiniz:

```
hop = Hop()
print(hop(claim="Stephen Curry is the best 3 pointer shooter ever in the human history"))
```

## LM kullanımını nasıl takip ederim?

!!! note "Sürüm Gereksinimi"
    LM kullanım takibi, DSPy 2.6.16 ve sonraki sürümlerinde mevcuttur.

DSPy, tüm modül çağrılarında dil modeli kullanımının yerleşik olarak izlenmesini sağlar. İzlemeyi etkinleştirmek için:

```python
dspy.configure(track_usage=True)
```

Etkinleştirildikten sonra, herhangi bir `dspy.Prediction` nesnesi üzerinden kullanım istatistiklerine erişebilirsiniz:

```python
usage = prediction_instance.get_lm_usage()
```

Kullanım verileri, her bir dil modeli adını kendi kullanım istatistikleriyle eşleyen bir sözlük (dictionary) olarak döndürülür. İşte tam bir örnek:

```python
import dspy

# DSPy'ı izleme (tracking) etkinleştirilmiş olarak yapılandırın
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

# Birden fazla LM çağrısı yapan basit bir program tanımlayın
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

# Programı çalıştırın ve kullanımı kontrol edin
program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())
```

Bu kod, aşağıdakine benzer kullanım istatistikleri çıktılayacaktır:

```python
{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        }
    }
}
```

DSPy'nin önbelleğe alma (caching) özelliklerini kullandığınızda (litellm aracılığıyla bellek içi veya diskte), önbelleğe alınan yanıtlar kullanım istatistiklerine dahil edilmez. Örneğin:

```python
# Önbelleğe almayı (caching) etkinleştirin
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# İlk çağrı - kullanım istatistiklerini gösterecektir
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Token kullanımını gösterir

# İkinci çağrı - aynı soru, önbelleği kullanacaktır
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Boş sözlük gösterir: {}
```
