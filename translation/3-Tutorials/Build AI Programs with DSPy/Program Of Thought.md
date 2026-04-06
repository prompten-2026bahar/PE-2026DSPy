# Eğitim: ProgramOfThought

`dspy.ProgramOfThought`, aşağı akış görevlerini çözmek için Python kodunu otomatik olarak üretir ve iyileştirir.

En güncel DSPy sürümünü `pip install -U dspy` ile kurun ve birlikte ilerleyin.

## 1) LocalSandbox Kullanımı

`ProgramOfThought`, LM'ler tarafından üretilen kodu çalıştırmak için uyarlanmış bir Python sandbox ile entegre olur. 

Sandbox'ın nasıl çalıştığını göstermek için kısa bir örnek olarak, bir `dspy.LocalSandbox` örneği oluşturacağız ve `ProgramOfThought`'un temel yürütmesini göstereceğiz.

```python
import dspy
sandbox = dspy.LocalSandbox()
expr = "value = 2*5 + 4\nvalue"
answer = sandbox.execute(expr)
answer
```

## 2) ProgramOfThought'u Gösterme

 Bir örnek olarak, bir girdi sorusu ve bir çıktı cevabı içeren bir imza tanımlayacağız. Ardından, önce soruyu temsil edecek kodu üretmek için bir LM kullanan, kodu yorumlayıcı ile çalıştıran ve son sonucu sorunun cevabı olarak veren `ProgramOfThought` programını oluşturup çağıracağız.

Meta'nın `Llama-3-70b-Instruct` modelini kullanalım. Bunu kolayca [diğer sağlayıcılar veya yerel modellerle](https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb) değiştirebilirsiniz.

```python
llama31_70b = dspy.LM("openai/meta-llama/Meta-Llama-3-70b-Instruct", api_base="API_BASE", api_key="None")

dspy.configure(lm=llama31_70b)
```

Şimdi, girdi sorusunu ve çıktı cevabını belirten kısa bir imza ile modülümüzü tanımlayalım. Ardından imza üzerinde `ProgramOfThought`'u çağırabilir ve örnek problemimizi geçirebiliriz.

```python
class BasicGenerateAnswer(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

pot = dspy.ProgramOfThought(BasicGenerateAnswer)
problem = "2*5 + 4"
pot(question=problem).answer
```

Harika! Modül aynı doğru cevabı başarıyla üretti. Şimdi bunu yapmak için LM'yi tam olarak nasıl kullandığına bakalım:

```python
dspy.inspect_history()
```

Üretilen Python kodunun ara hesaplamalar için bir fonksiyon tanımladığını ve `LocalSandbox` üzerinden çalıştırıldığında doğru cevabı vererek son cevabı döndürdüğünü görüyoruz.

## 3) ChainOfThought ile Karşılaştırma

Şimdi `ProgramOfThought` modülünün nasıl yardımcı olabileceğini göstermek için daha karmaşık bir probleme geçelim. 

Problem: **12! / 1 ile 30 arasındaki asal sayıların toplamını hesaplayın.**

Bu oldukça zorlayıcı bir hesaplamadır. Önce `ChainOfThought`'un nasıl performans gösterdiğine bakalım:

```python
problem = "Compute 12! / sum of prime numbers between 1 and 30."

cot = dspy.ChainOfThought(BasicGenerateAnswer)
cot(question=problem).answer
```

```python
dspy.inspect_history()
```

Görüyoruz ki `ChainOfThought`, adımları akıl yürüterek ilerlemede oldukça iyi performans gösteriyor; hem 12! için hem de 1-30 arasındaki yalnızca asal sayıların toplamı için doğru değeri buluyor. 

Ancak son bölme adımında başarısız oluyor ve doğru cevap `3713190.69767` iken (gerçek bir hesap makinesiyle doğrulandı!) `479,001,600 / 129 = 3,710,009` işlemini yanlış hesaplıyor.

Şimdi `ProgramOfThought`'un nasıl performans gösterdiğine bakalım:

```python
pot(question=problem).answer
```

```python
dspy.inspect_history()
```

Python yorumlayıcısı kodu doğru biçimde çalıştırdığı için `ProgramOfThought`, `ChainOfThought` içinde başarısız olabilecek hesaplama hatalarını azaltır ve özellikle sayısal ve mantıksal sorgular için doğruluğu artırır.

## 3) Bağlamsal Akıl Yürütme ile Hesaplama

Şimdi karmaşık matematiksel sözel problemlerde hesaplama yapmanın daha karmaşık bir örneğini deneyelim. 

### Adım 1: Wikipedia'da arama yapmak için yardımcı bir fonksiyon tanımlayın
Wikipedia'dan en iyi eşleşmeleri almak ve bunları `ProgramOfThought` hattı içinde ayrıştırmak için bir `dspy.ColBERTv2` sunucusu kullanacağız.

```python
def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]
```

### Adım 2: ProgramOfThought ile Çok Adımlı Arama
[Multi-Hop Search görevi](https://dspy.ai/tutorials/multihop_search/)'nden ilham alacağız ve yalnızca son `generate_answer` katmanını, bir soru ve getirilen bağlam verildiğinde doğru hesaplamaları güvence altına almak için `ChainOfThought` yerine `ProgramOfThought` kullanacak şekilde uyarlayacağız.

Bilgi toplamak için getirme gerektiren ve ardından gerçekleri kullanarak hesaplama yapıp nihai sonucu üretmeyi gerektiren zorlayıcı bir sözel problem soruyoruz. 

```python
class GenerateAnswer(dspy.Signature):
    """Soruları kısa, olgusal cevaplarla cevaplayın."""

    context = dspy.InputField(desc="ilgili gerçekleri içerebilir")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="çoğunlukla 1 ile 5 kelime arasındadır")


class GenerateSearchQuery(dspy.Signature):
    """Karmaşık bir sorunun sayısal olmayan bileşenlerini cevaplamaya yardımcı olacak basit bir arama sorgusu yazın."""

    context = dspy.InputField(desc="ilgili gerçekleri içerebilir")
    question = dspy.InputField()
    query = dspy.OutputField()

from dspy.dsp.utils import deduplicate

class MultiHopSearchWithPoT(dspy.Module):
    def __init__(self, num_hops):
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ProgramOfThought(GenerateAnswer, max_iters=3)

    def forward(self, question):
        context = []
        for _ in range(self.num_hops):
            query = self.generate_query(context=context, question=question).query
            context = deduplicate(context + search_wikipedia(query))
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

multi_hop_pot = MultiHopSearchWithPoT(num_hops=2)
question = (
    "What is the square of the total sum of the atomic number of the metal "
    "that makes up the gift from France to the United States in the late "
    "19th century and the sum of the number of digits in the first 10 prime numbers?"
)
multi_hop_pot(question=question).answer
```

```python
dspy.inspect_history()
```

Getirilen bağlamın Özgürlük Heykeli ve bakır hakkında pasajlar içerdiğine dikkat edin. Bu getirme işlemi, sorunun ilk kısmını yanıtlamaya yardımcı olur; Özgürlük Heykeli'ni 19. yüzyılın sonlarında Fransa'dan ABD'ye verilen hediye olarak tanımlar, onun bakırdan yapıldığını belirler ve adım adım akıl yürütme yoluyla bakırın atom numarasını (29) getirir.

Sorunun ikinci kısmı ise Python mantığına ayrıştırılır ve ilk 10 asal sayıdaki rakam sayısı programatik olarak toplanır.

Bu iki alt problemi birleştirerek çözüm sonuçları doğru şekilde toplar ve nihai cevabı üretir: **2025**.
