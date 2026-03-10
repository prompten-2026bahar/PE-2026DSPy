# İmzalar (Signatures)

DSPy'de dil modellerine (LM) görev atadığımızda, ihtiyacımız olan davranışı bir **İmza (Signature)** olarak belirtiriz.

**İmza, bir DSPy modülünün girdi/çıktı davranışının bildirimsel (declarative) bir spesifikasyonudur.** İmzalar, LM'ye bunu nasıl yapmasını söylemek yerine, *ne* yapması gerektiğini söylemenize olanak tanır.



Girdi ve çıktı argümanlarını ve bunların türlerini belirten fonksiyon imzalarına muhtemelen aşinasınızdır. DSPy imzaları buna benzer ancak birkaç farkı vardır. Tipik fonksiyon imzaları sadece şeyleri *tanımlarken*, DSPy İmzaları modüllerin *davranışını beyan eder ve başlatır*. Dahası, DSPy İmzalarında alan (field) isimleri önemlidir. Anlamsal rolleri sade bir İngilizceyle (veya ilgili dilde) ifade edersiniz: bir `question` (soru) `answer`dan (cevap) farklıdır; bir `sql_query`, `python_code`dan farklıdır.

## Neden bir DSPy İmzası kullanmalıyım?

Modüler ve temiz bir kod için, LM çağrıları yüksek kaliteli istemlere (veya otomatik ince ayarlara - finetunes) optimize edilebilir. Çoğu insan, LM'leri uzun ve kırılgan istemler (prompts) yazarak görevleri yapmaya zorlar. Ya da ince ayar için veri toplar/üretir. İmza yazmak; istemlerle veya ince ayarlarla uğraşmaktan çok daha modüler, uyarlanabilir ve yeniden üretilebilirdir. DSPy derleyicisi (compiler); imzanız için, verileriniz üzerinde ve boru hattınız (pipeline) dahilinde LM'niz için yüksek düzeyde optimize edilmiş bir istemin nasıl oluşturulacağını (veya küçük LM'nize ince ayar yapılacağını) çözecektir. Pek çok durumda, derlemenin insanların yazdığından daha iyi istemlere yol açtığını gördük. Bunun nedeni DSPy optimize edicilerinin insanlardan daha yaratıcı olması değil, sadece daha fazla şeyi deneyebilmeleri ve metrikleri doğrudan ayarlayabilmeleridir.

## **Satır İçi (Inline)** DSPy İmzaları

İmzalar; girdiler/çıktılar için anlamsal rolleri tanımlayan argüman adları ve isteğe bağlı türler içeren kısa bir dize (string) olarak tanımlanabilir.

1. Soru Cevaplama: `"question -> answer"`. Bu, varsayılan tür her zaman `str` olduğu için `"question: str -> answer: str"` ifadesine eşdeğerdir.

2. Duygu Sınıflandırması: `"sentence -> sentiment: bool"`, örneğin olumluysa `True`.

3. Özetleme: `"document -> summary"`

İmzalarınız ayrıca türleri olan birden fazla girdi/çıktı alanına sahip olabilir:

4. Erişim Destekli Soru Cevaplama (RAG): `"context: list[str], question: str -> answer: str"`

5. Akıl Yürütme ile Çoktan Seçmeli Soru Cevaplama: `"question, choices: list[str] -> reasoning: str, selection: int"`

**İpucu:** Alanlar için geçerli herhangi bir değişken adı çalışır! Alan adları anlamsal olarak anlamlı olmalıdır, ancak basit başlayın ve anahtar kelimeleri vaktinden önce optimize etmeye çalışmayın! Bu tür "hackleme" işlerini DSPy derleyicisine bırakın. Örneğin, özetleme için `"document -> summary"`, `"text -> gist"` veya `"long_context -> tldr"` demek muhtemelen yeterli olacaktır.

Ayrıca, çalışma zamanında değişkenleri kullanabilen satır içi imzanıza talimatlar ekleyebilirsiniz. İmzanıza talimat eklemek için `instructions` anahtar kelime argümanını kullanın.

```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
comment = "you are beautiful."
toxicity(comment=comment).toxic
```

**Çıktı:**
```text
False
```


### Örnek A: Duygu Sınıflandırması

```python
sentence = "it's a charming and often affecting journey."  # SST-2 veri kümesinden bir örnek.

classify = dspy.Predict('sentence -> sentiment: bool')  # Literal[] içeren bir örneği daha sonra göreceğiz.
classify(sentence=sentence).sentiment
```
**Çıktı:**
```text
True
```

### Örnek B: Özetleme

```python
# XSum veri kümesinden bir örnek.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**Olası Çıktı:**
```text
The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Pek çok DSPy modülü (`dspy.Predict` hariç), imzanızı perde arkasında genişleterek yardımcı bilgiler döndürür.

Örneğin, `dspy.ChainOfThought` modülü, çıktı olan `summary`yi (özet) üretmeden önce LM'nin akıl yürütme sürecini içeren bir `reasoning` (akıl yürütme) alanı ekler.

```python
print("Reasoning:", response.reasoning)
```
**Olası Çıktı**
```text
Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Sınıf Tabanlı** DSPy İmzaları

Bazı gelişmiş görevler için daha ayrıntılı imzalara ihtiyaç duyarsınız. Bu genellikle şu amaçlarla yapılır:

1. Görevin doğası hakkında bir durumu netleştirmek (aşağıda `docstring` olarak ifade edilmiştir).

2. `dspy.InputField` için bir `desc` (açıklama) anahtar kelime argümanı olarak ifade edilen, girdi alanının doğasına ilişkin ipuçları sağlamak.

3. `dspy.OutputField` için bir `desc` anahtar kelime argümanı olarak ifade edilen, çıktı alanı üzerindeki kısıtlamaları belirtmek.

Sınıf tabanlı imza örneğini ve kod bloğunu çevirmemi ister misin Melike Nur? Hazır olduğunda gönderebilirsin.

### Örenk C :Sınıflandırma

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""
    
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
**Olası Çıktı**
```text
Prediction(
    sentiment='fear'
)
```

**İpucu:** LM'ye yönelik isteklerinizi daha net bir şekilde belirtmenizde yanlış bir şey yoktur. Sınıf tabanlı İmzalar bu konuda size yardımcı olur. Ancak, imzanızın anahtar kelimelerini manuel olarak vaktinden önce ayarlamaya (tune etmeye) çalışmayın. DSPy optimize edicileri muhtemelen bu işi daha iyi yapacaktır (ve bu optimizasyonlar farklı LM'ler arasında daha iyi aktarılacaktır).

### Örnek D: Alıntılara sadakati değerlendiren bir metrik

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Olası Çıktı :**
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Örnek E: Çok modlu (multi-modal) görüntü sınıflandırması

```python
class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```

**Olası Çıktı:**

```text
Prediction(
    answer='Labrador Retriever'
)
```

## İmzalarda Tür Çözümleme (Type Resolution)

DSPy imzaları çeşitli anotasyon türlerini destekler:

1. `str`, `int`, `bool` gibi **temel türler**
2. `list[str]`, `dict[str, int]`, `Optional[float]`, `Union[str, int]` gibi **yazım (typing) modülü türleri**
3. Kodunuzda tanımlanan **özel türler (custom types)**
4. Uygun yapılandırma ile iç içe geçmiş türler için **nokta notasyonu**
5. `dspy.Image`, `dspy.History` gibi **özel veri türleri**



### Özel Türlerle Çalışma

```python
# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float

signature = dspy.Signature("query: str -> result: QueryResult")

class MyContainer:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = dspy.Signature("query: MyContainer.Query -> score: MyContainer.Score")
```

## Modül Oluşturmak ve Bunları Derlemek için İmzaları Kullanma

İmzalar, yapılandırılmış girdi ve çıktılarla prototip oluşturmak için kullanışlı olsa da, onları kullanmanın tek nedeni bu değildir!

Birden fazla imzayı bir araya getirerek daha büyük [DSPy modülleri](modules.md) oluşturmalı ve bu modülleri [optimize edilmiş istemlere](../optimization/optimizers.md) veya ince ayarlara (finetunes) **derlemelisiniz**.
