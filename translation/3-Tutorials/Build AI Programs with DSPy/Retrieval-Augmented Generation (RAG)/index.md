## Eğitim: Bilgi Geri Getirme Destekli Nesil (RAG)

DSPy'de **bilgi geri getirme destekli nesil** (RAG) kullanarak ve kullanmadan **temel soru-cevaplama** işlemlerine dair hızlı bir örnek üzerinden geçelim. Özellikle, Linux veya iPhone uygulamaları gibi teknoloji sorularını yanıtlayan bir sistem inşa edelim.

`pip install -U dspy` komutuyla en son DSPy sürümünü yükleyin ve takip edin. Eğer bunun yerine DSPy hakkında kavramsal bir genel bakış arıyorsanız, [bu güncel ders](https://dspy.ai/learn/intro/concepts/) başlamak için iyi bir yerdir. Ayrıca `pip install datasets` komutunu da çalıştırmanız gerekmektedir.


### DSPy Ortamını Yapılandırma

Modüllerimizde OpenAI'nın `gpt-4o-mini` modelini kullanacağımızı DSPy'ye bildirelim. Kimlik doğrulaması için DSPy, `OPENAI_API_KEY` çevresel değişkeninize bakacaktır. Bunu diğer sağlayıcılar veya yerel modellerle kolayca değiştirebilirsiniz.

```python
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

### Temel DSPy Modüllerini Keşfetme

LM'yi her zaman doğrudan `lm(prompt="prompt")` veya `lm(messages=[...])` aracılığıyla istemleyebilirsiniz. Ancak DSPy, LM fonksiyonlarınızı tanımlamanız için daha iyi bir yol olarak size **Modülleri (Modules)** sunar.

En basit modül `dspy.Predict` modülüdür. Bu modül bir **DSPy İmzası (Signature)**, yani yapılandırılmış bir giriş/çıkış şeması alır ve belirttiğiniz davranış için size çağrılabilir bir fonksiyon döndürür. Giriş olarak bir `question` (str tipinde) alan ve çıktı olarak bir `response` üreten bir modül tanımlamak için imzaların "satır içi" (in-line) gösterimini kullanalım.


```python
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")

print(response.response)
```

```
In Linux, "high memory" and "low memory" refer to different regions of the system's memory address space, particularly in the context of 32-bit architectures.

- **Low Memory**: This typically refers to the memory that is directly accessible by the kernel. In a 32-bit system, this is usually the first 896 MB of RAM (from 0 to 896 MB). The kernel can directly map this memory, making it faster for the kernel to access and manage. Low memory is used for kernel data structures and for user processes that require direct access to memory.

- **High Memory**: This refers to the memory above the low memory limit, which is not directly accessible by the kernel in a 32-bit system. This area is typically above 896 MB. The kernel cannot directly access this memory without using special mechanisms, such as mapping it into the kernel's address space when needed. High memory is used for user processes that require more memory than what is available in low memory.

In summary, low memory is directly accessible by the kernel, while high memory requires additional steps for the kernel to access it, especially in 32-bit systems. In 64-bit systems, this distinction is less significant as the kernel can address a much larger memory space directly.
```

İmzada belirttiğimiz değişken isimlerinin, giriş ve çıkış argüman adlarımızı ve rollerini nasıl tanımladığına dikkat edin.

Peki, DSPy bu `qa` modülünü oluşturmak için ne yaptı? Bu örnekte henüz çok karmaşık bir şey yok. Modül; imzanızı, Dil Modelini (LM) ve girişleri bir **Bağdaştırıcıya (Adapter)** iletti. Bağdaştırıcı, girişleri yapılandırmayı ve yapılandırılmış çıktıları imzanıza uyacak şekilde ayrıştırmayı yöneten bir katmandır.


Bunu doğrudan görelim. DSPy tarafından gönderilen son `n` istemi (prompt) kolayca inceleyebilirsiniz. Alternatif olarak, yukarıda MLflow İzlemeyi (Tracing) etkinleştirdiyseniz, her program yürütmesi için tam LLM etkileşimlerini bir ağaç görünümünde görebilirsiniz.


```python
dspy.inspect_history(n=1)
```

```
[2024-11-23T23:16:35.966534]

System message:

Your input fields are:
1. `question` (str)

Your output fields are:
1. `response` (str)

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## response ## ]]
{response}

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Given the fields `question`, produce the fields `response`.


User message:

[[ ## question ## ]]
what are high memory and low memory on linux?

Respond with the corresponding output fields, starting with the field `[[ ## response ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## response ## ]]
In Linux, "high memory" and "low memory" refer to different regions of the system's memory address space, particularly in the context of 32-bit architectures.

- **Low Memory**: This typically refers to the memory that is directly accessible by the kernel. In a 32-bit system, this is usually the first 896 MB of RAM (from 0 to 896 MB). The kernel can directly map this memory, making it faster for the kernel to access and manage. Low memory is used for kernel data structures and for user processes that require direct access to memory.

- **High Memory**: This refers to the memory above the low memory limit, which is not directly accessible by the kernel in a 32-bit system. This area is typically above 896 MB. The kernel cannot directly access this memory without using special mechanisms, such as mapping it into the kernel's address space when needed. High memory is used for user processes that require more memory than what is available in low memory.

In summary, low memory is directly accessible by the kernel, while high memory requires additional steps for the kernel to access it, especially in 32-bit systems. In 64-bit systems, this distinction is less significant as the kernel can address a much larger memory space directly.

[[ ## completed ## ]]

```

DSPy; `dspy.ChainOfThought`, `dspy.ProgramOfThought` ve `dspy.ReAct` gibi çeşitli yerleşik modüllere sahiptir. Bunlar temel `dspy.Predict` ile birbirinin yerine kullanılabilir: Görevinize özel imzanızı alırlar ve ona genel amaçlı istem teknikleri (prompting techniques) ile çıkarım zamanı stratejileri (inference-time strategies) uygularlar.

Örneğin, `dspy.ChainOfThought`, imzanızda talep edilen çıktılara karar vermeden önce Dil Modelinizden (LM) bir **akıl yürütme (reasoning)** süreci elde etmenin kolay bir yoludur.

Aşağıdaki örnekte, varsayılan tür dizgi (string) olduğu için `str` türlerini belirtmeyeceğiz. Diğer alanlar ve türlerle denemeler yapmaktan çekinmeyin; örneğin `topics: list[str]` veya `is_realistic: bool` gibi türleri deneyebilirsiniz.


```python
cot = dspy.ChainOfThought('question -> response')
cot(question="should curly braces appear on their own line?")
```

```
Prediction(
    reasoning='The placement of curly braces on their own line depends on the coding style and conventions being followed. In some programming languages and style guides, such as the Allman style, curly braces are placed on their own line to enhance readability. In contrast, other styles, like K&R style, place the opening brace on the same line as the control statement. Ultimately, it is a matter of personal or team preference, and consistency within a project is key.',
    response='Curly braces can appear on their own line depending on the coding style you are following. If you prefer a style that enhances readability, such as the Allman style, then yes, they should be on their own line. However, if you are following a different style, like K&R, they may not need to be. Consistency is important, so choose a style and stick with it.'
)
```

İlginç bir şekilde, akıl yürütme istemek bu durumda çıktıdaki `response` kısmını daha kısa hale getirebilir. Bu iyi bir şey mi yoksa kötü bir şey mi? Bu, neye ihtiyacınız olduğuna bağlıdır: "bedava öğle yemeği yoktur", ancak DSPy size farklı stratejileri son derece hızlı bir şekilde denemeniz için araçlar sunar.

Bu arada, `dspy.ChainOfThought`, `dspy.Predict` kullanılarak DSPy içinde uygulanmıştır. Merak ediyorsanız burası `dspy.inspect_history` yapmak için iyi bir yerdir.



### DSPy'yi iyi kullanmak, değerlendirme ve yinelemeli geliştirmeyi içerir.

Bu noktada DSPy hakkında çok şey biliyorsunuz. Tek istediğiniz hızlı betik yazmaksa, DSPy'nin bu kadarı bile çok şeyi mümkün kılar. Python kontrol akışınıza DSPy imzalarını ve modüllerini serpiştirmek, LM'lerle işlerinizi halletmenin oldukça ergonomik bir yoludur.

Bununla birlikte, muhtemelen yüksek kaliteli bir sistem kurmak ve onu zamanla geliştirmek istediğiniz için buradasınız. Bunu DSPy'de yapmanın yolu, sisteminizin kalitesini değerlendirerek ve DSPy'nin Optimizasyon Araçları (Optimizers) gibi güçlü araçlarını kullanarak hızlıca yineleme yapmaktır.

### DSPy'de Örnekleri (Examples) Yönetme.

DSPy sisteminizin kalitesini ölçmek için şunlara ihtiyacınız vardır: (1) örneğin `questions` gibi bir dizi giriş değeri ve (2) sisteminizin çıktısının kalitesini puanlayabilen bir `metric`. Metrikler büyük ölçüde farklılık gösterir. Bazı metrikler, örneğin sınıflandırma veya soru-cevaplama için ideal çıktıların gerçek etiketlerine (ground-truth labels) ihtiyaç duyar. Diğer metrikler ise kendi kendini denetler (self-supervised); örneğin sadakati (faithfulness) veya halüsinasyon eksikliğini, belki de bu nitelikleri değerlendirmek için bir DSPy programını yargıç olarak kullanarak kontrol eder.


Soru ve onların (oldukça uzun) altın yanıtlarından (gold answers) oluşan bir veri kümesini yükleyelim. Bu deftere teknoloji sorularını yanıtlayan bir sistem kurma hedefiyle başladığımız için, `RAG-QA Arena` veri kümesinden bir dizi StackExchange tabanlı soru ve bunların doğru yanıtlarını elde ettik.


```python
import orjson
from dspy.utils import download

# Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [orjson.loads(line) for line in f]
```

```python
# Inspect one datapoint.
data[0]
```


```
{'question': 'why igp is used in mpls?',
 'response': "An IGP exchanges routing prefixes between gateways/routers.  \nWithout a routing protocol, you'd have to configure each route on every router and you'd have no dynamic updates when routes change because of link failures. \nFuthermore, within an MPLS network, an IGP is vital for advertising the internal topology and ensuring connectivity for MP-BGP inside the network.",
 'gold_doc_ids': [2822, 2823]}
```

Bu basit sözlük yapısını temel alarak, DSPy'de eğitim (veya test) veri noktalarını taşıyan veri tipi olan `dspy.Example` listesini oluşturalım.

Bir `dspy.Example` oluştururken, hangi alanların giriş olduğunu belirtmek için genellikle `.with_inputs("alan1", "alan2", ...)` yöntemini kullanmalısınız. Diğer alanlar etiket (label) veya meta veri olarak kabul edilir.


```python
data = [dspy.Example(**d).with_inputs('question') for d in data]

# Let's pick an `example` here from the data.
example = data[2]
example
```

```
Example({'question': 'why are my text messages coming up as maybe?', 'response': 'This is part of the Proactivity features new with iOS 9: It looks at info in emails to see if anyone with this number sent you an email and if it finds the phone number associated with a contact from your email, it will show you "Maybe". \n\nHowever, it has been suggested there is a bug in iOS 11.2 that can result in "Maybe" being displayed even when "Find Contacts in Other Apps" is disabled.', 'gold_doc_ids': [3956, 3957, 8034]}) (input_keys={'question'})
```

Şimdi verileri şu şekilde bölelim:

### Eğitim (ve Doğrulama) Seti:
Bunlar genellikle DSPy optimize edicilerine verdiğiniz bölümlerdir. Optimize ediciler genellikle doğrudan eğitim örneklerinden öğrenir ve ilerlemelerini doğrulama örneklerini kullanarak kontrol ederler.
* Eğitim ve doğrulama için her birinden 30-300 örnek olması iyidir.
* Özellikle istem (prompt) optimize edicileri için, eğitime kıyasla **daha fazla** doğrulama örneği geçirmek genellikle daha iyidir.
* Aşağıda toplamda 200 adet kullanacağız. Eğer bir doğrulama seti (valset) geçmezseniz, MIPROv2 bunları %20 eğitim ve %80 doğrulama olarak bölecektir.

### Geliştirme ve Test Setleri:
Geri kalanlar (genellikle 30-1000 civarında) şunlar için kullanılabilir:
* **Geliştirme** (sisteminiz üzerinde yinelenirken onları inceleyebilirsiniz) ve
* **Test** (nihai, sistemin hiç görmediği değerlendirme).


```python
import random

random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]

len(trainset), len(devset), len(testset)
```

```
(200, 300, 500)
```

## DSPy'de Değerlendirme

Soru-cevaplama görevimize ne tür bir metrik uygun olabilir? Birçok seçenek var, ancak yanıtlar uzun olduğu için şunu sorabiliriz: Sistem yanıtı, altın yanıttaki (gold response) tüm temel gerçekleri ne kadar iyi **kapsıyor**? Ve tam tersi, sistem yanıtı altın yanıtta olmayan şeyleri **söylememeyi** ne kadar iyi başarıyor?



Bu metrik özünde bir **Semantik F1**'dir, bu yüzden DSPy'den bir `SemanticF1` metriği yükleyelim. Bu metrik aslında, üzerinde çalıştığımız herhangi bir LM'yi kullanan **çok basit bir DSPy modülü** olarak uygulanmıştır.

```python
from dspy.evaluate import SemanticF1

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")
```

```
Question: 	 why are my text messages coming up as maybe?

Gold Response: 	 This is part of the Proactivity features new with iOS 9: It looks at info in emails to see if anyone with this number sent you an email and if it finds the phone number associated with a contact from your email, it will show you "Maybe". 

However, it has been suggested there is a bug in iOS 11.2 that can result in "Maybe" being displayed even when "Find Contacts in Other Apps" is disabled.

Predicted Response: 	 Your text messages are showing up as "maybe" because your messaging app is uncertain about the sender's identity. This typically occurs when the sender's number is not saved in your contacts or if the message is from an unknown number. To resolve this, you can save the contact in your address book or check the message settings in your app.

Semantic F1 Score: 0.33
```

Yukarıdaki son DSPy modülü çağrısı aslında `metric` (metrik) içerisinde gerçekleşir. Bu örnek için semantik F1'in nasıl ölçüldüğünü merak ediyor olabilirsiniz.

```python
dspy.inspect_history(n=1)
```


```
[2024-11-23T23:16:36.149518]

System message:

Your input fields are:
1. `question` (str)
2. `ground_truth` (str)
3. `system_response` (str)

Your output fields are:
1. `reasoning` (str)
2. `ground_truth_key_ideas` (str): enumeration of key ideas in the ground truth
3. `system_response_key_ideas` (str): enumeration of key ideas in the system response
4. `discussion` (str): discussion of the overlap between ground truth and system response
5. `recall` (float): fraction (out of 1.0) of ground truth covered by the system response
6. `precision` (float): fraction (out of 1.0) of system response covered by the ground truth

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## ground_truth ## ]]
{ground_truth}

[[ ## system_response ## ]]
{system_response}

[[ ## reasoning ## ]]
{reasoning}

[[ ## ground_truth_key_ideas ## ]]
{ground_truth_key_ideas}

[[ ## system_response_key_ideas ## ]]
{system_response_key_ideas}

[[ ## discussion ## ]]
{discussion}

[[ ## recall ## ]]
{recall}        # note: the value you produce must be a single float value

[[ ## precision ## ]]
{precision}        # note: the value you produce must be a single float value

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Compare a system's response to the ground truth to compute recall and precision of key ideas.
        You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision.


User message:

[[ ## question ## ]]
why are my text messages coming up as maybe?

[[ ## ground_truth ## ]]
This is part of the Proactivity features new with iOS 9: It looks at info in emails to see if anyone with this number sent you an email and if it finds the phone number associated with a contact from your email, it will show you "Maybe". 

However, it has been suggested there is a bug in iOS 11.2 that can result in "Maybe" being displayed even when "Find Contacts in Other Apps" is disabled.

[[ ## system_response ## ]]
Your text messages are showing up as "maybe" because your messaging app is uncertain about the sender's identity. This typically occurs when the sender's number is not saved in your contacts or if the message is from an unknown number. To resolve this, you can save the contact in your address book or check the message settings in your app.

Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## ground_truth_key_ideas ## ]]`, then `[[ ## system_response_key_ideas ## ]]`, then `[[ ## discussion ## ]]`, then `[[ ## recall ## ]]` (must be formatted as a valid Python float), then `[[ ## precision ## ]]` (must be formatted as a valid Python float), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## reasoning ## ]]
The system response provides a general explanation for why text messages may show up as "maybe," focusing on the uncertainty of the sender's identity. However, it does not mention the specific iOS features or the potential bug in iOS 11.2 that are highlighted in the ground truth. The ground truth explains that this feature is part of iOS 9's Proactivity features and mentions a specific bug in iOS 11.2, which is not addressed in the system response.

[[ ## ground_truth_key_ideas ## ]]
1. Proactivity features introduced in iOS 9.
2. The system checks emails to associate phone numbers with contacts.
3. "Maybe" is displayed if the number is not saved in contacts.
4. Mention of a bug in iOS 11.2 causing "Maybe" to appear incorrectly.

[[ ## system_response_key_ideas ## ]]
1. Text messages show up as "maybe" due to uncertainty about the sender's identity.
2. Occurs when the sender's number is not saved in contacts or is from an unknown number.
3. Suggests saving the contact or checking message settings.

[[ ## discussion ## ]]
There is some overlap between the ground truth and the system response regarding the uncertainty of the sender's identity and the suggestion to save the contact. However, the system response lacks specific details about the iOS features and the bug mentioned in the ground truth. The ground truth provides a more comprehensive explanation of the "maybe" feature, while the system response is more general and does not address the iOS version specifics.

[[ ## recall ## ]]
0.25

[[ ## precision ## ]]
0.5

[[ ## completed ## ]]


```
Değerlendirme için yukarıdaki metriği basit bir döngü içinde kullanabilir ve puanların ortalamasını alabilirsiniz. Ancak düzgün bir paralellik ve yardımcı araçlar için `dspy.Evaluate` yapısına güvenebiliriz.

```python
# Define an evaluator that we can re-use.
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                         display_progress=True, display_table=2)

# Evaluate the Chain-of-Thought program.
evaluate(cot)
```

```
Average Metric: 125.68 / 300 (41.9%): 100%|██████████| 300/300 [00:00<00:00, 666.96it/s]
2024/11/23 23:16:36 INFO dspy.evaluate.evaluate: Average Metric: 125.68228336477591 / 300 (41.9%)
```
Şu ana kadar soru-cevaplama için çok basit bir zincirleme düşünce (chain-of-thought) modülü oluşturduk ve bunu küçük bir veri kümesi üzerinde değerlendirdik.

Daha iyisini yapabilir miyiz? Bu kılavuzun geri kalanında, aynı görev için DSPy'de bir bilgi geri getirme destekli nesil (RAG) programı inşa edeceğiz. Bunun puanı nasıl önemli ölçüde artırabileceğini göreceğiz; ardından RAG programımızı daha yüksek kaliteli istemlere (prompts) **derlemek (compile)** için DSPy Optimize Edicilerinden birini kullanacağız ve puanlarımızı daha da yükselteceğiz.



### Temel Bilgi Geri Getirme Destekli Nesil (RAG)

İlk olarak, RAG araması için kullanacağımız derlem (corpus) verilerini indirelim. Bu eğitimin eski bir versiyonu tam derlemi (650.000 belge) kullanıyordu. Çalıştırmayı çok hızlı ve ucuz hale getirmek için derlemi sadece 28.000 belgeye indirdik.

```python
download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")
```
### Sistemin Geri Getiricisini (Retriever) Kurun

DSPy açısından, araçları veya geri getiricileri çağırmak için herhangi bir Python kodunu sisteme dahil edebilirsiniz. Burada, kolaylık sağlaması açısından sadece OpenAI Gömmeleri (Embeddings) kullanacağız ve yerel olarak en yakın K (top-K) araması yapacağız.



**Not:** Aşağıdaki adım, faiss kütüphanesinden kaçınmak için ya `pip install -U faiss-cpu` komutunu çalıştırmanızı ya da `dspy.retrievers.Embeddings` modülüne `brute_force_threshold=30_000` parametresini geçmenizi gerektirecektir.

```python
max_characters = 6000  # for truncating >99th percentile of documents
topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [orjson.loads(line)['text'][:max_characters] for line in f]
    print(f"Loaded {len(corpus)} documents. Will encode them below.")

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)
```
### İlk RAG Modülünüzü İnşa Edin

Önceki kılavuzda, `dspy.Predict("question -> answer")` gibi tekil DSPy modüllerine yalıtılmış olarak bakmıştık.

Peki ya birden fazla adımdan oluşan bir DSPy **programı** inşa etmek istersek? Aşağıdaki `dspy.Module` söz dizimi, birkaç parçayı birbirine bağlamanıza olanak tanır; bu durumda, tüm sistemin optimize edilebilmesi için geri getiricimizi (retriever) ve bir üretim modülümüzü bağlayacağız.



Somut olarak, `__init__` metodunda ihtiyaç duyacağınız tüm alt modülleri bildirirsiniz; bu örnekte bu, sadece geri getirilen bağlamı ve bir soruyu alıp bir yanıt üreten `dspy.ChainOfThought('context, question -> response')` modülüdür. `forward` metodunda ise, modüllerinizi kullanarak istediğiniz herhangi bir Python kontrol akışını basitçe ifade edersiniz. Bu durumda, önce daha önce tanımlanan `search` fonksiyonunu çağırıyoruz ve ardından `self.respond` ChainOfThought modülünü tetikliyoruz.

```python
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```


```python
rag = RAG()
rag(question="what are high memory and low memory on linux?")
```


```python
dspy.inspect_history()
```

Daha önce bir CoT (Zincirleme Düşünce) modülü ile geliştirme setimizde (devset) semantik F1 açısından %40 civarında bir sonuç elde etmiştik. Peki, bu **RAG** modülü daha iyi bir puan alabilir mi?


```python
evaluate(RAG())
```

```
Average Metric: 166.54 / 300 (55.5%): 100%|██████████| 300/300 [00:04<00:00, 61.40it/s] 
2024/11/23 23:16:54 INFO dspy.evaluate.evaluate: Average Metric: 166.53601368289284 / 300 (55.5%)
55.51
```
## RAG İstemini İyileştirmek İçin Bir DSPy Optimize Edici Kullanma

Hazır haliyle `RAG` modülümüz %55 puan alıyor. Onu daha güçlü hale getirmek için seçeneklerimiz nelerdir? DSPy'nin sunduğu çeşitli seçeneklerden biri, boru hattımızdaki (pipeline) istemleri (prompts) optimize etmektir.



Eğer programınızda birçok alt modül varsa, bunların hepsi birlikte optimize edilecektir. Bu durumda sadece bir tane var: `self.respond = dspy.ChainOfThought('context, question -> response')`.

DSPy'nin MIPRO (v2) optimize edicisini kuralım ve kullanalım. Aşağıdaki çalıştırmanın maliyeti yaklaşık 1.5$ (orta seviye otomatik ayar için) civarındadır ve iş parçacığı sayınıza bağlı olarak yaklaşık 20-30 dakika sürebilir.

```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small

optimized_rag = tp.compile(RAG(), trainset=trainset,
                           max_bootstrapped_demos=2, max_labeled_demos=2)
```

Buradaki istem optimizasyonu süreci oldukça sistematiktir; süreç hakkında daha fazla bilgi edinmek için örneğin bu makaleyi inceleyebilirsiniz. Önemli bir nokta, bunun sihirli bir düğme olmadığıdır. Örneğin, eğitim setinize aşırı uyum (overfitting) sağlaması ve ayrılmış bir test setine iyi genellenememesi oldukça olasıdır. Bu durum, programlarımızı yinelemeli olarak doğrulamamızı (validation) zorunlu kılar.

Şimdi bir örnek üzerinden kontrol edelim: Aynı soruyu hem optimize edilmemiş temel rag = RAG() programına hem de istem optimizasyonundan sonraki optimized_rag = MIPROv2(..)(..) programına soralım.

```python
baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
print(baseline.response)
```

```python
pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
print(pred.response)
```
Optimizasyon **öncesi** ve **sonrası** RAG istemini görüntülemek için `dspy.inspect_history(n=2)` komutunu kullanabilirsiniz.

Somut olarak, bu not defterinin çalıştırılmalarından birinde, optimize edilmiş istem aşağıdakileri yapar (daha sonraki bir yeniden çalıştırmada farklılık gösterebileceğini unutmayın).

1- Aşağıdaki talimatı oluşturur:



> Sağlanan `context` ve `question` alanlarını kullanarak, kapsamlı ve bilgilendirici bir `response` oluşturmak için bilgileri adım adım analiz edin. Yanıtın ilgili kavramları net bir şekilde açıkladığından, temel ayrımları vurguladığından ve bağlamda belirtilen karmaşıklıkları ele aldığından emin olun.

2- Ve sentetik akıl yürütme ile yanıtlar içeren, tam olarak işlenmiş iki RAG örneği ekler; örneğin: *whatsapp sesli mesajı bilgisayara nasıl aktarılır?*.

Şimdi tüm geliştirme seti (devset) üzerinde değerlendirme yapalım.

```python
evaluate(optimized_rag)
```

```
Average Metric: 183.32 / 300 (61.1%): 100%|██████████| 300/300 [00:02<00:00, 104.48it/s]
2024/11/23 23:17:21 INFO dspy.evaluate.evaluate: Average Metric: 183.3194433591069 / 300 (61.1%)
61.11
```

### Maliyete Göz Kulak Olmak

DSPy, çağrılarınızın maliyetini izlemek için kullanılabilecek programlarınızın maliyetini takip etmenize olanak tanır. Burada, DSPy ile programlarınızın maliyetini nasıl takip edeceğinizi göstereceğiz.


```python
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # in USD, as calculated by LiteLLM for certain providers
```
### Kaydetme ve Yükleme

Optimize edilmiş programın içinde oldukça basit bir yapısı vardır. Onu keşfetmekten çekinmeyin.
Burada, `optimized_rag`'ı kaydedeceğiz, böylece daha sonra sıfırdan optimize etmek zorunda kalmadan tekrar yükleyebiliriz.

```python
optimized_rag.save("optimized_rag.json")

loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")

loaded_rag(question="cmd+tab does not work on hidden or minimized windows")
```

### Sırada Ne Var?

Bu görevde, `SemanticF1` açısından yaklaşık %42'den yaklaşık %61'e çıkmak oldukça kolaydı. Ancak DSPy, sisteminizin kalitesi üzerinde yineleme yapmaya devam etmeniz için size yollar sunar ve biz henüz sadece yüzeyi çizdik.

Genel olarak şu araçlara sahipsiniz:

* **Daha iyi sistem mimarilerini keşfedin:** Örneğin, LM'den geri getirici (retriever) için arama sorguları oluşturmasını istersek ne olur? Örn: DSPy ile oluşturulan **STORM boru hattını** inceleyin.
* **Farklı istem (prompt) veya ağırlık optimize edicileri keşfedin:** Optimize Edici Belgelerine (Optimizers Docs) bakın.
* **Çıkarım zamanı hesaplamasını ölçeklendirin:** Birden fazla optimizasyon sonrası programı birleştirerek (ensembling) DSPy Optimize Edicilerini kullanın.
* **Maliyeti düşürün:** İstem veya ağırlık optimizasyonu yoluyla daha küçük bir LM'ye damıtma (distillation) yapın.



**Hangisiyle devam edeceğinize nasıl karar verirsiniz?**

İlk adım, sistem çıktılarınıza bakmaktır; bu, varsa düşük performansın kaynaklarını belirlemenize olanak tanır. Tüm bunları yaparken, metriğinizi iyileştirmeye devam ettiğinizden (örneğin, kendi yargılarınıza göre optimize ederek) ve daha fazla (veya daha gerçekçi) veri topladığınızdan (ilgili alanlardan veya sisteminizin bir demosunu kullanıcıların önüne koyarak) emin olun.