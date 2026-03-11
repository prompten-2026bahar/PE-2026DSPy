## DSPy Modüllerini Özelleştirerek Yapay Zeka Uygulamaları İnşa Etmek

Bu kılavuzda, `dspy.Module` yapısını özelleştirerek bir GenAI uygulamasının nasıl inşa edileceği konusunda size rehberlik edeceğiz.

Bir **DSPy modülü**, DSPy programlarının temel yapı taşıdır.

* Her yerleşik modül, bir istem tekniğini (zincirleme düşünce veya ReAct gibi) soyutlar. En önemli özellikleri, herhangi bir imzayı (signature) işleyebilecek şekilde genelleştirilmiş olmalarıdır.
* Bir DSPy modülü, öğrenilebilir parametrelere (yani istemi oluşturan küçük parçalar ve LM ağırlıkları) sahiptir; girdileri işlemek ve çıktıları döndürmek için çağrılabilir.
* Birden fazla modül, daha büyük modüller (programlar) oluşturacak şekilde birleştirilebilir. DSPy modülleri doğrudan PyTorch'taki NN modüllerinden esinlenmiştir, ancak LM programlarına uygulanmıştır.


Özel bir modül uygulamadan da bir DSPy programı oluşturabilseniz de, mantığınızı özel bir modül içine yerleştirmenizi şiddetle tavsiye ederiz. Bu sayede DSPy optimizer veya MLflow DSPy tracing gibi diğer DSPy özelliklerini kullanabilirsiniz.

Başlamadan önce DSPy'nin yüklü olduğundan emin olun:

```python
!pip install dspy
```

## DSPy Modülünü Özelleştirme

DSPy modülünü özelleştirerek özel istem (prompting) mantığı uygulayabilir ve harici araçları veya servisleri entegre edebilirsiniz. Bunu başarmak için `dspy.Module` sınıfından alt sınıf türetin ve şu iki ana yöntemi uygulayın:

* **__init__**: Bu, programınızın özniteliklerini ve alt modüllerini tanımladığınız kurucu (constructor) metodudur.
* **forward**: Bu metod, DSPy programınızın temel mantığını içerir.

`forward()` metodu içinde sadece diğer DSPy modüllerini çağırmakla sınırlı değilsiniz; Langchain/Agno ajanları, MCP araçları, veritabanı işleyicileri ve daha fazlasıyla etkileşim kurmak için kullanılan standart Python fonksiyonlarını da entegre edebilirsiniz.

Özel bir DSPy modülü için temel yapı şu şekildedir:

```python
class MyProgram(dspy.Module):
    
    def __init__(self, ...):
        # Define attributes and sub-modules here
        {constructor_code}

    def forward(self, input_name1, input_name2, ...):
        # Implement your program's logic here
        {custom_logic_code}
```

Bunu pratik bir kod örneğiyle somutlaştıralım. Birkaç aşamadan oluşan basit bir Bilgi Geri Getirme Destekli Nesil (RAG) uygulaması inşa edeceğiz:

1. **Sorgu Oluşturma (Query Generation):** İlgili bağlamı geri getirmek için kullanıcının sorusuna dayalı uygun bir sorgu oluşturun.
2. **Bağlam Geri Getirme (Context Retrieval):** Oluşturulan sorguyu kullanarak bağlamı getirin.
3. **Yanıt Oluşturma (Answer Generation):** Geri getirilen bağlam ve orijinal soruya dayanarak nihai bir yanıt üretin.

Bu çok aşamalı programın kod uygulaması aşağıda gösterilmiştir:


```python
import dspy

class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=1)
    return [x["text"] for x in results]

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("question,context->answer")

    def forward(self, question, **kwargs):
        query = self.query_generator(question=question).query
        context = search_wikipedia(query)[0]
        return self.answer_generator(question=question, context=context).answer
```

`forward` metoduna bir göz atalım. Önce soruyu, bağlamı geri getirmek için sorguyu almak üzere bir `dspy.Predict` olan `self.query_generator`'a gönderiyoruz. Ardından, ColBERT'i çağırmak için bu sorguyu kullanıyoruz ve geri getirilen ilk bağlamı saklıyoruz. Son olarak, nihai yanıtı oluşturmak için soruyu ve bağlamı bir `dspy.ChainOfThought` olan `self.answer_generator`'a gönderiyoruz.



Ardından, programı çalıştırmak için `RAG` modülümüzün bir örneğini oluşturacağız.

**Önemli:** Özel bir DSPy modülünü çağırırken, `forward()` metodunu açıkça çağırmak yerine doğrudan modül örneğini kullanmalısınız (bu, dahili olarak `__call__` metodunu çağırır). `__call__` metodu, `forward` mantığını yürütmeden önce gerekli dahili işlemleri yönetir.

```python
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
rag = RAG()
print(rag(question="Is Lebron James the basketball GOAT?"))
```

```
The question of whether LeBron James is the basketball GOAT is subjective and depends on personal opinions. Many consider him one of the greatest due to his achievements and impact on the game, but others may argue for different players like Michael Jordan.
```

İşte bu kadar! Özetle, GenAI uygulamalarınızı oluşturmak için özel mantığı `forward()` metoduna yerleştiriyoruz, ardından bir modül örneği oluşturup bu örneği doğrudan çağırıyoruz.

### Neden Modül Özelleştirilir?

DSPy, hafif bir yazma ve optimizasyon çerçevesidir; odak noktamız, sağlam yapay zeka sistemleri için istem mühendisliği (prompt engineering) karmaşasını, istem tabanlı (metin gir, metin al) LLM kullanımından programlama tabanlı (yapılandırılmış girdi gir, yapılandırılmış çıktı al) LLM kullanımına dönüştürerek çözmektir.



Yapay zeka uygulamalarınızı oluşturmayı kolaylaştırmak için akıl yürütme için `dspy.ChainOfThought` veya araç çağıran ajanlar için `dspy.ReAct` gibi özel istem mantığına sahip önceden oluşturulmuş modüller sunsak da, ajanları nasıl oluşturduğunuzu standartlaştırmayı hedeflemiyoruz.

DSPy'de, uygulama mantığınız sadece özel Modülünüzün `forward` metoduna gider ve Python kodu yazdığınız sürece herhangi bir kısıtlama yoktur. Bu düzen sayesinde, DSPy'ye diğer çerçevelerden veya yalın SDK kullanımından geçiş yapmak kolaydır ve özünde sadece Python kodu olduğu için DSPy'den ayrılmak da kolaydır.