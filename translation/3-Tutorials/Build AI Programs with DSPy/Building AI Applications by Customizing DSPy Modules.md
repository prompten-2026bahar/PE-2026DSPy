# DSPy Modüllerini Özelleştirerek Yapay Zeka Uygulamaları Geliştirme (Building AI Applications by Customizing DSPy Modules)

Bir DSPy modülü, DSPy programları için temel yapı taşıdır.

* Her yerleşik (built-in) modül, bir istem (prompting) tekniğini (örneğin chain of thought veya ReAct) soyutlar. En önemlisi, herhangi bir imzayı (signature) işleyebilecek şekilde genelleştirilmişlerdir.
* Bir DSPy modülü öğrenilebilir parametrelere (yani istemi ve dil modeli ağırlıklarını oluşturan küçük parçalara) sahiptir ve girdileri işleyip çıktıları döndürmek üzere çağrılabilir (invoke/call).
* Birden fazla modül, daha büyük modüllere (programlara) dönüştürülmek üzere bir araya getirilebilir. DSPy modülleri doğrudan PyTorch'taki sinir ağı (NN) modüllerinden ilham almış, ancak dil modeli (LM) programlarına uyarlanmıştır.

Özel bir modül uygulamadan da bir DSPy programı oluşturabilseniz de, mantığınızı özel bir modül içine koymanızı şiddetle tavsiye ederiz; böylece DSPy optimizer veya MLflow DSPy tracing gibi diğer DSPy özelliklerini kullanabilirsiniz.

MLflow, DSPy ile yerel olarak entegre olan, açıklanabilirlik ve deney takibi sunan bir LLMOps aracıdır. Bu eğitimde, DSPy'nin davranışını daha iyi anlamak amacıyla istemleri (prompts) ve optimizasyon ilerlemesini izler (traces) olarak görselleştirmek için MLflow'u kullanabilirsiniz. Aşağıdaki dört adımı izleyerek MLflow'u kolayca kurabilirsiniz.

**MLflow Trace**

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
4. İzlemeyi (tracing) etkinleştirin.
```python
mlflow.dspy.autolog()
```
Entegrasyon hakkında daha fazla bilgi edinmek için MLflow DSPy Dokümantasyonunu da ziyaret edebilirsiniz.

## DSPy Modülünü Özelleştirme (Customize DSPy Module)

Özel bir istem (prompting) mantığı uygulayabilir ve bir DSPy modülünü özelleştirerek harici araçları veya hizmetleri entegre edebilirsiniz. Bunu başarmak için, `dspy.Module` sınıfından alt sınıf (subclass) oluşturun ve aşağıdaki iki temel metodu uygulayın:

* `__init__` : Bu, programınızın niteliklerini ve alt modüllerini tanımladığınız yapıcıdır (constructor).
* `forward` : Bu metod, DSPy programınızın çekirdek mantığını içerir.

`forward()` metodu içerisinde sadece diğer DSPy modüllerini çağırmakla sınırlı değilsiniz; Langchain/Agno ajanları, MCP araçları, veritabanı işleyicileri ve daha fazlasıyla etkileşim kurmak için olanlar gibi herhangi bir standart Python fonksiyonunu da entegre edebilirsiniz.

Özel bir DSPy modülünün temel yapısı şuna benzer:

```python
class MyProgram(dspy.Module):
    def __init__(self, ...):
        # Define attributes and sub-modules here
        {constructor_code}

    def forward(self, input_name1, input_name2, ...):
        # Implement your program's logic here
        {custom_logic_code}
```

Bunu pratik bir kod örneğiyle gösterelim. Çoklu aşamalara sahip basit bir Retrieval-Augmented Generation (RAG) uygulaması oluşturacağız:

* **Query Generation (Sorgu Üretimi):** İlgili bağlamı (context) getirmek için kullanıcının sorusuna dayalı uygun bir sorgu üretir.
* **Context Retrieval (Bağlam Getirme):** Üretilen sorguyu kullanarak bağlamı getirir.
* **Answer Generation (Cevap Üretimi):** Getirilen bağlama ve orijinal soruya dayalı olarak nihai bir cevap üretir.

Bu çok aşamalı programın kod uygulaması aşağıda gösterilmiştir.

```python
import dspy

class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    results = dspy.ColBERTv2(url='[http://20.102.90.50:2017/wiki17_abstracts')(query](http://20.102.90.50:2017/wiki17_abstracts')(query), k=1)
    return [x["text"] for x in results]

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        query = self.query_generator(question=question).query
        context = search_wikipedia(query)
        answer = self.answer_generator(context=context, question=question).answer
        return dspy.Prediction(answer=answer)
```

Gelin `forward` metoduna bir göz atalım. Önce soruyu, bağlam getirme sorgusunu elde etmek için bir `dspy.Predict` olan `self.query_generator`'a gönderiyoruz. Ardından ColBERT'i çağırmak için sorguyu kullanıyor ve getirilen ilk bağlamı tutuyoruz. Son olarak, nihai cevabı üretmek için soruyu ve bağlamı, bir `dspy.ChainOfThought` olan `self.answer_generator`'a gönderiyoruz.

Sırada, programı çalıştırmak için RAG modülümüzün bir örneğini (instance) oluşturmak var.

**Önemli:** Özel bir DSPy modülünü çağırırken, `forward()` metodunu açıkça çağırmak yerine doğrudan modül örneğini kullanmalısınız (bu, arka planda `__call__` metodunu çağırır). `__call__` metodu, `forward` mantığını yürütmeden önce gerekli iç işlemleri halleder.

```python
import os

os.environ["OPENAI_API_KEY"] = "{kendi_openai_api_anahtariniz}"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

rag = RAG()
response = rag(question="Who won the 2022 FIFA World Cup?")
print(response.answer)
```

Özetle, GenAI uygulamalarınızı oluşturmak için özel mantığı `forward()` metoduna koyar, ardından bir modül örneği oluşturur ve örneğin kendisini çağırırız.

## Neden Modülü Özelleştirmeliyiz? (Why Customizing Module?)

DSPy hafif bir oluşturma (authoring) ve optimizasyon framework'üdür ve odak noktamız, dayanıklı yapay zeka sistemleri için istemleyen (string içeri, string dışarı) LLM'leri programlayan (yapılandırılmış girdiler içeri, yapılandırılmış çıktılar dışarı) LLM'lere dönüştürerek istem mühendisliğinin (prompt engineering) yarattığı karmaşayı çözmektir. Akıl yürütme için `dspy.ChainOfThought` ve AI uygulamalarınızı oluşturmanızı kolaylaştıracak araç çağırma ajanı (tool calling agent) için `dspy.ReAct` gibi özel istem mantıklarına sahip önceden oluşturulmuş modüller sunsak da, ajanları nasıl inşa edeceğinizi standartlaştırmayı hedeflemiyoruz.

DSPy'de, uygulamanızın mantığı basitçe özel Modülünüzün (Module) `forward` metoduna girer ve Python kodu yazdığınız sürece herhangi bir kısıtlaması yoktur. Bu düzenle, DSPy'ye diğer framework'lerden veya standart SDK kullanımından geçiş yapmak çok kolaydır ve DSPy'den ayrılmak da kolaydır çünkü temelde bu sadece Python kodudur.