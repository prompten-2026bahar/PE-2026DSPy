# Streaming

Bu kılavuzda, DSPy programınızda streaming’i nasıl etkinleştireceğinizi adım adım inceleyeceğiz. DSPy Streaming iki bölümden oluşur:

- **Çıktı Token Streaming**: Tam yanıtı beklemek yerine, üretildikleri anda tek tek token’ları akıtın.
- **Ara Durum Streaming**: Programın yürütme durumu hakkında gerçek zamanlı güncellemeler sağlayın (ör. "Web araması çağrılıyor...", "Sonuçlar işleniyor...").

## Çıktı Token Streaming

DSPy’nin token streaming özelliği, boru hattınızdaki yalnızca son çıktı ile sınırlı değildir; herhangi bir modülle çalışır. Tek gereklilik, stream edilecek alanın `str` türünde olmasıdır. Token streaming’i etkinleştirmek için:

1. Programınızı `dspy.streamify` ile sarın
2. Hangi alanların stream edileceğini belirtmek için bir veya daha fazla `dspy.streaming.StreamListener` nesnesi oluşturun

İşte temel bir örnek:

```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# 'answer' alanı için streaming'i etkinleştir
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
```

Stream edilen çıktıyı tüketmek için:

```python
import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```

Bu, aşağıdakine benzer bir çıktı üretir:

```
StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' other')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' side of the frying pan!')
Prediction(
    answer='To get to the other side of the frying pan!'
)
```

Not: `dspy.streamify` async generator döndürdüğü için bunu async bağlam içinde kullanmanız gerekir. Jupyter veya Google Colab gibi zaten bir event loop’a (async bağlama) sahip ortamlarda generator’ı doğrudan kullanabilirsiniz.

Yukarıdaki streaming içinde iki farklı varlık olduğunu fark etmiş olabilirsiniz: `StreamResponse` ve `Prediction.` `StreamResponse`, dinlenen alandaki stream edilen token’lar için sarmalayıcıdır; bu örnekte bu alan `answer` alanıdır. `Prediction` ise programın nihai çıktısıdır. DSPy’de streaming, yan kanal (sidecar) tarzında uygulanır: LM üzerinde streaming etkinleştirilir, böylece LM token akışı üretir. Bu token’lar, kullanıcı tanımlı listener’lar tarafından sürekli okunan bir yan kanala gönderilir. Listener’lar akışı yorumlamaya devam eder ve dinledikleri `signature_field_name` alanının görünmeye başlayıp tamamlanıp tamamlanmadığına karar verir. Alanın görünmeye başladığına karar verdiklerinde, kullanıcıların okuyabildiği async generator’a token’ları vermeye başlarlar. Listener’ların iç mekanizması arka plandaki adapter’a göre değişir ve genellikle bir alanın tamamlandığına ancak bir sonraki alanı gördükten sonra karar verebildiğimiz için, listener son generator’a göndermeden önce çıktı token’larını tamponlar; bu yüzden çoğu zaman `StreamResponse` türündeki son parçanın birden fazla token içerdiğini görürsünüz. Programın çıktısı da akışa yazılır; bu da yukarıdaki örnek çıktıda görülen `Prediction` parçasıdır.

Bu farklı türleri ele almak ve özel mantık uygulamak için:

```python
import asyncio

async def read_output_stream():
  output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

  return_value = None
  async for chunk in output_stream:
    if isinstance(chunk, dspy.streaming.StreamResponse):
      print(f"Field {chunk.signature_field_name} için çıktı token’ı: {chunk.chunk}")
    elif isinstance(chunk, dspy.Prediction):
      return_value = chunk
  return return_value


program_output = asyncio.run(read_output_stream())
print("Nihai çıktı: ", program_output)
```

### `StreamResponse` Yapısını Anlama

`StreamResponse` (`dspy.streaming.StreamResponse`), stream edilen token’ların sarmalayıcı sınıfıdır. 3 alan içerir:

- `predict_name`: `signature_field_name` alanını barındıran predict’in adıdır. Bu ad, `your_program.named_predictors()` çalıştırdığınızda gördüğünüz anahtar adlarıyla aynıdır. Yukarıdaki kodda `answer`, `predict` nesnesinin kendisinden geldiği için `predict_name`, `self` olarak görünür; bu da `predict.named_predictors()` çalıştırıldığında gördüğünüz tek anahtardır.
- `signature_field_name`: Bu token’ların eşlendiği çıktı alanıdır. `predict_name` ve `signature_field_name` birlikte alanın benzersiz kimliğini oluşturur. Bu kılavuzun ilerleyen bölümünde birden fazla alanın stream edilmesini ve yinelenen alan adlarının nasıl ele alınacağını göstereceğiz.
- `chunk`: Stream parçasının değeridir.

### Önbellekle Streaming

Önbelleğe alınmış bir sonuç bulunduğunda, akış tek tek token’ları atlar ve yalnızca nihai `Prediction` değerini döndürür. Örneğin:

```
Prediction(
    answer='To get to the other side of the dinner plate!'
)
```

### Birden Fazla Alanı Stream Etme

Her alan için bir `StreamListener` oluşturarak birden fazla alanı izleyebilirsiniz. Aşağıda çok modüllü bir programla örnek verilmiştir:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)

async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value

program_output = asyncio.run(read_output_stream())
print("Nihai çıktı: ", program_output)
```

Çıktı şu şekilde görünür:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk='To')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' reach')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' the')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' other side of the recipe!')
Final output:  Prediction(
    simplified_answer='To reach the other side of the recipe!'
)
```

### Aynı Alanı Birden Fazla Kez Stream Etme (`dspy.ReAct` gibi)

Varsayılan olarak bir `StreamListener`, tek bir streaming oturumunu tamamladıktan sonra kendini otomatik olarak kapatır. Bu tasarım performans sorunlarını önlemeye yardımcı olur; çünkü her token tüm yapılandırılmış stream listener’lara yayınlanır ve çok fazla etkin listener önemli ek yük getirebilir.

Ancak bir DSPy modülünün döngü içinde tekrar tekrar kullanıldığı durumlarda — örneğin `dspy.ReAct` ile — her tahminde aynı alanı, her kullanıldığında stream etmek isteyebilirsiniz. Bu davranışı etkinleştirmek için `StreamListener` oluştururken `allow_reuse=True` ayarlayın. Aşağıdaki örneğe bakın:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


def fetch_user_info(user_name: str):
    """Ad, doğum günü vb. kullanıcı bilgilerini getir."""
    return {
        "name": user_name,
        "birthday": "2009-05-16",
    }


def get_sports_news(year: int):
    """Belirli bir yıl için spor haberlerini getir."""
    if year == 2009:
        return "Usane Bolt broke the world record in the 100m race."
    return None


react = dspy.ReAct("question->answer", tools=[fetch_user_info, get_sports_news])

stream_listeners = [
    # dspy.ReAct'in yerleşik bir çıktı alanı vardır: "next_thought".
    dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
]
stream_react = dspy.streamify(react, stream_listeners=stream_listeners)


async def read_output_stream():
    output = stream_react(question="What sports news happened in the year Adam was born?")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


print(asyncio.run(read_output_stream()))
```

Bu örnekte, `StreamListener` içinde `allow_reuse=True` ayarlayarak `"next_thought"` için streaming’in yalnızca ilk yinelemede değil, her yinelemede kullanılabilir olmasını sağlarsınız. Bu kodu çalıştırdığınızda, `next_thought` alanı her üretildiğinde ona ait stream token’larını görürsünüz.

#### Yinelenen Alan Adlarını Ele Alma

Farklı modüllerden aynı adlı alanları stream ederken, `StreamListener` içinde hem `predict` hem de `predict_name` belirtin:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("question, answer->answer, score")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict1,
        predict_name="predict1"
    ),
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict2,
        predict_name="predict2"
    ),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)


async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


program_output = asyncio.run(read_output_stream())
print("Nihai çıktı: ", program_output)
```

Çıktı şu şekilde olacaktır:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk="I'm")
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' ready')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' assist')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' you')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk='! Please provide a question.')
Final output:  Prediction(
    answer="I'm ready to assist you! Please provide a question.",
    score='N/A'
)
```

## Ara Durum Streaming

Durum streaming, kullanıcıları programın ilerleyişi hakkında bilgilendirir; bu özellikle araç çağrıları veya karmaşık AI boru hatları gibi uzun süren işlemlerde yararlıdır. Durum streaming’i uygulamak için:

1. `dspy.streaming.StatusMessageProvider` sınıfından türeyen özel bir durum mesajı sağlayıcısı oluşturun
2. Özel durum mesajları sağlamak için istediğiniz hook metotlarını override edin
3. Sağlayıcınızı `dspy.streamify` içine geçin

Örnek:

```python
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Girdiler {inputs} ile LM çağrılıyor..."

    def lm_end_status_message(self, outputs):
        return f"Araç şu çıktı ile tamamlandı: {outputs}!"
```

Kullanılabilir hook’lar:

- lm_start_status_message: `dspy.LM` çağrısının başlangıcındaki durum mesajı.
- lm_end_status_message: `dspy.LM` çağrısının sonundaki durum mesajı.
- module_start_status_message: `dspy.Module` çağrısının başlangıcındaki durum mesajı.
- module_end_status_message: `dspy.Module` çağrısının sonundaki durum mesajı.
- tool_start_status_message: `dspy.Tool` çağrısının başlangıcındaki durum mesajı.
- tool_end_status_message: `dspy.Tool` çağrısının sonundaki durum mesajı.

Her hook, durum mesajını içeren bir string döndürmelidir.

Mesaj sağlayıcısını oluşturduktan sonra bunu `dspy.streamify` içine geçirmeniz yeterlidir; böylece hem durum mesajı streaming’ini hem de çıktı token streaming’ini etkinleştirebilirsiniz. Aşağıdaki örneğe bakın. Ara durum mesajı `dspy.streaming.StatusMessage` sınıfıyla temsil edilir; bu nedenle onu yakalamak için ek bir koşul kontrolü yapmamız gerekir.

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.tool = dspy.Tool(lambda x: 2 * x, name="double_the_number")
        self.predict = dspy.ChainOfThought("num1, num2->sum")

    def forward(self, num, **kwargs):
        num2 = self.tool(x=num)
        return self.predict(num1=num, num2=num2)


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def tool_start_status_message(self, instance, inputs):
        return f"Araç {instance.name}, girdiler {inputs} ile çağrılıyor..."

    def tool_end_status_message(self, outputs):
        return f"Araç şu çıktı ile tamamlandı: {outputs}!"


predict = MyModule()
stream_listeners = [
    # dspy.ChainOfThought'in yerleşik bir çıktı alanı vardır: "reasoning".
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
    status_message_provider=MyStatusMessageProvider(),
)


async def read_output_stream():
    output = stream_predict(num=3)

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            print(chunk)
    return return_value


program_output = asyncio.run(read_output_stream())
print("Nihai çıktı: ", program_output)
```

Örnek çıktı:

```
StatusMessage(message='Calling tool double_the_number...')
StatusMessage(message='Tool calling finished! Querying the LLM with tool calling results...')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='To')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' find')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' sum')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' of')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' two')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' numbers')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' we')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' simply')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' add')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' them')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' together')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='.')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' Here')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' ')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='3')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' plus')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 6 equals 9.')
Final output:  Prediction(
    reasoning='To find the sum of the two numbers, we simply add them together. Here, 3 plus 6 equals 9.',
    sum='9'
)
```

## Senkron Streaming

Varsayılan olarak streamify edilmiş bir DSPy programını çağırmak async generator üretir. Senkron generator elde etmek için `async_streaming=False` bayrağını ayarlayabilirsiniz:

```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# 'answer' alanı için streaming'i etkinleştir
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,
)

output = stream_predict(question="why did a chicken cross the kitchen?")

program_output = None
for chunk in output:
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk)
    elif isinstance(chunk, dspy.Prediction):
        program_output = chunk
print(f"Program output: {program_output}")
```
