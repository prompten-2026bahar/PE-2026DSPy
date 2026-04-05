# /translation/streamify.md

## dspy.streamify

```python
dspy.streamify(program: Module, status_message_provider: StatusMessageProvider | None = None, stream_listeners: list[StreamListener] | None = None, include_final_prediction_in_output_stream: bool = True, is_async_program: bool = False, async_streaming: bool = True) -> Callable[[Any, Any], Awaitable[Any]]
```

Bir DSPy programını, çıktılarını tek seferde döndürmek yerine artımlı (incremental) olarak akış (stream) şeklinde sunacak şekilde sarmalar (wraps).



Ayrıca, programın ilerlemesini belirtmek için kullanıcıya durum mesajları sağlar. Kullanıcılar durum mesajlarını ve hangi modül için durum mesajı üretileceğini özelleştirmek üzere kendi durum mesajı sağlayıcılarını (status message provider) uygulayabilirler.

**Parametreler:**

| İsim (Name) | Tip (Type) | Açıklama (Description) | Varsayılan (Default) |
| :--- | :--- | :--- | :--- |
| `program` | `Module` | Akış işlevselliği ile sarmalanacak DSPy programı. | **Gerekli (required)** |
| `status_message_provider` | `StatusMessageProvider \| None` | Varsayılanın yerine kullanılacak özel bir durum mesajı oluşturucu. Kullanıcılar durum mesajlarını ve hangi modül için durum mesajı üretileceğini özelleştirmek üzere kendi durum mesajı oluşturucularını uygulayabilirler. | `None` |
| `stream_listeners` | `list[StreamListener] \| None` | Programdaki alt tahminlerin (sub predicts) belirli alanlarının akış çıktısını yakalamak için kullanılacak akış dinleyicileri listesi. Sağlandığında, yalnızca hedef tahmindeki hedef alanlar kullanıcıya akış olarak iletilecektir. | `None` |
| `include_final_prediction_in_output_stream` | `bool` | Nihai tahmini (final prediction) çıktı akışına dahil edip etmeyeceği; yalnızca `stream_listeners` sağlandığında yararlıdır. `False` ise, nihai tahmin çıktı akışına dahil edilmez. Program önbelleğe (cache) takıldığında veya hiçbir dinleyici hiçbir şey yakalamadığında, bu değer `False` olsa bile nihai tahmin çıktı akışına dahil edilecektir. | `True` |
| `is_async_program` | `bool` | Programın asenkron olup olmadığı. `False` ise program `asyncify` ile sarmalanır, aksi takdirde program `acall` ile çağrılır. | `False` |
| `async_streaming` | `bool` | Asenkron bir üreteç (async generator) mi yoksa senkron bir üreteç mi döndürüleceği. `False` ise, akış işlemi senkron bir üretece dönüştürülür. | `True` |

**Dönüş Değerleri (Returns):**

| Tip (Type) | Açıklama (Description) |
| :--- | :--- |
| `Callable[[Any, Any], Awaitable[Any]]` | Orijinal programla aynı argümanları alan, ancak programın çıktılarını artımlı olarak üreten (yield) asenkron bir üreteç döndüren bir fonksiyon. |

---

## Örnekler (Examples)

**Temel Kullanım:**

```python
import asyncio
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Programı oluşturun ve akış işlevselliği ile sarmalayın
program = dspy.streamify(dspy.Predict("q->a"))

# Programı akış çıktısıyla kullanın
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
            
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

**Özel durum mesajı sağlayıcısı ile örnek:**

```python
import asyncio
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyStatusMessageProvider(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        return f"Predicting..."
        
    def tool_end_status_message(self, outputs):
        return f"Tool calling finished with output: {outputs}!"

# Programı oluşturun ve akış işlevselliği ile sarmalayın
program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

# Programı akış çıktısıyla kullanın
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
            
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

**Akış dinleyicileri (stream listeners) ile örnek:**

```python
import asyncio
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

# Programı oluşturun ve akış işlevselliği ile sarmalayın
predict = dspy.Predict("question->answer, reasoning")

stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]

stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

async def use_streaming():
    output = stream_predict(
        question="why did a chicken cross the kitchen?",
        include_final_prediction_in_output_stream=False,
    )
    
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
            
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

Konsol çıktısında akış parçalarını (streaming chunks) `dspy.streaming.StreamResponse` formatında görmelisiniz.

*Kaynak kod: `.venv/lib/python3.14/site-packages/dspy/streaming/streamify.py`*
"""