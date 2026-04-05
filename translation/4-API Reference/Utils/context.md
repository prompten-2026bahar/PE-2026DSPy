# /translation/context.md

## dspy.context

Tek bir `with` bloğu içinde DSPy ayarlarını geçersiz kılın (override).

```python
with dspy.context(**kwargs):
    ...  # geçersiz kılmalar burada aktiftir
# orijinal ayarlar geri yüklenir
```

Programınızın bir bölümü için işlem genelindeki (process-wide) varsayılanları `dspy.configure`'dan değiştirmeden farklı bir LM, adaptör veya bayrak (flag) kullanmanız gerektiğinde `dspy.context(...)` kullanın. Blok, mevcut her ayarı devralır, yalnızca aktardığınız anahtarları geçersiz kılar ve çıkışta orijinallerini geri yükler.

`dspy.context(...)`, `dspy.configure` ile aynı ayarları kabul eder.

---

## Örnekler (Examples)

### Tek bir blok için farklı bir LM kullanma

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print("default:", result.answer)

with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-6")):
    result = qa(question="What is the capital of France?")
    print("temporary:", result.answer)

print("restored:", dspy.settings.lm.model)  # openai/gpt-5-mini
```

### Tek bir blok için farklı bir adaptör kullanma

```python
import dspy

dspy.configure(lm=dspy.LM("gemini/gemini-3-flash-preview"))

qa = dspy.Predict("question -> answer")

with dspy.context(adapter=dspy.JSONAdapter()):
    result = qa(question="What is the capital of France?")
    print(result.answer)
```

### Asenkron araç dönüştürmeyi geçici olarak etkinleştirme

```python
import asyncio
import dspy

async def async_tool(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 2

tool = dspy.Tool(async_tool)

with dspy.context(allow_tool_async_sync_conversion=True):
    print(tool(x=5))
```

### İç içe geçmiş bloklar (Nested blocks)

İçteki bloklar dıştakileri geçersiz kılar. Her biri çıkışta temiz bir şekilde geri yüklenir:

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-5-mini"))

with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-6")):
    print(dspy.settings.lm.model)        # anthropic/claude-sonnet-4-6
    
    with dspy.context(track_usage=True):
        print(dspy.settings.lm.model)    # anthropic/claude-sonnet-4-6 (devralındı/inherited)
        print(dspy.settings.track_usage) # True
        
    print(dspy.settings.track_usage)     # False (geri yüklendi/restored)

print(dspy.settings.lm.model)            # openai/gpt-5-mini (geri yüklendi/restored)
```

---

## İş Parçacığı Güvenliği (Thread safety)

`dspy.configure`'un aksine, `dspy.context(...)` fonksiyonunu *herhangi* bir iş parçacığından veya asenkron görevden (async task) çağırabilirsiniz. Bu, onu `dspy.Parallel`, `asyncio.gather` veya herhangi bir eşzamanlı (concurrent) kod içindeki geçersiz kılmalar için doğru araç yapar.

Bir `dspy.context(...)` bloğu içindeki ayarlar diğer iş parçacıklarına veya görevlere sızmaz.

---

## Ayrıca Bakınız (See Also)
* **`dspy.configure`** — işlem genelindeki varsayılanları ayarlar.  https://dspy.ai/api/utils/configure/
* **`dspy.LM`** — `lm` olarak aktaracağınız dil modelini oluşturur.  https://dspy.ai/api/models/LM/
* **Dil Modelleri (Language Models)** — LM yapılandırmasına genel bakış.  https://dspy.ai/learn/programming/language_models/
* **Adaptörler (Adapters)** — adaptörlerin istemleri nasıl formatladığı ve yanıtları nasıl ayrıştırdığı.  https://dspy.ai/learn/programming/adapters/
